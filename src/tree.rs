use std::collections::HashMap;
use std::iter;

use burn::prelude::*;
use burn::tensor::cast::ToElement;
use itertools::Itertools;
use rand::prelude::*;
use rand_distr::{Dirichlet, Distribution};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;
use shakmaty::{Chess, Position};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot::channel;

use crate::agent::AlphaZero;
use crate::chess::{board_to_text, index_to_move, move_to_index, play_move, to_tensor, GameResult};
use crate::memory::ChessStateKey;
use crate::parameters::{ACTION_SPACE, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, NUM_SIMULATIONS, TEMPERATURE};
use crate::training::{InferenceRequest, InferenceResult, PriorsCache};

#[derive(Debug, Clone)]
pub struct MCTree {
    pub nodes: HashMap<usize, Box<MCTree>>,
    pub moves: Vec<usize>,
    pub state: Chess,

    pub policy: Box<[f32; ACTION_SPACE]>,
    pub visits: Box<[f32; ACTION_SPACE]>,
    pub scores: Box<[f32; ACTION_SPACE]>
}

impl MCTree {
    pub fn init<B: Backend>(model: &AlphaZero<B>, game: Chess, apply_noise: bool) -> Self {
        let legal_moves = game.legal_moves()
            .iter()
            .map(|m| move_to_index(m, game.turn()))
            .collect::<Vec<_>>();

        let (policy_tensor, _value_tensor) = model.forward(to_tensor(&game));
        let mut policy = policy_tensor.into_data()
            .into_vec()
            .unwrap()
            .try_into()
            .unwrap();

        if apply_noise {
            apply_dirichlet_noise(&mut policy, &legal_moves)
        }

        Self {
            nodes: HashMap::new(),
            moves: legal_moves,
            state: game,

            policy: Box::new(policy),
            visits: Box::new([0.0; ACTION_SPACE]),
            scores: Box::new([0.0; ACTION_SPACE])
        }
    }

    pub async fn async_init(sender: &UnboundedSender<InferenceRequest>, cache: &PriorsCache, game: &Chess, apply_noise: bool) -> Self {
        let cache_reader = cache.read().await;
        let state_key = ChessStateKey::new(game.clone());
        if let Some(entry) = cache_reader.get(&state_key) {
            Self::new(*entry.policy, game, apply_noise)
        }
        else {
            drop(cache_reader);

            let (response_sender, response_receiver) = channel();
            sender.send(InferenceRequest {
                state: game.clone(),
                response_sender: response_sender,
            }).expect("Failed to send request!");

            let InferenceResult { policy, .. } = response_receiver.await.expect("No result provided!");
            Self::new(*policy, game, apply_noise)
        }
    }

    pub fn new(mut policy: [f32; ACTION_SPACE], state: &Chess, apply_noise: bool) -> Self {
        let legal_moves = state.legal_moves()
            .iter()
            .map(|m| move_to_index(m, state.turn()))
            .collect::<Vec<_>>();

        if apply_noise {
            apply_dirichlet_noise(&mut policy, &legal_moves)
        }

        Self {
            nodes: HashMap::new(),
            moves: legal_moves,
            state: state.clone(),

            policy: Box::new(policy),
            visits: Box::new([0.0; ACTION_SPACE]),
            scores: Box::new([0.0; ACTION_SPACE])
        }
    }

    pub fn monte_carlo_tree_search<B: Backend>(&mut self, model: &AlphaZero<B>) -> Box<[f32; ACTION_SPACE]> {
        for _i in 1..=NUM_SIMULATIONS{
            self.simulation(model);
        }
        let weights = self.visits.iter().map(|n| n.powf(1.0 / TEMPERATURE));
        let weights_sum = weights.clone().sum::<f32>();
        let improved_policy: Vec<f32> = weights.map(|x| x / weights_sum).collect();

        improved_policy.into_boxed_slice().try_into().unwrap()
    }

    fn simulation<B: Backend>(&mut self, model: &AlphaZero<B>) -> f32 {
        let mut max_value = f32::NEG_INFINITY;
        let mut max_index = 0;

        let total_visits = self.visits.iter().sum::<f32>() + 1.0;

        for &i in &self.moves {
            let u_value = C_PUCT * self.policy[i] * total_visits.sqrt() / (1.0 + self.visits[i]);
            let q_value = if self.visits[i] > 0.0 { self.scores[i] / self.visits[i] } else { 0.0 };
            let value = q_value + u_value;

            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        let value = if let Some(node) = self.nodes.get_mut(&max_index) {
            -node.simulation(model)
        }
        else {
            -self.expand(model, max_index)
        };

        self.scores[max_index] += value;
        self.visits[max_index] += 1.0;
        value
    }

    fn expand<B: Backend>(&mut self, model: &AlphaZero<B>, max_index: usize) -> f32 {
        let mut leaf_state = self.state.clone();
        let action = index_to_move(max_index, &leaf_state).expect("Illegal move!");
        match play_move(&mut leaf_state, action) {
            Ok(GameResult::Ongoing) => {
                let (policy_tensor, value_tensor) = model.forward(to_tensor(&leaf_state));
                let policy = policy_tensor.into_data()
                    .into_vec()
                    .unwrap()
                    .try_into()
                    .unwrap();

                let value = value_tensor.into_scalar().to_f32();
                self.nodes.insert(max_index, Box::new(Self::new(policy, &leaf_state, false)));

                value
            }
            Ok(GameResult::Draw) => 0.0,
            Ok(_) => -1.0,
            Err(_) => unreachable!("Illegal move picked: {max_index}")
        }
    }

    pub async fn async_monte_carlo_tree_search(&mut self, sender: &UnboundedSender<InferenceRequest>, cache: &PriorsCache) -> Box<[f32; ACTION_SPACE]> {
        for _i in 1..=NUM_SIMULATIONS {
            self.async_simulation(sender, cache).await;
        }
        let weights = self.visits.iter().map(|n| n.powf(1.0 / TEMPERATURE));
        let weights_sum = weights.clone().sum::<f32>();
        let improved_policy: Vec<f32> = weights.map(|x| x / weights_sum).collect();

        improved_policy.into_boxed_slice().try_into().unwrap()
    }

    async fn async_simulation(&mut self, sender: &UnboundedSender<InferenceRequest>, cache: &PriorsCache) -> f32 {
        let mut max_value = f32::NEG_INFINITY;
        let mut max_index = 0;

        let total_visits = self.visits.iter().sum::<f32>() + 1.0;

        for &i in &self.moves {
            let u_value = C_PUCT * self.policy[i] * total_visits.sqrt() / (1.0 + self.visits[i]);
            let q_value = if self.visits[i] > 0.0 { self.scores[i] / self.visits[i] } else { 0.0 };
            let value = q_value + u_value;

            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        let value = if let Some(node) = self.nodes.get_mut(&max_index) {
            -Box::pin(node.async_simulation(sender, cache)).await
        }
        else {
            -self.async_expand(sender, cache, max_index).await
        };

        self.scores[max_index] += value;
        self.visits[max_index] += 1.0;
        value
    }

    async fn async_expand(&mut self, sender: &UnboundedSender<InferenceRequest>, cache: &PriorsCache, max_index: usize) -> f32 {
        let mut leaf_state = self.state.clone();
        let action = index_to_move(max_index, &leaf_state).expect("Illegal move!");
        match play_move(&mut leaf_state, action) {
            Ok(GameResult::Ongoing) => {
                let cache_reader = cache.read().await;
                let state_key = ChessStateKey::new(leaf_state.clone());
                if let Some(entry) = cache_reader.get(&state_key) {
                    self.nodes.insert(max_index, Box::new(Self::new(*entry.policy, &leaf_state, false)));
                    entry.value
                }
                else {
                    drop(cache_reader);

                    let (response_sender, response_receiver) = channel();
                    sender.send(InferenceRequest {
                        state: leaf_state.clone(),
                        response_sender: response_sender,
                    }).expect("Failed to send request!");

                    let InferenceResult { policy, value } = response_receiver.await.expect("No result provided!");
                    self.nodes.insert(max_index, Box::new(Self::new(*policy, &leaf_state, false)));
                    value
                }
            }
            Ok(GameResult::Draw) => 0.0,
            Ok(_) => -1.0,
            Err(_) => unreachable!("Illegal move picked: {max_index}")
        }
    }

    pub fn traverse_new(mut self, action: usize, apply_noise: bool) -> Self {
        let mut new_tree_data = self.nodes.remove(&action)
            .expect("Attempted to traverse to a non-existent child node.");

        if apply_noise {
            apply_dirichlet_noise(&mut new_tree_data.policy, &new_tree_data.moves)
        }

        Self {
            nodes: HashMap::new(),
            moves: new_tree_data.moves,
            state: new_tree_data.state,

            policy: new_tree_data.policy,
            visits: Box::new([0.0; ACTION_SPACE]),
            scores: Box::new([0.0; ACTION_SPACE])
        }
    }

    pub fn max_subtree_depth(&self) -> usize {
        let max_child_depth = self.nodes
            .iter()
            .map(|opt_node| opt_node.1)
            .map(|child_node| child_node.max_subtree_depth())
            .max();

        match max_child_depth {
            Some(depth) => 1 + depth,
            None => 0,
        }
    }
}

fn apply_dirichlet_noise(policy: &mut [f32; ACTION_SPACE], legal_moves: &Vec<usize>) {
    if legal_moves.len() < 2 {
        return;
    }
    
    let mut rng = thread_rng();
    let dirichlet = Dirichlet::new_with_size(DIRICHLET_ALPHA, legal_moves.len())
        .expect("Failed to create Dirichlet distribution!");
    let noise_vector = dirichlet.sample(&mut rng);

    for i in 0..ACTION_SPACE {
        policy[i] *= 1.0 - DIRICHLET_EPSILON;
    }

    for (i, &move_idx) in legal_moves.iter().enumerate() {
        policy[move_idx] += DIRICHLET_EPSILON * noise_vector[i];
    }
}

/////////////////////////////////////////////////
/// Tree Visualisation Tools
/////////////////////////////////////////////////


#[derive(Debug, Clone)]
pub struct TreeViewerState {
    tree: MCTree,
    path_indices: Vec<(usize, String, usize)>,
    scroll_state: TableState,
}

impl TreeViewerState {
    pub fn new(search_tree: MCTree) -> Self {
        Self {
            tree: search_tree,
            path_indices: vec![],
            scroll_state: TableState::new().with_selected(0)
        }
    }

    pub fn current_node(&self) -> &MCTree {
        let mut current = &self.tree;
        for &(_, _, move_index) in &self.path_indices {
            current = current.nodes.get(&move_index)
                .expect("Path contained an index to a non-existent node");
        }
        current
    }

    pub fn next_action(&mut self) {
        let node = self.current_node();
        let list_len = (0..ACTION_SPACE).filter(|&i| node.moves.contains(&i) || node.policy[i] >= 0.01).count();
        self.scroll_state.select(Some(
            (self.scroll_state.selected().expect("No raw selected!") + 1) % list_len)
        );
    }

    pub fn previous_action(&mut self) {
        let node = self.current_node();
        let list_len = (0..ACTION_SPACE).filter(|&i| node.moves.contains(&i) || node.policy[i] >= 0.01).count();
        self.scroll_state.select(Some(
            (self.scroll_state.selected().expect("No raw selected!") + list_len - 1) % list_len)
        );
    }

    pub fn descend(&mut self) -> bool {
        let current_node = self.current_node();
        let select_index = self.scroll_state.selected().expect("No raw selected!");
        let action_index = (0..ACTION_SPACE)
            .filter(|&i| current_node.moves.contains(&i) || current_node.policy[i] >= 0.01)
            .sorted_by_key(|&i| current_node.visits[i] as usize)
            .nth(select_index)
            .expect("Invalid list index");

        if current_node.nodes.get(&action_index).is_some() {
            let move_str = index_to_move(action_index, &current_node.state)
                .map(|m| format!("{m}"))
                .unwrap_or(format!("IM{action_index}"));
            self.path_indices.push((select_index, move_str, action_index));
            self.scroll_state.select(Some(0));
            true
        } else {
            false
        }
    }

    pub fn ascend(&mut self) -> bool {
        if let Some((last_select_index, _, _)) = self.path_indices.pop() {
            self.scroll_state.select(Some(last_select_index));
            true
        } else {
            false
        }
    }
}


pub fn draw_board_view(frame: &mut Frame, area: Rect, state: &TreeViewerState) {
    let main_block = Block::default()
        .title("Node Info & Board State")
        .borders(Borders::ALL);
    
    let inner_area = main_block.inner(area);
    frame.render_widget(main_block, area);

    let node = state.current_node();
    let total_visits = node.visits.iter().sum::<f32>();

    let move_display = iter::once("Root".to_string()).chain(
        state.path_indices
            .iter()
            .map(|(_, move_str, _)| move_str.clone())
    ).collect::<Vec<_>>().join(" → ");

    let bold_style = Style::default().bold();

    let board_text = board_to_text(&node.state);
    let board_height = board_text.height() as u16;

    let stats_height = 1u16;

    let content_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(stats_height),
            Constraint::Min(0),
        ])
        .split(inner_area);

    let stats_line = Line::from(vec![
        Span::styled("Depth: ", bold_style),
        Span::raw(format!("{}", state.path_indices.len())),
        Span::raw(" | "),
        Span::styled("Total Visits: ", bold_style),
        Span::raw(format!("{:.0}", total_visits)),
        Span::raw(" | "),
        Span::styled("Move Sequence: ", bold_style),
        Span::raw(move_display),
    ]);

    let stats_paragraph = Paragraph::new(stats_line)
        .style(Style::default().fg(Color::White))
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    frame.render_widget(stats_paragraph, content_chunks[0]);

    let board_area = content_chunks[1];
    let vertical_center_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(board_height),
            Constraint::Min(0),
        ])
        .split(board_area);

    let board_paragraph = Paragraph::new(board_text)
        .alignment(Alignment::Center);

    frame.render_widget(board_paragraph, vertical_center_layout[1]);
}

pub fn draw_stats_table(frame: &mut Frame, area: Rect, state: &mut TreeViewerState) {
    let node = state.current_node();
    let selected_style = Style::default().add_modifier(Modifier::REVERSED);
    
    let header_cells = ["Move", "Visits", "Q-Value", "P-Value", "U-Value", "Depth", "Select Val"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
    
    let header = Row::new(header_cells)
        .style(Style::new().bg(Color::DarkGray))
        .height(1)
        .top_margin(1)
        .bottom_margin(1);

    let total_visits = node.visits.iter().sum::<f32>() + 1.0;
    let sqrt_total_visits = if total_visits > 0.0 { total_visits.sqrt() } else { 0.0 };

    let rows = (0..ACTION_SPACE)
        .filter(|&i| node.moves.contains(&i) || node.policy[i] >= 0.01)
        .sorted_by_key(|&i| node.visits[i] as usize)
        .map(|i| {
            let q_value = if node.visits[i] > 0.0 { node.scores[i] / node.visits[i] } else { 0.0 };
            let n_value = node.visits[i];
            let p_value = node.policy[i];
            let is_legal = node.moves.contains(&i);

            let u_value = if is_legal && n_value >= 0.0 {
                C_PUCT * p_value * sqrt_total_visits / (1.0 + n_value)
            } else {
                0.0
            };

            let max_depth = if let Some(child_node) = node.nodes.get(&i) {
                child_node.max_subtree_depth()
            } else {
                0
            };

            let selection_value = if is_legal { q_value + u_value } else { f32::NEG_INFINITY };
            let move_marker = if !is_legal { "x" } else if node.nodes.get(&i).is_some() { "*" } else { " " };
            let move_str = index_to_move(i, &node.state)
                    .map(|m| format!("{m}"))
                    .unwrap_or(format!("IM{i}"));

            let data = [
                format!("{} {}", move_marker, move_str),
                format!("{:.1}", n_value),
                format!("{:.4}", q_value),
                format!("{:.4}", p_value),
                format!("{:.4}", u_value),
                format!("{}", max_depth),
                format!("{:.4}", selection_value),
            ];
            Row::new(data)
        });


    let table = Table::new(
        rows,
        [
            Constraint::Percentage(20), // Move
            Constraint::Percentage(10), // Visits
            Constraint::Percentage(15), // Q-Value
            Constraint::Percentage(15), // P-Value
            Constraint::Percentage(15), // U-Value
            Constraint::Percentage(10), // Max Depth
            Constraint::Percentage(15), // Select Val
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Action Statistics (*: expanded, x: illegal, ↑/↓/←/→)"))
    .row_highlight_style(selected_style)
    .highlight_symbol(">> ");

    frame.render_stateful_widget(table, area, &mut state.scroll_state);
}