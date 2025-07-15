use std::iter;

use burn::prelude::*;
use burn::tensor::cast::ToElement;
use rand::{prelude::*, rng};
use rand_distr::Dirichlet;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot::channel;

use crate::connect4::Connect4;
use crate::agent::AlphaZero;
use crate::parameters::{C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, NUM_SIMULATIONS, TEMPERATURE};
use crate::training::{InferenceRequest, InferenceResult, PriorsCache};

#[derive(Debug, Clone)]
pub struct MCTree {
    pub nodes: [Option<Box<MCTree>>; 7],
    pub moves: Vec<usize>,
    pub state: Connect4,

    pub policy: [f32; 7],
    pub visits: [f32; 7],
    pub scores: [f32; 7]
}

impl MCTree {
    pub fn init<B: Backend>(model: &AlphaZero<B>, game: Connect4, apply_noise: bool) -> Self {
        let legal_moves = game.get_legal_moves();
        let (policy_tensor, _value_tensor) = model.forward(game.to_tensor());
        let policy = policy_tensor.into_data()
            .into_vec()
            .unwrap()
            .try_into()
            .unwrap();

        let policy = if apply_noise {
            apply_dirichlet_noise(policy)
        } else {
            policy
        };

        Self {
            nodes: [const { None }; 7],
            moves: legal_moves,
            state: game,

            policy,
            visits: [0.0; 7],
            scores: [0.0; 7]
        }
    }

    pub async fn async_init(sender: &UnboundedSender<InferenceRequest>, cache: &PriorsCache, game: &Connect4, apply_noise: bool) -> Self {
        let cache_reader = cache.read().await;
        if let Some(entry) = cache_reader.get(&game.get_state()) {
            Self::new(entry.policy, game, apply_noise)
        }
        else {
            drop(cache_reader);

            let (response_sender, response_receiver) = channel();
            sender.send(InferenceRequest {
                state: game.clone(),
                response_sender: response_sender,
            }).expect("Failed to send request!");

            let InferenceResult { policy, .. } = response_receiver.await.expect("No result provided!");
            Self::new(policy, game, apply_noise)
        }
    }

    pub fn new(policy: [f32; 7], state: &Connect4, apply_noise: bool) -> Self {
        let legal_moves = state.get_legal_moves();
        let policy = if apply_noise {
            apply_dirichlet_noise(policy)
        } else {
            policy
        };
        Self {
            nodes: [const { None }; 7],
            moves: legal_moves,
            state: state.clone(),

            policy,
            visits: [0.0; 7],
            scores: [0.0; 7]
        }
    }

    pub fn monte_carlo_tree_search<B: Backend>(&mut self, model: &AlphaZero<B>) -> [f32; 7] {
        for _i in 1..=NUM_SIMULATIONS{
            self.simulation(model);
        }
        let weights = self.visits.iter().map(|n| n.powf(1.0 / TEMPERATURE));
        let weights_sum = weights.clone().sum::<f32>();
        let improved_policy: Vec<f32> = weights.map(|x| x / weights_sum).collect();

        improved_policy.try_into().unwrap()
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

        let value = if let Some(node) = &mut self.nodes[max_index] {
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
        let action = max_index;
        match leaf_state.play(action) {
            Ok("Game in progress") => {
                let (policy_tensor, value_tensor) = model.forward(leaf_state.to_tensor());
                let policy = policy_tensor.into_data()
                    .into_vec()
                    .unwrap()
                    .try_into()
                    .unwrap();

                let value = value_tensor.into_scalar().to_f32();
                self.nodes[max_index] = Some(Box::new(Self::new(policy, &leaf_state, false)));

                value
            }
            Ok("Draw") => 0.0,
            Ok(_) => -1.0,
            Err(_) => unreachable!("Illegal move picked: {max_index}")
        }
    }

    pub async fn async_monte_carlo_tree_search(&mut self, sender: &UnboundedSender<InferenceRequest>, cache: &PriorsCache) -> [f32; 7] {
        for _i in 1..=NUM_SIMULATIONS {
            self.async_simulation(sender, cache).await;
        }
        let weights = self.visits.iter().map(|n| n.powf(1.0 / TEMPERATURE));
        let weights_sum = weights.clone().sum::<f32>();
        let improved_policy: Vec<f32> = weights.map(|x| x / weights_sum).collect();

        improved_policy.try_into().unwrap()
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

        let value = if let Some(node) = &mut self.nodes[max_index] {
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
        let action = max_index;
        match leaf_state.play(action) {
            Ok("Game in progress") => {
                let cache_reader = cache.read().await;
                if let Some(entry) = cache_reader.get(&leaf_state.get_state()) {
                    self.nodes[max_index] = Some(Box::new(Self::new(entry.policy, &leaf_state, false)));
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
                    self.nodes[max_index] = Some(Box::new(Self::new(policy, &leaf_state, false)));
                    value
                }
            }
            Ok("Draw") => 0.0,
            Ok(_) => -1.0,
            Err(_) => unreachable!("Illegal move picked: {max_index}")
        }
    }

    pub fn traverse_new(mut self, action: usize, apply_noise: bool) -> Self {
        let new_tree_data = self.nodes[action].take()
            .expect("Attempted to traverse to a non-existent child node.");

        let policy = if apply_noise {
            apply_dirichlet_noise(new_tree_data.policy)
        } else {
            new_tree_data.policy
        };

        Self {
            nodes: [const { None }; 7],
            moves: new_tree_data.moves,
            state: new_tree_data.state,

            policy,
            visits: [0.0; 7],
            scores: [0.0; 7]
        }
    }

    pub fn max_subtree_depth(&self) -> usize {
        let max_child_depth = self.nodes
            .iter()
            .filter_map(|opt_node| opt_node.as_ref())
            .map(|child_node| child_node.max_subtree_depth())
            .max();

        match max_child_depth {
            Some(depth) => 1 + depth,
            None => 0,
        }
    }
}

fn apply_dirichlet_noise(policy: [f32; 7]) -> [f32; 7] {
    let mut rng = rng();
    let dirichlet = Dirichlet::new([DIRICHLET_ALPHA; 7]).expect("Failed to create Dirichlet distribution!");
    let noise = dirichlet.sample(&mut rng);

    let mut noisy_policy = [0.0; 7];
    for i in 0..7 {
        noisy_policy[i] = (1.0 - DIRICHLET_EPSILON) * policy[i] + DIRICHLET_EPSILON * noise[i];
    }

    noisy_policy
}

/////////////////////////////////////////////////
/// Tree Visualisation Tools
/////////////////////////////////////////////////


#[derive(Debug, Clone)]
pub struct TreeViewerState {
    tree: MCTree,
    path_indices: Vec<usize>,
    selected_action_index: usize,
}

impl TreeViewerState {
    pub fn new(search_tree: MCTree) -> Self {
        Self {
            tree: search_tree,
            path_indices: vec![],
            selected_action_index: 0,
        }
    }

    pub fn current_node(&self) -> &MCTree {
        let mut current = &self.tree;
        for &index in &self.path_indices {
            current = current.nodes[index]
                .as_ref()
                .expect("Path contained an index to a non-existent node");
        }
        current
    }

    pub fn next_action(&mut self) {
        self.selected_action_index = (self.selected_action_index + 1) % 7;
    }

    pub fn previous_action(&mut self) {
        self.selected_action_index = (self.selected_action_index + 6) % 7;
    }

    pub fn descend(&mut self) -> bool {
        let current_node = self.current_node();
        if current_node.nodes[self.selected_action_index].is_some() {
            self.path_indices.push(self.selected_action_index);
            self.selected_action_index = 0;
            true
        } else {
            false
        }
    }

    pub fn ascend(&mut self) -> bool {
        if let Some(last_action_index) = self.path_indices.pop() {
            self.selected_action_index = last_action_index;
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
            .map(|i| i.to_string())
    ).collect::<Vec<_>>().join(" → ");

    let bold_style = Style::default().bold();

    let board_text = node.state.to_string();
    let board_lines: Vec<Line> = board_text.lines().map(Line::from).collect();
    let board_height = board_lines.len() as u16;

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

    let board_paragraph = Paragraph::new(board_lines)
        .alignment(Alignment::Center);

    frame.render_widget(board_paragraph, vertical_center_layout[1]);
}

pub fn draw_stats_table(frame: &mut Frame, area: Rect, state: &TreeViewerState) {
    let node = state.current_node();
    let selected_style = Style::default().add_modifier(Modifier::REVERSED);
    
    let header_cells = ["Move", "Visits", "Q-Value", "P-Value", "U-Value", "Max Depth", "Select Val"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
    
    let header = Row::new(header_cells)
        .style(Style::new().bg(Color::DarkGray))
        .height(1)
        .top_margin(1)
        .bottom_margin(1);

    let total_visits = node.visits.iter().sum::<f32>() + 1.0;
    let sqrt_total_visits = if total_visits > 0.0 { total_visits.sqrt() } else { 0.0 };

    let rows = (0..7).map(|i| {
        let q_value = if node.visits[i] > 0.0 { node.scores[i] / node.visits[i] } else { 0.0 };
        let n_value = node.visits[i];
        let p_value = node.policy[i];
        let is_legal = node.moves.contains(&i);

        let u_value = if is_legal && n_value >= 0.0 {
            C_PUCT * p_value * sqrt_total_visits / (1.0 + n_value)
        } else {
            0.0
        };

        let max_depth = if let Some(child_node) = node.nodes[i].as_ref() {
            child_node.max_subtree_depth()
        } else {
            0
        };

        let selection_value = if is_legal { q_value + u_value } else { f32::NEG_INFINITY };
        let move_marker = if !is_legal { "x" } else if node.nodes[i].is_some() { "*" } else { " " };

        let data = [
            format!("{} {}", move_marker, i),
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
            Constraint::Percentage(10), // Move
            Constraint::Percentage(15), // Visits
            Constraint::Percentage(15), // Q-Value
            Constraint::Percentage(15), // P-Value
            Constraint::Percentage(15), // U-Value
            Constraint::Percentage(15), // Max Depth
            Constraint::Percentage(15), // Select Val
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Action Statistics (*: expanded, x: illegal, ↑/↓/←/→)"))
    .row_highlight_style(selected_style)
    .highlight_symbol(">> ");

    let mut table_state = TableState::default();
    table_state.select(Some(state.selected_action_index));

    frame.render_stateful_widget(table, area, &mut table_state);
}