use std::{collections::{HashMap, VecDeque}, fs::File, io::{BufReader, BufWriter}, path::Path};

use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use rand::prelude::*;
use ratatui::{layout::{Alignment, Constraint, Direction, Layout, Rect}, style::{Color, Modifier, Style}, widgets::{BarChart, Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap}, Frame};
use serde::{Deserialize, Serialize};
use shakmaty::{Chess, Position, Square};

use crate::{chess::{board_to_text, index_to_move}, logger::TuiState, parameters::{ACTION_SPACE, REPLAY_BUFFER_SIZE}, training::EpisodeStep};

#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub state: Chess,
    pub policy: Box<[f32; ACTION_SPACE]>,
    pub value: f32,
}

#[derive(Hash, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct ChessStateKey {
    game: Chess,
    ep_square: Option<Square>
}

impl ChessStateKey {
    pub fn new(game: Chess) -> Self {
        let ep_square = game.legal_ep_square();
        Self {
            game, ep_square
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MemoryEntry {
    policy: Box<[f32; ACTION_SPACE]>,
    value: f32,
    visit_count: usize,
}

#[derive(Serialize, Deserialize)]
pub struct ReplayBuffer {
    buffer: HashMap<ChessStateKey, MemoryEntry>,
    order: VecDeque<Chess>,
}

impl ReplayBuffer {

    pub fn new() -> Self {
        Self {
            buffer: HashMap::with_capacity(REPLAY_BUFFER_SIZE),
            order: VecDeque::with_capacity(REPLAY_BUFFER_SIZE),
        }
    }

    pub fn add(&mut self, step: EpisodeStep) -> usize {
        let state_key = ChessStateKey::new(step.state.clone());

        let x = step.state.fen();

        if let Some(entry) = self.buffer.get_mut(&state_key) {
            let old_count = entry.visit_count as f32;
            let new_total_count = old_count + 1.0;

            entry.value = (entry.value * old_count + step.final_value) / new_total_count;

            for i in 0..entry.policy.len() {
                entry.policy[i] = (entry.policy[i] * old_count + step.improved_policy[i]) / new_total_count;
            }

            entry.visit_count += 1;

            0
        } else {
            if self.order.len() >= REPLAY_BUFFER_SIZE {
                if let Some(oldest_position) = self.order.pop_front() {
                    let state_key = ChessStateKey::new(oldest_position);
                    self.buffer.remove(&state_key);
                }
            }

            let new_entry = MemoryEntry {
                policy: step.improved_policy,
                value: step.final_value,
                visit_count: 1,
            };

            self.buffer.insert(state_key, new_entry);
            self.order.push_back(step.state);
            
            1
        }
    }

    pub fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<TrainingSample> {
        let effective_batch_size = std::cmp::min(batch_size, self.order.len());
        if effective_batch_size == 0 {
            return Vec::new();
        }

        self.order
            .iter()
            .collect::<Vec<_>>()
            .choose_multiple(rng, effective_batch_size)
            .map(|game| {
                let state_key = ChessStateKey::new((*game).clone());
                let entry = self.buffer.get(&state_key).expect("Key from `order` should exist in `buffer`");
                TrainingSample {
                    state: (*game).clone(),
                    policy: entry.policy.clone(),
                    value: entry.value,
                }
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Saves the ReplayBuffer to a compressed binary file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let encoder = &mut GzEncoder::new(writer, Compression::default());
        bincode::serde::encode_into_std_write(self, encoder, bincode::config::standard())?;
        Ok(())
    }

    /// Loads a ReplayBuffer from a compressed binary file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let decoder = &mut GzDecoder::new(reader);
        let buffer = bincode::serde::decode_from_std_read(decoder, bincode::config::standard())?;
        Ok(buffer)
    }
}


/////////////////////////////////////////////////
/// Replay Memory Visualisation Tools
/////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct ReplaySampleWithPrediction {
    pub sample: TrainingSample,
    pub predicted_policy: Box<[f32; ACTION_SPACE]>,
    pub predicted_value: f32
}

#[derive(Debug, Clone)]
pub struct ReplayViewerState {
    pub scroll_state: TableState,
}

impl ReplayViewerState {
    pub fn new() -> Self {
        let mut state = Self {
            scroll_state: TableState::default(),
        };
        state.scroll_state.select(Some(0));
        state
    }

    pub fn selected(&self) -> Option<usize> {
        self.scroll_state.selected()
    }

    pub fn next(&mut self, num_items: usize) {
        if num_items == 0 { return; }
        let i = self.selected().map_or(0, |i| (i + 1) % num_items);
        self.scroll_state.select(Some(i));
    }

    pub fn previous(&mut self, num_items: usize) {
        if num_items == 0 { return; }
        let i = self.selected().map_or(0, |i| {
            if i == 0 { num_items - 1 } else { i - 1 }
        });
        self.scroll_state.select(Some(i));
    }
}

fn render_centered_board(frame: &mut Frame, area: Rect, state: &Chess) {
    let main_block = Block::default().title("Board State").borders(Borders::ALL);
    let inner_area = main_block.inner(area);
    frame.render_widget(main_block, area);
    
    let board_text = board_to_text(state);
    let board_height = board_text.height() as u16;

    let vertical_center_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(board_height),
            Constraint::Min(0),
        ])
        .split(inner_area);

    let board_paragraph = Paragraph::new(board_text)
        .alignment(Alignment::Center);

    frame.render_widget(board_paragraph, vertical_center_layout[1]);
}

fn render_replay_buffer_list(frame: &mut Frame, area: Rect, state: &mut TuiState) {
    let selected_style = Style::default().add_modifier(Modifier::REVERSED);
    
    let header_cells = ["#", "Value", "Predic.", "Turn"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
    
    let header = Row::new(header_cells)
        .style(Style::new().bg(Color::DarkGray))
        .height(1)
        .bottom_margin(1);

    let rows = state.replay_buffer_sample.iter().enumerate().map(|(i, data)| {
        let turn_char = if data.sample.state.turn().is_white() { "W" } else { "B" };
        Row::new(vec![
            Cell::from(i.to_string()),
            Cell::from(format!("{:.2}", data.sample.value)),
            Cell::from(format!("{:.2}", data.predicted_value)),
            Cell::from(turn_char.to_string()),
        ])
    });

    let constraints = [
        Constraint::Percentage(16),
        Constraint::Percentage(28),
        Constraint::Percentage(28),
        Constraint::Percentage(28)
    ];

    let table = Table::new(rows, constraints)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title("Buffer Samples (↑/↓)"))
        .row_highlight_style(selected_style)
        .highlight_symbol(">> ");

    frame.render_stateful_widget(table, area, &mut state.replay_viewer_state.scroll_state);
}

pub fn render_replay_buffer_panel(frame: &mut Frame, area: Rect, state: &mut TuiState) {
    if state.replay_buffer_sample.is_empty() {
        let msg = Paragraph::new(
            "No replay buffer data available.\n\nA sample will be sent here after self-play completes.\nPress 'Tab' to return to the Stats Panel.",
        )
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true })
        .block(Block::default().title("Replay Buffer Visualizer").borders(Borders::ALL));
        frame.render_widget(msg, area);
        return;
    }

    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(area);

    let top_panel_area = main_chunks[0];
    let bottom_panel_area = main_chunks[1];

    let top_panel_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(top_panel_area);
    
    let list_area = top_panel_chunks[0];
    let board_area = top_panel_chunks[1];

    render_replay_buffer_list(frame, list_area, state);

    let selected_data = state.replay_viewer_state.selected()
        .and_then(|i| state.replay_buffer_sample.get(i));
    
    if let Some(data) = selected_data {
        render_centered_board(frame, board_area, &data.sample.state);
        
        let bottom_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(bottom_panel_area);

        const TOP_MOVES_TO_SHOW: usize = 7;

        let mut top_target_moves = data.sample.policy.iter().enumerate().collect::<Vec<_>>();
        top_target_moves.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let target_labels: Vec<String> = top_target_moves.iter()
            .take(TOP_MOVES_TO_SHOW)
            .map(|(idx, _)| index_to_move(*idx, &data.sample.state)
                .map(|m| format!("{m}"))
                .unwrap_or(format!("IM{idx}"))
            )
            .collect();
            
        let target_policy_data: Vec<(&str, u64)> = target_labels.iter()
            .zip(top_target_moves.iter().take(TOP_MOVES_TO_SHOW))
            .map(|(label, (_, p))| (label.as_str(), (**p * 100.0).round() as u64))
            .collect();

        let target_barchart = BarChart::default()
            .block(Block::default().title("MCTS Policy (Target)").borders(Borders::ALL))
            .data(&target_policy_data)
            .bar_width(6) // Reduced width for potentially wider labels
            .bar_gap(2)
            .bar_style(Style::default().fg(Color::Cyan))
            .value_style(Style::default().fg(Color::Black).bg(Color::Cyan))
            .label_style(Style::default().fg(Color::White))
            .max(100);
        
        frame.render_widget(target_barchart, bottom_chunks[0]);

        let mut top_predicted_moves = data.predicted_policy.iter().enumerate().collect::<Vec<_>>();
        top_predicted_moves.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let predicted_labels: Vec<String> = top_predicted_moves.iter()
            .take(TOP_MOVES_TO_SHOW)
            .map(|(idx, _)| index_to_move(*idx, &data.sample.state)
                .map(|m| format!("{m}"))
                .unwrap_or(format!("IM{idx}"))
            )
            .collect();

        let predicted_policy_data: Vec<(&str, u64)> = predicted_labels.iter()
            .zip(top_predicted_moves.iter().take(TOP_MOVES_TO_SHOW))
            .map(|(label, (_, p))| (label.as_str(), (**p * 100.0).round() as u64))
            .collect();
        
        let predicted_barchart = BarChart::default()
            .block(Block::default().title("Model Policy (Prediction)").borders(Borders::ALL))
            .data(&predicted_policy_data)
            .bar_width(6)
            .bar_gap(2)
            .bar_style(Style::default().fg(Color::Magenta))
            .value_style(Style::default().fg(Color::Black).bg(Color::Magenta))
            .label_style(Style::default().fg(Color::White))
            .max(100);

        frame.render_widget(predicted_barchart, bottom_chunks[1]);
    } else {
        frame.render_widget(Block::default().borders(Borders::ALL).title("Board State"), top_panel_chunks[1]);
        let block = Block::default().title("Policy Distributions").borders(Borders::ALL);
        frame.render_widget(block, bottom_panel_area);
    }
}