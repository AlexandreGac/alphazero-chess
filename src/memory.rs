use std::collections::{HashMap, VecDeque};

use rand::prelude::*;
use ratatui::{layout::{Alignment, Constraint, Direction, Layout, Rect}, style::{Color, Modifier, Style}, text::Line, widgets::{BarChart, Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap}, Frame};

use crate::{connect4::Connect4, logger::TuiState, parameters::REPLAY_BUFFER_SIZE, training::EpisodeStep};

#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub state: Connect4,
    pub policy: [f32; 7],
    pub value: f32,
}

struct MemoryEntry {
    policy: [f32; 7],
    value: f32,
    visit_count: usize,
}

pub struct ReplayBuffer {
    buffer: HashMap<[[i32; 6]; 7], MemoryEntry>,
    order: VecDeque<Connect4>,
}

impl ReplayBuffer {

    pub fn new() -> Self {
        Self {
            buffer: HashMap::with_capacity(REPLAY_BUFFER_SIZE),
            order: VecDeque::with_capacity(REPLAY_BUFFER_SIZE),
        }
    }

    pub fn add(&mut self, step: EpisodeStep) -> usize {
        let state_key = step.state.get_state();

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
                    self.buffer.remove(&oldest_position.get_state());
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
                let entry = self.buffer.get(&game.get_state()).expect("Key from `order` should exist in `buffer`");
                TrainingSample {
                    state: (*game).clone(),
                    policy: entry.policy,
                    value: entry.value,
                }
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}


/////////////////////////////////////////////////
/// Replay Memory Visualisation Tools
/////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct ReplaySampleWithPrediction {
    pub sample: TrainingSample,
    pub predicted_policy: [f32; 7],
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

fn render_centered_connect4_board(frame: &mut Frame, area: Rect, state: &Connect4) {
    let main_block = Block::default().title("Board State").borders(Borders::ALL);
    let inner_area = main_block.inner(area);
    frame.render_widget(main_block, area);
    
    let board_text = state.to_string();
    let board_lines: Vec<Line> = board_text.lines().map(Line::from).collect();
    let board_height = board_lines.len() as u16;

    let vertical_center_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(board_height),
            Constraint::Min(0),
        ])
        .split(inner_area);

    let board_paragraph = Paragraph::new(board_lines)
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
        let turn_char = if data.sample.state.get_turn() > 0.0 { "X" } else { "O" };
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
        render_centered_connect4_board(frame, board_area, &data.sample.state);
        
        let bottom_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(bottom_panel_area);

        const POLICY_LABELS: [&'static str; 7] = ["0", "1", "2", "3", "4", "5", "6"];

        let target_policy_data: Vec<(&str, u64)> = POLICY_LABELS.iter()
            .zip(data.sample.policy.iter())
            .map(|(label, p)| (*label, (p * 100.0).round() as u64))
            .collect();

        let target_barchart = BarChart::default()
            .block(Block::default().title("MCTS Policy (Target)").borders(Borders::ALL))
            .data(&target_policy_data).bar_width(5)
            .bar_style(Style::default().fg(Color::Cyan))
            .value_style(Style::default().fg(Color::Black).bg(Color::Cyan))
            .label_style(Style::default().fg(Color::White)).max(100);
        
        frame.render_widget(target_barchart, bottom_chunks[0]);

        let predicted_policy_data: Vec<(&str, u64)> = POLICY_LABELS.iter()
            .zip(data.predicted_policy.iter())
            .map(|(label, p)| (*label, (p * 100.0).round() as u64))
            .collect();
        
        let predicted_barchart = BarChart::default()
            .block(Block::default().title("Model Policy (Prediction)").borders(Borders::ALL))
            .data(&predicted_policy_data).bar_width(5)
            .bar_style(Style::default().fg(Color::Magenta))
            .value_style(Style::default().fg(Color::Black).bg(Color::Magenta))
            .label_style(Style::default().fg(Color::White)).max(100);

        frame.render_widget(predicted_barchart, bottom_chunks[1]);
    } else {
        frame.render_widget(Block::default().borders(Borders::ALL).title("Board State"), top_panel_chunks[1]);
        let block = Block::default().title("Policy Distributions").borders(Borders::ALL);
        frame.render_widget(block, bottom_panel_area);
    }
}