use std::{
    collections::VecDeque,
    io::{self, stdout},
    sync::{mpsc::{channel, Sender, Receiver}, Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::{
        Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Row, Table, Wrap
    }
};

use crate::{memory::{render_replay_buffer_panel, ReplaySampleWithPrediction, ReplayViewerState}, parameters::{BATCH_SIZE, SKIP_VALIDATION, WIN_RATE_THRESHOLD}, tree::{draw_board_view, draw_stats_table, MCTree, TreeViewerState}};

const LOG_BUFFER_SIZE: usize = 512;
const CHART_HISTORY_SIZE: usize = 512;

#[derive(Debug, Clone)]
pub enum ViewMode {
    Stats,
    Mcts,
    ReplayBuffer
}

#[derive(Debug, Clone)]
pub enum MetricUpdate {
    IterationStart {
        iteration: usize,
    },
    SelfPlayFinished {
        duration: Duration,
        replay_buffer_size: usize,
        new_unique: usize,
        avg_search_depth: f32,
        avg_batch_size: f32
    },
    TrainingFinished {
        duration: Duration,
        avg_policy_loss: f32,
        avg_value_loss: f32,
    },
    ValidationFinished {
        duration: Duration,
        win_rate: f32,
        avg_batch_size: f32
    },
    EvaluationFinished {
        duration: Duration,
        elos: Vec<f32>,
        avg_batch_size: f32,
        winrate_matrix: Vec<Vec<f32>>
    },
    ReplayBufferSample(Vec<ReplaySampleWithPrediction>),
    GeneralLog(String),
    MctsTree(MCTree),
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct TuiState {
    pub start_time: Instant,
    pub current_iteration: usize,
    pub current_phase: String,
    pub phase_start_time: Instant,
    pub should_quit: bool,
    pub log_messages: VecDeque<String>,
    pub log_scroll: usize,
    pub log_autoscroll: bool,

    pub policy_loss_history: VecDeque<(f64, f64)>,
    pub value_loss_history: VecDeque<(f64, f64)>,
    pub elo_history: VecDeque<(f64, f64)>,

    pub last_self_play_duration: Option<Duration>,
    pub last_train_duration: Option<Duration>,
    pub last_val_duration: Option<Duration>,
    pub last_eval_duration: Option<Duration>,
    pub replay_buffer_size: usize,
    pub new_unique: usize,
    pub avg_search_depth: f32,
    pub latest_win_rate: f32,
    pub latest_elo: f32,

    pub latest_eval_avg_batch_size: f32,
    pub latest_valid_avg_batch_size: f32,
    pub latest_sp_avg_batch_size: f32,

    pub view_mode: ViewMode,
    pub tree_viewer_state: Option<TreeViewerState>,
    pub replay_buffer_sample: Vec<ReplaySampleWithPrediction>,
    pub replay_viewer_state: ReplayViewerState,
}

impl TuiState {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            current_iteration: 0,
            current_phase: "Initializing...".to_string(),
            phase_start_time: Instant::now(),
            should_quit: false,
            log_messages: VecDeque::with_capacity(LOG_BUFFER_SIZE),
            log_scroll: 0,
            log_autoscroll: true,

            policy_loss_history: VecDeque::new(),
            value_loss_history: VecDeque::new(),
            elo_history: VecDeque::from(vec![]),
            last_self_play_duration: None,
            last_train_duration: None,
            last_val_duration: None,
            last_eval_duration: None,
            replay_buffer_size: 0,
            new_unique: 0,
            avg_search_depth: 0.0,
            latest_win_rate: 0.0,
            latest_elo: 0.0,

            latest_eval_avg_batch_size: 0.0,
            latest_valid_avg_batch_size: 0.0,
            latest_sp_avg_batch_size: 0.0,

            view_mode: ViewMode::Stats,
            tree_viewer_state: None,
            replay_buffer_sample: vec![],
            replay_viewer_state: ReplayViewerState::new(),
        }
    }

    fn add_log(&mut self, msg: String) {
        if self.log_messages.len() == LOG_BUFFER_SIZE {
            self.log_messages.pop_front();
        }
        self.log_messages.push_back(msg);
    }
    
    fn add_point(history: &mut VecDeque<(f64, f64)>, x: f64, y: f64) {
        if history.len() >= CHART_HISTORY_SIZE {
            history.pop_front();
        }
        history.push_back((x, y));
    }

    fn update(&mut self, update: MetricUpdate) {
        let mut set_phase = |new_phase: String| {
            self.current_phase = new_phase;
            self.phase_start_time = Instant::now();
        };

        match update {
            MetricUpdate::IterationStart { iteration } => {
                self.current_iteration = iteration;
                set_phase("Self-Play".to_string());
                self.add_log(String::new());
                self.add_log(format!("Starting Iteration {}", iteration));
                self.add_log(String::new());
            }
            MetricUpdate::SelfPlayFinished {
                duration,
                replay_buffer_size,
                new_unique,
                avg_search_depth,
                avg_batch_size
            } => {
                set_phase("Training".to_string());
                self.last_self_play_duration = Some(duration);
                self.replay_buffer_size = replay_buffer_size;
                self.new_unique = new_unique;
                self.avg_search_depth = avg_search_depth;
                self.latest_sp_avg_batch_size = avg_batch_size;
            }
            MetricUpdate::TrainingFinished {
                duration,
                avg_policy_loss,
                avg_value_loss,
            } => {
                set_phase(if SKIP_VALIDATION { "Evaluation" } else { "Validation" }.to_string());
                self.last_train_duration = Some(duration);
                let iter_f64 = self.current_iteration as f64;
                Self::add_point(&mut self.policy_loss_history, iter_f64, avg_policy_loss as f64);
                Self::add_point(&mut self.value_loss_history, iter_f64, avg_value_loss as f64);
            }
            MetricUpdate::ValidationFinished {
                duration,
                win_rate,
                avg_batch_size
            } => {
                let next_phase = if win_rate >= WIN_RATE_THRESHOLD { "Evaluation" } else { "Pending Next Iteration" }.to_string();
                set_phase(next_phase);
                self.latest_valid_avg_batch_size = avg_batch_size;
                self.last_val_duration = Some(duration);
                self.latest_win_rate = win_rate;
            }
            MetricUpdate::EvaluationFinished { duration, elos, avg_batch_size, winrate_matrix } => {
                set_phase("Pending Next Iteration".to_string());
                let model_index = elos.len() - 1;
                self.last_eval_duration = Some(duration);
                self.latest_elo = elos[model_index];
                self.latest_eval_avg_batch_size = avg_batch_size;
                Self::add_point(&mut self.elo_history, self.current_iteration as f64, elos[model_index] as f64);
                self.add_log(
                    format!("Winrate against evaluator ({:.0} ELO): {:.1}%. New ELO: {:.0}",
                    elos[model_index - 1],
                    winrate_matrix[model_index][model_index - 1] * 100.0,
                    elos[model_index]
                ));
            }
            MetricUpdate::ReplayBufferSample(sample) => {
                match self.view_mode {
                    ViewMode::ReplayBuffer => {
                        self.add_log(format!(
                            "Received {} samples from replay buffer. Discarded.",
                            sample.len()
                        ));
                    }
                    _ => {
                        self.add_log(format!(
                            "Received {} samples from replay buffer. Press 'Tab' to view.",
                            sample.len()
                        ));
                        self.replay_buffer_sample = sample;
                        if let Some(selected) = self.replay_viewer_state.selected() {
                            if selected >= self.replay_buffer_sample.len() {
                                self.replay_viewer_state.scroll_state.select(
                                    self.replay_buffer_sample.len().checked_sub(1)
                                );
                            }
                        } else if !self.replay_buffer_sample.is_empty() {
                            self.replay_viewer_state.scroll_state.select(Some(0));
                        }
                    }
                }
            },
            MetricUpdate::GeneralLog(msg) => self.add_log(msg),
            MetricUpdate::MctsTree(tree) => {
                match self.view_mode {
                    ViewMode::Mcts => {
                        self.add_log("MCTS tree received. Discarded".to_string());
                    }
                    _ => {
                        self.tree_viewer_state = Some(TreeViewerState::new(tree));
                        self.add_log("MCTS tree received. Press 'Tab' to view.".to_string());
                    }
                }
            }
            MetricUpdate::Shutdown => {
                self.should_quit = true;
                self.add_log("Shutdown signal received.".to_string());
            }
        }
    }
}

pub struct TuiLogger {
    sender: Sender<MetricUpdate>,
    handle: Option<JoinHandle<io::Result<()>>>,
}

impl TuiLogger {
    pub fn new() -> Self {
        let (sender, receiver) = channel();
        let state = Arc::new(Mutex::new(TuiState::new()));

        let ui_state = Arc::clone(&state);
        let handle = thread::spawn(move || run_ui_loop(receiver, ui_state));

        TuiLogger {
            sender,
            handle: Some(handle),
        }
    }

    pub fn log(&self, update: MetricUpdate) {
        self.sender.send(update).ok();
    }

    pub fn stop(mut self) {
        self.log(MetricUpdate::Shutdown);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap().unwrap();
        }
    }
}

fn run_ui_loop(
    receiver: Receiver<MetricUpdate>,
    state: Arc<Mutex<TuiState>>,
) -> io::Result<()> {
    stdout().execute(EnterAlternateScreen)?;
    enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    terminal.clear()?;

    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(33);

    loop {
        while let Ok(update) = receiver.try_recv() {
            let is_shutdown = matches!(update, MetricUpdate::Shutdown);
            state.lock().unwrap().update(update);
            if is_shutdown {
                break;
            }
        }

        terminal.draw(|frame| {
            let mut state_guard = state.lock().unwrap();
            ui(frame, &mut state_guard);
        })?;

        if crossterm::event::poll(tick_rate.saturating_sub(last_tick.elapsed()))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == event::KeyEventKind::Press {
                    let mut state_guard = state.lock().unwrap();
                    if key.code == KeyCode::Char('q') {
                        state_guard.should_quit = true;
                    }
                    if key.code == KeyCode::Tab {
                        state_guard.view_mode = match state_guard.view_mode  {
                            ViewMode::Stats => ViewMode::Mcts,
                            ViewMode::Mcts => ViewMode::ReplayBuffer,
                            ViewMode::ReplayBuffer => ViewMode::Stats,
                        };
                    }

                    match state_guard.view_mode {
                        ViewMode::Stats => {
                            match key.code {
                                KeyCode::Down => {
                                    state_guard.log_scroll = state_guard.log_scroll.saturating_add(1);
                                    state_guard.log_autoscroll = false;
                                }
                                KeyCode::Up => {
                                    state_guard.log_scroll = state_guard.log_scroll.saturating_sub(1);
                                    state_guard.log_autoscroll = false;
                                }
                                _ => {}
                            }
                        }
                        ViewMode::Mcts => {
                            if let Some(tree_viewer) = state_guard.tree_viewer_state.as_mut() {
                                match key.code {
                                    KeyCode::Down => tree_viewer.next_action(),
                                    KeyCode::Up => tree_viewer.previous_action(),
                                    KeyCode::Left  => {
                                        tree_viewer.ascend();
                                    }
                                    KeyCode::Right => {
                                        tree_viewer.descend();
                                    }
                                    _ => {}
                                }
                            }
                        }
                        ViewMode::ReplayBuffer => {
                            let num_items = state_guard.replay_buffer_sample.len();
                             match key.code {
                                KeyCode::Down => state_guard.replay_viewer_state.next(num_items),
                                KeyCode::Up => state_guard.replay_viewer_state.previous(num_items),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }

        if state.lock().unwrap().should_quit {
            break;
        }
    }

    stdout().execute(LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

fn ui(frame: &mut Frame, state: &mut TuiState) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(frame.area());

    let title = Paragraph::new(format!(
        "AlphaZero Training | {} (Tab to switch panel) | Iteration: {} | Phase: {} | Elapsed: {:.1?}",
        match state.view_mode {
            ViewMode::Stats        => "General Statistics",
            ViewMode::Mcts         => "Tree Search Viewer",
            ViewMode::ReplayBuffer => "Replay Memory View",
        },
        state.current_iteration, state.current_phase, state.start_time.elapsed()
    ))
    .style(Style::default().fg(Color::Yellow))
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(title, main_layout[0]);

    let content_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(60),
        ])
        .split(main_layout[1]);
    
    let left_column_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),
            Constraint::Min(0),
        ])
        .split(content_layout[0]);

    render_stats_table(frame, left_column_layout[0], state);

    match state.view_mode {
        ViewMode::Stats => {
            render_charts(frame, content_layout[1], state);
        },
        ViewMode::Mcts => {
            render_mcts_panel(frame, content_layout[1], state);
        },
        ViewMode::ReplayBuffer => {
            render_replay_buffer_panel(frame, content_layout[1], state);
        },
    }
    
    let log_text: Vec<Line> = state
        .log_messages
        .iter()
        .map(|msg| Line::from(msg.as_str()))
        .collect();

    let log_panel_height = left_column_layout[1].height.saturating_sub(2) as usize;
    let max_scroll = log_text.len().saturating_sub(log_panel_height);

    if state.log_scroll >= max_scroll {
        state.log_autoscroll = true;
    }
    
    if state.log_autoscroll {
        state.log_scroll = max_scroll;
    } else {
        state.log_scroll = state.log_scroll.min(max_scroll);
    }

    let log_title = if matches!(state.view_mode, ViewMode::Stats) {
        "Logs (q: quit, ↑/↓: scroll)"
    } else {
        "Logs (q: quit)"
    };

    let log_panel = Paragraph::new(log_text)
        .wrap(Wrap { trim: true })
        .block(Block::default().title(log_title).borders(Borders::ALL))
        .scroll((state.log_scroll as u16, 0));

    frame.render_widget(log_panel, left_column_layout[1]);
}


fn render_stats_table(frame: &mut Frame, area: Rect, state: &TuiState) {
    const SPINNER_CHARS: [char; 8] = ['⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇'];
    let block = Block::default().title("Live Metrics").borders(Borders::ALL);

    let elapsed_since_phase_start = state.phase_start_time.elapsed();
    let spinner_idx = (elapsed_since_phase_start.as_millis() / 75) as usize % SPINNER_CHARS.len();
    let spinner_char = SPINNER_CHARS[spinner_idx];

    let get_time_display = |phase_name: &str, last_duration: Option<Duration>| -> String {
        if state.current_phase == phase_name {
            format!(
                "{} {:.1?} (running)",
                spinner_char, elapsed_since_phase_start
            )
        } else {
            match phase_name {
                "Evaluation" => last_duration.map_or("N/A".to_string(), |d| format!("{:.1?} (BS: {:.1})", d, state.latest_eval_avg_batch_size)),
                "Validation" => last_duration.map_or("N/A".to_string(), |d| format!("{:.1?} (BS: {:.1})", d, state.latest_valid_avg_batch_size)),
                "Self-Play" => last_duration.map_or("N/A".to_string(), |d| format!("{:.1?} (BS: {:.1})", d, state.latest_sp_avg_batch_size)),
                _ => last_duration.map_or("N/A".to_string(), |d| format!("{:.1?} (BS: {:.1})", d, BATCH_SIZE)),
            }
        }
    };

    let rows = vec![
        Row::new(vec![String::from("Replay Buffer Size"), state.replay_buffer_size.to_string()]),
        Row::new(vec![String::from("New Unique Positions"), state.new_unique.to_string()]),
        Row::new(vec![String::from("Win Rate % (vs prev)"), format!("{:.1}", state.latest_win_rate * 100.0)]),
        Row::new(vec![String::from("Average Search Depth"), format!("{:.1}", state.avg_search_depth)]),
        Row::new(vec![String::from("Current Model ELO"), format!("{:.0}", state.latest_elo)]),
        Row::new(vec![String::from("--------------------"), String::from("--------------------")]),
        Row::new(vec![String::from("Self-Play Time"), get_time_display("Self-Play", state.last_self_play_duration)]),
        Row::new(vec![String::from("Training Time"), get_time_display("Training", state.last_train_duration)]),
        Row::new(vec![String::from("Validation Time"), get_time_display("Validation", state.last_val_duration)]),
        Row::new(vec![String::from("Evaluation Time"), get_time_display("Evaluation", state.last_eval_duration)]),
    ];

    let table = Table::new(rows, [Constraint::Percentage(60), Constraint::Percentage(40)])
        .block(block)
        .style(Style::default().fg(Color::White));

    frame.render_widget(table, area);
}

fn render_charts(frame: &mut Frame, area: Rect, state: &TuiState) {
    let chart_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let policy_loss_data = state.policy_loss_history.iter().cloned().collect::<Vec<_>>();
    let value_loss_data = state.value_loss_history.iter().cloned().collect::<Vec<_>>();

    let datasets_loss = vec![
        Dataset::default()
            .name("Policy")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&policy_loss_data),
        Dataset::default()
            .name("Value")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Magenta))
            .data(&value_loss_data),
    ];
    
    let [min_x, max_x] = get_x_bounds(&[&state.policy_loss_history, &state.value_loss_history]);
    let x_labels = generate_axis_labels(min_x, max_x, 5, 0);

    let y_bounds_loss = [0.0, 2.3];
    let y_labels_loss = generate_axis_labels(y_bounds_loss[0], y_bounds_loss[1], 5, 1); // 5 labels, 1 decimal place

    let loss_chart = Chart::new(datasets_loss)
        .block(Block::default().title("Loss History").borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .title("Iteration")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_x, max_x])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .style(Style::default().fg(Color::Gray))
                .bounds(y_bounds_loss)
                .labels(y_labels_loss),
        );
    frame.render_widget(loss_chart, chart_layout[0]);
    
    let elo_history_data = state.elo_history.iter().cloned().collect::<Vec<_>>();
    let datasets_elo = vec![
        Dataset::default()
            .name("ELO")
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&elo_history_data),
    ];

    let [min_x_elo, max_x_elo] = get_x_bounds(&[&state.elo_history]);
    let x_labels_elo = generate_axis_labels(min_x_elo, max_x_elo, 5, 0);

    let y_min_elo = state.elo_history.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max_elo = state.elo_history.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    
    let y_bounds_elo = if (y_max_elo - y_min_elo).abs() < 1.0 {
        [y_min_elo - 50.0, y_max_elo + 50.0]
    } else {
        [y_min_elo, y_max_elo]
    };

    let y_labels_elo = generate_axis_labels(y_bounds_elo[0], y_bounds_elo[1], 5, 0);

    let elo_chart = Chart::new(datasets_elo)
        .block(Block::default().title("Model ELO from Tournament").borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .title("Iteration")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_x_elo, max_x_elo])
                .labels(x_labels_elo),
        )
        .y_axis(
            Axis::default()
                .title("ELO")
                .style(Style::default().fg(Color::Gray))
                .bounds(y_bounds_elo)
                .labels(y_labels_elo),
        );

    frame.render_widget(elo_chart, chart_layout[1]);
}

fn render_mcts_panel(frame: &mut Frame, area: Rect, state: &mut TuiState) {
    let chart_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    if let Some(tree_viewer_state) = &mut state.tree_viewer_state {
        draw_board_view(frame, chart_layout[0], tree_viewer_state);
        draw_stats_table(frame, chart_layout[1], tree_viewer_state);
    }
    else {
        let msg = Paragraph::new(
            "No MCTS tree data available.
                \n\nA tree will be sent here after a game or validation.
                \nPress TAB to change panel.",
        )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(Block::default().title("MCTS Viewer").borders(Borders::ALL));

        frame.render_widget(&msg, area);
    }
}

fn get_x_bounds(histories: &[&VecDeque<(f64, f64)>]) -> [f64; 2] {
    let first_x = histories.iter().filter_map(|h| h.front().map(|(x, _)| *x)).fold(f64::INFINITY, f64::min);
    let last_x = histories.iter().filter_map(|h| h.back().map(|(x, _)| *x)).fold(f64::NEG_INFINITY, f64::max);
    
    if first_x.is_infinite() || last_x.is_infinite() {
        [0.0, 1.0]
    } else {
        [first_x, last_x.max(first_x + 1.0)]
    }
}

fn generate_axis_labels(min: f64, max: f64, count: usize, precision: usize) -> Vec<Span<'static>> {
    if count < 2 || max <= min {
        return vec![
            Span::from(format!("{:.*}", precision, min)),
            Span::from(format!("{:.*}", precision, max)),
        ];
    }

    let step = (max - min) / (count - 1) as f64;
    (0..count)
        .map(|i| {
            let value = min + (i as f64 * step);
            Span::from(format!("{:.*}", precision, value))
        })
        .collect()
}
