pub const ACTION_SPACE: usize = 64 * 8 * 8;
pub const CACHE_CAPACITY: u64 = 500_000;

pub const SEED: u64 = 42;
pub const NUM_RES_BLOCKS: usize = 10;
pub const NUM_FILTERS: usize = 128;

pub const REPLAY_BUFFER_SIZE: usize = 100_000;
pub const MIN_REPLAY_SIZE: usize = 20_000;
pub const NUM_ITERATIONS: usize = 10_000;
pub const NUM_EPISODES: usize = 100;
pub const NUM_THREADS: usize = 8;

pub const NUM_TRAIN_STEPS: usize = 40;
pub const BATCH_SIZE: usize = 512;

pub const BASE_LEARNING_RATE: f64 = 1e-3;
pub const MAX_LEARNING_RATE: f64 = 1e-2;
pub const FULL_CYCLE: usize = 20;
pub const HALF_CYCLE: usize = 10;
pub const DECAY_INTERVAL: usize = 1000;

pub const VALUE_LOSS_WEIGHT: f32 = 0.5;
pub const WEIGHT_DECAY: f32 = 1e-4;

pub const DIRICHLET_ALPHA: f32 = 0.3;
pub const DIRICHLET_EPSILON: f32 = 0.25;

pub const TEMPERATURE_ANNEALING: u32 = 15;    // 15 moves each
pub const NUM_SIMULATIONS: usize = 256;
pub const TEMPERATURE: f32 = 1.0;
pub const C_PUCT: f32 = 3.0;

pub const SKIP_VALIDATION: bool = true;
pub const EVALUATION_GAMES: usize = 256;
pub const WIN_RATE_THRESHOLD: f32 = 0.55;