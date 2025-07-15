use core::time;
use std::{collections::HashMap, sync::Arc, thread, time::Instant, vec};

use burn::{grad_clipping::GradientClippingConfig, module::{AutodiffModule, Module}, optim::{AdamWConfig, GradientsParams, Optimizer}, prelude::Backend, record::{FullPrecisionSettings, NamedMpkFileRecorder}, tensor::{backend::AutodiffBackend, cast::ToElement, Tensor}};
use rand::prelude::*;
use rand::distr::weighted::WeightedIndex;
use tokio::{runtime, sync::{mpsc::{self, UnboundedSender}, oneshot::Sender, RwLock}, task::JoinSet};

use crate::{agent::AlphaZero, connect4::Connect4, load_model, logger::{MetricUpdate, TuiLogger}, memory::{ReplayBuffer, ReplaySampleWithPrediction}, parameters::{BASE_LEARNING_RATE, BATCH_SIZE, DECAY_INTERVAL, FULL_CYCLE, HALF_CYCLE, MAX_LEARNING_RATE, MIN_REPLAY_SIZE, NUM_EPISODES, NUM_ITERATIONS, NUM_THREADS, NUM_TRAIN_STEPS, SEED, SKIP_VALIDATION, TEMPERATURE_ANNEALING, VALUE_LOSS_WEIGHT, WEIGHT_DECAY, WIN_RATE_THRESHOLD}, ratings::compute_elo_rankings, tree::MCTree, validation::{evaluate, sample_search_tree, Player}};

pub struct EpisodeStep {
    pub state: Connect4,
    pub improved_policy: [f32; 7],
    pub final_value: f32,
    pub search_depth: usize
}

#[derive(Clone, Copy, Debug)]
pub struct CacheEntry {
    pub policy: [f32; 7],
    pub value: f32,
}

pub struct InferenceRequest {
    pub state: Connect4,
    pub response_sender: Sender<InferenceResult>,
}


#[derive(Debug)]
pub struct InferenceResult {
    pub policy: [f32; 7],
    pub value: f32,
}

pub type PriorsCache = Arc<RwLock<HashMap<[[i32; 6]; 7], CacheEntry>>>;

pub fn train<B: AutodiffBackend>() {
    B::seed(SEED);
    let mut rng = rand::rng();
    let logger = TuiLogger::new();
    let runtime = runtime::Builder::new_multi_thread()
        .worker_threads(NUM_THREADS)
        .enable_all()
        .build()
        .expect("Enable to build async runtime!");

    let mut model = AlphaZero::<B>::new();
    let evaluator = load_model::<B>("artifacts/evaluator/evaluator_elo_716.mpk");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
        .with_weight_decay(WEIGHT_DECAY)
        .init();

    logger.log(MetricUpdate::GeneralLog(format!("Device: {:?}", B::Device::default())));

    let mut replay_buffer = ReplayBuffer::new();
    
    for iteration in 0..NUM_ITERATIONS {
        logger.log(MetricUpdate::IterationStart { iteration });

        let start = Instant::now();
        let mut new_unique_states = 0;
        let mut avg_search_depth;
        let mut avg_batch_size;
        loop {
            let (avg_bs, new_steps) = runtime.block_on(
                run_all_episodes(&model.valid())
            );

            avg_batch_size = avg_bs;
            avg_search_depth = 0.0;
            let len_new_steps = new_steps.len() as f32;
            for step in new_steps {
                avg_search_depth += step.search_depth as f32;
                
                let symmetric_step = generate_symmetry(&step);
                new_unique_states += replay_buffer.add(symmetric_step);
                new_unique_states += replay_buffer.add(step);
            }
            avg_search_depth /= len_new_steps;

            if replay_buffer.len() < MIN_REPLAY_SIZE {
                logger.log(MetricUpdate::GeneralLog(format!(
                    "Filling replay buffer: {}/{} unique samples.",
                    replay_buffer.len(),
                    MIN_REPLAY_SIZE
                )));
                continue;
            }

            break
        }

        logger.log(MetricUpdate::SelfPlayFinished {
            duration: start.elapsed(),
            replay_buffer_size: replay_buffer.len(),
            new_unique: new_unique_states,
            avg_search_depth,
            avg_batch_size
        });

        let start = Instant::now();
        let mut new_model = model.clone();
        
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;

        for batch_number in 0..NUM_TRAIN_STEPS {
            let batch = replay_buffer.sample(BATCH_SIZE, &mut rng);
            let states = batch.iter().map(|e|
                e.state.to_tensor::<B>()
            ).collect::<Vec<_>>();

            let target_policies = batch.iter()
                .map(|e| Tensor::<B, 1>::from_floats(e.policy, &B::Device::default()))
                .map(|tensor| tensor.unsqueeze::<2>())
                .collect::<Vec<_>>();

            let target_values = batch.iter()
                .map(|e| e.value)
                .collect::<Vec<_>>();

            let state_tensor = Tensor::cat(states, 0);
            let target_policy_tensor = Tensor::cat(target_policies, 0);
            let target_value_tensor = Tensor::<B, 1>::from_floats(
                target_values.as_slice(), 
                &B::Device::default()
            );

            let (predicted_policy_tensor, predicted_value_tensor) = new_model.forward(state_tensor);

            if batch_number == 0 {
                let predicted_policies = predicted_policy_tensor.clone().iter_dim(0).map(|p| {
                    p.into_data()
                        .to_vec()
                        .unwrap()
                        .try_into()
                        .unwrap()
                }).collect::<Vec<[f32; 7]>>();

                let predicted_values = predicted_value_tensor.clone().iter_dim(0).map(|v| {
                    v.into_scalar().to_f32()
                }).collect::<Vec<f32>>();

                let samples_with_predictions: Vec<ReplaySampleWithPrediction> = batch.into_iter()
                    .zip(predicted_policies.into_iter().zip(predicted_values))
                    .map(|(sample, (predicted_policy, predicted_value))| ReplaySampleWithPrediction {
                        sample,
                        predicted_policy,
                        predicted_value
                    })
                    .collect();

                logger.log(MetricUpdate::ReplayBufferSample(samples_with_predictions));
            }

            let (policy_loss, value_loss, gradients) = compute_gradients(
                predicted_policy_tensor, target_policy_tensor,
                predicted_value_tensor, target_value_tensor
            );

            let grads = GradientsParams::from_grads(gradients, &new_model);
            new_model = optimizer.step(
                get_cyclical_lr(iteration),
                new_model, grads
            );

            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
        }

        logger.log(MetricUpdate::TrainingFinished {
            duration: start.elapsed(),
            avg_policy_loss: total_policy_loss / NUM_TRAIN_STEPS as f32,
            avg_value_loss: total_value_loss / NUM_TRAIN_STEPS as f32,
        });

        let start = Instant::now();
        let is_better = SKIP_VALIDATION || {
            let eval = runtime.block_on(
                evaluate(&Player::BaseModel(new_model.valid()), &Player::BaseModel(model.valid()))
            );

            logger.log(MetricUpdate::ValidationFinished {
                duration: start.elapsed(),
                win_rate: eval.winrate,
                avg_batch_size: eval.avg_batch_size
            });

            logger.log(MetricUpdate::GeneralLog(format!(
                "Model performance: winrate {:.1}% (W: {:.1}%, D: {:.1}%, L: {:.1}%)",
                eval.winrate * 100.0,
                eval.p1_winrate * 100.0,
                eval.drawrate * 100.0,
                eval.p2_winrate * 100.0
            )));

            eval.winrate >= WIN_RATE_THRESHOLD
        };

        if is_better {
            model = new_model;

            let start = Instant::now();
            let players = vec![
                Player::BaseModel(evaluator.valid()),
                Player::BaseModel(model.valid())
            ];
            let model_index = players.len() - 1;
            let (avg_batch_size, elo_ranks, winrate_matrix) = compute_elo_rankings(
                players, 700.0, &runtime, false
            );
            
            let new_elo = elo_ranks[model_index];
            logger.log(MetricUpdate::EvaluationFinished { 
                duration: start.elapsed(),
                elos: elo_ranks,
                winrate_matrix,
                avg_batch_size,
            });

            let opponent = Player::MiniMax(4);
            if iteration % 2 == 0 {
                if let Some(sampled_tree) = sample_search_tree(&Player::MctsModel(model.valid()), &opponent, 6) {
                    logger.log(MetricUpdate::MctsTree(sampled_tree));
                }
            }
            else {
                if let Some(sampled_tree) = sample_search_tree(&opponent, &Player::MctsModel(model.valid()), 6) {
                    logger.log(MetricUpdate::MctsTree(sampled_tree));
                }
            }

            let file_path = format!("artifacts/models/iteration_{}_elo_{:.0}.mpk", iteration, new_elo);
            model.clone().save_file(file_path.clone(), &recorder).expect("Unable to save model!");
        }
    }

    logger.stop();
}

fn compute_gradients<B: AutodiffBackend>(
    predicted_policy: Tensor<B, 2>, target_policy: Tensor<B, 2>,
    predicted_value: Tensor<B, 1>, target_value: Tensor<B, 1>
) -> (f32, f32, B::Gradients) {
    let difference = predicted_value - target_value.detach();
    let policy_loss = -(target_policy.detach() * predicted_policy.log()).sum_dim(1).mean();
    let value_loss = (difference.clone() * difference).mean();
    let loss = policy_loss.clone() + value_loss.clone() * VALUE_LOSS_WEIGHT;

    let policy_loss_val = policy_loss.into_scalar().to_f32();
    let value_loss_val = value_loss.into_scalar().to_f32();

    let gradients = loss.backward();
    (policy_loss_val, value_loss_val, gradients)
}

async fn run_episode(mut search_tree: MCTree, sender: &UnboundedSender<InferenceRequest>, cache: &PriorsCache) -> Vec<EpisodeStep> {
    let mut game = Connect4::new();
    let mut history = vec![];

    let result = loop {
        let improved_policy = search_tree.async_monte_carlo_tree_search(sender, cache).await;
        let search_depth = search_tree.max_subtree_depth();
        let turn = game.get_turn();

        history.push(EpisodeStep {
            state: game.clone(),
            improved_policy,
            final_value: turn,
            search_depth
        });

        let action_index = if game.get_total_moves() >= TEMPERATURE_ANNEALING {
            improved_policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap()
        }
        else {
            let distribution = WeightedIndex::new(&improved_policy).unwrap();
            distribution.sample(&mut rand::rng())
        };

        match game.play(action_index) {
            Ok("Game in progress") => search_tree = search_tree.traverse_new(action_index, true),
            Ok("Draw") => break 0.0,
            Ok(_) => break turn,
            Err(_) => panic!("The model played an illegal move!")
        }
    };

    for episode_step in &mut history {
        episode_step.final_value *= result;
    }

    history
}

async fn run_all_episodes<B: Backend>(model: &AlphaZero<B>) -> (f32, Vec<EpisodeStep>) {
    let (sender, mut receiver) = mpsc::unbounded_channel::<InferenceRequest>();
    let cache: PriorsCache = Arc::new(RwLock::new(HashMap::new()));

    let game = Connect4::new();
    let (policy_tensor, _value_tensor) = model.forward(game.to_tensor());
    let policy = policy_tensor.into_data()
        .into_vec()
        .unwrap()
        .try_into()
        .unwrap();

    let mut tasks = JoinSet::new();
    for _episode in 0..NUM_EPISODES {
        let game = game.clone();
        let sender = sender.clone();
        let cache = cache.clone();
        tasks.spawn(async move {
            let search_tree = MCTree::new(policy, &game, true);
            run_episode(search_tree, &sender, &cache).await
        });
    }

    drop(sender);

    let mut num_inferences = 0.0;
    let mut avg_batch_size = 0.0;
    let mut requests_batch = vec![];
    
    while receiver.recv_many(&mut requests_batch, 1024).await > 0 {
        num_inferences += 1.0;
        avg_batch_size += process_batch(&mut requests_batch, model, &cache).await;
        thread::sleep(time::Duration::new(0, 1000));
    }

    avg_batch_size /= num_inferences;
    let result = tasks.join_all().await;
    (avg_batch_size, result.into_iter().flatten().collect())
}

pub async fn process_batch<B: Backend>(requests_batch: &mut Vec<InferenceRequest>, model: &AlphaZero<B>, cache: &PriorsCache) -> f32 {
    let batch_size = requests_batch.len() as f32;
    let states_to_infer = requests_batch.iter()
        .map(|req| req.state.to_tensor())
        .collect::<Vec<_>>();

    let raw_states = requests_batch.iter()
        .map(|req| req.state.get_state())
        .collect::<Vec<_>>();
    
    let response_senders = requests_batch.drain(..)
        .map(|req| req.response_sender)
        .collect::<Vec<_>>();

    let state_tensor = Tensor::cat(states_to_infer, 0);

    let (policy_tensor, value_tensor) = model.forward(state_tensor);

    let policies = policy_tensor.iter_dim(0).map(|p| {
        p.into_data()
            .into_vec()
            .unwrap()
            .try_into()
            .unwrap()
    }).collect::<Vec<[f32; 7]>>();

    let values = value_tensor.iter_dim(0).map(|v| {
        v.into_scalar().to_f32()
    }).collect::<Vec<f32>>();

    let mut cache_writer = cache.write().await;

    for (i, response_sender) in response_senders.into_iter().enumerate() {
        cache_writer.insert(raw_states[i], CacheEntry { policy: policies[i], value: values[i] });
        let response = InferenceResult {
            policy: policies[i],
            value: values[i],
        };
        response_sender.send(response).expect("An error occured!");
    }

    batch_size
}

pub fn get_cyclical_lr(iteration: usize) -> f64 {
    let decay_factor = (iteration / DECAY_INTERVAL) as u32;
    let decay_multiplier = 10f64.powi(-(decay_factor as i32));

    let base_lr = BASE_LEARNING_RATE * decay_multiplier;
    let max_lr = MAX_LEARNING_RATE * decay_multiplier;

    let current_iter_in_cycle = iteration % FULL_CYCLE;
    let lr_range = max_lr - base_lr;

    if current_iter_in_cycle <= HALF_CYCLE {
        let progress = current_iter_in_cycle as f64 / HALF_CYCLE as f64;
        base_lr + progress * lr_range
    } else {
        let progress = (current_iter_in_cycle - HALF_CYCLE) as f64 / HALF_CYCLE as f64;
        max_lr - progress * lr_range
    }
}

pub fn generate_symmetry(step: &EpisodeStep) -> EpisodeStep {
    let mut new_improved_policy = step.improved_policy.clone();
    new_improved_policy.reverse();

    EpisodeStep {
        state: step.state.symmetrize(),
        improved_policy: new_improved_policy,
        final_value: step.final_value,
        search_depth: step.search_depth,
    }
}