use std::{collections::HashMap, io, sync::Arc};

use crate::{agent::AlphaZero, connect4::Connect4, parameters::EVALUATION_GAMES, training::{process_batch, InferenceRequest, InferenceResult, PriorsCache}, tree::MCTree};
use burn::prelude::*;
use rand::prelude::*;
use rand::{distr::weighted::WeightedIndex, seq::IndexedRandom};
use tokio::{sync::{mpsc::{self, UnboundedSender}, oneshot::channel, RwLock}, task::JoinSet};

pub struct EvaluationResult {
    pub num_inferences: f32,
    pub avg_batch_size: f32,
    pub winrate: f32,
    pub p1_winrate: f32,
    pub p2_winrate: f32,
    pub drawrate: f32
}

pub enum Player<B: Backend> {
    MctsModel(AlphaZero<B>),
    BaseModel(AlphaZero<B>),
    MiniMax(u32),
    Random,
    Human,
}

pub fn play_one_game<B: Backend>(player1: &Player<B>, player2: &Player<B>, num_stochastic_moves: usize) -> f32 {
    let mut game = Connect4::new();
    let mut rng = rand::rng();
    let players = [player1, player2];

    loop {
        let current_player_idx = if game.get_turn() > 0.0 { 0 } else { 1 };
        let current_player = players[current_player_idx];

        let action_index = match current_player {
            Player::MctsModel(model) => {
                let mut search_tree = MCTree::init(model, game.clone(), false);
                let improved_policy = search_tree.monte_carlo_tree_search(model);

                if game.get_total_moves() >= num_stochastic_moves {
                    improved_policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                }
                else {
                    let distribution = WeightedIndex::new(&improved_policy).unwrap();
                    distribution.sample(&mut rng)
                }
            }
            Player::BaseModel(model) => {
                let legal_moves = game.get_legal_moves();
                let (policy_tensor, _value_tensor) = model.forward(game.to_tensor());
                let mut policy: [f32; 7] = policy_tensor.into_data()
                    .into_vec()
                    .unwrap()
                    .try_into()
                    .unwrap();

                for i in 0..7 {
                    if !legal_moves.contains(&i) {
                        policy[i] = 0.0;
                    }
                }

                if game.get_total_moves() >= num_stochastic_moves {
                    policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                } else {
                    let distribution = WeightedIndex::new(&policy).unwrap();
                    distribution.sample(&mut rng)
                }
            }
            Player::Human => {
                game.display();
                let legal_moves = game.get_legal_moves();
                let current_player_symbol = if game.get_turn() == 1.0 { 'X' } else { 'O' };
                println!("Turn: Player {}", current_player_symbol);
                println!("Enter column (0-6): ");

                loop {
                    let mut input = String::new();
                    io::stdin().read_line(&mut input).expect("Failed to read line");
                    match input.trim().parse::<usize>() {
                        Ok(num) => {
                            if num <= 6 {
                                if legal_moves.contains(&num) {
                                    println!("");
                                    break num
                                }
                                else {
                                    println!("You made an illegal move!");
                                }
                            } else {
                                println!("Invalid column. Please enter a number between 0 and 6.");
                            }
                        }
                        Err(_) => {
                            println!("Invalid input. Please enter a number.");
                        }
                    }
                }
            }
            Player::MiniMax(n) => {
                game.find_best_move(*n).expect("No best move found!")
            }
            Player::Random => {
                let legal_moves = game.get_legal_moves();
                *legal_moves.choose(&mut rng).unwrap()
            }
        };

        match game.play(action_index) {
            Ok("Draw") => return 0.0,
            Ok("P1 Win") => return 1.0,
            Ok("P2 Win") => return -1.0,
            Err(_) => {
                let player_name = match current_player {
                    Player::MiniMax(_) => "MiniMax Bot",
                    Player::MctsModel(_) => "MCTS AI",
                    Player::BaseModel(_) => "Base AI",
                    Player::Human => "Human player",
                    Player::Random => "Random Bot",
                };
                panic!("{} played an illegal move!", player_name);
            }
            Ok(_) => continue,
        }
    }
}

pub enum AsyncPlayer {
    MctsModel(UnboundedSender<InferenceRequest>, PriorsCache),
    BaseModel(UnboundedSender<InferenceRequest>, PriorsCache),
    MiniMax(u32),
    Random
}

pub async fn evaluate<B: Backend>(player_1: &Player<B>, player_2: &Player<B>) -> EvaluationResult {
    let (sender_1, mut receiver_1) = mpsc::unbounded_channel::<InferenceRequest>();
    let (sender_2, mut receiver_2) = mpsc::unbounded_channel::<InferenceRequest>();
    let cache_1: PriorsCache = Arc::new(RwLock::new(HashMap::new()));
    let cache_2: PriorsCache = Arc::new(RwLock::new(HashMap::new()));

    let mut tasks = JoinSet::new();
    for i in 0..EVALUATION_GAMES {
        let async_player_1 = match player_1 {
            Player::MctsModel(_) => {
                AsyncPlayer::MctsModel(sender_1.clone(), Arc::clone(&cache_1))
            }
            Player::BaseModel(_) => {
                AsyncPlayer::BaseModel(sender_1.clone(), Arc::clone(&cache_1))
            }
            Player::MiniMax(n) => {
                AsyncPlayer::MiniMax(*n)
            },
            Player::Random => {
                AsyncPlayer::Random
            },
            _ => panic!("Unable to compute winrate of this player!")
        };

        let async_player_2 = match player_2 {
            Player::MctsModel(_) => {
                AsyncPlayer::MctsModel(sender_2.clone(), Arc::clone(&cache_2))
            }
            Player::BaseModel(_) => {
                AsyncPlayer::BaseModel(sender_2.clone(), Arc::clone(&cache_2))
            }
            Player::MiniMax(n) => {
                AsyncPlayer::MiniMax(*n)
            },
            Player::Random => {
                AsyncPlayer::Random
            },
            _ => panic!("Unable to compute winrate of this player!")
        };

        tasks.spawn(async move {
           if i % 2 == 0 {
                play_evaluation_game(&async_player_1, &async_player_2, 6).await
            } else {
                -play_evaluation_game(&async_player_2, &async_player_1, 6).await
            }
        });
    }

    drop(sender_1);
    drop(sender_2);

    let mut finished_1 = false;
    let mut finished_2 = false;
    let mut num_inferences = 0.0;
    let mut avg_batch_size = 0.0;
    let mut requests_batch_1 = vec![];
    let mut requests_batch_2 = vec![];
    while !(finished_1 && finished_2) {
        tokio::select! {
            received = receiver_1.recv_many(&mut requests_batch_1, 1024) => {
                if received > 0 {
                    match player_1 {
                        Player::MctsModel(model_1) => {
                            num_inferences += 1.0;
                            avg_batch_size += process_batch(&mut requests_batch_1, model_1, &cache_1).await;
                        }
                        Player::BaseModel(model_1) => {
                            num_inferences += 1.0;
                            avg_batch_size += process_batch(&mut requests_batch_1, model_1, &cache_1).await;
                        }
                        _ => unreachable!("No request if no model!")
                    }
                }
                else {
                    finished_1 = true;
                }
            },
            received = receiver_2.recv_many(&mut requests_batch_2, 1024) => {
                if received > 0 {
                    match player_2 {
                        Player::MctsModel(model_2) => {
                            num_inferences += 1.0;
                            avg_batch_size += process_batch(&mut requests_batch_2, model_2, &cache_2).await;
                        }
                        Player::BaseModel(model_2) => {
                            num_inferences += 1.0;
                            avg_batch_size += process_batch(&mut requests_batch_2, model_2, &cache_2).await;
                        }
                        _ => unreachable!("No request if no model!")
                    }
                }
                else {
                    finished_2 = true;
                }
            }
        };
    }

    avg_batch_size /= num_inferences;

    let results = tasks.join_all().await;
    let mut p1_wins = 0.0;
    let mut draws = 0.0;

    for result in results {
        if result > 0.0 {
            p1_wins += 1.0;
        } else if result == 0.0 {
            draws += 1.0;
        }
    }

    let winrate = (p1_wins + draws / 2.0) / EVALUATION_GAMES as f32;
    let p2_wins = EVALUATION_GAMES as f32 - p1_wins - draws;
    let p1_winrate = p1_wins / EVALUATION_GAMES as f32;
    let drawrate = draws / EVALUATION_GAMES as f32;
    let p2_winrate = p2_wins / EVALUATION_GAMES as f32;

    EvaluationResult {
        num_inferences,
        avg_batch_size,
        winrate,
        p1_winrate,
        p2_winrate,
        drawrate
    }
}

async fn play_evaluation_game(player_1: &AsyncPlayer, player_2: &AsyncPlayer, num_stochastic_moves: usize) -> f32 {
    let mut game = Connect4::new();
    let players = [player_1, player_2];

    loop {
        let current_player_idx = if game.get_turn() > 0.0 { 0 } else { 1 };
        let current_player = players[current_player_idx];

        let action_index = match current_player {
            AsyncPlayer::MctsModel(sender, cache) => {
                let mut search_tree = MCTree::async_init(sender, cache, &game, false).await;
                let improved_policy = search_tree.async_monte_carlo_tree_search(sender, cache).await;

                if game.get_total_moves() >= num_stochastic_moves {
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
                }
            }
            AsyncPlayer::BaseModel(sender, cache) => {
                let cache_reader = cache.read().await;
                let mut policy = if let Some(entry) = cache_reader.get(&game.get_state()) {
                    entry.policy
                }
                else {
                    drop(cache_reader);

                    let (response_sender, response_receiver) = channel();
                    sender.send(InferenceRequest {
                        state: game.clone(),
                        response_sender: response_sender,
                    }).expect("Failed to send request!");

                    let InferenceResult { policy, .. } = response_receiver.await.expect("No result provided!");
                    policy
                };

                let legal_moves = game.get_legal_moves();
                for i in 0..7 {
                    if !legal_moves.contains(&i) {
                        policy[i] = 0.0;
                    }
                }

                if game.get_total_moves() >= num_stochastic_moves {
                    policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                } else {
                    let distribution = match WeightedIndex::new(&policy) {
                        Ok(dist) => dist,
                        Err(_) => panic!("Distribution creation failed with {policy:?}"),
                    };
                    distribution.sample(&mut rand::rng())
                }
            }
            AsyncPlayer::MiniMax(n) => {
                game.find_best_move(*n).expect("No best move found!")
            }
            AsyncPlayer::Random => {
                let legal_moves = game.get_legal_moves();
                *legal_moves.choose(&mut rand::rng()).unwrap()
            }
        };

        match game.play(action_index) {
            Ok("Draw") => return 0.0,
            Ok("P1 Win") => return 1.0,
            Ok("P2 Win") => return -1.0,
            Err(_) => {
                let player_name = match current_player {
                    AsyncPlayer::MctsModel(_, _) => "MCTS AI",
                    AsyncPlayer::BaseModel(_, _) => "Base AI",
                    AsyncPlayer::MiniMax(_) => "MiniMax Bot",
                    AsyncPlayer::Random => "Random Bot"
                };
                panic!("{} played an illegal move!", player_name);
            }
            Ok(_) => continue,
        }
    }
}

pub fn sample_search_tree<B: Backend>(player1: &Player<B>, player2: &Player<B>, num_stochastic_moves: usize) -> Option<MCTree> {
    let mut game = Connect4::new();
    let mut rng = rand::rng();
    let players = [player1, player2];
    let mut model_generated_trees: Vec<MCTree> = Vec::new();

    loop {
        let current_player_idx = if game.get_turn() > 0.0 { 0 } else { 1 };
        let current_player = players[current_player_idx];

        let action_index = match current_player {
            Player::MctsModel(model) => {
                let mut search_tree = MCTree::init(model, game.clone(), false);
                let improved_policy = search_tree.monte_carlo_tree_search(model);
                model_generated_trees.push(search_tree);

                if game.get_total_moves() >= num_stochastic_moves {
                    improved_policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                }
                else {
                    let distribution = WeightedIndex::new(&improved_policy).unwrap();
                    distribution.sample(&mut rng)
                }
            }
            Player::BaseModel(model) => {
                let legal_moves = game.get_legal_moves();
                let (policy_tensor, _value_tensor) = model.forward(game.to_tensor());
                let mut policy: [f32; 7] = policy_tensor.into_data()
                    .into_vec()
                    .unwrap()
                    .try_into()
                    .unwrap();

                for i in 0..7 {
                    if !legal_moves.contains(&i) {
                        policy[i] = 0.0;
                    }
                }

                if game.get_total_moves() >= num_stochastic_moves {
                    policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                } else {
                    let distribution = WeightedIndex::new(&policy).unwrap();
                    distribution.sample(&mut rng)
                }
            }
            Player::Human => {
                game.display();
                let legal_moves = game.get_legal_moves();
                let current_player_symbol = if game.get_turn() == 1.0 { 'X' } else { 'O' };
                println!("Turn: Player {}", current_player_symbol);
                println!("Enter column (0-6): ");

                loop {
                    let mut input = String::new();
                    io::stdin().read_line(&mut input).expect("Failed to read line");
                    match input.trim().parse::<usize>() {
                        Ok(num) => {
                            if num <= 6 {
                                if legal_moves.contains(&num) {
                                    println!("");
                                    break num
                                }
                                else {
                                    println!("You made an illegal move!");
                                }
                            } else {
                                println!("Invalid column. Please enter a number between 0 and 6.");
                            }
                        }
                        Err(_) => {
                            println!("Invalid input. Please enter a number.");
                        }
                    }
                }
            }
            Player::MiniMax(n) => {
                game.find_best_move(*n).expect("No best move found!")
            }
            Player::Random => {
                let legal_moves = game.get_legal_moves();
                *legal_moves.choose(&mut rng).unwrap()
            }
        };

        match game.play(action_index) {
            Ok("Draw") => break,
            Ok("P1 Win") => break,
            Ok("P2 Win") => break,
            Err(_) => {
                let player_name = match current_player {
                    Player::MiniMax(_) => "MiniMax Bot",
                    Player::MctsModel(_) => "MCTS AI",
                    Player::BaseModel(_) => "Base AI",
                    Player::Human => "Human player",
                    Player::Random => "Random Bot",
                };
                panic!("{} played an illegal move!", player_name);
            }
            Ok(_) => continue,
        }
    }

    if model_generated_trees.is_empty() {
        None
    } else {
    let index = rng.random_range(0..model_generated_trees.len());
        Some(model_generated_trees.remove(index))
    }
}