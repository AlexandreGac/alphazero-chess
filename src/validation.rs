use std::{io::{self, Write}, str::FromStr};

use crate::{agent::AlphaZero, chess::{board_to_string, get_best_move, index_to_move, move_to_index, play_move, to_tensor, GameResult}, parameters::{ACTION_SPACE, BATCH_SIZE, CACHE_CAPACITY, EVALUATION_GAMES, TEMPERATURE_ANNEALING}, training::{process_batch, CacheEntry, InferenceRequest, InferenceResult}, tree::MCTree};
use burn::prelude::*;
use moka::future::Cache;
use rand::prelude::*;
use rand::distributions::weighted::WeightedIndex;
use shakmaty::{fen::Fen, uci::UciMove, Chess, Color, Position};
use tokio::{sync::{mpsc::{self, UnboundedSender}, oneshot::channel}, task::JoinSet};

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

pub fn play_one_game<B: Backend>(player1: &Player<B>, player2: &Player<B>, num_stochastic_moves: u32) -> f32 {
    let mut game = Chess::new();
    let mut rng = thread_rng();
    let players = [player1, player2];

    loop {
        let current_player_idx = if game.turn() == Color::White { 0 } else { 1 };
        let current_player = players[current_player_idx];

        let action_index = match current_player {
            Player::MctsModel(model) => {
                let mut search_tree = MCTree::init(model, game.clone(), false);
                let improved_policy = search_tree.monte_carlo_tree_search(model);

                if game.fullmoves().get() > num_stochastic_moves {
                    improved_policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                }
                else {
                    let distribution = WeightedIndex::new(*improved_policy).unwrap();
                    distribution.sample(&mut rng)
                }
            }
            Player::BaseModel(model) => {
                let legal_moves = game.legal_moves()
                    .iter()
                    .map(|m| move_to_index(m, game.turn()))
                    .collect::<Vec<_>>();

                let (policy_tensor, _value_tensor) = model.forward(to_tensor(&game));
                let mut policy: [f32; ACTION_SPACE] = policy_tensor.into_data()
                    .into_vec()
                    .unwrap()
                    .try_into()
                    .unwrap();

                for i in 0..ACTION_SPACE {
                    if !legal_moves.contains(&i) {
                        policy[i] = 0.0;
                    }
                }

                if game.fullmoves().get() > num_stochastic_moves {
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
                println!("{}", board_to_string(&game));

                let action = loop {
                    print!("Your move (e.g., e2e4, g1f3, e7e8q for promotion): ");
                    io::stdout().flush().unwrap();

                    let mut input = String::new();
                    io::stdin().read_line(&mut input).expect("Failed to read line");

                    let input = input.trim();
                    
                    match UciMove::from_str(input) {
                        Ok(uci) => {
                            match uci.to_move(&game) {
                                Ok(legal_move) => break legal_move,
                                Err(_) => println!("That's an illegal move. Try again."),
                            }
                        }
                        Err(_) => {
                            println!("Invalid move format. Please use UCI format (e.g., 'e2e4').");
                        }
                    }
                };

                move_to_index(&action, game.turn())
            }
            Player::MiniMax(n) => {
                move_to_index(
                    &get_best_move(&game, *n).expect("No best move found!"),
                    game.turn()
                )
            }
            Player::Random => {
                let legal_moves = game.legal_moves()
                    .iter()
                    .map(|m| move_to_index(m, game.turn()))
                    .collect::<Vec<_>>();
                *legal_moves.choose(&mut rng).unwrap()
            }
        };

        let action = index_to_move(action_index, &game).expect("Invalid move!");
        match play_move(&mut game, action) {
            Ok(GameResult::Draw) => return 0.0,
            Ok(GameResult::WhiteWins) => return 1.0,
            Ok(GameResult::BlackWins) => return -1.0,
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
    MctsModel(UnboundedSender<InferenceRequest>, Cache<Fen, CacheEntry>),
    BaseModel(UnboundedSender<InferenceRequest>, Cache<Fen, CacheEntry>),
    MiniMax(u32),
    Random
}

pub async fn evaluate<B: Backend>(player_1: &Player<B>, player_2: &Player<B>) -> EvaluationResult {
    let (sender_1, mut receiver_1) = mpsc::unbounded_channel::<InferenceRequest>();
    let (sender_2, mut receiver_2) = mpsc::unbounded_channel::<InferenceRequest>();
    let cache_1 = Cache::new(CACHE_CAPACITY / 2);
    let cache_2 = Cache::new(CACHE_CAPACITY / 2);

    let mut tasks = JoinSet::new();
    for i in 0..EVALUATION_GAMES {
        let async_player_1 = match player_1 {
            Player::MctsModel(_) => {
                AsyncPlayer::MctsModel(sender_1.clone(), cache_1.clone())
            }
            Player::BaseModel(_) => {
                AsyncPlayer::BaseModel(sender_1.clone(), cache_1.clone())
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
                AsyncPlayer::MctsModel(sender_2.clone(), cache_2.clone())
            }
            Player::BaseModel(_) => {
                AsyncPlayer::BaseModel(sender_2.clone(), cache_2.clone())
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
                play_evaluation_game(&async_player_1, &async_player_2, TEMPERATURE_ANNEALING).await
            } else {
                -play_evaluation_game(&async_player_2, &async_player_1, TEMPERATURE_ANNEALING).await
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
            received = receiver_1.recv_many(&mut requests_batch_1, BATCH_SIZE) => {
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
            received = receiver_2.recv_many(&mut requests_batch_2, BATCH_SIZE) => {
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

async fn play_evaluation_game(player_1: &AsyncPlayer, player_2: &AsyncPlayer, num_stochastic_moves: u32) -> f32 {
    let mut game = Chess::new();
    let players = [player_1, player_2];

    loop {
        let current_player_idx = if game.turn() == Color::White { 0 } else { 1 };
        let current_player = players[current_player_idx];

        let action_index = match current_player {
            AsyncPlayer::MctsModel(sender, cache) => {
                let mut search_tree = MCTree::async_init(sender, cache, &game, false).await;
                let improved_policy = search_tree.async_monte_carlo_tree_search(sender, cache).await;

                if game.fullmoves().get() > num_stochastic_moves {
                    improved_policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                }
                else {
                    let distribution = WeightedIndex::new(*improved_policy).unwrap();
                    distribution.sample(&mut thread_rng())
                }
            }
            AsyncPlayer::BaseModel(sender, cache) => {
                let state_key = Fen::from_position(&game, shakmaty::EnPassantMode::PseudoLegal);
                let mut policy = if let Some(entry) = cache.get(&state_key).await {
                    entry.policy.clone()
                }
                else {
                    let (response_sender, response_receiver) = channel();
                    sender.send(InferenceRequest {
                        state: game.clone(),
                        response_sender: response_sender,
                    }).expect("Failed to send request!");

                    let InferenceResult { policy, .. } = response_receiver.await.expect("No result provided!");
                    policy
                };

                let legal_moves = game.legal_moves()
                    .iter()
                    .map(|m| move_to_index(m, game.turn()))
                    .collect::<Vec<_>>();

                for i in 0..ACTION_SPACE {
                    if !legal_moves.contains(&i) {
                        policy[i] = 0.0;
                    }
                }

                if game.fullmoves().get() > num_stochastic_moves {
                    policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                } else {
                    let distribution = match WeightedIndex::new(*policy) {
                        Ok(dist) => dist,
                        Err(_) => panic!("Distribution creation failed with {policy:?}"),
                    };
                    distribution.sample(&mut thread_rng())
                }
            }
            AsyncPlayer::MiniMax(n) => {
                move_to_index(
                    &get_best_move(&game, *n).expect("No best move found!"),
                    game.turn()
                )
            }
            AsyncPlayer::Random => {
                let legal_moves = game.legal_moves()
                    .iter()
                    .map(|m| move_to_index(m, game.turn()))
                    .collect::<Vec<_>>();
                *legal_moves.choose(&mut thread_rng()).unwrap()
            }
        };

        let action = index_to_move(action_index, &game).expect("Invalid move!");
        match play_move(&mut game, action) {
            Ok(GameResult::Draw) => return 0.0,
            Ok(GameResult::WhiteWins) => return 1.0,
            Ok(GameResult::BlackWins) => return -1.0,
            Err(_) => {
                let player_name = match current_player {
                    AsyncPlayer::MiniMax(_) => "MiniMax Bot",
                    AsyncPlayer::MctsModel(_, _) => "MCTS AI",
                    AsyncPlayer::BaseModel(_, _) => "Base AI",
                    AsyncPlayer::Random => "Random Bot",
                };
                panic!("{} played an illegal move!", player_name);
            }
            Ok(_) => continue,
        }
    }
}

pub fn sample_search_tree<B: Backend>(player1: &Player<B>, player2: &Player<B>, num_stochastic_moves: u32) -> Option<MCTree> {
    let mut game = Chess::new();
    let mut rng = thread_rng();
    let players = [player1, player2];
    let mut model_generated_trees: Vec<MCTree> = Vec::new();

    loop {
        let current_player_idx = if game.turn() == Color::White { 0 } else { 1 };
        let current_player = players[current_player_idx];

        let action_index = match current_player {
            Player::MctsModel(model) => {
                let mut search_tree = MCTree::init(model, game.clone(), false);
                let improved_policy = search_tree.monte_carlo_tree_search(model);
                model_generated_trees.push(search_tree);

                if game.fullmoves().get() > num_stochastic_moves {
                    improved_policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap()
                }
                else {
                    let distribution = WeightedIndex::new(*improved_policy).unwrap();
                    distribution.sample(&mut rng)
                }
            }
            Player::BaseModel(model) => {
                let legal_moves = game.legal_moves()
                    .iter()
                    .map(|m| move_to_index(m, game.turn()))
                    .collect::<Vec<_>>();

                let (policy_tensor, _value_tensor) = model.forward(to_tensor(&game));
                let mut policy: [f32; ACTION_SPACE] = policy_tensor.into_data()
                    .into_vec()
                    .unwrap()
                    .try_into()
                    .unwrap();

                for i in 0..ACTION_SPACE {
                    if !legal_moves.contains(&i) {
                        policy[i] = 0.0;
                    }
                }

                if game.fullmoves().get() > num_stochastic_moves {
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
                println!("{}", board_to_string(&game));

                let action = loop {
                    print!("Your move (e.g., e2e4, g1f3, e7e8q for promotion): ");
                    io::stdout().flush().unwrap();

                    let mut input = String::new();
                    io::stdin().read_line(&mut input).expect("Failed to read line");

                    let input = input.trim();
                    
                    match UciMove::from_str(input) {
                        Ok(uci) => {
                            match uci.to_move(&game) {
                                Ok(legal_move) => break legal_move,
                                Err(_) => println!("That's an illegal move. Try again."),
                            }
                        }
                        Err(_) => {
                            println!("Invalid move format. Please use UCI format (e.g., 'e2e4').");
                        }
                    }
                };

                move_to_index(&action, game.turn())
            }
            Player::MiniMax(n) => {
                move_to_index(
                    &get_best_move(&game, *n).expect("No best move found!"),
                    game.turn()
                )
            }
            Player::Random => {
                let legal_moves = game.legal_moves()
                    .iter()
                    .map(|m| move_to_index(m, game.turn()))
                    .collect::<Vec<_>>();
                *legal_moves.choose(&mut rng).unwrap()
            }
        };

        let action = index_to_move(action_index, &game).expect("Invalid move!");
        match play_move(&mut game, action) {
            Ok(GameResult::Draw) => break,
            Ok(GameResult::WhiteWins) => break,
            Ok(GameResult::BlackWins) => break,
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
    let index = rng.gen_range(0..model_generated_trees.len());
        Some(model_generated_trees.remove(index))
    }
}