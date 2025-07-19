#![recursion_limit = "256"]

mod chess;
mod tree;
mod agent;
mod training;
mod validation;
mod inference;
mod ratings;
mod logger;
mod memory;
mod parameters;

use burn::prelude::*;
use burn::{backend::{Autodiff, Cuda}, prelude::Backend, record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder}};
use clap::{Parser, Subcommand};
use tokio::runtime;

use crate::parameters::NUM_THREADS;
use crate::ratings::compute_elo_rankings;
use crate::validation::Player;
use crate::{agent::AlphaZero, inference::play_against_model, training::train};

#[derive(Parser, Debug)]
#[command(author, version, about = "A Rust-based AlphaZero implementation for Chess.", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a new model from scratch
    Train,
    
    /// Play against an opponent in the terminal
    Play {
        /// The opponent to play against. Can be a path to a model file, or one of: "Human", "MiniMax", "Random"
        opponent: String,
    },

    /// Compute Elo ratings for a list of players
    Elo {
        /// Player identifiers to be rated. Can be a path to a model file, or one of: "Random", "MiniMax"
        #[arg(required = true, num_args = 1..)]
        players: Vec<String>,

        /// The fixed Elo rating of the first player
        #[arg(required = true, long)]
        initial_elo: f32,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train => {
            println!("Mode: Training");
            train::<Autodiff<Cuda>>();
        }

        Commands::Play { opponent } => {
            println!("Mode: Play against AI");
            println!("Loading opponent: {}", opponent);
            play_against_model::<Cuda>(opponent);
        }

        Commands::Elo { players, initial_elo} => {
            println!("Mode: Compute Elo Rankings");
            println!("Rating {} players...", players.len());
            
            let runtime = runtime::Builder::new_multi_thread()
                .worker_threads(NUM_THREADS)
                .enable_all()
                .build()
                .expect("Unable to build async runtime!");

            let player_pool: Vec<Player<Cuda>> = players
                .iter()
                .map(|identifier| {
                    println!("- Preparing player: {}", identifier);
                    create_player(identifier)
                })
                .collect();
            
            compute_elo_rankings(player_pool, *initial_elo, &runtime, true);
        }
    }
}

pub fn create_player<B: Backend>(identifier: &str) -> Player<B> {
    match identifier {
        "Random" => Player::Random,
        "MiniMax" => Player::MiniMax(4),
        "Human" => panic!("A 'Human' player cannot be used in Elo computation."),
        model_path => Player::MctsModel(load_model(model_path)),
    }
}

fn load_model<B: Backend>(model_path: &str) -> AlphaZero<B> {
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &B::Device::default())
        .expect("Should be able to load the model weights from the provided file");

    let model = AlphaZero::<B>::new().load_record(record);
    model
}