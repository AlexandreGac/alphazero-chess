use crate::validation::{evaluate, Player};
use burn::prelude::*;
use tokio::runtime::Runtime;

pub fn compute_elo_rankings<B: Backend>(players: Vec<Player<B>>, base_elo: f32, runtime: &Runtime, pretty_print: bool) -> (f32, Vec<f32>, Vec<Vec<f32>>) {
    let num_players = players.len();
    let mut winrate_matrix = vec![vec![0.5; num_players]; num_players];
    let mut num_inferences = 0.0;
    let mut avg_batch_size = 0.0;
    for i in 0..num_players {
        for j in 0..i {
            if pretty_print {
                println!("Computing match {i} vs {j}");
            }
            let result = runtime.block_on(evaluate(&players[i], &players[j]));
            avg_batch_size = result.avg_batch_size * result.num_inferences + avg_batch_size * num_inferences;
            num_inferences += result.num_inferences;
            avg_batch_size /= num_inferences;

            let winrate = result.winrate;
            winrate_matrix[j][i] = 1.0 - winrate;
            winrate_matrix[i][j] = winrate;
        }
    }

    let elo_rankings = compute_elos(&winrate_matrix, base_elo);

    if pretty_print {
        println!();
        let mut max_name_len = 0;
        let player_names: Vec<String> = (0..num_players)
            .map(|i| {
                let name = match &players[i] {
                    Player::MctsModel(_) => format!("AZ{}", i),
                    Player::MiniMax(_) => format!("MM{}", i),
                    Player::Random => format!("RD{}", i),
                    _ => panic!("Invalid player!")
                };
                max_name_len = usize::max(max_name_len, name.len());
                name
            })
            .collect();

        println!("\n\n{:-^width$}", " Tournament Results ", width = 80);

        println!("\n--- Winrate Matrix (%) ---");
        const CELL_WIDTH: usize = 7;

        print!("{:<width$}", "", width = max_name_len + 2);
        for name in &player_names {
            print!("{:^width$}", name, width = CELL_WIDTH);
        }
        println!();

        for i in 0..num_players {
            print!("{:<width$}|", player_names[i], width = max_name_len + 1);
            for j in 0..num_players {
                if i == j {
                    print!("{:^width$}", "-", width = CELL_WIDTH);
                } else {
                    let percent = winrate_matrix[i][j] * 100.0;
                    print!("{:^width$.1}", percent, width = CELL_WIDTH);
                }
            }
            println!();
        }

        println!("\n--- Final Elo Rankings ---");

        let mut ranked_players: Vec<_> = player_names
            .into_iter()
            .zip(elo_rankings.clone().into_iter())
            .collect();

        ranked_players.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!(
            "{:<5} | {:<width$} | {:<10}",
            "Rank",
            "Player",
            "Elo",
            width = max_name_len
        );
        println!(
            "{:-<5}-+-{:-<width$}-+-{:-<10}",
            "",
            "",
            "",
            width = max_name_len
        );

        for (i, (name, elo)) in ranked_players.iter().enumerate() {
            let elo_str = if name.ends_with(" 0") {
                format!("{:.0} (Baseline)", elo)
            } else {
                format!("{:.0}", elo.round())
            };

            println!(
                "{:<5} | {:<width$} | {}",
                format!("{}.", i + 1),
                name,
                elo_str,
                width = max_name_len
            );
        }
        println!("{:-^width$}", " End of Report ", width = 80);
    }

    (avg_batch_size, elo_rankings, winrate_matrix)
}

fn compute_elos(winrate_matrix: &Vec<Vec<f32>>, base_elo: f32) -> Vec<f32> {
    let num_players = winrate_matrix.len();
    const LEARNING_RATE: f32 = 8.0;
    const ITERATIONS: usize = 1000;

    let mut elos = vec![base_elo; num_players];

    for _ in 0..ITERATIONS {
        let prev_elos = elos.clone();

        for i in 1..num_players {
            let mut actual_score = 0.0;
            let mut expected_score = 0.0;

            for j in 0..num_players {
                if i == j {
                    continue;
                }

                actual_score += winrate_matrix[i][j];

                let elo_diff = prev_elos[j] - prev_elos[i];
                expected_score += 1.0 / (1.0 + 10.0_f32.powf(elo_diff / 400.0));
            }

            let elo_update = LEARNING_RATE * (actual_score - expected_score);
            elos[i] += elo_update;
        }
    }

    elos
}