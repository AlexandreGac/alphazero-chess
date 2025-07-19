use burn::prelude::Backend;

use crate::{create_player, parameters::TEMPERATURE_ANNEALING, validation::{play_one_game, Player}};

pub fn play_against_model<B: Backend>(model_path: &String) {
    let opponent = create_player::<B>(&model_path);
    
    println!("==== Game starts ====");
    let human_player = Player::Human;

    let human_start = true;
    let result = if human_start {
        play_one_game(&human_player, &opponent, TEMPERATURE_ANNEALING)
    } else {
        -play_one_game(&opponent, &human_player, TEMPERATURE_ANNEALING)
    };

    if result > 0.0 {
        println!("Human Win!");
    } else if result == 0.0 {
        println!("Draw!");
    } else {
        println!("Opponent Win!");
    }
}