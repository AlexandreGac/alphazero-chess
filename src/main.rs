// Added Piece and Role for the visual print_board function
use shakmaty::{Chess, Color, File, Move, Position, Rank, Square, Piece, Role};
use shakmaty::uci::UciMove;
use rand::prelude::*; // Using your exact import
use std::io::{self, Write};
use std::str::FromStr;

// --- VISUAL ENHANCEMENTS START HERE ---
// This is the only section that has been changed.

const SEARCH_DEPTH: u8 = 3;

const RESET_COLOR: &str = "\x1b[0m";
const LIGHT_SQUARE_BG: &str = "\x1b[48;5;222m";
const DARK_SQUARE_BG: &str = "\x1b[48;5;173m";

/// Returns the Unicode character for a given chess piece.
fn get_piece_symbol(piece: &Piece) -> &'static str {
    match (piece.role, piece.color) {
        (Role::Pawn, Color::White) => "♙", (Role::Pawn, Color::Black) => "♟",
        (Role::Knight, Color::White) => "♘", (Role::Knight, Color::Black) => "♞",
        (Role::Bishop, Color::White) => "♗", (Role::Bishop, Color::Black) => "♝",
        (Role::Rook, Color::White) => "♖", (Role::Rook, Color::Black) => "♜",
        (Role::Queen, Color::White) => "♕", (Role::Queen, Color::Black) => "♛",
        (Role::King, Color::White) => "♔", (Role::King, Color::Black) => "♚",
    }
}

/// Prints the board to the console with Unicode characters and colors.
fn print_board(pos: &Chess) {
    println!("\n     a  b  c  d  e  f  g  h");
    println!("   +------------------------+");
    for rank_idx in (0..8).rev() {
        print!(" {} |", rank_idx + 1);
        for file_idx in 0..8 {
            let bg_color = if (rank_idx + file_idx) % 2 == 0 { DARK_SQUARE_BG } else { LIGHT_SQUARE_BG };
            let square = Square::from_coords(File::new(file_idx), Rank::new(rank_idx));
            
            let symbol = match pos.board().piece_at(square) {
                Some(piece) => get_piece_symbol(&piece),
                None => " ",
            };
            
            print!("{} {} {}", bg_color, symbol, RESET_COLOR);
        }
        println!("| {}", rank_idx + 1);
    }
    println!("   +------------------------+");
    println!("     a  b  c  d  e  f  g  h\n");
}

// --- VISUAL ENHANCEMENTS END HERE ---
// The rest of the code is your version, untouched.

/// Prompts the user for a move and returns a legal move if one is entered.
fn get_player_move(pos: &Chess) -> Move {
    loop {
        print!("Your move (e.g., e2e4, g1f3, e7e8q for promotion): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");

        let input = input.trim();
        
        match UciMove::from_str(input) {
            Ok(uci) => {
                match uci.to_move(pos) {
                    Ok(legal_move) => return legal_move,
                    Err(_) => println!("That's an illegal move. Try again."),
                }
            }
            Err(_) => {
                println!("Invalid move format. Please use UCI format (e.g., 'e2e4').");
            }
        }
    }
}

// We use centipawns to avoid floating point numbers. 100 centipawns = 1 pawn.
const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 330;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;
const MATE_SCORE: i32 = 20000;


fn evaluate_material(pos: &Chess) -> i32 {
    let board = pos.board();
    let mut score = 0;
    let us = pos.turn();
    let them = !us;

    for (role, value) in [(Role::Pawn, PAWN_VALUE), (Role::Knight, KNIGHT_VALUE), (Role::Bishop, BISHOP_VALUE), (Role::Rook, ROOK_VALUE), (Role::Queen, QUEEN_VALUE)] {
        score += (*board.material_side(us).get(role) as i32) * value;
        score -= (*board.material_side(them).get(role) as i32) * value;
    }
    score
}

/// The recursive Negamax function.
/// Returns the score of the position from the perspective of the current player to move.
fn negamax(pos: &Chess, depth: u8) -> i32 {
    // Base case: if we've reached max depth or the game is over
    if depth == 0 || pos.is_game_over() {
        // Handle checkmate/stalemate explicitly
        let outcome = pos.outcome();
        if outcome.is_known() {
            if let Some(_) = outcome.winner() {
                // We have been checkmated, which is the worst possible outcome.
                // We add the depth to the score to prefer longer mates.
                return -MATE_SCORE - (depth as i32); 
            }
            else {
                return 0; // A draw is a neutral score
            }
        }

        // Otherwise, return the static material evaluation
        return evaluate_material(pos);
    }

    let mut max = i32::MIN;
    let legal_moves = pos.legal_moves();

    // Go through each legal move
    for m in legal_moves {
        let mut child_pos = pos.clone();
        child_pos.play_unchecked(m);
        // The score is the opposite of the opponent's best score
        let score = -negamax(&child_pos, depth - 1);
        if score > max {
            max = score;
        }
    }
    max
}


/// The top-level function that initiates the search.
fn get_ai_move(pos: &Chess, depth: u8) -> Option<Move> {
    let legal_moves = pos.legal_moves();
    if legal_moves.is_empty() {
        return None;
    }

    let mut best_score = i32::MIN;
    // We'll store all moves that result in the best score to add some variety.
    let mut best_moves: Vec<Move> = Vec::new();

    for m in legal_moves {
        let mut temp_pos = pos.clone();
        temp_pos.play_unchecked(m);
        // The score is from the opponent's perspective, so we negate it to get our score.
        let score = -negamax(&temp_pos, depth - 1);

        if score > best_score {
            best_score = score;
            best_moves.clear();
            best_moves.push(m.clone());
        } else if score == best_score {
            best_moves.push(m.clone());
        }
    }
    
    // Choose randomly from the set of best moves.
    best_moves.choose(&mut rand::rng()).cloned()
}



fn main() {
    let mut game = Chess::default();

    loop {
        print_board(&game);

        if game.is_game_over() {
            println!("Game Over!");
            let outcome = game.outcome();
            if outcome.is_known() {
                match outcome.winner() {
                    Some(color) => println!("{} wins!", color),
                    None => println!("It's a draw!"),
                }
            }
            break;
        }

        match game.turn() {
            Color::White => {
                println!("It's your turn (White).");
                let player_move = get_player_move(&game);
                // Using game.play_unchecked(player_move) as per your code
                game.play_unchecked(player_move);
            }
            Color::Black => {
                println!("It's the AI's turn (Black).");
                if let Some(ai_move) = get_ai_move(&game, SEARCH_DEPTH) {
                    println!("AI plays: {}", ai_move.to_string());
                    // Using game.play_unchecked(ai_move) as per your code
                    game.play_unchecked(ai_move);
                }
            }
        }
    }
}