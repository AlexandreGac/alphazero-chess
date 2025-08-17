use std::{collections::HashMap, fmt::Write, str::FromStr};

use burn::prelude::*;
use rand::prelude::*;
use ratatui::{style::Style, text::{Line, Span, Text}};
use shakmaty::{uci::UciMove, CastlingSide, Chess, Color, File, Move, Piece, Position, Rank, Role, Square};


pub const NUM_HALFMOVES: u32 = 100;
pub const NUM_FULLMOVES: u32 = 200;
pub const REPETITIONS: usize = 3;

#[derive(Debug, Clone)]
pub struct GameState {
    pub position: Chess,
    pub pos_count: HashMap<Chess, usize>
}

impl GameState {
    pub fn new() -> Self {
        let position = Chess::new();
        let mut pos_count = HashMap::new();
        pos_count.insert(position.clone(), 1);

        Self { position, pos_count }
    }
}

pub enum GameResult {
    Ongoing,
    Draw,
    WhiteWins,
    BlackWins
}

pub fn play_move(state: &mut GameState, action: Move) -> Result<GameResult, &'static str> {
    let GameState { position: game, pos_count } = state;
    if !game.is_legal(action) {
        Err("Illegal move")
    }
    else {
        game.play_unchecked(action);
        let outcome = game.outcome();
        if outcome.is_known() {
            Ok(match outcome.winner() {
                Some(Color::White) => GameResult::WhiteWins,
                Some(Color::Black) => GameResult::BlackWins,
                None => GameResult::Draw,
            })
        }
        else {
            let count = pos_count.entry(game.clone()).or_insert(0);
            *count += 1;

            if *count < REPETITIONS && game.halfmoves() < NUM_HALFMOVES && game.fullmoves().get() < NUM_FULLMOVES {
                Ok(GameResult::Ongoing)
            }
            else {
                Ok(GameResult::Draw)
            }
        }
    }
}

/// Converts a valid `Move` into a policy index (0-4095).
///
/// # Move Type Planes (64 total)
/// - **Planes 0-7: Knight Moves** (Clockwise from NNE)
/// - **Planes 8-63: Queen-like Moves** (8 directions * 7 distances)
///   - Directions are ordered clockwise: N, NE, E, SE, S, SW, W, NW.
///   - This single set of planes handles moves for Queens, Rooks, Bishops,
///     Kings (dist 1), and Pawns (including promotions to Queen).
pub fn move_to_index(m: &Move, turn: Color) -> usize {
    let from_square = m.from().expect("The move has no starting square!");
    let dest_square = m.to();

    let file = from_square.file().to_usize();
    let rank = match turn {
        Color::Black => 7 - from_square.rank().to_usize(),
        Color::White => from_square.rank().to_usize(),
    };

    let dest_file = dest_square.file().to_usize();
    let dest_rank = match turn {
        Color::Black => 7 - dest_square.rank().to_usize(),
        Color::White => dest_square.rank().to_usize(),
    };

    let df = dest_file as isize - file as isize;
    let dr = dest_rank as isize - rank as isize;

    let move_plane = match (df, dr) {
        // Knight moves
        ( 1,  2) => 0,
        ( 2,  1) => 1,
        ( 2, -1) => 2,
        ( 1, -2) => 3,
        (-1, -2) => 4,
        (-2, -1) => 5,
        (-2,  1) => 6,
        (-1,  2) => 7,
        // Queen moves
        ( 0     ,  1..=7 ) => 7 + dr,
        ( 1..=7 ,  1..=7 ) => 14 + dr,
        ( 1..=7 ,  0     ) => 21 + df,
        ( 1..=7 , -7..=-1) => 28 + df,
        ( 0     , -7..=-1) => 35 - dr,
        (-7..=-1, -7..=-1) => 42 - dr,
        (-7..=-1,       0) => 49 - df,
        (-7..=-1,  1..=7 ) => 56 - df,
        _ => unreachable!("Illegal move!")
    };

    let index = move_plane as usize * 64 + rank * 8 + file;
    index
}

pub fn index_to_move(index: usize, game: &Chess) -> Option<Move> {
    let move_plane = index / 64;
    let square_index = index % 64;
    let from_file = square_index % 8;
    let canonical_rank = square_index / 8;

    let from_rank = match game.turn() {
        Color::Black => 7 - canonical_rank,
        Color::White => canonical_rank,
    };

    let (df, dr) = match move_plane {
        // Knight moves
        0 => ( 1,  2),
        1 => ( 2,  1),
        2 => ( 2, -1),
        3 => ( 1, -2),
        4 => (-1, -2),
        5 => (-2, -1),
        6 => (-2,  1),
        7 => (-1,  2),
        // Queen moves
         8..15 => (0, move_plane as isize - 7),
        15..22 => (move_plane as isize - 14, move_plane as isize - 14),
        22..29 => (move_plane as isize - 21, 0),
        29..36 => (move_plane as isize - 28, 28 - move_plane as isize),
        36..43 => (0, 35 - move_plane as isize),
        43..50 => (42 - move_plane as isize, 42 - move_plane as isize),
        50..57 => (49 - move_plane as isize, 0),
        57..64 => (56 - move_plane as isize, move_plane as isize - 56),
        _ => unreachable!("Impossible index!")
    };

    let dr = match game.turn() {
        Color::Black => -dr,
        Color::White => dr,
    };

    let dest_file = from_file as isize + df;
    let dest_rank = from_rank as isize + dr;

    if dest_file < 0 || dest_file > 7 || dest_rank < 0 || dest_rank > 7 {
        return None;
    }

    let from_square = Square::from_coords(File::new(from_file as u32), Rank::new(from_rank as u32));
    let dest_square = Square::from_coords(File::new(dest_file as u32), Rank::new(dest_rank as u32));
    let promotion = if game.board().role_at(from_square)? == Role::Pawn && (dest_rank == 0 || dest_rank == 7) {
        "q"
    } else { "" };

    let result = UciMove::from_str(format!("{}{}{}", from_square, dest_square, promotion).as_str());
    result.ok()?.to_move(game).ok()
}

/// Converts a chess game state into a tensor suitable for a neural network.
///
/// The tensor representation is inspired by AlphaZero and is always from the
/// perspective of the current player.
///
/// The output tensor has a shape of `[1, 19, 8, 8]` (batch, channels, rank, file).
/// The 19 feature planes are structured as follows:
///
/// -   **Planes 0-5:** Current player's pieces (Pawn, Knight, Bishop, Rook, Queen, King).
/// -   **Planes 6-11:** Opponent's pieces (Pawn, Knight, Bishop, Rook, Queen, King).
/// -   **Plane 12:** Current player can castle kingside (1.0 if yes, 0.0 if no).
/// -   **Plane 13:** Current player can castle queenside (1.0 if yes, 0.0 if no).
/// -   **Plane 14:** Opponent can castle kingside (1.0 if yes, 0.0 if no).
/// -   **Plane 15:** Opponent can castle queenside (1.0 if yes, 0.0 if no).
/// -   **Plane 16:** En-passant square. A single 1.0 on the valid en-passant target square, 0.0 otherwise.
/// -   **Plane 17:** Halfmove clock (for 50-move rule), normalized by dividing by NUM_HALFMOVES.
/// -   **Plane 18:** Fullmove number, normalized by dividing by NUM_FULLMOVES (max game length).
///
pub fn to_tensor<B: Backend>(game: &Chess) -> Tensor<B, 4> {
    let mut tensor_data = [[[[0.0f32; 8]; 8]; 19]; 1];

    let us = game.turn();
    let them = !us;
    for square in Square::ALL {
        if let Some(piece) = game.board().piece_at(square) {
            let offset = if piece.color == us { 0 } else { 6 };
            let plane = match piece.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            };

            let file = square.file().to_usize();
            let rank = match us {
                Color::Black => 7 - square.rank().to_usize(),
                Color::White => square.rank().to_usize(),
            };

            tensor_data[0][plane + offset][rank][file] = 1.0;
        }
    }

    let castling_rights = game.castles();
    if castling_rights.has(us, CastlingSide::KingSide) {
        tensor_data[0][12] = [[1.0f32; 8]; 8];
    }
    if castling_rights.has(us, CastlingSide::QueenSide) {
        tensor_data[0][13] = [[1.0f32; 8]; 8];
    }
    if castling_rights.has(them, CastlingSide::KingSide) {
        tensor_data[0][14] = [[1.0f32; 8]; 8];
    }
    if castling_rights.has(them, CastlingSide::QueenSide) {
        tensor_data[0][15] = [[1.0f32; 8]; 8];
    }

    if let Some(square) = game.pseudo_legal_ep_square() {
        let file = square.file().to_usize();
        let rank = match us {
            Color::Black => 7 - square.rank().to_usize(),
            Color::White => square.rank().to_usize(),
        };
        tensor_data[0][16][rank][file] = 1.0;
    }

    tensor_data[0][17] = [[game.halfmoves() as f32 / NUM_HALFMOVES as f32; 8]; 8];
    tensor_data[0][18] = [[game.fullmoves().get() as f32 / NUM_FULLMOVES as f32; 8]; 8];

    Tensor::<B, 4>::from_floats(tensor_data, &B::Device::default())
}


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

fn negamax(pos: &Chess, depth: u32) -> i32 {
    if depth == 0 || pos.is_game_over() {
        let outcome = pos.outcome();
        if outcome.is_known() {
            if let Some(_) = outcome.winner() {
                return -MATE_SCORE - (depth as i32); 
            }
            else {
                return 0;
            }
        }

        return evaluate_material(pos);
    }

    let mut max = i32::MIN;
    let legal_moves = pos.legal_moves();

    for m in legal_moves {
        let mut child_pos = pos.clone();
        child_pos.play_unchecked(m);
        let score = -negamax(&child_pos, depth - 1);
        if score > max {
            max = score;
        }
    }
    max
}


pub fn get_best_move(pos: &Chess, depth: u32) -> Option<Move> {
    let legal_moves = pos.legal_moves();
    if legal_moves.is_empty() {
        return None;
    }

    let mut best_score = i32::MIN;
    let mut best_moves: Vec<Move> = Vec::new();

    for m in legal_moves {
        let mut temp_pos = pos.clone();
        temp_pos.play_unchecked(m);
        let score = -negamax(&temp_pos, depth - 1);

        if score > best_score {
            best_score = score;
            best_moves.clear();
            best_moves.push(m.clone());
        } else if score == best_score {
            best_moves.push(m.clone());
        }
    }
    
    best_moves.choose(&mut thread_rng()).cloned()
}


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

pub fn board_to_string(pos: &Chess) -> String {
    const RESET_COLOR: &str = "\x1b[0m";
    const LIGHT_SQUARE_BG: &str = "\x1b[48;5;222m";
    const DARK_SQUARE_BG: &str = "\x1b[48;5;173m";
    
    let mut output = String::with_capacity(1024);

    writeln!(output, "     a  b  c  d  e  f  g  h     ").unwrap();
    writeln!(output, "   ╭────────────────────────╮   ").unwrap();

    for rank_idx in (0..8).rev() {
        write!(output, " {} │", rank_idx + 1).unwrap();

        for file_idx in 0..8 {
            let bg_color = if (rank_idx + file_idx) % 2 == 0 { DARK_SQUARE_BG } else { LIGHT_SQUARE_BG };
            let square = Square::from_coords(File::new(file_idx), Rank::new(rank_idx));
            
            let symbol = match pos.board().piece_at(square) {
                Some(piece) => get_piece_symbol(&piece),
                None => " ",
            };
            write!(output, "{} {} {}", bg_color, symbol, RESET_COLOR).unwrap();
        }
        writeln!(output, "│ {} ", rank_idx + 1).unwrap();
    }

    writeln!(output, "   ╰────────────────────────╯   ").unwrap();
    writeln!(output, "     a  b  c  d  e  f  g  h     ").unwrap();
    writeln!(output, "Turn: {:?}", pos.turn()).unwrap();
    
    output
}

pub fn board_to_text(pos: &Chess) -> Text<'static> {
    const LIGHT_SQUARE_BG: ratatui::style::Color = ratatui::style::Color::Indexed(222); // A light beige
    const DARK_SQUARE_BG: ratatui::style::Color = ratatui::style::Color::Indexed(173);  // A brownish pink
    const BORDER_COLOR: ratatui::style::Color = ratatui::style::Color::Rgb(150, 150, 150); // A nice gray for the border

    let mut lines: Vec<Line> = Vec::with_capacity(14);

    lines.push(Line::from(format!("Turn: {:?}", pos.turn())));
    lines.push(Line::from(vec![
        Span::raw("   "),
        Span::styled("╭────────────────────────╮", Style::default().fg(BORDER_COLOR)),
    ]));

    for rank_idx in (0..8).rev() {
        let mut spans_for_rank: Vec<Span> = Vec::with_capacity(20);

        spans_for_rank.push(Span::from(format!(" {} ", rank_idx + 1)));
        spans_for_rank.push(Span::styled("│", Style::default().fg(BORDER_COLOR)));

        for file_idx in 0..8 {
            let bg_color = if (rank_idx + file_idx) % 2 == 0 {
                DARK_SQUARE_BG
            } else {
                LIGHT_SQUARE_BG
            };
            
            let style = Style::default().bg(bg_color);
            let square = Square::from_coords(File::new(file_idx), Rank::new(rank_idx));
            
            let symbol = match pos.board().piece_at(square) {
                Some(piece) => get_piece_symbol(&piece),
                None => " ",
            };

            spans_for_rank.push(Span::styled(" ", style));
            spans_for_rank.push(Span::styled(symbol, style));
            spans_for_rank.push(Span::styled(" ", style));
        }

        spans_for_rank.push(Span::styled("│", Style::default().fg(BORDER_COLOR)));
        
        lines.push(Line::from(spans_for_rank));
    }

    lines.push(Line::from(vec![
        Span::raw("   "),
        Span::styled("╰────────────────────────╯", Style::default().fg(BORDER_COLOR)),
    ]));
    lines.push(Line::from("     a  b  c  d  e  f  g  h  "));

    Text::from(lines)
}