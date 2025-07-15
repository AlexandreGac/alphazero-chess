use std::{fmt, vec};
use burn::prelude::*;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Connect4 {
    board: [[i32; 6]; 7],
    total_moves: usize,
    turn: f32
}

impl Connect4 {
    pub fn new() -> Self {
        return Self {
            board: [[0; 6]; 7],
            total_moves: 0,
            turn: 1.0
        }
    }

    fn won(&self, col: usize, row: usize) -> bool {
        let player = self.board[col][row];
        if player == 0 {
            return false;
        }

        let directions = [(1, 0), (0, 1), (1, 1), (1, -1)];

        for (dc, dr) in directions.into_iter() {
            let count = 1 +
                self.count_in_direction(col, row, dc, dr) +
                self.count_in_direction(col, row, -dc, -dr);

            if count >= 4 {
                return true;
            }
        }

        false
    }

    fn count_in_direction(&self, start_col: usize, start_row: usize, dc: i32, dr: i32) -> usize {
        let player = self.board[start_col][start_row];
        let mut count = 0;
        let mut c = start_col as i32 + dc;
        let mut r = start_row as i32 + dr;

        while c >= 0 && c < 7 && r >= 0 && r < 6 {
            if self.board[c as usize][r as usize] == player {
                count += 1;
            } else {
                break;
            }
            c += dc;
            r += dr;
        }
        count
    }

    pub fn play(&mut self, column: usize) -> Result<&'static str, &'static str> {
        if column >= 7 {
            return Err("Invalid column index!");
        }

        for row in 0..6 {
            if self.board[column][row] as f32 == 0.0 {
                self.board[column][row] = self.turn as i32;
                self.total_moves += 1;

                if self.won(column, row) {
                    let winner_message = if self.turn == 1.0 {
                        "P1 Win"
                    } else {
                        "P2 Win"
                    };
                    return Ok(winner_message);
                }

                if self.total_moves == 7 * 6 {
                    return Ok("Draw");
                }

                self.turn = -self.turn;
                return Ok("Game in progress");
            }
        }

        Err("Illegal move: Column is full!")
    }

    pub fn get_legal_moves(&self) -> Vec<usize> {
        let mut moves = vec![];
        for column in 0..7 {
            if self.board[column][5] == 0 {
                moves.push(column);
            }
        }
        moves
    }

    pub fn get_total_moves(&self) -> usize {
        self.total_moves
    }

    pub fn get_turn(&self) -> f32 {
        self.turn
    }

    pub fn get_state(&self) -> [[i32; 6]; 7] {
        self.board
    }

    pub fn to_tensor<B: Backend>(&self) -> Tensor<B, 4> {
        let mut tensor_data = [[[[0.0f32; 6]; 7]; 2]; 1]; // Two channels

        let my_piece = self.turn as i32;
        let opponent_piece = -self.turn as i32;

        for c in 0..7 {
            for r in 0..6 {
                if self.board[c][r] == my_piece {
                    // Channel 0: My pieces
                    tensor_data[0][0][c][r] = 1.0;
                } else if self.board[c][r] == opponent_piece {
                    // Channel 1: Opponent's pieces
                    tensor_data[0][1][c][r] = 1.0;
                }
            }
        }

        Tensor::<B, 4>::from_floats(tensor_data, &B::Device::default())
    }

    pub fn display(&self) {
        println!("{}", self.to_string());
    }

    pub fn find_best_move(&self, depth: u32) -> Option<usize> {
        let legal_moves = self.get_legal_moves();
        if legal_moves.is_empty() {
            return None;
        }

        let current_player_turn = self.get_turn();
        for &mov in &legal_moves {
            let mut temp_game = self.clone();
            if let Ok(status) = temp_game.play(mov) {
                let is_win_for_p1 = current_player_turn == 1.0 && status == "P1 Win";
                let is_win_for_p2 = current_player_turn == -1.0 && status == "P2 Win";

                if is_win_for_p1 || is_win_for_p2 {
                    return Some(mov);
                }
            }
        }

        let mut rng = rand::rng();
        if depth == 0 {
            let choice = *legal_moves.choose(&mut rng).unwrap();
            return Some(choice)
        }
        
        let mut winning_moves = Vec::new();
        let mut safe_moves = Vec::new();
        let mut losing_moves = Vec::new();

        for &mov in &legal_moves {
            let mut next_game = self.clone();
            next_game.play(mov).ok()?;

            let score = -self.negamax(&next_game, depth - 1);

            match score {
                1 => winning_moves.push(mov),
                0 => safe_moves.push(mov),
                -1 => losing_moves.push(mov),
                _ => unreachable!(),
            }
        }

        if !winning_moves.is_empty() {
            let choice = *winning_moves.choose(&mut rng).unwrap();
            Some(choice)
        } else if !safe_moves.is_empty() {
            let choice = *safe_moves.choose(&mut rng).unwrap();
            Some(choice)
        } else if !losing_moves.is_empty() {
            let choice = *losing_moves.choose(&mut rng).unwrap();
            Some(choice)
        } else {
            None
        }
    }

    fn negamax(&self, game: &Connect4, depth: u32) -> i8 {
        if game.get_total_moves() == 42 {
            return 0;
        }

        let legal_moves = game.get_legal_moves();
        for &mov in &legal_moves {
            let mut temp_game = game.clone();
            if let Ok(status) = temp_game.play(mov) {
                if status.contains("Win") {
                    return 1;
                }
            }
        }
        
        if depth == 0 {
            return 0;
        }
        
        let mut max_score = -1; 
        for &mov in &legal_moves {
            let mut next_game = game.clone();
            next_game.play(mov).unwrap();

            let score = -self.negamax(&next_game, depth - 1);

            if score > max_score {
                max_score = score;
            }
            
            if max_score == 1 {
                return 1;
            }
        }

        max_score
    }

    pub fn symmetrize(&self) -> Self {
        let mut new_board = self.board.clone();
        new_board.reverse();
        
        Self {
            board: new_board,
            total_moves: self.total_moves,
            turn: self.turn
        }
    }
}

impl fmt::Display for Connect4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  ╭───────────────╮")?;

        for row in (0..6).rev() {
            write!(f, "  │ ")?;
            for column in 0..7 {
                let piece = self.board[column][row];
                let symbol = match piece {
                    1 => 'X',
                    -1 => 'O',
                    _ => '·',
                };
                write!(f, "{} ", symbol)?;
            }
            writeln!(f, "│")?;
        }

        writeln!(f, "  ├───────────────┤")?;
        writeln!(f, "  │ 0 1 2 3 4 5 6 │")?;
        writeln!(f, "  ╰───────────────╯")?;

        let current_player = if self.turn == 1.0 { "X (P1)" } else { "O (P2)" };
        write!(f, "Turn: {} ({})", self.total_moves + 1, current_player)?;

        Ok(())
    }
}