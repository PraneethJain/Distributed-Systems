#[macro_use]
extern crate memoffset;

use mpi::{
    datatype::{Equivalence, UserDatatype},
    traits::*,
    Address,
};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[repr(C)]
struct NonZero {
    col: i32,
    val: f64,
}

unsafe impl Equivalence for NonZero {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(NonZero, col) as Address,
                offset_of!(NonZero, val) as Address,
            ],
            &[i32::equivalent_datatype(), f64::equivalent_datatype()],
        )
    }
}

type SparseRow = Vec<NonZero>;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let mut dims = [0, 0, 0]; // [n, m, p]
    let local_a_rows: Vec<SparseRow>;
    let b_matrix: Vec<SparseRow>;

    let root_process = world.process_at_rank(0);

    if rank == 0 {
        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!(
                "[Rank 0] ERROR: Please provide an input file name as a command-line argument."
            );
            world.abort(1);
        }
        let filename = &args[1];

        let (a_matrix, b_mat, n, m, p) = read_input_from_file(filename).unwrap();
        dims = [n, m, p];
        b_matrix = b_mat;

        root_process.broadcast_into(&mut dims);
        local_a_rows = scatter_a_matrix(&world, &a_matrix);
        broadcast_b_matrix(&world, &b_matrix);
    } else {
        root_process.broadcast_into(&mut dims);
        local_a_rows = receive_a_rows(&world);
        b_matrix = receive_b_matrix(&world, dims[1]);
    }

    let start = std::time::Instant::now();

    let local_c_rows = multiply_local_sparse(&local_a_rows, &b_matrix);

    let duration = start.elapsed();
    eprintln!("ExecutionTime: {:.6}", duration.as_secs_f64());


    if rank == 0 {
        gather_and_print_c_matrix(&world, local_c_rows, dims[0]);
    } else {
        send_c_rows_to_root(&world, &local_c_rows);
    }
}

fn read_input_from_file(
    filename: &str,
) -> io::Result<(Vec<SparseRow>, Vec<SparseRow>, i32, i32, i32)> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines.next().unwrap()?;
    let dims: Vec<i32> = first_line
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let (n, m, p) = (dims[0], dims[1], dims[2]);

    let mut a = Vec::with_capacity(n as usize);
    for _ in 0..n {
        a.push(parse_sparse_row(lines.next().unwrap()?));
    }
    let mut b = Vec::with_capacity(m as usize);
    for _ in 0..m {
        b.push(parse_sparse_row(lines.next().unwrap()?));
    }
    Ok((a, b, n, m, p))
}

fn parse_sparse_row(line: String) -> SparseRow {
    let mut parts = line.split_whitespace();
    let k: usize = parts.next().unwrap().parse().unwrap();
    let mut row = Vec::with_capacity(k);
    for _ in 0..k {
        let col: i32 = parts.next().unwrap().parse().unwrap();
        let val: f64 = parts.next().unwrap().parse().unwrap();
        row.push(NonZero { col, val });
    }
    row
}

fn get_row_distribution(n_total: i32, size: i32, rank: i32) -> (usize, usize) {
    let n = n_total as usize;
    let s = size as usize;
    let r = rank as usize;
    (r * n / s, (r + 1) * n / s)
}

fn scatter_a_matrix(world: &impl Communicator, a_matrix: &[SparseRow]) -> Vec<SparseRow> {
    let size = world.size();
    let n = a_matrix.len() as i32;

    for i in 1..size {
        let (start, end) = get_row_distribution(n, size, i);
        let rows_to_send = &a_matrix[start..end];

        let flat_data: Vec<NonZero> = rows_to_send.iter().flatten().cloned().collect();
        let row_lengths: Vec<i32> = rows_to_send.iter().map(|r| r.len() as i32).collect();

        world.process_at_rank(i).send(&(rows_to_send.len() as i32));
        world.process_at_rank(i).send(row_lengths.as_slice());
        world.process_at_rank(i).send(flat_data.as_slice());
    }

    let (start, end) = get_row_distribution(n, size, 0);
    a_matrix[start..end].to_vec()
}

fn receive_a_rows(world: &impl Communicator) -> Vec<SparseRow> {
    let (num_rows, _) = world.process_at_rank(0).receive::<i32>();
    let (row_lengths, _) = world.process_at_rank(0).receive_vec::<i32>();
    let (flat_data, _) = world.process_at_rank(0).receive_vec::<NonZero>();

    let mut local_a = Vec::with_capacity(num_rows as usize);
    let mut pos = 0;
    for &len in &row_lengths {
        let end_pos = pos + len as usize;
        local_a.push(flat_data[pos..end_pos].to_vec());
        pos = end_pos;
    }
    local_a
}

fn broadcast_b_matrix(world: &impl Communicator, b_matrix: &[SparseRow]) {
    let root_process = world.process_at_rank(0);
    let mut b_row_lengths: Vec<i32> = b_matrix.iter().map(|r| r.len() as i32).collect();
    root_process.broadcast_into(&mut b_row_lengths);

    let mut b_flat: Vec<NonZero> = b_matrix.iter().cloned().flatten().collect();
    root_process.broadcast_into(&mut b_flat);
}

fn receive_b_matrix(world: &impl Communicator, m: i32) -> Vec<SparseRow> {
    let root_process = world.process_at_rank(0);
    let mut b_row_lengths = vec![0; m as usize];
    root_process.broadcast_into(&mut b_row_lengths);

    let total_b_non_zeros = b_row_lengths.iter().sum::<i32>() as usize;
    let mut b_flat = vec![NonZero::default(); total_b_non_zeros];
    root_process.broadcast_into(&mut b_flat);

    let mut b_matrix = Vec::with_capacity(m as usize);
    let mut current_pos = 0;
    for &len in &b_row_lengths {
        let end_pos = current_pos + len as usize;
        b_matrix.push(b_flat[current_pos..end_pos].to_vec());
        current_pos = end_pos;
    }
    b_matrix
}

fn multiply_local_sparse(local_a: &[SparseRow], b_matrix: &[SparseRow]) -> Vec<SparseRow> {
    local_a
        .iter()
        .map(|a_row| {
            let mut c_row_map: HashMap<i32, f64> = HashMap::new();

            for a_elem in a_row {
                if let Some(b_row) = b_matrix.get(a_elem.col as usize) {
                    for b_elem in b_row {
                        *c_row_map.entry(b_elem.col).or_insert(0.0) += a_elem.val * b_elem.val;
                    }
                }
            }

            // Convert and sort for consistent output
            let mut result: Vec<NonZero> = c_row_map
                .into_iter()
                .filter(|&(_, val)| val.abs() > 1e-9)
                .map(|(col, val)| NonZero { col, val })
                .collect();
            result.sort_unstable_by_key(|elem| elem.col);
            result
        })
        .collect()
}

fn send_c_rows_to_root(world: &impl Communicator, local_c_rows: &[SparseRow]) {
    let local_c_row_lengths: Vec<i32> = local_c_rows.iter().map(|r| r.len() as i32).collect();
    let local_c_flat: Vec<NonZero> = local_c_rows.iter().cloned().flatten().collect();

    world.process_at_rank(0).send(&(local_c_rows.len() as i32));
    world
        .process_at_rank(0)
        .send(local_c_row_lengths.as_slice());
    world.process_at_rank(0).send(local_c_flat.as_slice());
}

fn gather_and_print_c_matrix(world: &impl Communicator, root_local_c: Vec<SparseRow>, n: i32) {
    let size = world.size();
    let mut c_matrix = vec![Vec::new(); n as usize];

    let (start_root, end_root) = get_row_distribution(n, size, 0);
    for (i, row) in (start_root..end_root).zip(root_local_c.into_iter()) {
        c_matrix[i] = row;
    }

    for i in 1..size {
        let (_num_rows, _) = world.process_at_rank(i).receive::<i32>();
        let (row_lengths, _) = world.process_at_rank(i).receive_vec::<i32>();
        let (data_flat, _) = world.process_at_rank(i).receive_vec::<NonZero>();

        let (start_worker, end_worker) = get_row_distribution(n, size, i);
        let mut current_pos = 0;
        for (j, &len) in (start_worker..end_worker).zip(row_lengths.iter()) {
            let end_pos = current_pos + len as usize;
            if j < c_matrix.len() {
                c_matrix[j] = data_flat[current_pos..end_pos].to_vec();
            }
            current_pos = end_pos;
        }
    }
    print_sparse_matrix(&c_matrix);
}

fn print_sparse_matrix(matrix: &[SparseRow]) {
    for row in matrix {
        print!("{}", row.len());
        for elem in row {
            print!(" {} {}", elem.col, elem.val);
        }
        println!();
    }
}
