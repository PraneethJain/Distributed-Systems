use kmeans_mapreduce::{
    combine_assignment_files, convert_flat_to_points, find_nearest_center, read_points_from_txt,
    squared_euclidean_distance, write_assignments, write_centers, Point,
};
use mpi::traits::*;
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    if args.len() != 6 {
        if rank == 0 {
            eprintln!(
                "Usage: {} <num_centers> <points_file> <centers_file> <max_iterations> <output_dir>",
                args.get(0).map_or("kmeans_mapreduce", |s| s.as_str())
            );
        }
        return Ok(());
    }

    let num_centers_arg: usize = args[1].parse()?;
    let points_path = &args[2];
    let centers_path = &args[3];
    let max_iterations: usize = args[4].parse()?;
    let output_dir = &args[5];

    let (num_points, num_centers, dimensions) =
        broadcast_parameters(&world, rank, points_path, centers_path, num_centers_arg)?;
    let tolerance = 1e-4;

    let local_points =
        scatter_points_to_workers(&world, rank, size, num_points, dimensions, points_path)?;
    let mut centers = broadcast_centers(&world, rank, num_centers, dimensions, centers_path)?;

    let mut iteration = 0;
    let final_assignments = loop {
        if rank == 0 {
            // println!("Process {}: Starting iteration {}", rank, iteration);
        }

        world.barrier(); // Synchronize before starting map phase timing
        let map_start = std::time::Instant::now();

        // MAP PHASE
        let (local_assignments, local_sums, local_counts) =
            map_and_combine(&local_points, &centers, num_centers, dimensions);

        world.barrier(); // Synchronize before stopping map phase timing
        let map_duration = map_start.elapsed();
        if rank == 0 {
            println!("[METRICS] map_time_ms={}", map_duration.as_millis());
        }

        world.barrier(); // Synchronize before starting reduce phase timing
        let reduce_start = std::time::Instant::now();

        // REDUCE PHASE (only master gets global results)
        let global_results = reduce_phase(
            &world,
            &local_sums,
            &local_counts,
            rank,
            num_centers,
            dimensions,
        )?;

        world.barrier(); // Synchronize after reduce phase
        let reduce_duration = reduce_start.elapsed();
        if rank == 0 {
            println!("[METRICS] reduce_time_ms={}", reduce_duration.as_millis());
        }

        // MASTER: Compute new centers and check convergence
        let (new_centers, converged) = if rank == 0 {
            let (global_sums, global_counts) = global_results.unwrap();
            let (new_centers, converged) = compute_new_centers_and_check_convergence(
                &global_sums,
                &global_counts,
                &centers,
                num_centers,
                dimensions,
                tolerance,
            );
            (Some(new_centers), Some(converged))
        } else {
            (None, None)
        };

        // Broadcast new centers and convergence status
        let (updated_centers, is_converged) = broadcast_centers_and_convergence(
            &world,
            rank,
            new_centers,
            converged,
            num_centers,
            dimensions,
        )?;

        centers = updated_centers;

        if is_converged {
            break Some(local_assignments);
        }

        iteration += 1;
        if iteration >= max_iterations {
            break None;
        }
    };

    println!("Process {}: K-means complete", rank);

    if let Some(final_assignments) = final_assignments {
        println!(
            "Process {}: Converged after {} iterations",
            rank,
            iteration + 1
        );
        write_assignments(&local_points, &final_assignments, rank, output_dir)?;
    } else {
        println!(
            "Process {}: Reached max iterations ({})",
            rank, max_iterations
        );
    }

    if rank == 0 {
        write_centers(&centers, output_dir)?;
        combine_assignment_files(size, output_dir)?;
    }

    Ok(())
}

fn scatter_points_to_workers(
    world: &mpi::topology::SimpleCommunicator,
    rank: usize,
    size: usize,
    total_points: usize,
    dimensions: usize,
    points_path: &str,
) -> Result<Vec<Point>, Box<dyn Error>> {
    let points_per_worker = total_points.div_ceil(size);
    let total_elements_per_worker = points_per_worker * dimensions;
    let total_elements = size * total_elements_per_worker;

    let mut local_buffer = vec![0.0; total_elements_per_worker];

    if rank == 0 {
        let all_points = read_points_from_txt(points_path);

        let mut flat_points: Vec<f64> = all_points.into_iter().flat_map(|point| point).collect();
        flat_points.resize(total_elements, 0.0);
        world
            .process_at_rank(0)
            .scatter_into_root(&flat_points, &mut local_buffer);
    } else {
        world.process_at_rank(0).scatter_into(&mut local_buffer);
    }

    let start_idx = rank * points_per_worker;
    let actual_count = std::cmp::min(points_per_worker, total_points.saturating_sub(start_idx));

    let local_points = convert_flat_to_points(&local_buffer, dimensions, actual_count);
    Ok(local_points)
}

fn broadcast_centers(
    world: &mpi::topology::SimpleCommunicator,
    rank: usize,
    num_centers: usize,
    dimensions: usize,
    centers_path: &str,
) -> Result<Vec<Point>, Box<dyn Error>> {
    let total_elements = num_centers * dimensions;
    let mut buffer = vec![0.0; total_elements];

    if rank == 0 {
        let centers = read_points_from_txt(centers_path);

        let flat_centers: Vec<f64> = centers.into_iter().flat_map(|point| point).collect();

        buffer.copy_from_slice(&flat_centers);
    }

    world.process_at_rank(0).broadcast_into(&mut buffer);

    let centers = convert_flat_to_points(&buffer, dimensions, num_centers);
    Ok(centers)
}

fn map_and_combine(
    local_points: &[Point],
    centers: &[Point],
    num_centers: usize,
    dimensions: usize,
) -> (Vec<usize>, Vec<Vec<f64>>, Vec<i32>) {
    let mut local_assignments = Vec::with_capacity(local_points.len());

    let mut local_sums = vec![vec![0.0; dimensions]; num_centers];
    let mut local_counts = vec![0; num_centers];

    // Assign each local point to nearest center
    for point in local_points {
        let nearest_cluster = find_nearest_center(point, centers);
        local_assignments.push(nearest_cluster);

        for d in 0..dimensions {
            local_sums[nearest_cluster][d] += point[d];
        }
        local_counts[nearest_cluster] += 1;
    }

    (local_assignments, local_sums, local_counts)
}

fn reduce_phase(
    world: &mpi::topology::SimpleCommunicator,
    local_sums: &[Vec<f64>],
    local_counts: &[i32],
    rank: usize,
    num_centers: usize,
    dimensions: usize,
) -> Result<Option<(Vec<Vec<f64>>, Vec<i32>)>, Box<dyn Error>> {
    if rank == 0 {
        // Master: receive global results
        let mut global_counts = vec![0i32; num_centers];
        let mut global_sums = vec![vec![0.0f64; dimensions]; num_centers];

        // Reduce cluster counts to master
        world.process_at_rank(0).reduce_into_root(
            local_counts,
            &mut global_counts,
            mpi::collective::SystemOperation::sum(),
        );

        // Reduce cluster sums to master (cluster by cluster)
        for cluster in 0..num_centers {
            world.process_at_rank(0).reduce_into_root(
                &local_sums[cluster],
                &mut global_sums[cluster],
                mpi::collective::SystemOperation::sum(),
            );
        }

        Ok(Some((global_sums, global_counts)))
    } else {
        // Workers: send their local data to master
        world
            .process_at_rank(0)
            .reduce_into(local_counts, mpi::collective::SystemOperation::sum());

        // Send local sums to master (cluster by cluster)
        for cluster in 0..num_centers {
            world.process_at_rank(0).reduce_into(
                &local_sums[cluster],
                mpi::collective::SystemOperation::sum(),
            );
        }

        // println!("Process {}: Sent local data to master for reduction", rank);
        Ok(None)
    }
}

fn compute_new_centers_and_check_convergence(
    global_sums: &[Vec<f64>],
    global_counts: &[i32],
    old_centers: &[Point],
    num_centers: usize,
    dimensions: usize,
    tolerance: f64,
) -> (Vec<Point>, bool) {
    let mut new_centers = vec![vec![0.0; dimensions]; num_centers];

    // Compute new centers
    for cluster in 0..num_centers {
        if global_counts[cluster] > 0 {
            for d in 0..dimensions {
                new_centers[cluster][d] = global_sums[cluster][d] / global_counts[cluster] as f64;
            }
        } else {
            // Keep old center if no points assigned
            new_centers[cluster] = old_centers[cluster].clone();
            println!(
                "Warning: Cluster {} has no assigned points, keeping old center",
                cluster
            );
        }
    }

    // Check convergence
    let mut converged = true;
    for (_cluster, (old, new)) in old_centers.iter().zip(new_centers.iter()).enumerate() {
        let distance = squared_euclidean_distance(old, new);
        // println!(
        //     "Master: Cluster {} center moved by distance: {:.6}",
        //     _cluster, distance
        // );
        if distance > tolerance {
            converged = false;
        }
    }

    // println!("Master: Convergence check - converged: {}", converged);
    (new_centers, converged)
}

fn broadcast_centers_and_convergence(
    world: &mpi::topology::SimpleCommunicator,
    rank: usize,
    new_centers: Option<Vec<Point>>,
    converged: Option<bool>,
    num_centers: usize,
    dimensions: usize,
) -> Result<(Vec<Point>, bool), Box<dyn Error>> {
    // Broadcast new centers
    let total_elements = num_centers * dimensions;
    let mut buffer = vec![0.0; total_elements];

    if rank == 0 {
        let centers = new_centers.unwrap();
        let flat_centers: Vec<f64> = centers
            .iter()
            .flat_map(|point| point.iter().cloned())
            .collect();
        buffer.copy_from_slice(&flat_centers);
        // println!("Master: Broadcasting new centers");
    }

    world.process_at_rank(0).broadcast_into(&mut buffer);
    let centers = convert_flat_to_points(&buffer, dimensions, num_centers);

    // Broadcast convergence status
    let mut converged_buf = if rank == 0 { converged.unwrap() } else { false };
    world.process_at_rank(0).broadcast_into(&mut converged_buf);

    // if rank != 0 {
    //     println!(
    //         "Process {}: Received new centers and convergence status: {}",
    //         rank, converged_buf
    //     );
    // }

    Ok((centers, converged_buf))
}

fn broadcast_parameters(
    world: &mpi::topology::SimpleCommunicator,
    rank: usize,
    points_path: &str,
    centers_path: &str,
    num_centers_arg: usize,
) -> Result<(usize, usize, usize), Box<dyn Error>> {
    let mut params = [0, 0, 0];

    if rank == 0 {
        let points = read_points_from_txt(points_path);
        let centers = read_points_from_txt(centers_path);

        let num_points = points.len();
        let num_centers = if !centers.is_empty() {
            centers.len()
        } else {
            num_centers_arg
        };
        let dimensions = if !points.is_empty() {
            points[0].len()
        } else if !centers.is_empty() {
            centers[0].len()
        } else {
            0
        };

        params = [num_points, num_centers, dimensions];
    }

    world.process_at_rank(0).broadcast_into(&mut params);

    Ok((params[0], params[1], params[2]))
}
