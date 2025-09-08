pub type Point = Vec<f64>;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use csv;

mod output;

pub fn read_points_from_csv(path: &str) -> Vec<Point> {
    csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .expect("Failed to open CSV file")
        .records()
        .filter_map(|result| {
            result.ok().map(|record| {
                record
                    .iter()
                    .map(|s| s.parse::<f64>().expect("Failed to parse float"))
                    .collect::<Point>()
            })
        })
        .collect()
}

pub fn read_points_from_txt(path: &str) -> Vec<Point> {
    println!("Reading points from TXT file: {}", path);
    let file = File::open(path).expect("Failed to open TXT file");
    let reader = BufReader::new(file);

    reader
        .lines()
        .filter_map(|line| line.ok())
        .map(|line| {
            line.split_whitespace()
                .map(|s| s.parse::<f64>().expect("Failed to parse float"))
                .collect::<Point>()
        })
        .collect()
}

pub fn convert_flat_to_points(
    flat_data: &[f64],
    dimensions: usize,
    actual_count: usize,
) -> Vec<Point> {
    (0..actual_count)
        .map(|i| {
            (0..dimensions)
                .map(|d| flat_data[i * dimensions + d])
                .collect()
        })
        .collect()
}

pub fn squared_euclidean_distance(a: &Point, b: &Point) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

pub fn find_nearest_center(point: &Point, centers: &[Point]) -> usize {
    let mut min_distance = f64::INFINITY;
    let mut nearest = 0;

    for (i, center) in centers.iter().enumerate() {
        let distance = squared_euclidean_distance(point, center);
        if distance < min_distance {
            min_distance = distance;
            nearest = i;
        }
    }

    nearest
}

pub fn write_assignments(
    local_points: &[Point],
    assignments: &[usize],
    rank: usize,
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    use std::fs::create_dir_all;
    use std::io::Write;

    create_dir_all(output_dir)?;
    let filename = format!("{}/assignments_rank_{}.txt", output_dir, rank);
    let mut file = std::fs::File::create(&filename)?;

    for (i, &cluster_id) in assignments.iter().enumerate() {
        writeln!(file, "{},{}", i + rank * local_points.len(), cluster_id)?;
    }

    println!(
        "Process {}: Wrote {} assignments to {}",
        rank,
        assignments.len(),
        filename
    );
    Ok(())
}

pub fn write_centers(centers: &[Point], output_dir: &str) -> Result<(), Box<dyn Error>> {
    use std::fs::create_dir_all;
    use std::io::Write;

    create_dir_all(output_dir)?;
    let filename = format!("{}/final_centers.txt", output_dir);
    let mut file = std::fs::File::create(&filename)?;

    for center in centers {
        let line = center
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{}", line)?;
    }

    println!("Master: Wrote {} centers to {}", centers.len(), filename);
    Ok(())
}

pub fn combine_assignment_files(
    num_processes: usize,
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    use std::fs::{remove_file, File};
    use std::io::{BufRead, BufReader, Write};

    let combined_filename = format!("{}/final_assignments.txt", output_dir);
    let mut combined_file = File::create(&combined_filename)?;

    let mut total_assignments = 0;

    // Read and combine all individual assignment files
    for rank in 0..num_processes {
        let individual_filename = format!("{}/assignments_rank_{}.txt", output_dir, rank);

        match File::open(&individual_filename) {
            Ok(file) => {
                let reader = BufReader::new(file);
                let mut rank_assignments = 0;

                for line in reader.lines() {
                    let line = line?;
                    if !line.trim().is_empty() {
                        writeln!(combined_file, "{}", line)?;
                        rank_assignments += 1;
                        total_assignments += 1;
                    }
                }

                println!(
                    "Master: Combined {} assignments from rank {}",
                    rank_assignments, rank
                );

                // Remove individual file after combining
                if let Err(e) = remove_file(&individual_filename) {
                    println!(
                        "Warning: Could not remove file {}: {}",
                        individual_filename, e
                    );
                }
            }
            Err(e) => {
                println!(
                    "Warning: Could not read assignments from rank {}: {}",
                    rank, e
                );
            }
        }
    }

    println!(
        "Master: Combined {} total assignments into {}",
        total_assignments, combined_filename
    );
    Ok(())
}
