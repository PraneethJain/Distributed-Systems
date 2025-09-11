use std::env;
use std::fs;
use std::io::{self, Write};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

pub mod matrix {
    tonic::include_proto!("matrix");
}
use matrix::{
    client_request::Command, matrix_service_client::MatrixServiceClient, ClientRequest,
    DeterminantQuery, RankQuery, ResetRequest, Row,
};

#[derive(Debug, Clone)]
struct ClientMatrixPart {
    rows: Vec<Vec<f64>>,
    client_id: String,
}

fn print_menu(client_id: &str, rows_count: usize) {
    println!(
        "\n=== Matrix Client {} Menu ({} rows) ===",
        client_id, rows_count
    );
    println!("1. Send next row");
    println!("2. Send all my rows");
    println!("3. Query rank");
    println!("4. Query determinant");
    println!("5. Reset server");
    println!("6. Show my rows");
    println!("7. Exit");
    print!("Choose an option (1-7): ");
    io::stdout().flush().unwrap();
}

fn read_input() -> String {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn read_matrix_part_from_file(
    filepath: &str,
    client_id: &str,
) -> Result<ClientMatrixPart, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(filepath)?;
    let lines: Vec<&str> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();

    if lines.is_empty() {
        return Err("File is empty or contains no valid rows".into());
    }

    let mut rows = Vec::new();
    let mut expected_dimension = None;

    for (line_num, line) in lines.iter().enumerate() {
        let values: Result<Vec<f64>, _> =
            line.split_whitespace().map(|s| s.parse::<f64>()).collect();

        let row = values.map_err(|e| format!("Invalid number on line {}: {}", line_num + 1, e))?;

        if row.is_empty() {
            return Err(format!("Line {} is empty", line_num + 1).into());
        }

        // All rows should have the same dimension
        if let Some(dim) = expected_dimension {
            if row.len() != dim {
                return Err(format!(
                    "Line {} has {} elements, expected {} (dimension set by first row)",
                    line_num + 1,
                    row.len(),
                    dim
                )
                .into());
            }
        } else {
            expected_dimension = Some(row.len());
        }

        rows.push(row);
    }

    Ok(ClientMatrixPart {
        rows,
        client_id: client_id.to_string(),
    })
}

fn display_client_rows(matrix_part: &ClientMatrixPart, sent_rows: usize) {
    println!("\n=== Client {} Matrix Part ===", matrix_part.client_id);
    println!(
        "My rows: {} (dimension: {})",
        matrix_part.rows.len(),
        matrix_part.rows.get(0).map(|r| r.len()).unwrap_or(0)
    );
    println!("Rows sent: {}/{}", sent_rows, matrix_part.rows.len());
    println!("My matrix rows:");

    for (i, row) in matrix_part.rows.iter().enumerate() {
        let status = if i < sent_rows {
            "✓ SENT"
        } else {
            "  pending"
        };
        println!("  Row {}: {:?} {}", i + 1, row, status);
    }
    println!("================================\n");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("Usage: {} <matrix_part_file> [client_id]", args[0]);
        eprintln!("Example: {} client1_rows.txt client1", args[0]);
        eprintln!("         {} client2_rows.txt client2", args[0]);
        eprintln!("If client_id is not provided, it will be derived from the filename");
        std::process::exit(1);
    }

    let filepath = &args[1];

    let client_id = if args.len() >= 3 {
        args[2].clone()
    } else {
        std::path::Path::new(filepath)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("client")
            .to_string()
    };

    println!(
        "Client {} reading matrix part from file: {}",
        client_id, filepath
    );

    let matrix_part = match read_matrix_part_from_file(filepath, &client_id) {
        Ok(part) => {
            println!(
                "✓ Successfully loaded {} rows (dimension: {}) for client {}",
                part.rows.len(),
                part.rows.get(0).map(|r| r.len()).unwrap_or(0),
                client_id
            );
            part
        }
        Err(e) => {
            eprintln!("Error reading matrix part file: {}", e);
            eprintln!("\nFile format should be:");
            eprintln!("- Each line contains one row of the matrix");
            eprintln!("- Values separated by spaces");
            eprintln!("- All rows must have the same dimension");
            eprintln!("- This client contributes only some rows to the full matrix");
            eprintln!("\nExample file for client contributing 2 rows to a 4x4 matrix:");
            eprintln!("1.0 2.0 3.0 4.0");
            eprintln!("5.0 6.0 7.0 8.0");
            std::process::exit(1);
        }
    };

    let mut client = MatrixServiceClient::connect("http://[::1]:50051").await?;
    println!("✓ Client {} connected to server.", client_id);

    let (tx, rx) = mpsc::channel(32);
    let request_stream = ReceiverStream::new(rx);
    let mut response_stream = client.interact(request_stream).await?.into_inner();

    let response_client_id = client_id.clone();

    let response_handle = tokio::spawn(async move {
        while let Some(response) = response_stream.next().await {
            match response {
                Ok(res) => match res.response {
                    Some(matrix::server_response::Response::Result(result)) => {
                        println!(
                            "[CLIENT {}] ✓ Query Result: {} - {}",
                            response_client_id,
                            if result.success { "SUCCESS" } else { "FAILED" },
                            result.description
                        );
                    }
                    Some(matrix::server_response::Response::Error(error)) => {
                        println!(
                            "[CLIENT {}] ✗ Server Error: {}",
                            response_client_id, error.message
                        );
                    }
                    None => {
                        println!(
                            "[CLIENT {}] Received empty response from server",
                            response_client_id
                        );
                    }
                },
                Err(e) => {
                    eprintln!("[CLIENT {}] Error from server: {}", response_client_id, e);
                    break;
                }
            }
        }
    });

    let mut sent_rows = 0;

    display_client_rows(&matrix_part, sent_rows);

    // Main interactive loop
    loop {
        print_menu(&client_id, matrix_part.rows.len());
        let choice = read_input();

        match choice.as_str() {
            "1" => {
                // Send next row
                if sent_rows >= matrix_part.rows.len() {
                    println!("All my rows have already been sent!");
                    continue;
                }

                let row = &matrix_part.rows[sent_rows];
                println!(
                    "[CLIENT {}] Sending row {}: {:?}",
                    client_id,
                    sent_rows + 1,
                    row
                );

                if let Err(e) = tx
                    .send(ClientRequest {
                        command: Some(Command::Row(Row {
                            values: row.clone(),
                        })),
                    })
                    .await
                {
                    eprintln!("Failed to send row: {}", e);
                    break;
                }

                sent_rows += 1;
                println!(
                    "✓ Row sent. My progress: {}/{}",
                    sent_rows,
                    matrix_part.rows.len()
                );
            }

            "2" => {
                // Send all my rows
                if sent_rows >= matrix_part.rows.len() {
                    println!("All my rows have already been sent!");
                    continue;
                }

                let remaining = matrix_part.rows.len() - sent_rows;
                println!(
                    "[CLIENT {}] Sending all my remaining {} rows...",
                    client_id, remaining
                );

                for i in sent_rows..matrix_part.rows.len() {
                    let row = &matrix_part.rows[i];
                    println!("  [CLIENT {}] Sending row {}: {:?}", client_id, i + 1, row);

                    if let Err(e) = tx
                        .send(ClientRequest {
                            command: Some(Command::Row(Row {
                                values: row.clone(),
                            })),
                        })
                        .await
                    {
                        eprintln!("Failed to send row {}: {}", i + 1, e);
                        break;
                    }

                    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
                }

                sent_rows = matrix_part.rows.len();
                println!("✓ [CLIENT {}] All my rows sent!", client_id);
            }

            "3" => {
                // Query rank
                print!("Enter minimum rank to check: ");
                io::stdout().flush().unwrap();
                let input = read_input();

                match input.parse::<i32>() {
                    Ok(rank) => {
                        println!("[CLIENT {}] Querying: Is rank >= {}?", client_id, rank);
                        if let Err(e) = tx
                            .send(ClientRequest {
                                command: Some(Command::RankQuery(RankQuery { r: rank })),
                            })
                            .await
                        {
                            eprintln!("Failed to send rank query: {}", e);
                            break;
                        }
                        // Wait for user to acknowledge the response
                        let _ = read_input();
                    }
                    Err(_) => println!("Error: Invalid rank value"),
                }
            }

            "4" => {
                // Query determinant
                print!("Enter minimum determinant to check: ");
                io::stdout().flush().unwrap();
                let input = read_input();

                match input.parse::<f64>() {
                    Ok(det) => {
                        println!(
                            "[CLIENT {}] Querying: Is determinant >= {}?",
                            client_id, det
                        );
                        if let Err(e) = tx
                            .send(ClientRequest {
                                command: Some(Command::DetQuery(DeterminantQuery { d: det })),
                            })
                            .await
                        {
                            eprintln!("Failed to send determinant query: {}", e);
                            break;
                        }
                        // Wait for user to acknowledge the response
                        let _ = read_input();
                    }
                    Err(_) => println!("Error: Invalid determinant value"),
                }
            }

            "5" => {
                // Reset server
                println!("[CLIENT {}] Resetting server state...", client_id);
                match client.reset(ResetRequest {}).await {
                    Ok(_) => {
                        println!("✓ [CLIENT {}] Server reset successfully", client_id);
                        sent_rows = 0; // Reset local tracking too
                    }
                    Err(e) => println!("✗ [CLIENT {}] Failed to reset server: {}", client_id, e),
                }
            }

            "6" => {
                // Show my matrix part info
                display_client_rows(&matrix_part, sent_rows);
            }

            "7" => {
                // Exit
                println!("[CLIENT {}] Goodbye!", client_id);
                break;
            }

            _ => {
                println!("Invalid option. Please choose 1-7.");
            }
        }
    }

    drop(tx);
    response_handle.abort();

    Ok(())
}
