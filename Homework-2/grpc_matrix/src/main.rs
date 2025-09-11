use futures::stream::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status, Streaming};

pub mod matrix {
    tonic::include_proto!("matrix");
}
mod matrix_ops;
use matrix::{
    client_request::Command,
    matrix_service_server::{MatrixService, MatrixServiceServer},
    server_response::Response as ServerResponseType,
    ClientRequest, QueryResult, ResetRequest, ResetResponse, ServerError, ServerResponse,
};

#[derive(Default, Debug)]
struct AppState {
    dimension: Option<usize>,
    matrix_rows: Vec<Vec<f64>>,
    cached_rank: Option<i32>,
    cached_determinant: Option<f64>,
}

#[derive(Debug)]
pub struct MyMatrixService {
    state: Arc<Mutex<AppState>>,
}

type ServerStream = Pin<Box<dyn Stream<Item = Result<ServerResponse, Status>> + Send>>;

#[tonic::async_trait]
impl MatrixService for MyMatrixService {
    type InteractStream = ServerStream;

    async fn interact(
        &self,
        request: Request<Streaming<ClientRequest>>,
    ) -> Result<Response<Self::InteractStream>, Status> {
        let mut in_stream = request.into_inner();
        let state = self.state.clone();

        let (tx, rx) = mpsc::channel(4);

        tokio::spawn(async move {
            while let Some(result) = in_stream.message().await.transpose() {
                let req = match result {
                    Ok(req) => req,
                    Err(e) => {
                        eprintln!("Error from client stream: {}", e);
                        break;
                    }
                };

                let mut state = state.lock().await;
                let response = match req.command {
                    Some(Command::Row(row)) => {
                        if state.cached_rank.is_some() {
                            Some(ServerResponse {
                                response: Some(ServerResponseType::Error(ServerError {
                                    message: "Matrix is already complete. Please reset.".into(),
                                })),
                            })
                        } else {
                            handle_row(&mut state, row.values);
                            None
                        }
                    }

                    Some(Command::RankQuery(query)) => Some(handle_query(
                        &state,
                        state.cached_rank,
                        query.r as f64,
                        "rank",
                    )),

                    Some(Command::DetQuery(query)) => Some(handle_query(
                        &state,
                        state.cached_determinant,
                        query.d,
                        "determinant",
                    )),
                    _ => None,
                };

                if let Some(resp) = response {
                    if tx.send(Ok(resp)).await.is_err() {
                        eprintln!("Client disconnected, cannot send response.");
                        break;
                    }
                }
            }
            println!("Client stream closed.");
        });

        let out_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(out_stream) as Self::InteractStream))
    }

    async fn reset(
        &self,
        _request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        let mut state = self.state.lock().await;
        *state = AppState::default();
        println!("Server state has been reset.");
        Ok(Response::new(ResetResponse {}))
    }
}

fn handle_row(state: &mut AppState, values: Vec<f64>) {
    let dim = *state.dimension.get_or_insert(values.len());

    if values.len() != dim || dim == 0 {
        println!("Ignoring row with invalid dimension.");
        return;
    }

    state.matrix_rows.push(values);
    println!("Added row. Total rows: {}/{}", state.matrix_rows.len(), dim);

    if state.matrix_rows.len() == dim {
        println!("Matrix is complete! Calculating rank and determinant...");

        state.cached_rank = Some(matrix_ops::calculate_rank(&state.matrix_rows) as i32);
        state.cached_determinant = Some(matrix_ops::calculate_determinant(&state.matrix_rows));

        println!(
            "Calculation complete. Rank: {:?}, Determinant: {:?}",
            state.cached_rank, state.cached_determinant
        );
    }
}

fn handle_query(
    state: &AppState,
    cached_value: Option<impl Into<f64>>,
    query_value: f64,
    query_type: &str,
) -> ServerResponse {
    if let Some(val) = cached_value {
        let val: f64 = val.into();
        let success = val >= query_value;
        ServerResponse {
            response: Some(ServerResponseType::Result(QueryResult {
                success,
                description: format!(
                    "Query: Is {} >= {}? Actual: {}. Result: {}",
                    query_type, query_value, val, success
                ),
            })),
        }
    } else {
        ServerResponse {
            response: Some(ServerResponseType::Error(ServerError {
                message: format!(
                    "Matrix not complete yet. Have {}/{} rows.",
                    state.matrix_rows.len(),
                    state.dimension.unwrap_or(0)
                ),
            })),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let app_state = Arc::new(Mutex::new(AppState::default()));
    let matrix_service = MyMatrixService { state: app_state };

    println!("MatrixService server listening on {}", addr);

    Server::builder()
        .add_service(MatrixServiceServer::new(matrix_service))
        .serve(addr)
        .await?;

    Ok(())
}
