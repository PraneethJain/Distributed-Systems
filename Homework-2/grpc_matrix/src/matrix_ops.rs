/// Calculate the rank of a matrix using Gaussian elimination
pub fn calculate_rank(matrix: &[Vec<f64>]) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }

    let n = matrix.len();
    let m = matrix[0].len();

    let mut mat: Vec<Vec<f64>> = matrix.to_vec();

    let mut rank = 0;
    let epsilon = 1e-10;

    for col in 0..m.min(n) {
        let mut pivot_row = rank;
        for row in rank..n {
            if mat[row][col].abs() > mat[pivot_row][col].abs() {
                pivot_row = row;
            }
        }

        if mat[pivot_row][col].abs() < epsilon {
            continue;
        }

        if pivot_row != rank {
            mat.swap(rank, pivot_row);
        }

        let pivot = mat[rank][col];
        for j in col..m {
            mat[rank][j] /= pivot;
        }

        for row in 0..n {
            if row != rank && mat[row][col].abs() > epsilon {
                let factor = mat[row][col];
                for j in col..m {
                    mat[row][j] -= factor * mat[rank][j];
                }
            }
        }

        rank += 1;
    }

    rank
}

pub fn calculate_determinant(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();

    if n == 0 {
        return 1.0;
    }

    if n == 1 {
        return matrix[0][0];
    }

    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    let mut mat: Vec<Vec<f64>> = matrix.to_vec();
    let mut det = 1.0;
    let epsilon = 1e-10;

    for i in 0..n {
        let mut pivot_row = i;
        for k in i + 1..n {
            if mat[k][i].abs() > mat[pivot_row][i].abs() {
                pivot_row = k;
            }
        }

        if mat[pivot_row][i].abs() < epsilon {
            return 0.0;
        }

        if pivot_row != i {
            mat.swap(i, pivot_row);
            det = -det;
        }

        det *= mat[i][i];

        for k in i + 1..n {
            let factor = mat[k][i] / mat[i][i];
            for j in i + 1..n {
                mat[k][j] -= factor * mat[i][j];
            }
            mat[k][i] = 0.0;
        }
    }

    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_identity() {
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert_eq!(calculate_rank(&matrix), 3);
    }

    #[test]
    fn test_rank_singular() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        assert_eq!(calculate_rank(&matrix), 1);
    }

    #[test]
    fn test_determinant_identity() {
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert!((calculate_determinant(&matrix) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_2x2() {
        let matrix = vec![vec![3.0, 8.0], vec![4.0, 6.0]];
        let det = calculate_determinant(&matrix);
        assert!((det - (3.0 * 6.0 - 8.0 * 4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_3x3() {
        let matrix = vec![
            vec![6.0, 1.0, 1.0],
            vec![4.0, -2.0, 5.0],
            vec![2.0, 8.0, 7.0],
        ];
        let det = calculate_determinant(&matrix);
        // Expected determinant is -306
        assert!((det - (-306.0)).abs() < 1e-10);
    }
}
