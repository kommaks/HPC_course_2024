#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100 // Number of variables
#define MAX_ITER 10000
#define TOL 1e-10

void generate_diagonally_dominant_matrix(double A[N][N]) {

    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                A[i][j] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
                sum += fabs(A[i][j]);
            }
        }
        // Set the diagonal element to ensure the row is diagonally dominant
        A[i][i] = sum + ((double)rand() / RAND_MAX) * 10.0 + 1.0;
    }
}

void initialize_vector_b(double b[N]) {
    for (int i = 0; i < N; i++) {
        b[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
    }
}

void jacobi(double A[N][N], double b[N], double x[N]) {
    double x_new[N], diff, sum;
    int iter = 0;
    
    // Initial guess
    for (int i = 0; i < N; i++) {
        x[i] = b[i];
    }

    do {
        #pragma omp parallel for private(sum) shared(x, x_new) num_threads(8)
        for (int i = 0; i < N; i++) {
            sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < N; i++) {
            diff += fabs(x_new[i] - x[i]);
            x[i] = x_new[i];
        }

        iter++;
    } while (diff > TOL && iter < MAX_ITER);

    printf("Converged after %d iterations.\n", iter);
}

int main() {
    double A[N][N], b[N], x[N];
    generate_diagonally_dominant_matrix(A);
    initialize_vector_b(b);

    double start, end;
    start = omp_get_wtime();
    jacobi(A, b, x);
    end = omp_get_wtime();
    printf("Time = %f seconds\n", end - start);

    printf("Solution vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    return 0;
}
