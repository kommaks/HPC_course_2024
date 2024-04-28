#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 10000
#define LEARNING_RATE 0.00001
#define MAX_ITER 1000
#define TOL 1e-6

void generate_data(double x[], double y[], double a, double b, int n) {
    double noise_amp = 1.0;
    srand(42);

    for (int i = 0; i < n; i++) {
        x[i] = (double)i / 100.0;
        double noise = noise_amp * ((double)rand() / RAND_MAX - 0.5);
        y[i] = a * x[i] + b + noise;
    }
}

void gradient_descent(double x[], double y[], int n, double *a, double *b) {
    double a_grad = 0, b_grad = 0;
    double a_current = 0, b_current = 0;
    int iter = 0;

    // Dynamically determine the best number of threads
    int num_procs = omp_get_num_procs();
    omp_set_num_threads(num_procs);

    do {
        a_grad = 0;
        b_grad = 0;

        #pragma omp parallel for reduction(+:a_grad, b_grad)
        for (int i = 0; i < n; i++) {
            double r = (a_current * x[i] + b_current) - y[i];
            a_grad += r * x[i];
            b_grad += r;
        }

        a_current -= LEARNING_RATE * a_grad / n;
        b_current -= LEARNING_RATE * b_grad / n;

        if (sqrt(a_grad * a_grad + b_grad * b_grad) / n < TOL)
            break;

        iter++;
    } while (iter < MAX_ITER);

    *a = a_current;
    *b = b_current;
}

int main() {
    double x[N], y[N];
    double a_est, b_est;
    double a_true = 2.0, b_true = 1.0;
    double start, end;

    generate_data(x, y, a_true, b_true, N);

    //start = omp_get_wtime();
    gradient_descent(x, y, N, &a_est, &b_est);
    //end = omp_get_wtime();
    //printf("Time = %f\n", end - start);

    printf("Estimated coefficients: a = %f, b = %f\n", a_est, b_est);
    return 0;
}
