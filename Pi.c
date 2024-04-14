#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
    omp_set_num_threads(10);
    
    const size_t N = 100000;
    double step;
    int tid;

    double x, pi, sum = 0.;

    step = 1. / (double)N;
    
    #pragma omp parallel for private(x) reduction(+:sum)
    for (int i = 0; i < N; ++i)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1. + x * x);
        //tid = omp_get_thread_num();
        //printf("tid = %d\n", tid);
    }

    pi = step * sum;

    printf("pi = %.16f\n", pi);

    return 0;
}
