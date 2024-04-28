#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RULE_30 0b00011110

int getNewState(int pattern, unsigned int rule) {
    return (rule >> pattern) & 1;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;        // Total number of cells
    int steps = 100;     // Number of time steps
    unsigned int rule = RULE_30; // Default to Rule 30
    int boundaryType = 0; // 0 for periodic, 1 for constant

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);
    if (argc > 3) rule = strtoul(argv[3], NULL, 2);
    if (argc > 4) boundaryType = atoi(argv[4]);

    int local_n = N / size;
    int *local_cur = malloc((local_n + 2) * sizeof(int));
    int *local_next = malloc((local_n + 2) * sizeof(int));

    // Random initialization based on rank to ensure different seeds
    srand(time(NULL) + rank);
    for (int i = 1; i <= local_n; i++) {
        local_cur[i] = rand() % 2;
    }

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    MPI_Status status;

    double startTime = MPI_Wtime();

    FILE *file;
    if (rank == 0) {
        file = fopen("output_states.txt", "w");
        if (file == NULL) {
            fprintf(stderr, "Failed to open file for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int *global_state = malloc(N * sizeof(int));

    for (int step = 0; step < steps; step++) {
        // Exchange ghost cells based on boundary type
        if (boundaryType == 0) {  // Periodic
            MPI_Sendrecv(&local_cur[1], 1, MPI_INT, left, 0,
                         &local_cur[local_n + 1], 1, MPI_INT, right, 0,
                         MPI_COMM_WORLD, &status);
            MPI_Sendrecv(&local_cur[local_n], 1, MPI_INT, right, 1,
                         &local_cur[0], 1, MPI_INT, left, 1,
                         MPI_COMM_WORLD, &status);
        } else {  // Constant boundaries
            if (rank == 0) local_cur[0] = 0;
            if (rank == size - 1) local_cur[local_n + 1] = 0;
        }

        // Update states
        for (int i = 1; i <= local_n; i++) {
            int pattern = (local_cur[i - 1] << 2) | (local_cur[i] << 1) | local_cur[i + 1];
            local_next[i] = getNewState(pattern, rule);
        }

        // Swap pointers for the next iteration
        int *temp = local_cur;
        local_cur = local_next;
        local_next = temp;

        // Gather the results at the root process
        MPI_Gather(local_cur + 1, local_n, MPI_INT, global_state, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        // Write the global state to file
        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                fprintf(file, "%d ", global_state[i]);
            }
            fprintf(file, "\n");
        }
    }

    double endTime = MPI_Wtime();

    if (rank == 0) {
        printf("Simulation took %f seconds\n", endTime - startTime);
        fclose(file);
    }

    free(local_cur);
    free(local_next);
    free(global_state);

    MPI_Finalize();
    return 0;
}
