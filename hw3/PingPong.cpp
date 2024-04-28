#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 10; // Number of passes, adjustable
    char ball[256] = ""; // String to accumulate the names (ranks)
    int count = 0; // Count how many passes have been made

    // Seed the random number generator to vary the selection of processors
    srand(time(NULL) + rank);

    if (rank == 0) {
        // Processor 0 starts the game
        sprintf(ball, "Proc %d ", rank); // Append processor 0's rank
        int next = rand() % size;
        while (next == rank) { // Ensure not passing to itself
            next = rand() % size;
        }
        MPI_Ssend(ball, strlen(ball)+1, MPI_CHAR, next, 0, MPI_COMM_WORLD);
        count++;
    }

    while (count < N) {
        MPI_Status status;
        MPI_Recv(ball, sizeof(ball), MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        
        // Append this processor's rank to the ball's history
        char procName[20];
        sprintf(procName, "Proc %d ", rank);
        strcat(ball, procName);
        count++;

        if (count >= N) {
            printf("Game Over. Final path: %sn", ball);
            break;
        }

        // Select the next processor to pass the ball to
        int next = rand() % size;
        while (next == rank) { // Ensure not passing to itself
            next = rand() % size;
        }
        MPI_Ssend(ball, strlen(ball)+1, MPI_CHAR, next, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
