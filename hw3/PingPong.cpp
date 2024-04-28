#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_passes>\n";
        return 1;
    }

    MPI_Init(&argc, &argv);

    int N = atoi(argv[1]);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);
    std::vector<int> data;

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This game requires at least 2 processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    int step = 0;
    MPI_Status status;
    const int tag_game_on = 0;
    const int tag_game_over = 1;

    if (rank == 0) {
        // Processor 0 starts the game.
        data.push_back(rank);
        int dest = rand() % (size - 1) + 1; // Pick a random target that isn't itself
        MPI_Ssend(data.data(), data.size(), MPI_INT, dest, tag_game_on, MPI_COMM_WORLD);
        std::cout << "Processor " << rank << " started the game, passing to Processor " << dest << std::endl;
    }

    bool game_over = false;
    while (!game_over) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        // Check if the game is over based on the tag
        if (status.MPI_TAG == tag_game_over) {
            game_over = true;
            MPI_Recv(nullptr, 0, MPI_INT, MPI_ANY_SOURCE, tag_game_over, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank != 0) {
                // Inform others the game is over
                for (int i = 1; i < size; ++i) {
                    if (i != rank) {
                        MPI_Send(nullptr, 0, MPI_INT, i, tag_game_over, MPI_COMM_WORLD);
                    }
                }
            }
            break;
        }

        int count;
        MPI_Get_count(&status, MPI_INT, &count);
        data.resize(count);
        MPI_Recv(data.data(), count, MPI_INT, MPI_ANY_SOURCE, tag_game_on, MPI_COMM_WORLD, &status);

        step = data.size();
        if (step >= N) {
            // Game over, notify others
            for (int i = 0; i < size; ++i) {
                if (i != rank) {
                    MPI_Send(nullptr, 0, MPI_INT, i, tag_game_over, MPI_COMM_WORLD);
                }
            }
            game_over = true;
            break;
        }

        // Add current rank to the path and choose next recipient
        data.push_back(rank);
        int dest = rank;
        while (dest == rank) {
            dest = rand() % size; // Ensure not sending to self.
        }

        MPI_Ssend(data.data(), data.size(), MPI_INT, dest, tag_game_on, MPI_COMM_WORLD);
        std::cout << "Processor " << rank << " passed the ball to Processor " << dest << "; Pass #" << step+1 << std::endl;
    }

    MPI_Finalize();
    if (rank == 0) {
        std::cout << "Game concluded after " << N << " passes." << std::endl;
    }

    return 0;
}