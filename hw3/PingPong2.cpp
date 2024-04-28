#include <iostream>
#include <fstream>
#include <mpi.h>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_passes>\n";
        return 1;
    }
    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        std::cerr << "This program requires at least two MPI processes." << std::endl;
        MPI_Finalize();
        return -1;
    }

    int max_iterations = atoi(argv[1]); // Number of ping-pong iterations for averaging

    std::vector<int> message_sizes;
    message_sizes.push_back(0); // Start with 0 as requested

    for (int value = 1; value < 100000.; value *= 2) {
        message_sizes.push_back(value);
    }
    //for (int value = 32; value <= 100000; value += 32) {
    //    message_sizes.push_back(value);
    //}



    int n_message_sizes = message_sizes.size();

    double start_time, end_time, elapsed_time, latency, bandwidth;

    std::ofstream outputFile;
    // Only process 0 writes to the file
    if (rank == 0) {
        outputFile.open("mpi_ping_pong_results.csv");
        outputFile << "MessageSize(Bytes),TimePerMessage(µs),Bandwidth(MB/s)\n";
    }

    for (int i = 0; i < n_message_sizes; ++i) {
        int msg_size = message_sizes[i];
        std::vector<char> buffer(msg_size, 0);

        MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing

        start_time = MPI_Wtime();

        for(int j = 0; j < max_iterations; ++j) {
            if (rank == 0) {
                MPI_Send(&buffer[0], msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&buffer[0], msg_size, MPI_CHAR, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(&buffer[0], msg_size, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&buffer[0], msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        end_time = MPI_Wtime();
        elapsed_time = (end_time - start_time) / (2 * max_iterations);

        if (rank == 0) {
            latency = (msg_size == 0) ? elapsed_time : latency;
            bandwidth = (msg_size > 0) ? (msg_size / elapsed_time) / 1e6 : 0; // Convert to MB/s
            std::cout << "Message size: " << msg_size << " Bytes, Time per message: " << elapsed_time * 1e6 << " µs, Bandwidth: " << bandwidth << " MB/s" << std::endl;
            // Write the results to file.
            outputFile << msg_size << "," << elapsed_time * 1e6 << "," << bandwidth << "\n";
        }
    }

    // Close the file after writing is complete
    if (rank == 0) {
        outputFile.close();
    }

    MPI_Finalize();
    return 0;
}