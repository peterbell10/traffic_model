
#include <mpi.h>
#include <cstdio>
#include <random>
#include <algorithm>
#include <cmath>

constexpr size_t N = 20;
constexpr size_t iters = 2000;
int rank = 0, size = 0;

bool road1[N], road2[N];

void print_road_segment(bool * road) {
    for (int i = 0; i < N; ++i) {
        printf("%c", road[i] ? 'X' : '-');
    }
    printf("|");
}

void print_road(bool * road_print, bool * road_scratch) {
    if (rank == 0) {
        print_road_segment(road_print);

        for (int i = 1; i < size; ++i) {
            MPI_Recv(road_scratch, N, MPI_C_BOOL, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print_road_segment(road_scratch);
        }
        printf("\n");
    } else {
        MPI_Send(road_print, N, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD);
    }
}

bool road_update(bool prev, bool cur, bool next) {
    return (cur & next) | (prev & !cur);
}

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int left_neighbour = (rank + size - 1) % size;
    const int right_neighbour = (rank + 1) % size;

    std::mt19937 rng{std::random_device{}()};
    std::bernoulli_distribution dist(.6);
    std::generate_n(road1, N, [&]{ return dist(rng); });

    bool * road_old = road1;
    bool * road_new = road2;

    for (int i = 0; i < iters; ++i) {
        // Initiate non-blocking communication
        MPI_Request send_request[2];
        MPI_Issend(&road_old[0],   1, MPI_C_BOOL, left_neighbour, 0, comm, &send_request[0]);
        MPI_Issend(&road_old[N-1], 1, MPI_C_BOOL, right_neighbour, 1, comm, &send_request[1]);

        MPI_Request recv_request[2];
        bool halo_left, halo_right;
        MPI_Irecv(&halo_left,  1, MPI_C_BOOL, left_neighbour, 1, comm, &recv_request[0]);
        MPI_Irecv(&halo_right, 1, MPI_C_BOOL, right_neighbour, 0, comm, &recv_request[1]);

        // Update centre cells
        for (size_t j = 1; j + 1 < N; ++j) {
            road_new[j] = road_update(road_old[j-1], road_old[j], road_old[j+1]);
        }

        MPI_Waitall(2, recv_request, MPI_STATUSES_IGNORE);

        // Update boundary cells
        road_new[0] = road_update(halo_left, road_old[0], road_old[1]);
        road_new[N-1] = road_update(road_old[N-2], road_old[N-1], halo_right);

        MPI_Waitall(2, send_request, MPI_STATUSES_IGNORE);

        if (i % (size * N) == 0) {
            print_road(road_new, road_old);
        }

        std::swap(road_old, road_new);
    }

    MPI_Finalize();
}
