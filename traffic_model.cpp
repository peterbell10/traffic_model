
#include <mpi.h>
#include <cstdio>
#include <random>
#include <algorithm>
#include <cmath>

constexpr size_t N = 10000;
int rank = 0, size = 0;

bool road1[N], road2[N];

int main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);

	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	const int left_neighbour = (rank + size - 1) % size;
	const int right_neighbour = (rank + 1) % size;

	std::mt19937 rng{std::random_device{}()};
	std::bernoulli_distribution dist(.2);
	std::generate_n(road1, N, [&]{ return dist(rng); });

	bool * road_old = road1;
	bool * road_new = road2;

	bool halo_left, halo_right;

	for (int i = 0; i < size-1; ++i) {
		MPI_Request send_request[2];
		MPI_Request recv_request[2];
		MPI_Issend(&road_old[0], 1, MPI_C_BOOL, left_neighbour, 0, comm, &send_request[0]);
		MPI_Issend(&road_old[N-1], 1, MPI_C_BOOL, right_neighbour, 0, comm, &send_request[1]);
		MPI_Irecv(&halo_left, 1, MPI_C_BOOL, left_neighbour, 0, comm, &recv_request[0]);
		MPI_Irecv(&halo_right, 1, MPI_C_BOOL, right_neighbour, 0, comm, &recv_request[1]);

		// Update centre cells
		for (size_t j = 1; j + 1 < N; ++j) {
			road_new[j] =
				(road_old[j] & road_old[j+1]) |
				(road_old[j-1] & !road_old[j]);
		}

		MPI_Wait(&recv_request[0], MPI_STATUS_IGNORE);
		MPI_Wait(&recv_request[1], MPI_STATUS_IGNORE);

		// Update boundary cells
		road_new[0] =
			(road_old[0] & road_old[1]) |
			(halo_left & !road_old[0]);

		road_new[N-1] =
			(road_old[N-1] & halo_right) |
			(road_old[N-2] & !road_old[N-1]);

		MPI_Wait(&send_request[0], MPI_STATUS_IGNORE);
		MPI_Wait(&send_request[1], MPI_STATUS_IGNORE);

		std::swap(road_old, road_new);
	}

    printf("Road on rank %d", rank);
    for (int i = 0; i < N; ++i) {
	    printf("%c", road_old[i] ? 'X' : '-');
    }
    puts("\n");

	MPI_Finalize();
}
