# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MPI (Message Passing Interface) training project demonstrating parallel computing concepts. Contains example programs for learning distributed computing with OpenMPI.

## Build and Run Commands

```bash
# Compile
mpicc -o hello_mpi hello_mpi.c
mpicc -O3 -o pi_mpi pi_mpi.c

# Run with N processes
mpirun -np 4 ./hello_mpi
mpirun -np 4 ./pi_mpi
```

## Prerequisites

```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev
which mpicc mpirun  # verify installation
```

## Code Structure

- `hello_mpi.c` - Basic MPI program showing process initialization and rank identification
- `pi_mpi.c` - Parallel pi calculation using numerical integration with MPI_Reduce for aggregation
- `doc/mpi_setup.md` - Setup guide and MPI function reference

## MPI Programming Patterns

- Initialize with `MPI_Init`, finalize with `MPI_Finalize`
- Get process identity: `MPI_Comm_rank` (rank) and `MPI_Comm_size` (total)
- Distribute work by dividing ranges based on rank; last process handles remainder
- Aggregate results with `MPI_Reduce`
- Use `MPI_Wtime` for timing instead of standard C time functions
- Only rank 0 should print final output to avoid duplication
