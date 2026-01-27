#include <stdio.h>
#include <mpi.h>

// 円周率の計算: ∫[0,1] 4/(1+x^2) dx = π
// 各プロセスが担当区間を計算し、MPI_Reduceで集約

int main(int argc, char *argv[]) {
    int rank, size;
    long long n = 1000000000;  // 分割数（10億）
    double h, sum, x, pi, local_sum;
    long long i, local_start, local_end;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime();

    h = 1.0 / (double)n;  // 区間幅

    // 各プロセスの担当範囲を計算
    local_start = (n / size) * rank;
    local_end = (n / size) * (rank + 1);
    if (rank == size - 1) {
        local_end = n;  // 最後のプロセスは余りも担当
    }

    // 各プロセスが担当区間の積分を計算
    local_sum = 0.0;
    for (i = local_start; i < local_end; i++) {
        x = (i + 0.5) * h;  // 中点
        local_sum += 4.0 / (1.0 + x * x);
    }
    local_sum *= h;

    // 全プロセスの結果を集約
    MPI_Reduce(&local_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("=== MPI 円周率計算 ===\n");
        printf("プロセス数: %d\n", size);
        printf("分割数: %lld\n", n);
        printf("計算結果: %.15f\n", pi);
        printf("真の値:   3.141592653589793\n");
        printf("誤差:     %.2e\n", pi - 3.14159265358979323846);
        printf("計算時間: %.3f 秒\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
