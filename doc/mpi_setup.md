# MPI環境構築と並列計算

## 環境構築

### MPIのインストール

Ubuntu/Debian系でOpenMPIをインストール:

```bash
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

### インストール確認

```bash
which mpicc mpirun
```

## Hello World

### ソースコード (hello_mpi.c)

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello World from rank %d of %d processes\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

### コンパイルと実行

```bash
mpicc -o hello_mpi hello_mpi.c
mpirun -np 4 ./hello_mpi
```

### 出力例

```
Hello World from rank 0 of 4 processes
Hello World from rank 1 of 4 processes
Hello World from rank 2 of 4 processes
Hello World from rank 3 of 4 processes
```

## 円周率の並列計算

### アルゴリズム

数値積分法を使用:

```
∫[0,1] 4/(1+x²) dx = π
```

各プロセスが担当区間を計算し、`MPI_Reduce`で結果を集約する。

### ソースコード (pi_mpi.c)

```c
#include <stdio.h>
#include <mpi.h>

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
```

### コンパイルと実行

```bash
mpicc -O3 -o pi_mpi pi_mpi.c
mpirun -np 4 ./pi_mpi
```

### 実行結果

| プロセス数 | 計算時間 | 速度向上 | 誤差 |
|-----------|---------|---------|------|
| 1 | 1.114秒 | 1.0x | 1.78e-13 |
| 2 | 0.552秒 | 2.0x | 1.08e-13 |
| 4 | 0.276秒 | 4.0x | 2.75e-14 |

ほぼ理想的なスケーリング（線形高速化）を達成。

## 主要なMPI関数

| 関数 | 説明 |
|------|------|
| `MPI_Init` | MPI環境の初期化 |
| `MPI_Finalize` | MPI環境の終了 |
| `MPI_Comm_rank` | 自プロセスのランク取得 |
| `MPI_Comm_size` | 総プロセス数取得 |
| `MPI_Wtime` | 時間計測 |
| `MPI_Reduce` | 集約演算（SUM, MAX, MINなど） |
| `MPI_Send` | ポイントツーポイント送信 |
| `MPI_Recv` | ポイントツーポイント受信 |
| `MPI_Bcast` | ブロードキャスト |
| `MPI_Scatter` | データ分散 |
| `MPI_Gather` | データ収集 |
