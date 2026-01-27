#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <gmp.h>

// BBP公式による円周率計算:
// π = Σ(k=0 to ∞) [1/16^k] × [4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)]
// 各項が独立しているため並列化に最適

// 必要なビット精度を計算（1桁 ≈ 3.32ビット + マージン）
unsigned long calculate_precision(int digits) {
    return (unsigned long)(digits * 3.5 + 64);
}

// 必要な項数を計算（約1項で1桁の精度）
int calculate_num_terms(int digits) {
    return digits + 10;  // マージンを追加
}

// k番目のBBP項を計算
// term = (1/16^k) × [4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)]
void compute_bbp_term(mpf_t result, unsigned long k) {
    mpf_t power16, term1, term2, term3, term4, temp;
    unsigned long k8 = 8 * k;

    mpf_init(power16);
    mpf_init(term1);
    mpf_init(term2);
    mpf_init(term3);
    mpf_init(term4);
    mpf_init(temp);

    // 16^k を計算
    mpf_set_ui(power16, 16);
    mpf_pow_ui(power16, power16, k);

    // 4/(8k+1)
    mpf_set_ui(term1, 4);
    mpf_div_ui(term1, term1, k8 + 1);

    // 2/(8k+4)
    mpf_set_ui(term2, 2);
    mpf_div_ui(term2, term2, k8 + 4);

    // 1/(8k+5)
    mpf_set_ui(term3, 1);
    mpf_div_ui(term3, term3, k8 + 5);

    // 1/(8k+6)
    mpf_set_ui(term4, 1);
    mpf_div_ui(term4, term4, k8 + 6);

    // 4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)
    mpf_sub(temp, term1, term2);
    mpf_sub(temp, temp, term3);
    mpf_sub(temp, temp, term4);

    // (1/16^k) × [...]
    mpf_div(result, temp, power16);

    mpf_clear(power16);
    mpf_clear(term1);
    mpf_clear(term2);
    mpf_clear(term3);
    mpf_clear(term4);
    mpf_clear(temp);
}

// ローカル部分和を計算（ストライド分散方式）
void compute_local_sum(mpf_t local_sum, int rank, int size, int num_terms) {
    mpf_t term;
    mpf_init(term);
    mpf_set_ui(local_sum, 0);

    // ストライド分散: プロセスrankは k = rank, rank+size, rank+2*size, ... を担当
    for (int k = rank; k < num_terms; k += size) {
        compute_bbp_term(term, k);
        mpf_add(local_sum, local_sum, term);
    }

    mpf_clear(term);
}

// MPI_Send/Recvで結果を集約（GMPは文字列変換で送受信）
void aggregate_results(mpf_t pi, mpf_t local_sum, int rank, int size, int digits) {
    // 送受信用バッファサイズ（精度 + マージン）
    size_t buffer_size = digits + 100;
    char *buffer = (char *)malloc(buffer_size);

    if (rank == 0) {
        // rank 0 は自分の部分和から開始
        mpf_set(pi, local_sum);

        // 他のプロセスから部分和を受信して加算
        mpf_t received_sum;
        mpf_init(received_sum);

        for (int src = 1; src < size; src++) {
            MPI_Recv(buffer, buffer_size, MPI_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            mpf_set_str(received_sum, buffer, 10);
            mpf_add(pi, pi, received_sum);
        }

        mpf_clear(received_sum);
    } else {
        // 他のプロセスは部分和を文字列化して rank 0 に送信
        gmp_sprintf(buffer, "%.*Ff", digits + 20, local_sum);
        MPI_Send(buffer, strlen(buffer) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    free(buffer);
}

// 円周率を整形して出力
void print_pi(mpf_t pi, int digits) {
    // 指定桁数で出力
    gmp_printf("計算結果:\n3.");

    // 整数部分を除いた小数部分を取得
    mpf_t frac;
    mpf_init(frac);
    mpf_set(frac, pi);
    mpf_sub_ui(frac, frac, 3);  // 3を引く

    // 小数部分を文字列化
    char *str = (char *)malloc(digits + 100);
    gmp_sprintf(str, "%.*Ff", digits, frac);

    // "0."の部分をスキップして小数部分を出力（10桁ごとに区切り）
    char *decimal_part = str + 2;  // "0."をスキップ
    int len = strlen(decimal_part);
    if (len > digits) len = digits;

    for (int i = 0; i < len; i++) {
        putchar(decimal_part[i]);
        if ((i + 1) % 10 == 0 && i + 1 < len) {
            putchar(' ');
            if ((i + 1) % 50 == 0) {
                printf("\n  ");
            }
        }
    }
    printf("\n");

    free(str);
    mpf_clear(frac);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int digits = 100;  // デフォルト100桁
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // コマンドライン引数から桁数を取得
    if (argc > 1) {
        digits = atoi(argv[1]);
        if (digits < 1) {
            if (rank == 0) {
                fprintf(stderr, "使用法: mpirun -np N %s [桁数]\n", argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // GMP精度を設定
    unsigned long precision = calculate_precision(digits);
    mpf_set_default_prec(precision);

    int num_terms = calculate_num_terms(digits);

    start_time = MPI_Wtime();

    // 各プロセスがストライド分散で部分和を計算
    mpf_t local_sum, pi;
    mpf_init(local_sum);
    mpf_init(pi);

    compute_local_sum(local_sum, rank, size, num_terms);

    // rank 0 が MPI_Recv で全部分和を収集・合算
    aggregate_results(pi, local_sum, rank, size, digits);

    end_time = MPI_Wtime();

    // 結果を整形出力
    if (rank == 0) {
        printf("=== BBP公式によるMPI並列円周率計算 ===\n");
        printf("プロセス数: %d\n", size);
        printf("計算桁数: %d\n", digits);
        printf("計算項数: %d\n", num_terms);
        printf("GMP精度: %lu ビット\n", precision);
        printf("\n");
        print_pi(pi, digits);
        printf("\n計算時間: %.3f 秒\n", end_time - start_time);
    }

    mpf_clear(local_sum);
    mpf_clear(pi);

    MPI_Finalize();
    return 0;
}
