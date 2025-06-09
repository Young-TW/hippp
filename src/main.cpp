#include "benchmark/vector_add.hpp"
#include "benchmark/matrix_multiplication.hpp"

int main(int argc, char* argv[]) {
    benchmark_vector_add();
    benchmark_matrix_multiplication();
    return 0;
}
