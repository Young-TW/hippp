#include "benchmark/matrix_multiplication.hpp"
#include "benchmark/vector_add.hpp"

int main(int argc, char* argv[]) {
    benchmark_vector_add(100);
    benchmark_matrix_multiplication(100);
    return 0;
}
