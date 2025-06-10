#pragma once

#include <hip/hip_runtime.h>

#include <vector>

std::vector<float> matrix_multiplication_raii(const std::vector<float>& h_A,
                                              const std::vector<float>& h_B,
                                              int M, int N, int K);
