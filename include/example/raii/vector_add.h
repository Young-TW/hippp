#include <hip/hip_runtime.h>
#include <vector>

std::vector<float> vector_add_raii(const std::vector<float>& h_a, const std::vector<float>& h_b);
