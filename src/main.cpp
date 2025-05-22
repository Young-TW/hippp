#include <vector>

std::vector<float> vector_add_raii(const std::vector<float>& h_a, const std::vector<float>& h_b);
std::vector<float> vector_add_origin(const std::vector<float>& h_a, const std::vector<float>& h_b);

int main(int argc, char* argv[]) {
    std::vector<float> h_a(1024, 1.0f);
    std::vector<float> h_b(1024, 2.0f);

    std::vector<float> h_c_raii =  vector_add_raii(h_a, h_b);
    std::vector<float> h_c_origin = vector_add_origin(h_a, h_b);
    return 0;
}
