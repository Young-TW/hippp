#include <catch2/catch_test_macros.hpp>
#include "hip_raii/hip_raii.hpp"

TEST_CASE("HipContext constructs and destroys") {
    REQUIRE_NOTHROW(HipContext(0));
}

TEST_CASE("HipBuffer alloc/free and move semantics") {
    HipContext ctx(0);

    constexpr int N = 128;
    HipBuffer<float> buf(N);

    REQUIRE(buf.get() != nullptr);
    REQUIRE(buf.size() == N);

    auto p = buf.get();
    HipBuffer<float> moved(std::move(buf));
    REQUIRE(moved.get() == p);
    REQUIRE(buf.get() == nullptr);
}
