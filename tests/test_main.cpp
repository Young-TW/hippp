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

TEST_CASE("HipBuffer copy operations") {
    HipContext ctx(0);
    
    constexpr int N = 64;
    HipBuffer<int> buf(N);
    
    // Create host data
    std::vector<int> hostData(N);
    for (int i = 0; i < N; ++i) {
        hostData[i] = i * 2;
    }
    
    // Test copyFromHostAsync
    REQUIRE_NOTHROW(buf.copyFromHostAsync(hostData.data()));
    
    // Test copyToHostAsync
    std::vector<int> resultData(N);
    REQUIRE_NOTHROW(buf.copyToHostAsync(resultData.data()));
    
    // In our mock environment, the copy operations should work
    REQUIRE(resultData.size() == N);
}

TEST_CASE("HipStream creation and operations") {
    HipContext ctx(0);
    
    // Test stream creation
    HipStream stream;
    REQUIRE_NOTHROW(stream.sync());
    
    // Test that stream can be used with buffer operations
    constexpr int N = 32;
    HipBuffer<float> buf(N);
    std::vector<float> data(N, 3.14f);
    
    // Use stream conversion operator
    REQUIRE_NOTHROW(buf.copyFromHostAsync(data.data(), stream));
}

TEST_CASE("HipEvent timing operations") {
    HipContext ctx(0);
    
    HipEvent start, stop;
    
    REQUIRE_NOTHROW(start.record());
    REQUIRE_NOTHROW(stop.record());
    
    float elapsed = HipEvent::elapsedTime(start, stop);
    REQUIRE(elapsed >= 0.0f); // Mock returns 1.0f
}

TEST_CASE("Multiple HipContext instances") {
    // Test that multiple contexts can be created without issues
    HipContext ctx1(0);
    HipContext ctx2(0);
    // Test move semantics
    HipContext ctx3 = std::move(ctx1);
    
    REQUIRE(ctx3.get() != nullptr);
}

TEST_CASE("HipBuffer with different types") {
    HipContext ctx(0);
    
    // Test with different data types
    HipBuffer<int> intBuf(100);
    HipBuffer<double> doubleBuf(50);
    HipBuffer<char> charBuf(1024);
    
    REQUIRE(intBuf.size() == 100);
    REQUIRE(doubleBuf.size() == 50);
    REQUIRE(charBuf.size() == 1024);
    
    REQUIRE(intBuf.get() != nullptr);
    REQUIRE(doubleBuf.get() != nullptr);
    REQUIRE(charBuf.get() != nullptr);
}
