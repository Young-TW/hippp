# Building and Testing HIPPP

This project provides C++ RAII wrappers for AMD's HIP runtime API.

## Prerequisites

### For full functionality:
- ROCm/HIP installation
- AMD GPU hardware or compatible software stack

### For testing only:
- C++ compiler with C++23 support  
- CMake 3.21 or later

## Building

### Full Build (with ROCm/HIP)

```bash
mkdir build && cd build
cmake ..
make
```

### Testing Only (without ROCm/HIP)

When ROCm is not available or you only want to run tests, use:

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS_ONLY=ON
make hippp_tests
./hippp_tests
```

Or use the CMake test system:

```bash
make test
```

## Test Coverage

The test suite includes:
- HipContext creation and destruction
- HipBuffer allocation, deallocation, and move semantics
- Memory copy operations (host ↔ device)
- HipStream creation and operations
- HipEvent timing functionality
- Multiple context management
- Type safety for different data types

## Mock Environment

When `BUILD_TESTS_ONLY=ON` is used, the build system includes a mock HIP runtime environment that:
- Provides all necessary HIP API functions as inline mocks
- Uses regular malloc/free for memory operations
- Returns success for all operations to test the RAII logic
- Allows testing without actual GPU hardware