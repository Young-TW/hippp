# hippp - HIP programming with RAII

hippp is a C++ wrapper for the HIP API, designed to simplify the process of writing GPU code. It provides a RAII (Resource Acquisition Is Initialization) style interface, making it easier to manage resources and ensuring that they are properly released when no longer needed. This library is particularly useful for developers who want to write portable GPU code without dealing with the complexities of the HIP API directly.

## Benchmark

### Vector Addition

executed by Radeon RX 7600XT

Average RAII kernel time over 1000000 runs: 0.0724291 ms
Average Origin kernel time over 1000000 runs: 0.0726434 ms
