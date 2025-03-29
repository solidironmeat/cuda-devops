#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <operations/operations.h>

class OperationsTest : public ::testing::Test {
protected:
    static constexpr size_t N = 4;
    static constexpr size_t size = N * sizeof(float);
    float h_A[N] = { 1.0, 2.0, 3.0, 4.0 };
    float h_B[N] = { 5.0, 6.0, 7.0, 2.0 };
    float h_C[N] = { 0.0, 0.0, 0.0, 0.0 };
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    void SetUp() override {
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void copyResultToHost() {
        cudaError_t err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy h_C: " << cudaGetErrorString(err);
    }

};

// Test to check if CUDA is enabled and functional
TEST_F(OperationsTest, CheckCudaInitialization) {
    cudaError_t err = cudaFree(0);
    ASSERT_EQ(err, cudaSuccess) << "CUDA initialization failed: " << cudaGetErrorString(err);

    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);
    ASSERT_GT(deviceCount, 0) << "No CUDA-capable devices found";
}

TEST_F(OperationsTest, Addition) {
    add(d_A, d_B, d_C, N);
    copyResultToHost();
    ASSERT_FLOAT_EQ(h_C[0], 6.0) << "Addition failed at index 0";
    ASSERT_FLOAT_EQ(h_C[1], 8.0) << "Addition failed at index 1";
    ASSERT_FLOAT_EQ(h_C[2], 10.0) << "Addition failed at index 2";
    ASSERT_FLOAT_EQ(h_C[3], 6.0) << "Addition failed at index 3";
}

TEST_F(OperationsTest, Subtraction) {
    sub(d_A, d_B, d_C, N);
    copyResultToHost();
    ASSERT_FLOAT_EQ(h_C[0], -4.0) << "Subtraction failed at index 0";
    ASSERT_FLOAT_EQ(h_C[1], -4.0) << "Subtraction failed at index 1";
    ASSERT_FLOAT_EQ(h_C[2], -4.0) << "Subtraction failed at index 2";
    ASSERT_FLOAT_EQ(h_C[3], 2.0) << "Subtraction failed at index 3";
}

TEST_F(OperationsTest, Multiplication) {
    mul(d_A, d_B, d_C, N);
    copyResultToHost();
    ASSERT_FLOAT_EQ(h_C[0], 5.0) << "Multiplication failed at index 0";
    ASSERT_FLOAT_EQ(h_C[1], 12.0) << "Multiplication failed at index 1";
    ASSERT_FLOAT_EQ(h_C[2], 21.0) << "Multiplication failed at index 2";
    ASSERT_FLOAT_EQ(h_C[3], 8.0) << "Multiplication failed at index 3";
}

TEST_F(OperationsTest, Modulo) {
    mod(d_A, d_B, d_C, N);
    copyResultToHost();
    ASSERT_FLOAT_EQ(h_C[0], 1.0) << "Modulo failed at index 0";  // 1 % 5 = 1
    ASSERT_FLOAT_EQ(h_C[1], 2.0) << "Modulo failed at index 1";  // 2 % 6 = 2
    ASSERT_FLOAT_EQ(h_C[2], 3.0) << "Modulo failed at index 2";  // 3 % 7 = 3
    ASSERT_FLOAT_EQ(h_C[3], 0.0) << "Modulo failed at index 3";  // 4 % 2 = 0
}

TEST_F(OperationsTest, Division) {
    div(d_A, d_B, d_C, N);
    copyResultToHost();
    ASSERT_FLOAT_EQ(h_C[0], 0.2) << "Division failed at index 0";         // 1 / 5 = 0.2
    ASSERT_FLOAT_EQ(h_C[1], 0.33333333) << "Division failed at index 1";  // 2 / 6 ≈ 0.333
    ASSERT_FLOAT_EQ(h_C[2], 0.42857143) << "Division failed at index 2";  // 3 / 7 ≈ 0.429
    ASSERT_FLOAT_EQ(h_C[3], 2.0) << "Division failed at index 3";         // 4 / 2 = 2
}

TEST_F(OperationsTest, Power) {
    pow(d_A, d_B, d_C, N);
    copyResultToHost();
    ASSERT_FLOAT_EQ(h_C[0], 1.0) << "Power failed at index 0";    // 1^5 = 1
    ASSERT_FLOAT_EQ(h_C[1], 64.0) << "Power failed at index 1";   // 2^6 = 64
    ASSERT_FLOAT_EQ(h_C[2], 2187.0) << "Power failed at index 2"; // 3^7 = 2187
    ASSERT_FLOAT_EQ(h_C[3], 16.0) << "Power failed at index 3";   // 4^2 = 16
}