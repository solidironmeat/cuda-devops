#ifndef OPERATIONS_H
#define OPERATIONS_H

void add(const float* d_a, const float* d_b, float* d_c, size_t num_elements);
void sub(const float* d_a, const float* d_b, float* d_c, size_t num_elements);
void mul(const float* d_a, const float* d_b, float* d_c, size_t num_elements);
void mod(const float* d_a, const float* d_b, float* d_c, size_t num_elements);
void div(const float* d_a, const float* d_b, float* d_c, size_t num_elements);
void pow(const float* d_a, const float* d_b, float* d_c, size_t num_elements);

#endif  // OPERATIONS_H