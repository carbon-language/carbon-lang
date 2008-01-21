// RUN: %llvmgcc %s -S -o -

struct X { long double b; unsigned char c; double __attribute__((packed)) d; };
struct X x = { 3.0L, 5, 3.0 };

