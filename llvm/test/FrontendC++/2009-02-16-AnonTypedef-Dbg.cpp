// Test on debug info to make sure that anon typedef info is emitted.
// RUN: %llvmgcc -S --emit-llvm -x c++ -g %s -o - | grep composite
typedef struct { int a; long b; } foo;
foo x;

