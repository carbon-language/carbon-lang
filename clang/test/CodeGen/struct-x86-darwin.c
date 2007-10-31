// RUN: clang %s -emit-llvm | grep "STest1 = type { i32, \[4 x i16\], double }"
// Test struct layout for x86-darwin target
// FIXME : Enable this test for x86-darwin only. At the moment clang hard codes
// x86-darwin as the target

struct STest1 {int x; short y[4]; double z; } st1;
