// RUN: %clang -target mipsel-unknown-linux -ccc-clang-archs mipsel -O3 -S -o - -emit-llvm %s | FileCheck %s -check-prefix=O32
// RUN: %clang -target mips64el-unknown-linux -ccc-clang-archs mips64el -O3 -S -mabi=n64 -o - -emit-llvm %s | FileCheck %s -check-prefix=N64

// vectors larger than 16-bytes are returned via the hidden pointer argument. 
// N64/N32 returns vectors whose size is equal to or smaller than 16-bytes in
// integer registers. 
typedef float  v4sf __attribute__ ((__vector_size__ (16)));
typedef double v4df __attribute__ ((__vector_size__ (32)));

// O32: define void @test_v4sf(<4 x float>* noalias nocapture sret
// N64: define { i64, i64 } @test_v4sf
v4sf test_v4sf(float a) {
  return (v4sf){0.0f, a, 0.0f, 0.0f};
}

// O32: define void @test_v4df(<4 x double>* noalias nocapture sret
// N64: define void @test_v4df(<4 x double>* noalias nocapture sret
v4df test_v4df(double a) {
  return (v4df){0.0, a, 0.0, 0.0};
}

