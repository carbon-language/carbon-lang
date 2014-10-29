// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Test that C++ classes are correctly classified as homogeneous aggregates.

struct Base1 {
  int x;
};
struct Base2 {
  double x;
};
struct Base3 {
  double x;
};
struct D1 : Base1 {  // non-homogeneous aggregate
  double y, z;
};
struct D2 : Base2 {  // homogeneous aggregate
  double y, z;
};
struct D3 : Base1, Base2 {  // non-homogeneous aggregate
  double y, z;
};
struct D4 : Base2, Base3 {  // homogeneous aggregate
  double y, z;
};

// CHECK: define void @_Z7func_D12D1(%struct.D1* noalias sret %agg.result, [3 x i64] %x.coerce)
D1 func_D1(D1 x) { return x; }

// CHECK: define [3 x double] @_Z7func_D22D2([3 x double] %x.coerce)
D2 func_D2(D2 x) { return x; }

// CHECK: define void @_Z7func_D32D3(%struct.D3* noalias sret %agg.result, [4 x i64] %x.coerce)
D3 func_D3(D3 x) { return x; }

// CHECK: define [4 x double] @_Z7func_D42D4([4 x double] %x.coerce)
D4 func_D4(D4 x) { return x; }

