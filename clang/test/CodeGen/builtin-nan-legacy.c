// RUN: %clang -target mipsel-unknown-linux -mnan=legacy -emit-llvm -S %s -o - | FileCheck %s
// CHECK: float 0x7FF4000000000000, float 0x7FF8000000000000
// CHECK: double 0x7FF4000000000000, double 0x7FF8000000000000

float f[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};

double d[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};
