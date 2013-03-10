// RUN: %clang_cc1 %s -emit-llvm -o - -O0 -ffake-address-space-map -triple i686-pc-darwin | FileCheck %s

typedef struct {
  int cells[9];
} Mat3X3;

typedef struct {
  int cells[16];
} Mat4X4;

Mat4X4 __attribute__((noinline)) foo(Mat3X3 in) {
  Mat4X4 out;
  return out;
}

kernel void ker(global Mat3X3 *in, global Mat4X4 *out) {
  out[0] = foo(in[1]);
}

// Expect two mem copies: one for the argument "in", and one for
// the return value.
// CHECK: call void @llvm.memcpy.p0i8.p1i8.i32(i8*
// CHECK: call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)*
