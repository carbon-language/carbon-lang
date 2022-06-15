// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @test_builtin_clz(
// CHECK: tail call i32 @llvm.ctlz.i32(i32 %a, i1 true)
void test_builtin_clz(global int* out, int a)
{
  *out = __builtin_clz(a);
}

// CHECK-LABEL: @test_builtin_clzl(
// CHECK: tail call i64 @llvm.ctlz.i64(i64 %a, i1 true)
void test_builtin_clzl(global long* out, long a)
{
  *out = __builtin_clzl(a);
}

// CHECK: tail call i8 addrspace(5)* @llvm.frameaddress.p5i8(i32 0)
void test_builtin_frame_address(int *out) {
    *out = __builtin_frame_address(0);
}
