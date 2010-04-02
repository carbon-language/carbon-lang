// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// CHECK: call void @llvm.memset.p0i8.i32
// CHECK: call void @llvm.memset.p0i8.i32
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
// CHECK: call void @llvm.memmove.p0i8.p0i8.i32
// CHECK-NOT: __builtin
// CHECK: ret
int main(int argc, char **argv) {
  unsigned char a = 0x11223344;
  unsigned char b = 0x11223344;
  __builtin_bzero(&a, sizeof(a));
  __builtin_memset(&a, 0, sizeof(a));
  __builtin_memcpy(&a, &b, sizeof(a));
  __builtin_memmove(&a, &b, sizeof(a));
  return 0;
}
