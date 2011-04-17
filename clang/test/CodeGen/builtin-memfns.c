// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// CHECK: @test1
// CHECK: call void @llvm.memset.p0i8.i32
// CHECK: call void @llvm.memset.p0i8.i32
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
// CHECK: call void @llvm.memmove.p0i8.p0i8.i32
// CHECK-NOT: __builtin
// CHECK: ret
int test1(int argc, char **argv) {
  unsigned char a = 0x11223344;
  unsigned char b = 0x11223344;
  __builtin_bzero(&a, sizeof(a));
  __builtin_memset(&a, 0, sizeof(a));
  __builtin_memcpy(&a, &b, sizeof(a));
  __builtin_memmove(&a, &b, sizeof(a));
  return 0;
}

// rdar://9289468

// CHECK: @test2
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
char* test2(char* a, char* b) {
  return __builtin_memcpy(a, b, 4);
}

// CHECK: @test3
// CHECK: call void @llvm.memset
void test3(char *P) {
  __builtin___memset_chk(P, 42, 128, 128);
}

// CHECK: @test4
// CHECK: call void @llvm.memcpy
void test4(char *P, char *Q) {
  __builtin___memcpy_chk(P, Q, 128, 128);
}

// CHECK: @test5
// CHECK: call void @llvm.memmove
void test5(char *P, char *Q) {
  __builtin___memmove_chk(P, Q, 128, 128);
}
