// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10 < %s | FileCheck %s

struct Test1S {
 long NumDecls;
 long X;
 long Y;
};
struct Test2S {
 long NumDecls;
 long X;
};

// Make sure we don't generate extra memcpy for lvalues
void test1a(struct Test1S, struct Test2S);
// CHECK: define void @test1(
// CHECK-NOT: memcpy
// CHECK: call void @test1a
void test1(struct Test1S *A, struct Test2S *B) {
  test1a(*A, *B);
}
