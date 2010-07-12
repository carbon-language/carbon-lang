// RUN: %clang_cc1 -emit-llvm -o - -O0 -triple x86_64-apple-darwin10 %s | FileCheck %s

// Ensure that we don't emit available_externally functions at -O0.
int x;

inline void f0(int y) { x = y; }

// CHECK: define void @test()
// CHECK: declare void @f0(i32)
void test() {
  f0(17);
}
