// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

// PR5392
namespace PR5392 {
struct A
{
  static int a;
};

A a1;
void f()
{
  // CHECK: store i32 10, i32* @_ZN6PR53921A1aE
  a1.a = 10;
  // CHECK: store i32 20, i32* @_ZN6PR53921A1aE
  A().a = 20;
}

}
