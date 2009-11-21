// RUN: clang-cc -verify -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

// CHECK: @a = global i32 10
int a = 10;
// CHECK: @ar = global i32* @a
int &ar = a;

void f();
// CHECK: @fr = global void ()* @_Z1fv
void (&fr)() = f;

struct S { int& a; };
// CHECK: @s = global %0 { i32* @a }
S s = { a };

// PR5581
namespace PR5581 {
class C {
public:
  enum { e0, e1 };
  unsigned f;
};

// CHECK: @_ZN6PR55812g0E = global %1 { i32 1 }
C g0 = { C::e1 };
}
