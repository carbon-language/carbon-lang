// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i686-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck %s

#include <stdarg.h>

struct A {
  A(int a) : a(a) {}
  A(const A &o) : a(o.a) {}
  ~A() {}
  int a;
};

int foo(A a, ...) {
  va_list ap;
  va_start(ap, a);
  int sum = 0;
  for (int i = 0; i < a.a; ++i)
    sum += va_arg(ap, int);
  va_end(ap);
  return sum;
}

int main() {
  return foo(A(3), 1, 2, 3);
}
// CHECK-LABEL: define i32 @main()
// CHECK: %[[argmem_cast:[^ ]*]] = bitcast <{ %struct.A, i32, i32, i32 }>* %argmem to <{ %struct.A }>*
// CHECK: call i32 (<{ %struct.A }>*, ...)* @"\01?foo@@YAHUA@@ZZ"(<{ %struct.A }>* inalloca %[[argmem_cast]])
