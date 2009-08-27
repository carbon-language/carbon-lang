// RUN: clang-cc -fsyntax-only -verify %s
struct B { B(void* = 0); };

struct A {
  A(B b = B()) { }
};

void f() {
  (void)B();
  (void)A();
}
