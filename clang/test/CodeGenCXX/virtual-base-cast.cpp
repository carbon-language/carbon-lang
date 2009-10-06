// RUN: clang-cc -emit-llvm-only %s

struct A { virtual ~A(); };
struct B : A { virtual ~B(); };
struct C : virtual B { virtual ~C(); };

void f(C *c) {
  A* a = c;
}