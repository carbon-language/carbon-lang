// RUN: clang-cc -emit-llvm-only -verify %s

struct A {
  A& operator=(const A&);
};

struct B {
  A a;
  float b;
  int (A::*c)();
  _Complex float d;
};
void a(B& x, B& y) {
  x = y;
}

