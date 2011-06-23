// RUN: %clang_cc1 -std=c++0x -fsyntax-only %s

struct D;
struct B {
  D& operator = (const D&);
};
struct D : B {
  using B::operator=;
};
struct F : D {
};

void H () {
  F f;
  f = f;
}
