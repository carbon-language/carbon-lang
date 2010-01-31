// RUN: %clang_cc1 -fsyntax-only -verify %s

template <const int* p> struct X { };

int i = 42;
int* iptr = &i;
void test() {
  X<&i> x1;
  X<iptr> x2;
}
