// RUN: %clang_cc1 -verify %s
template<typename T>
void f0() {
  struct X;
  typedef struct Y {
    T (X::* f1())(int) { return 0; }
  } Y2;

  Y2 y = Y();
}

template void f0<int>();
