// RUN: clang-cc -fsyntax-only -verify %s

namespace N1 {
  struct X { };
  int& f(void*);
}

namespace N2 {
  template<typename T> struct Y { };
}

namespace N3 {
  void test() {
    int &ir = f((N2::Y<N1::X>*)0);
  }
}
