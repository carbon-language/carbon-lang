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

// PR5764
namespace PR5764 {
  class X {
    template <typename T>
    void Bar() {
      class Y {
        Y() {}
      };

      Y y;
    }
  };

  void test(X x) {
    x.Bar<int>();
  }
}

