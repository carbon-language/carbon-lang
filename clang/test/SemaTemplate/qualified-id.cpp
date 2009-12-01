// RUN: clang-cc -fsyntax-only -verify %s

// PR5061
namespace a {
  template <typename T> class C {};
}
namespace b {
  template<typename T> void f0(a::C<T> &a0) { }
}


namespace test1 {
  int a = 0;
  template <class T> class Base { };
  template <class T> class Derived : public Base<T> {
    int foo() {
      return test1::a;
    }
  };
}

namespace test2 {
  class Impl {
    int foo();
  };
  template <class T> class Magic : public Impl {
    int foo() {
      return Impl::foo();
    }
  };
}
