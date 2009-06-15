// RUN: clang-cc -fsyntax-only -verify %s
// XFAIL

// FIXME: This part is here to demonstrate the failure in looking up 'f', it can
// be removed once the whole test passes.
typedef int f; 
namespace N0 {
  struct A { 
    friend void f(); 
    void g() {
      int i = f(1);
    }
  };
}

namespace N1 {
  struct A { 
    friend void f(A &);
    operator int();
    void g(A a) {
      // ADL should not apply to the lookup of 'f', it refers to the typedef
      // above.
      int i = f(a);
    }
  };
}
