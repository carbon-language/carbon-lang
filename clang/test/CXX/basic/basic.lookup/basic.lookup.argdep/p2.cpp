// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace N {
  struct X { };
  
  X operator+(X, X);

  void f(X);
  void g(X); // expected-note{{candidate function}}

  void test_multiadd(X x) {
    (void)(x + x);
  }
}

namespace M {
  struct Y : N::X { };
}

void f();

void test_operator_adl(N::X x, M::Y y) {
  (void)(x + x);
  (void)(y + y);
}

void test_func_adl(N::X x, M::Y y) {
  f(x);
  f(y);
  (f)(x); // expected-error{{too many arguments to function call}}
  ::f(x); // expected-error{{too many arguments to function call}}
}

namespace N {
  void test_multiadd2(X x) {
    (void)(x + x);
  }
}


void test_func_adl_only(N::X x) {
  g(x);
}

namespace M {
  int g(N::X); // expected-note{{candidate function}}

  void test(N::X x) {
    g(x); // expected-error{{call to 'g' is ambiguous; candidates are:}}
    int i = (g)(x);

    int g(N::X);
    g(x); // okay; calls locally-declared function, no ADL
  }
}


void test_operator_name_adl(N::X x) {
  (void)operator+(x, x);
}

struct Z { };
int& f(Z);

namespace O {
  char &f();
  void test_global_scope_adl(Z z) {
    {
      int& ir = f(z);
    }
  }
}

