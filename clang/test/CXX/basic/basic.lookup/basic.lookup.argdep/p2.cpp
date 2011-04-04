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

void f(); // expected-note 2 {{'f' declared here}}

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

extern "C" {
  struct L { };
}

void h(L); // expected-note{{candidate function}}

namespace P {
  void h(L); // expected-note{{candidate function}}
  void test_transparent_context_adl(L l) {
    {
      h(l); // expected-error {{call to 'h' is ambiguous}}
    }
  }
}

namespace test5 {
  namespace NS {
    struct A;
    void foo(void (*)(A&));
  }
  void bar(NS::A& a);

  void test() {
    foo(&bar);
  }
}

// PR6762: __builtin_va_list should be invisible to ADL on all platforms.
void test6_function(__builtin_va_list &argv);
namespace test6 {
  void test6_function(__builtin_va_list &argv);

  void test() {
    __builtin_va_list args;
    test6_function(args);
  }
}
