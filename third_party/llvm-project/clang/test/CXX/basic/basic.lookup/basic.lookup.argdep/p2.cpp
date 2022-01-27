// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace N {
  struct X { };
  
  X operator+(X, X);

  void f(X); // expected-note 2 {{'N::f' declared here}}
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
  (f)(x); // expected-error{{too many arguments to function call, expected 0, have 1; did you mean 'N::f'?}}
  ::f(x); // expected-error{{too many arguments to function call, expected 0, have 1; did you mean 'N::f'?}}
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
    g(x); // expected-error{{call to 'g' is ambiguous}}
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
  struct L { int x; };
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

// PR13682: we might need to instantiate class temploids.
namespace test7 {
  namespace inner {
    class A {};
    void test7_function(A &);
  }
  template <class T> class B : public inner::A {};

  void test(B<int> &ref) {
    test7_function(ref);
  }
}

// Like test7, but ensure we don't complain if the type is properly
// incomplete.
namespace test8 {
  template <class T> class B;
  void test8_function(B<int> &);

  void test(B<int> &ref) {
    test8_function(ref);
  }
}



// [...] Typedef names and using-declarations used to specify the types
// do not contribute to this set.
namespace typedef_names_and_using_declarations {
  namespace N { struct S {}; void f(S); }
  namespace M { typedef N::S S; void g1(S); } // expected-note {{declared here}}
  namespace L { using N::S; void g2(S); } // expected-note {{declared here}}
  void test() {
	  M::S s;
    f(s);    // ok
    g1(s);   // expected-error {{use of undeclared}}
    g2(s);   // expected-error {{use of undeclared}}
  }
}
