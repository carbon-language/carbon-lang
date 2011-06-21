// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -verify %s
template<typename T> void f() {
  T t;
  t = 17;
}

// PR5407
struct A { A(); };
struct B { ~B(); };
void f() {
  A a;
  B b;
}

// PR5531
namespace PR5531 {
  struct A {
  };

  struct B {
    B(int);
  };

  struct C {
    ~C();
  };

  void test() {
    A();
    B(17);
    C();
  }
}

template<typename T>
struct X0 { };

template<typename T>
void test_dependent_init(T *p) {
  X0<int> i(p);
  (void)i;
}

namespace PR6948 {
  template<typename T> class X;
  
  void f() {
    X<char> str (read_from_file()); // expected-error{{use of undeclared identifier 'read_from_file'}}
  }
}

void unused_local_static() {
  static int x = 0;
  static int y = 0; // expected-warning{{unused variable 'y'}}
#pragma unused(x)
}

// PR10168
namespace PR10168 {
  // We expect a warning in the definition only for non-dependent variables, and
  // a warning in the instantiation only for dependent variables.
  template<typename T>
  struct S {
    void f() {
      int a; // expected-warning {{unused variable 'a'}}
      T b; // expected-warning 2{{unused variable 'b'}}
    }
  };

  template<typename T>
  void f() {
    int a; // expected-warning {{unused variable 'a'}}
    T b; // expected-warning 2{{unused variable 'b'}}
  }

  void g() {
    S<int>().f(); // expected-note {{here}}
    S<char>().f(); // expected-note {{here}}
    f<int>(); // expected-note {{here}}
    f<char>(); // expected-note {{here}}
  }
}
