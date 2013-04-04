// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fmodules %s

namespace test1 {
  int x; // expected-note {{previous definition is here}}
  static int y;
  void f() {} // expected-note {{previous definition is here}}

  extern "C" {
    extern int x; // expected-error {{declaration of 'x' has a different language linkage}}
    extern int y; // OK, has internal linkage, so no language linkage.
    void f(); // expected-error {{declaration of 'f' has a different language linkage}}
  }
}

// This is OK. Both test2_f don't have language linkage since they have
// internal linkage.
extern "C" {
  static void test2_f() {
  }
  static void test2_f(int x) {
  }
}

namespace test3 {
  extern "C" {
    namespace {
      extern int x2;
      void f2();
    }
  }
  namespace {
    int x2;
    void f2() {}
  }
}

namespace test4 {
  void dummy() {
    void Bar();
    class A {
      friend void Bar();
    };
  }
}

namespace test5 {
  static void g();
  void f()
  {
    void g();
  }
}

// pr14898
namespace test6 {
  template <class _Rp>
  class __attribute__ ((__visibility__("default"))) shared_future;
  template <class _Rp>
  class future {
    template <class> friend class shared_future;
    shared_future<_Rp> share();
  };
  template <class _Rp> future<_Rp>
  get_future();
  template <class _Rp>
  struct shared_future<_Rp&> {
    shared_future(future<_Rp&>&& __f); // expected-warning {{rvalue references are a C++11 extension}}
  };
  void f() {
    typedef int T;
    get_future<int>();
    typedef int& U;
    shared_future<int&> f1 = get_future<int&>();
  }
}

// This is OK. The variables have internal linkage and therefore no language
// linkage.
extern "C" {
  static int test7_x;
}
extern "C++" {
  extern int test7_x;
}
extern "C++" {
  static int test7_y;
}
extern "C" {
  extern int test7_y;
}
extern "C" { typedef int test7_F(); static test7_F test7_f; }
extern "C++" { extern test7_F test7_f; }

// FIXME: This should be invalid. The function has no language linkage, but
// the function type has, so this is redeclaring the function with a different
// type.
extern "C++" {
  static void test8_f();
}
extern "C" {
  extern void test8_f();
}
extern "C" {
  static void test8_g();
}
extern "C++" {
  extern void test8_g();
}

extern "C" {
  void __attribute__((overloadable)) test9_f(int c); // expected-note {{previous declaration is here}}
}
extern "C++" {
  void __attribute__((overloadable)) test9_f(int c); // expected-error {{declaration of 'test9_f' has a different language linkage}}
}

extern "C" {
  void __attribute__((overloadable)) test10_f(int);
  void __attribute__((overloadable)) test10_f(double);
}

extern "C" {
  void test11_f() {
    void  __attribute__((overloadable)) test11_g(int);
    void  __attribute__((overloadable)) test11_g(double);
  }
}

namespace test12 {
  const int n = 0;
  extern const int n;
  void f() {
    extern const int n;
  }
}

namespace test13 {
  static void a(void);
  extern void a();
  static void a(void) {}
}

namespace test14 {
  namespace {
    void a(void); // expected-note {{previous declaration is here}}
    static void a(void) {} // expected-error {{static declaration of 'a' follows non-static declaration}}
  }
}

namespace test15 {
  const int a = 5; // expected-note {{previous definition is here}}
  static const int a; // expected-error {{redefinition of 'a'}}
}
