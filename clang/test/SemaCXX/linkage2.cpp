// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test1 {
  int x; // expected-note {{previous definition is here}}
  static int y; // expected-note {{previous definition is here}}
  void f() {} // expected-note {{previous definition is here}}

  extern "C" {
    extern int x; // expected-error {{declaration of 'x' has a different language linkage}}
    extern int y; // expected-error {{declaration of 'y' has a different language linkage}}
    void f(); // expected-error {{declaration of 'f' has a different language linkage}}
  }
}

extern "C" {
  static void test2_f() { // expected-note {{previous definition is here}}
  }
  static void test2_f(int x) { // expected-error {{conflicting types for 'test2_f'}}
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
