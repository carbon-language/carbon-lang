// RUN: %clang_cc1 -fsyntax-only -verify %s

// Make sure we don't produce invalid IR.
// RUN: %clang_cc1 -emit-llvm-only %s

namespace test1 {
  static void foo(); // expected-warning {{function 'test1::foo' has internal linkage but is not defined}}
  template <class T> static void bar(); // expected-warning {{function 'test1::bar<int>' has internal linkage but is not defined}}

  void test() {
    foo(); // expected-note {{used here}}
    bar<int>(); // expected-note {{used here}}
  }
}

namespace test2 {
  namespace {
    void foo(); // expected-warning {{function 'test2::<anonymous namespace>::foo' has internal linkage but is not defined}}
    extern int var; // expected-warning {{variable 'test2::<anonymous namespace>::var' has internal linkage but is not defined}}
    template <class T> void bar(); // expected-warning {{function 'test2::<anonymous namespace>::bar<int>' has internal linkage but is not defined}}
  }
  void test() {
    foo(); // expected-note {{used here}}
    var = 0; // expected-note {{used here}}
    bar<int>(); // expected-note {{used here}}
  }
}

namespace test3 {
  namespace {
    void foo();
    extern int var;
    template <class T> void bar();
  }

  void test() {
    foo();
    var = 0;
    bar<int>();
  }

  namespace {
    void foo() {}
    int var = 0;
    template <class T> void bar() {}
  }
}

namespace test4 {
  namespace {
    struct A {
      A(); // expected-warning {{function 'test4::<anonymous namespace>::A::A' has internal linkage but is not defined}}
      ~A();// expected-warning {{function 'test4::<anonymous namespace>::A::~A' has internal linkage but is not defined}}
      virtual void foo(); // expected-warning {{function 'test4::<anonymous namespace>::A::foo' has internal linkage but is not defined}}
      virtual void bar() = 0;
      virtual void baz(); // expected-warning {{function 'test4::<anonymous namespace>::A::baz' has internal linkage but is not defined}}
    };
  }

  void test(A &a) {
    a.foo(); // expected-note {{used here}}
    a.bar();
    a.baz(); // expected-note {{used here}}
  }

  struct Test : A {
    Test() {} // expected-note 2 {{used here}}
  };
}

// rdar://problem/9014651
namespace test5 {
  namespace {
    struct A {};
  }

  template <class N> struct B {
    static int var; // expected-warning {{variable 'test5::B<test5::<anonymous>::A>::var' has internal linkage but is not defined}}
    static void foo(); // expected-warning {{function 'test5::B<test5::<anonymous>::A>::foo' has internal linkage but is not defined}}
  };

  void test() {
    B<A>::var = 0; // expected-note {{used here}}
    B<A>::foo(); // expected-note {{used here}}
  }
}
