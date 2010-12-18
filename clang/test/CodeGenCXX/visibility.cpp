// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -fvisibility hidden -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-HIDDEN

#define HIDDEN __attribute__((visibility("hidden")))
#define PROTECTED __attribute__((visibility("protected")))
#define DEFAULT __attribute__((visibility("default")))

// CHECK: @_ZN5Test425VariableInHiddenNamespaceE = hidden global i32 10
// CHECK: @_ZN5Test71aE = hidden global
// CHECK: @_ZN5Test71bE = global
// CHECK: @test9_var = global
// CHECK-HIDDEN: @test9_var = global
// CHECK: @_ZN6Test121A6hiddenE = external hidden global
// CHECK: @_ZN6Test121A7visibleE = external global
// CHECK-HIDDEN: @_ZN6Test121A6hiddenE = external hidden global
// CHECK-HIDDEN: @_ZN6Test121A7visibleE = external global
// CHECK: @_ZN6Test131B1aE = hidden global
// CHECK: @_ZN6Test131C1aE = global
// CHECK-HIDDEN: @_ZN6Test131B1aE = hidden global
// CHECK-HIDDEN: @_ZN6Test131C1aE = global
// CHECK: @_ZN6Test143varE = external global
// CHECK-HIDDEN: @_ZN6Test143varE = external global
// CHECK: @_ZN6Test154TempINS_1AEE5Inner6bufferE = external global [0 x i8]
// CHECK-HIDDEN: @_ZN6Test154TempINS_1AEE5Inner6bufferE = external global [0 x i8]
// CHECK: @_ZZN6Test193fooIiEEvvE1a = linkonce_odr global
// CHECK: @_ZGVZN6Test193fooIiEEvvE1a = linkonce_odr global i64
// CHECK-HIDDEN: @_ZZN6Test193fooIiEEvvE1a = linkonce_odr hidden global
// CHECK-HIDDEN: @_ZGVZN6Test193fooIiEEvvE1a = linkonce_odr hidden global i64
// CHECK-HIDDEN: @_ZTTN6Test161AIcEE = external constant
// CHECK-HIDDEN: @_ZTVN6Test161AIcEE = external constant
// CHECK: @_ZTVN5Test63fooE = weak_odr hidden constant 

namespace Test1 {
  // CHECK: define hidden void @_ZN5Test11fEv
  void HIDDEN f() { }
  
}

namespace Test2 {
  struct HIDDEN A {
    void f();
  };

  // A::f is a member function of a hidden class.
  // CHECK: define hidden void @_ZN5Test21A1fEv
  void A::f() { }
}
 
namespace Test3 {
  struct HIDDEN A {
    struct B {
      void f();
    };
  };

  // B is a nested class where its parent class is hidden.
  // CHECK: define hidden void @_ZN5Test31A1B1fEv
  void A::B::f() { }  
}

namespace Test4 HIDDEN {
  int VariableInHiddenNamespace = 10;

  // Test4::g is in a hidden namespace.
  // CHECK: define hidden void @_ZN5Test41gEv
  void g() { } 
  
  struct DEFAULT A {
    void f();
  };
  
  // A has default visibility.
  // CHECK: define void @_ZN5Test41A1fEv
  void A::f() { } 
}

namespace Test5 {

  namespace NS HIDDEN {
    // f is in NS which is hidden.
    // CHECK: define hidden void @_ZN5Test52NS1fEv()
    void f() { }
  }
  
  namespace NS {
    // g is in NS, but this NS decl is not hidden.
    // CHECK: define void @_ZN5Test52NS1gEv
    void g() { }
  }
}

// <rdar://problem/8091955>
namespace Test6 {
  struct HIDDEN foo {
    foo() { }
    void bonk();
    virtual void bar() = 0;

    virtual void zonk() {}
  };

  struct barc : public foo {
    barc();
    virtual void bar();
  };

  barc::barc() {}
}

namespace Test7 {
  class HIDDEN A {};
  A a; // top of file

  template <A&> struct Aref {
    static void foo() {}
  };

  class B : public A {};
  B b; // top of file

  // CHECK: define linkonce_odr hidden void @_ZN5Test74ArefILZNS_1aEEE3fooEv()
  void test() {
    Aref<a>::foo();
  }
}

namespace Test8 {
  void foo();
  void bar() {}
  // CHECK-HIDDEN: define hidden void @_ZN5Test83barEv()
  // CHECK-HIDDEN: declare void @_ZN5Test83fooEv()

  void test() {
    foo();
    bar();
  }
}

// PR8457
namespace Test9 {
  extern "C" {
    struct A { int field; };
    void DEFAULT test9_fun(struct A *a) { }
    struct A DEFAULT test9_var; // above
  }
  // CHECK: define void @test9_fun(
  // CHECK-HIDDEN: define void @test9_fun(

  void test() {
    A a = test9_var;
    test9_fun(&a);
  }
}

// PR8478
namespace Test10 {
  struct A;

  DEFAULT class B {
    void foo(A*);
  };

  // CHECK: define void @_ZN6Test101B3fooEPNS_1AE(
  // CHECK-HIDDEN: define void @_ZN6Test101B3fooEPNS_1AE(
  void B::foo(A*) {}
}

// PR8492
namespace Test11 {
  struct A {
    void foo() {}
    void DEFAULT bar() {}
  };

  void test() {
    A a;
    a.foo();
    a.bar();
  }

  // CHECK: define linkonce_odr void @_ZN6Test111A3fooEv(
  // CHECK: define linkonce_odr void @_ZN6Test111A3barEv(
  // CHECK-HIDDEN: define linkonce_odr hidden void @_ZN6Test111A3fooEv(
  // CHECK-HIDDEN: define linkonce_odr void @_ZN6Test111A3barEv(
}

// Tested at top of file.
namespace Test12 {
  struct A {
    // This is hidden in all cases: the explicit attribute takes
    // priority over -fvisibility on the parent.
    static int hidden HIDDEN;

    // This is default in all cases because it's only a declaration.
    static int visible;
  };

  void test() {
    A::hidden = 0;
    A::visible = 0;
  }
}

// Tested at top of file.
namespace Test13 {
  struct HIDDEN A {};

  // Should be hidden in all cases.
  struct B {
    static A a;
  };
  A B::a;

  // Should be default in all cases.
  struct DEFAULT C {
    static A a;
  };
  A C::a;
};

// Tested at top of file.
namespace Test14 {
  // Neither the visibility of the type nor -fvisibility=hidden should
  // apply to declarations.
  extern struct A *var;

  struct A *test() { return var; }
}

// rdar://problem/8613093
namespace Test15 {
  struct A {};
  template <class T> struct Temp {
    struct Inner {
      static char buffer[0];
    };
  };

  char *test() {
    return Temp<A>::Inner::buffer;
  }
}

namespace Test16 {
  struct Base1 { virtual void foo(); };
  struct Base2 : virtual Base1 { virtual void foo(); };
  template <class T> struct A : virtual Base1, Base2 {
    virtual void foo();
  };
  extern template struct A<char>;

  void test() {
    A<char> a;
    a.foo();
  }
}

namespace Test17 {
  struct HIDDEN A {
    static void foo();
    static void DEFAULT bar();
    static void HIDDEN baz();

    struct DEFAULT B {
      static void foo();
      static void DEFAULT bar();
      static void HIDDEN baz();
    };
  };

  void test() {
    A::foo();
    A::bar();
    A::baz();
    A::B::foo();
    A::B::bar();
    A::B::baz();
  }
  // CHECK: declare hidden void @_ZN6Test171A3fooEv()
  // CHECK: declare void @_ZN6Test171A3barEv()
  // CHECK: declare hidden void @_ZN6Test171A3bazEv()
  // CHECK: declare void @_ZN6Test171A1B3fooEv()
  // CHECK: declare void @_ZN6Test171A1B3barEv()
  // CHECK: declare hidden void @_ZN6Test171A1B3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test171A3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test171A3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test171A3bazEv()
  // CHECK-HIDDEN: declare void @_ZN6Test171A1B3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test171A1B3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test171A1B3bazEv()
}

namespace Test18 {
  template <class T> struct HIDDEN A {
    static void foo();
    static void DEFAULT bar();
    static void HIDDEN baz();

    struct DEFAULT B {
      static void foo();
      static void DEFAULT bar();
      static void HIDDEN baz();
    };
  };
  struct HIDDEN H;

  void test() {
    A<int>::foo();
    A<int>::bar();
    A<int>::baz();
    A<int>::B::foo();
    A<int>::B::bar();
    A<int>::B::baz();
    A<H>::foo();
    A<H>::bar();
    A<H>::baz();
    A<H>::B::foo();
    A<H>::B::bar();
    A<H>::B::baz();
  }
  // CHECK: declare hidden void @_ZN6Test181AIiE3fooEv()
  // CHECK: declare void @_ZN6Test181AIiE3barEv()
  // CHECK: declare hidden void @_ZN6Test181AIiE3bazEv()
  // CHECK: declare void @_ZN6Test181AIiE1B3fooEv()
  // CHECK: declare void @_ZN6Test181AIiE1B3barEv()
  // CHECK: declare hidden void @_ZN6Test181AIiE1B3bazEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE3fooEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE3barEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE3bazEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE1B3fooEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE1B3barEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE1B3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AIiE3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test181AIiE3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AIiE3bazEv()
  // CHECK-HIDDEN: declare void @_ZN6Test181AIiE1B3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test181AIiE1B3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AIiE1B3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE3fooEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE1B3fooEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE1B3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE1B3bazEv()
}

namespace Test19 {
  struct A { A(); ~A(); };

  // Tested at top of file.
  template <class T> void foo() {
    static A a;
  }

  void test() {
    foo<int>();
  }
}

// Various things with class template specializations.
namespace Test20 {
  template <unsigned> struct HIDDEN A {};

  // An explicit specialization inherits the explicit visibility of
  // the template.
  template <> struct A<0> {
    static void test0();
    static void test1();
  };

  // CHECK: define hidden void @_ZN6Test201AILj0EE5test0Ev()
  void A<0>::test0() {}

  // CHECK: declare hidden void @_ZN6Test201AILj0EE5test1Ev()
  void test1() {
    A<0>::test1();
  }

  // ...unless that's explicitly overridden.
  template <> struct DEFAULT A<1> {
    static void test2();
    static void test3();
  };

  // CHECK: define void @_ZN6Test201AILj1EE5test2Ev()
  void A<1>::test2() {}

  // CHECK: declare void @_ZN6Test201AILj1EE5test3Ev()
  void test3() {
    A<1>::test3();
  }

  // <rdar://problem/8778497>
  // But we should assume that an unknown specialization has the
  // explicit visibility settings of the template.
  template <class T> struct B {
    static void test4() {}
    static void test5();
  };

  // CHECK: define linkonce_odr hidden void @_ZN6Test201BINS_1AILj2EEEE5test4Ev()
  void test4() {
    B<A<2> >::test4();
  }

  // CHECK: declare void @_ZN6Test201BINS_1AILj2EEEE5test4Ev()
  // (but explicit visibility on a template argument doesn't count as
  //  explicit visibility for the template for purposes of deciding
  //  whether an external symbol gets visibility)
  void test5() {
    B<A<2> >::test5();
  }
}
