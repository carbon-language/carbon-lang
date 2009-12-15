// RUN: %clang_cc1 -fsyntax-only -verify %s

// We have to avoid ADL for this test.

template <unsigned N> class test {};

class foo {};	// expected-note {{candidate}}
test<0> foo(foo); // expected-note {{candidate}}

namespace Test0 {
  class foo { int x; };
  test<1> foo(class foo);

  namespace A {
    test<2> foo(class ::foo); // expected-note {{candidate}}

    void test0() {
      using ::foo;

      class foo a;
      test<0> _ = (foo)(a);
    }

    void test1() {
      using Test0::foo;

      class foo a;
      test<1> _ = (foo)(a);
    };

    void test2() {
      class ::foo a;
      
      // Argument-dependent lookup is ambiguous between B:: and ::.
      test<0> _0 = foo(a); // expected-error {{call to 'foo' is ambiguous}}

      // But basic unqualified lookup is not.
      test<2> _1 = (foo)(a);

      class Test0::foo b;
      test<2> _2 = (foo)(b); // expected-error {{no viable conversion from 'class Test0::foo' to 'class foo' is possible}}
    }
  }
}

namespace Test1 {
  namespace A {
    class a {};
  }

  namespace B {
    typedef class {} b;
  }

  namespace C {
    int c(); // expected-note {{target of using declaration}}
  }

  namespace D {
    using typename A::a;
    using typename B::b;
    using typename C::c; // expected-error {{'typename' keyword used on a non-type}}

    a _1 = A::a();
    b _2 = B::b();
  }
}
