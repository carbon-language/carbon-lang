// RUN: clang-cc -fsyntax-only -verify %s

// We have to avoid ADL for this test.

template <unsigned N> class test {};

class foo {};
test<0> foo(foo); // expected-note {{candidate}}

namespace A {
  class foo { int x; };
  test<1> foo(class foo);

  namespace B {
    test<2> foo(class ::foo); // expected-note {{candidate}}

    void test0() {
      using ::foo;

      class foo a;
      test<0> _ = (foo)(a);
    }

    void test1() {
      using A::foo;

      class foo a;
      test<1> _ = (foo)(a);
    };

    void test2() {
      class ::foo a;
      
      // Argument-dependent lookup is ambiguous between B:: and ::.
      test<0> _0 = foo(a); // expected-error {{call to 'foo' is ambiguous}}

      // But basic unqualified lookup is not.
      test<2> _1 = (foo)(a);

      class A::foo b;
      test<2> _2 = (foo)(b); // expected-error {{incompatible type passing}}
    }
  }
}
