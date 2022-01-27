// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// C++11 [class.mem]p2:
//   A class is considered a completely-defined object type (or
//   complete type) at the closing } of the class-specifier. Within
//   the class member-specification, the class is regarded as complete
//   within function bodies, default arguments,
//   exception-specifications, and brace-or-equal-initializers for
//   non-static data members (including such things in nested classes).
//   Otherwise it is regarded as incomplete within its own class
//   member-specification.

namespace test0 {
  struct A { // expected-note {{definition of 'test0::A' is not complete until the closing '}'}}
    A x; // expected-error {{field has incomplete type 'test0::A'}}
  };
}

namespace test1 {
  template <class T> struct A {
    A<int> x; // expected-error {{implicit instantiation of template 'test1::A<int>' within its own definition}}
  };
}

namespace test2 {
  template <class T> struct A;
  template <> struct A<int> {};
  template <class T> struct A {
    A<int> x;
  };
}

namespace test3 {
  struct A {
    struct B {
      void f() throw(A);
      void g() throw(B);
    };

    void f() throw(A);
    void g() throw(B);
  };

  template<typename T>
  struct A2 {
    struct B {
      void f1() throw(A2);
      void f2() throw(A2<T>);
      void g() throw(B);
    };

    void f1() throw(A2);
    void f2() throw(A2<T>);
    void g() throw(B);
  };

  template struct A2<int>;
}

namespace PR12629 {
  struct S {
    static int (f)() throw();
    static int ((((((g))))() throw(U)));
    int (*h)() noexcept(false);
    static int (&i)() noexcept(true);
    static int (*j)() throw(U); // expected-error {{unknown type name 'U'}}
    static int (k)() throw(U);

    struct U {};
  };
  static_assert(noexcept(S::f()), "");
  static_assert(!noexcept(S::g()), "");
  static_assert(!noexcept(S().h()), "");
  static_assert(noexcept(S::i()), "");
}

namespace PR12688 {
  struct S {
    // FIXME: Maybe suppress the "constructor cannot have a return type" error
    // if the return type is invalid.
    nonsense S() throw (more_nonsense); // \
    // expected-error {{'nonsense'}} \
    // expected-error {{constructor cannot have a return type}}
  };
}
