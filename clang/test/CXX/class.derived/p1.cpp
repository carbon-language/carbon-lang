// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++11

// base-clause:
//         : base-specifier-list
// base-specifier-list:
//         base-specifier ...[opt]
//         base-specifier-list , base-specifier ...[opt]
// base-specifier:
//         attribute-specifier-seq[opt] base-type-specifier
//         attribute-specifier-seq[opt] virtual access-specifier[opt] base-type-specifier
//         attribute-specifier-seq[opt] access-specifier virtual[opt] base-type-specifier
// class-or-decltype:
//         nested-name-specifier[opt] class-name
//         decltype-specifier
// base-type-specifier:
//         class-or-decltype
// access-specifier:
//         private
//         protected
//         public

namespace PR11216 {
  struct Base { };
  struct Derived : decltype(Base()) { };

  int func();
  struct Derived2 : decltype(func()) { }; // expected-error {{base specifier must name a class}}

  template<typename T>
  struct Derived3 : decltype(T().foo()) { };
  struct Foo { Base foo(); };
  Derived3<Foo> d;

  struct Derived4 : :: decltype(Base()) { }; // expected-error {{expected class name}}

  struct Derived5 : PR11216:: decltype(Base()) { }; // expected-error {{expected class name}}
}
