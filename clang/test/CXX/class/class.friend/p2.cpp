// RUN: clang-cc -fsyntax-only -verify %s

class A {
  friend class B {}; // expected-error {{cannot define a type in a friend declaration}}
  friend int; // expected-error {{friends can only be classes or functions}}
  friend B; // expected-error {{must specify 'class' in a friend class declaration}}
  friend class C; // okay
};
