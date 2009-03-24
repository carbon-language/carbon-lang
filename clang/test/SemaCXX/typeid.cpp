// RUN: clang-cc -fsyntax-only -verify %s

void f()
{
  (void)typeid(int); // expected-error {{error: you need to include <typeinfo> before using the 'typeid' operator}}
}

// FIXME: This should really include <typeinfo>, but we don't have that yet.
namespace std {
  class type_info;
}

void g()
{
  (void)typeid(int);
}
