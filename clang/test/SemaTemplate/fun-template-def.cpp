// RUN: clang-cc -fsyntax-only -verify %s

// Tests that dependent expressions are always allowed, whereas non-dependent
// are checked as usual.

#include <stddef.h>

// Fake typeid, lacking a typeinfo header.
namespace std { class type_info {}; }

struct dummy {};

template <typename T, typename U>
T f(T t1, U u1, int i1)
{
  T t2 = i1;
  t2 = i1 + u1;
  ++u1;
  u1++;
  int i2 = u1;

  i1 = t1[u1];
  i1 *= t1;

  i1(u1, t1); // error
  u1(i1, t1);

  U u2 = (T)i1;
  static_cast<void>(static_cast<U>(reinterpret_cast<T>(
    dynamic_cast<U>(const_cast<T>(i1)))));

  new U(i1, t1);
  new int(t1, u1); // expected-error {{initializer of a builtin type can only take one argument}}
  new (t1, u1) int;
  delete t1;

  dummy d1 = sizeof(t1); // expected-error {{cannot initialize 'd1'}}
  dummy d2 = offsetof(T, foo); // expected-error {{cannot initialize 'd2'}}
  dummy d3 = __alignof(u1); // expected-error {{cannot initialize 'd3'}}
  i1 = typeid(t1); // expected-error {{incompatible type assigning}}

  return u1;
}
