// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

enum X : short { A, B };
extern decltype(+A) x;
extern int x;

enum Y : long { C, D };
extern decltype(+C) y;
extern long y;

// An enum with a fixed underlying type has an integral promotion to that type,
// and to its promoted type.
enum B : bool { false_, true_ };
template<bool> struct T {};
T<false_> f;
T<true_> t;
T<+true_> t; // expected-error {{conversion from 'int' to 'bool'}}

enum B2 : bool {
  a = false,
  b = true,
  c = false_,
  d = true_,
  e = +false_ // expected-error {{conversion from 'int' to 'bool'}} \
              // FIXME: expected-error {{enumerator value 2 is not representable}}
};
