// RUN: %clang_cc1 -verify %s -ftemplate-depth 2

template<int N> struct S { };
// FIXME: We produce the same 'instantiation depth' error here many times
// (2^(depth+1) in total), due to additional lookups performed as part of
// error recovery in DiagnoseTwoPhaseOperatorLookup.
template<typename T> S<T() + T()> operator+(T, T); // expected-error 8{{}} expected-note 10{{}}
S<0> s;
int k = s + s; // expected-error {{invalid operands to binary expression}}
