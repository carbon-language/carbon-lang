// RUN: %clang_cc1 -fsyntax-only -verify %s

inline void __attribute__((dllexport)) foo1(){} // expected-warning{{dllexport attribute ignored}}
inline void __attribute__((dllimport)) foo2(){} // expected-warning{{dllimport attribute ignored}}

void __attribute__((dllimport)) foo3(){} // expected-error{{dllimport attribute can be applied only to symbol declaration}}

void __attribute__((dllimport, dllexport)) foo4(); // expected-warning{{dllimport attribute ignored}}

void __attribute__((dllexport)) foo5();
void __attribute__((dllimport)) foo5(); // expected-warning{{dllimport attribute ignored}}

typedef int __attribute__((dllexport)) type6; // expected-warning{{'dllexport' attribute only applies to variable and function types}}

typedef int __attribute__((dllimport)) type7; // expected-warning{{'dllimport' attribute only applies to variable and function}}

void __attribute__((dllimport)) foo6();
void foo6(){} // expected-warning {{'foo6' redeclared without dllimport attribute: previous dllimport ignored}}
