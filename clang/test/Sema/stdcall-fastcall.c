// RUN: %clang_cc1 -fsyntax-only -verify %s

// CC qualifier can be applied only to functions
int __attribute__((stdcall)) var1; // expected-warning{{'stdcall' only applies to function types; type here is 'int'}}
int __attribute__((fastcall)) var2; // expected-warning{{'fastcall' only applies to function types; type here is 'int'}}

// Different CC qualifiers are not compatible
void __attribute__((stdcall, fastcall)) foo3(void); // expected-error{{stdcall and fastcall attributes are not compatible}}
void __attribute__((stdcall)) foo4(); // expected-note{{previous declaration is here}}
void __attribute__((fastcall)) foo4(void); // expected-error{{function declared 'fastcall' here was previously declared 'stdcall'}}
