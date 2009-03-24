// RUN: clang-cc -fsyntax-only -verify %s

// CC qualifier can be applied only to functions
int __attribute__((stdcall)) var1; // expected-warning{{'stdcall' attribute only applies to function types}}
int __attribute__((fastcall)) var2; // expected-warning{{'fastcall' attribute only applies to function types}}

// Different CC qualifiers are not compatible
void __attribute__((stdcall, fastcall)) foo3(); // expected-error{{stdcall and fastcall attributes are not compatible}}
void __attribute__((stdcall)) foo4();
void __attribute__((fastcall)) foo4(); // expected-error{{fastcall and stdcall attributes are not compatible}}
