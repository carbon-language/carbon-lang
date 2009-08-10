// RUN: clang-cc -verify -fsyntax-only -triple x86_64-apple-darwin9 %s

int x __attribute__((section(
   42)));  // expected-error {{argument to section attribute was not a string literal}}


// rdar://4341926
int y __attribute__((section(
   "sadf"))); // expected-error {{mach-o section specifier requires a segment and section separated by a comma}}

