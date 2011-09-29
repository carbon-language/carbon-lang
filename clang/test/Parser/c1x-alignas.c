// RUN: %clang_cc1 -std=c1x -fsyntax-only -verify %s

_Alignas(4) char c1;
unsigned _Alignas(long) char c2;
char _Alignas(16) c3;

char c4 _Alignas(32); // expected-error {{expected ';' after top level declarator}}
