// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -DD2 -I. %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DD2 -I. -fmodules %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DC1 -I. %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DC1 -I. -fmodules %s -fmodules-cache-path=%t -verify

#ifdef D2
#include "d2.h"
void f() { return assert(true); } // expected-error {{undeclared identifier 'b'}}
#else
#include "c1.h"
void f() { return assert(true); } // expected-error {{undeclared identifier 'c'}}
#endif
