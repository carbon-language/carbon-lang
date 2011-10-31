// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify %s -include %s -include %s

// With PCH
// RUN: %clang_cc1 -fsyntax-only -verify %s -chain-include %s -chain-include %s

#ifndef HEADER1
#define HEADER1

@class I;

#elif !defined(HEADER2)
#define HEADER2

@interface I // expected-note {{previous}}
@end

#else

typedef int I; // expected-error {{redefinition}}

#endif
