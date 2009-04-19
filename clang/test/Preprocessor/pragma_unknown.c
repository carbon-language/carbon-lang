// RUN: clang-cc -E %s | grep '#pragma foo bar' &&
// RUN: clang-cc -fsyntax-only -Wunknown-pragmas -verify %s

// GCC doesn't expand macro args for unrecognized pragmas.
#define bar xX
#pragma foo bar   // expected-warning {{unknown pragma ignored}}

#pragma STDC FP_CONTRACT ON
#pragma STDC FP_CONTRACT OFF
#pragma STDC FP_CONTRACT DEFAULT
#pragma STDC FP_CONTRACT IN_BETWEEN

#pragma STDC FENV_ACCESS ON
#pragma STDC FENV_ACCESS OFF
#pragma STDC FENV_ACCESS DEFAULT
#pragma STDC FENV_ACCESS IN_BETWEEN

#pragma STDC CX_LIMITED_RANGE ON
#pragma STDC CX_LIMITED_RANGE OFF
#pragma STDC CX_LIMITED_RANGE DEFAULT 
#pragma STDC CX_LIMITED_RANGE IN_BETWEEN

#pragma STDC SO_GREAT  // expected-warning {{unknown pragma in STDC namespace}}
#pragma STDC   // expected-warning {{unknown pragma in STDC namespace}}

