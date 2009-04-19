// RUN: clang-cc -E %s | grep '#pragma foo bar' &&
// RUN: clang-cc -fsyntax-only -Wunknown-pragmas %s

// GCC doesn't expand macro args for unrecognized pragmas.
#define bar xX
#pragma foo bar   // expected-warning {{unknown pragma ignored}}

