// RUN: clang -fsyntax-only -verify -DAS_SOURCE %s &&
// RUN: env CPATH="" clang -fsyntax-only -verify -DAS_SOURCE %s &&
// RUN: env CPATH="xyz:xyz" clang -fsyntax-only -verify -DAS_SOURCE %s &&
// RUN: cd $(dirname %s) &&
// RUN: env CPATH="xyz::xyz" clang -fsyntax-only -verify -DSHOULD_FIND -DAS_SOURCE %s &&
// RUN: env CPATH="../Driver" clang -fsyntax-only -verify -DSHOULD_FIND -DAS_SOURCE %s

#ifdef AS_SOURCE
#undef AS_SOURCE

#define AS_INCLUDE

#ifdef SHOULD_FIND
#include <env-include-paths.c>
#else
/* expected-error {{file not found}} */ #include <env-include-paths.c> 
#endif

#undef AS_INCLUDE

#endif

#ifdef AS_INCLUDE

/* expected-warning {{Hello}} */ #warning "Hello"

#endif
