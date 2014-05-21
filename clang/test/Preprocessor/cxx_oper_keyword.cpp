// RUN: %clang_cc1 %s -E -verify -DOPERATOR_NAMES
// RUN: %clang_cc1 %s -E -verify -fno-operator-names

#ifndef OPERATOR_NAMES
//expected-error@+3 {{token is not a valid binary operator in a preprocessor subexpression}}
#endif
// Valid because 'and' is a spelling of '&&'
#if defined foo and bar
#endif

// Not valid in C++ unless -fno-operator-names is passed:

#ifdef OPERATOR_NAMES
//expected-error@+2 {{C++ operator 'and' (aka '&&') cannot be used as a macro name}}
#endif
#define and foo

#ifdef OPERATOR_NAMES
//expected-error@+2 {{C++ operator 'xor' (aka '^') cannot be used as a macro name}}
#endif
#if defined xor
#endif
