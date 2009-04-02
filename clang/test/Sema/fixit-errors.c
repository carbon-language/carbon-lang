// RUN: clang-cc -fsyntax-only -pedantic -fixit %s -o - | clang-cc -pedantic -Werror -x c -

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. Eventually,
   we would like to actually try to perform the suggested
   modifications and compile the result to test that no warnings
   remain. */
struct s; // expected-note{{previous use is here}}

union s *s1; // expected-error{{use of 's' with tag type that does not match previous declaration}}
