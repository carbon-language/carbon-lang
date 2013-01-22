// RUN: %clang -fsyntax-only %s -Xclang -verify -Werror=incompatible-pointer-types -Wno-error=incompatible-pointer-types-discards-qualifiers

// This test ensures that the subgroup of -Wincompatible-pointer-types warnings that
// concern discarding qualifers can be promoted (or not promoted) to an error *separately* from
// the other -Wincompatible-pointer-type warnings.
//
// <rdar://problem/13062738>
//

void foo(char *s); // expected-note {{passing argument to parameter 's' here}}
void baz(int *s); // expected-note {{passing argument to parameter 's' here}}

void bar(const char *s) {
  foo(s); // expected-warning {{passing 'const char *' to parameter of type 'char *' discards qualifiers}}
  baz(s); // expected-error {{incompatible pointer types passing 'const char *' to parameter of type 'int *'}}
}
