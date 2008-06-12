// RUN: clang -fsyntax-only -fno-caret-diagnostics -pedantic %s 2>&1 | grep warning | wc -l | grep 1 &&
// RUN: clang -fsyntax-only -verify -pedantic %s

char ((((                       /* expected-error {{to match this '('}} */
*X x ] ))));                    /* expected-error {{expected ')'}} */

;   // expected-warning {{ISO C does not allow an extra ';' outside of a function}}




struct S { void *X, *Y; };

struct S A = {
	&BADIDENT, 0     /* expected-error {{use of undeclared identifier}} */
};
