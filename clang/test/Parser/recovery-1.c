// RUN: clang -fsyntax-only -fno-caret-diagnostics -pedantic %s 2>&1 | grep warning | wc -l | grep 1
// RUN: clang -parse-ast-check %s

char ((((*X x  ]  /* expected-error {{expected ')'}} */
 ))));            /* expected-error {{to match this '('}} */

;   // expected-warning {{ISO C does not allow an extra ';' outside of a function}}
