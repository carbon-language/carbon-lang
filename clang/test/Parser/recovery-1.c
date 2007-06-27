// RUN: clang -parse-ast-check %s

char ((((*X x  ] )))); /* expected-error {{expected ')'}} \
                          expected-error {{to match this '('}} */

;   // expected-warning {{ISO C does not allow an extra ';' outside of a function}}
