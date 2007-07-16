// RUN: clang -parse-ast-check -std=c++0x %s 2>&1
int static_assert; /* expected-error {{expected identifier or '('}}} */
