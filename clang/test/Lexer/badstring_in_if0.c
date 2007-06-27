// RUN: clang -parse-ast-check %s
#if 0

  "

  '

#endif
/* expected-warning {{ISO C forbids an empty source file}} */
