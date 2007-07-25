// RUN: clang %s -parse-ast-check

typedef union <anonymous> __mbstate_t;  // expected-error: {{expected identifier or}}
