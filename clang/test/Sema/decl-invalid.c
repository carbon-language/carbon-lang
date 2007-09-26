// RUN: clang %s -parse-ast -verify

typedef union <anonymous> __mbstate_t;  // expected-error: {{expected identifier or}}
