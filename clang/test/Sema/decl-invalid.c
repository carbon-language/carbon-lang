// RUN: clang %s -fsyntax-only -verify

typedef union <anonymous> __mbstate_t;  // expected-error: {{expected identifier or}}
