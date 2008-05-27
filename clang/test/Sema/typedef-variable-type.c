// RUN: clang %s -verify -fsyntax-only -pedantic

typedef int (*a)[!.0]; // expected-error{{arrays with static storage duration must have constant integer length}}
