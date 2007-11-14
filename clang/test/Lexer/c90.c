// RUN: clang -std=c90 -fsyntax-only %s -verify

enum { cast_hex = (long) (
      0x0p-1   /* expected-error {{invalid suffix 'p' on integer constant}} */
     ) };
