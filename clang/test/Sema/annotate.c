// RUN: clang-cc %s -fsyntax-only -verify

void __attribute__((annotate("foo"))) foo(float *a) { 
  __attribute__((annotate("bar"))) int x;
  __attribute__((annotate(1))) int y; // expected-error {{argument to annotate attribute was not a string literal}}
  __attribute__((annotate("bar", 1))) int z; // expected-error {{attribute requires 1 argument(s)}}
}
