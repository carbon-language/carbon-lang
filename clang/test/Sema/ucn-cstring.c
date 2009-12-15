// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic

int printf(const char *, ...);

int main(void) {
  int a[sizeof("hello \u2192 \u2603 \u2190 world") == 24 ? 1 : -1];
  
  printf("%s (%d)\n", "hello \u2192 \u2603 \u2190 world", sizeof("hello \u2192 \u2603 \u2190 world"));
  printf("%s (%d)\n", "\U00010400\U0001D12B", sizeof("\U00010400\U0001D12B"));
  // Some error conditions...
  printf("%s\n", "\U"); // expected-error{{\u used with no following hex digits}}
  printf("%s\n", "\U00"); // expected-error{{incomplete universal character name}}
  printf("%s\n", "\U0001"); // expected-error{{incomplete universal character name}}
  printf("%s\n", "\u0001"); // expected-error{{invalid universal character}}
  return 0;
}

