// RUN: clang -verify -fsyntax-only %s

static void (*fp0)(void) __attribute__((noreturn));

static void __attribute__((noreturn)) f0(void) {
  fatal();
}

// On K&R
int f1() __attribute__((noreturn));

int g0 __attribute__((noreturn)); // expected-warning {{'noreturn' attribute only applies to function types}}

int f2() __attribute__((noreturn(1, 2))); // expected-error {{attribute requires 0 argument(s)}}
