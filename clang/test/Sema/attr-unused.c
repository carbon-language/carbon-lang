// RUN: clang-cc -verify -fsyntax-only %s

static void (*fp0)(void) __attribute__((unused));

static void __attribute__((unused)) f0(void);

// On K&R
int f1() __attribute__((unused));

int g0 __attribute__((unused));

int f2() __attribute__((unused(1, 2))); // expected-error {{attribute requires 0 argument(s)}}
