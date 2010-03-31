// RUN: %clang_cc1 -verify -Wunused-variable -fsyntax-only %s

static void (*fp0)(void) __attribute__((unused));

static void __attribute__((unused)) f0(void);

// On K&R
int f1() __attribute__((unused));

int g0 __attribute__((unused));

int f2() __attribute__((unused(1, 2))); // expected-error {{attribute requires 0 argument(s)}}

struct Test0_unused {} __attribute__((unused));
struct Test0_not_unused {};
typedef int Int_unused __attribute__((unused));
typedef int Int_not_unused;

void test0() {
  int x; // expected-warning {{unused variable}}

  Int_not_unused i0; // expected-warning {{unused variable}}
  Int_unused i1;

  struct Test0_not_unused s0; // expected-warning {{unused variable}}
  struct Test0_unused s1;
}
