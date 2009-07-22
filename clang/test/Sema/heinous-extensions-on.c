// RUN: clang-cc %s -verify -fheinous-gnu-extensions

void foo() {
  int a;
  // PR3788
  asm("nop" : : "m"((int)(a))); // expected-warning {{cast in a inline asm context requiring an l-value}}
  // PR3794
  asm("nop" : "=r"((unsigned)a)); // expected-warning {{cast in a inline asm context requiring an l-value}}
}
