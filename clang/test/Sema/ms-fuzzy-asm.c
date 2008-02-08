// RUN: clang %s -verify -fms-extensions

#define M __asm int 0x2c
#define M2 int

void t1(void) { M }
void t2(void) { __asm int 0x2c }
// FIXME? We don't support fuzzy parsing line-oriented __asm's where the body is partially defined in a macro.
void t3(void) { __asm M2 0x2c } // expected-error{{expected ';' after expression}} expected-warning{{expression result unused}}

