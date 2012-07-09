// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -fsyntax-only -verify %s

void foo() {
  // MIPS DSP Rev 1

  int a = 3;
  __builtin_mips_wrdsp(2052, a);  // expected-error{{argument to '__builtin_mips_wrdsp' must be a constant integer}}
  __builtin_mips_rddsp(a);        // expected-error{{argument to '__builtin_mips_rddsp' must be a constant integer}}
  __builtin_mips_wrdsp(2052, -1); // expected-error{{argument should be a value from 0 to 63}}
  __builtin_mips_rddsp(-1);       // expected-error{{argument should be a value from 0 to 63}}
  __builtin_mips_wrdsp(2052, 64); // expected-error{{argument should be a value from 0 to 63}}
  __builtin_mips_rddsp(64);       // expected-error{{argument should be a value from 0 to 63}}
}
