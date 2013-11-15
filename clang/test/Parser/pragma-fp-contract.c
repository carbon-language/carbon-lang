// RUN: %clang_cc1 -fsyntax-only -verify %s

void f1(void) {
  int x = 0;
/* expected-error@+1 {{'#pragma fp_contract' can only appear at file scope or at the start of a compound statement}} */
#pragma STDC FP_CONTRACT ON
}

void f2(void) {
  #pragma STDC FP_CONTRACT OFF
  #pragma STDC FP_CONTRACT ON 
}
