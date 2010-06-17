// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-store region -verify %s

char const *p;

void f0() {
  char const str[] = "This will change";
  p = str; // expected-warning {{Stack address was saved into a global variable.}}
}

void f1() {
  char const str[] = "This will change";
  p = str; 
  p = 0; // no-warning
}

void f2() {
  p = (const char *) __builtin_alloca(12); // expected-warning {{Stack address was saved into a global variable.}}
}

// PR 7383 - previosly the stack address checker would crash on this example
//  because it would attempt to do a direct load from 'pr7383_list'. 
static int pr7383(__const char *__)
{
  return 0;
}
extern __const char *__const pr7383_list[];
