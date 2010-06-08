// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-store region -analyzer-experimental-internal-checks -verify %s

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
