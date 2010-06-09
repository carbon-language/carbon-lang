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
