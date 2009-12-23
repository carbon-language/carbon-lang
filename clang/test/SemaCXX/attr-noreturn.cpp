// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5620
void f0() __attribute__((__noreturn__));
void f1(void (*)()); 
void f2() { f1(f0); }

// Taking the address of a noreturn function
void test_f0a() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// Taking the address of an overloaded noreturn function 
void f0(int) __attribute__((__noreturn__));

void test_f0b() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// No-returned function pointers
typedef void (* noreturn_fp)() __attribute__((noreturn));

void f3(noreturn_fp); // expected-note{{candidate function}}

void test_f3() {
  f3(f0); // okay
  f3(f2); // expected-error{{no matching function for call}}
}


class xpto {
  int blah() __attribute__((noreturn));
};

int xpto::blah() {
  return 3; // expected-warning {{function 'blah' declared 'noreturn' should not return}}
}
