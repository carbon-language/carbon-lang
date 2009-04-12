// RUN: clang-cc -fsyntax-only -verify -std=c++0x %s

int i = delete; // expected-error {{only functions can have deleted definitions}}

void fn() = delete; // expected-note {{candidate function has been explicitly deleted}}

void fn2(); // expected-note {{previous declaration is here}}
void fn2() = delete; // expected-error {{deleted definition must be first declaration}}

void fn3() = delete;
void fn3() {
  // FIXME: This definition should be invalid.
}

void ov(int) {} // expected-note {{candidate function}}
void ov(double) = delete; // expected-note {{candidate function has been explicitly deleted}}

struct WithDel {
  WithDel() = delete; // expected-note {{candidate function has been explicitly deleted}}
  void fn() = delete; // expected-note {{function has been explicitly marked deleted here}}
  operator int() = delete;
  void operator +(int) = delete;

  int i = delete; // expected-error {{only functions can have deleted definitions}}
};

void test() {
  fn(); // expected-error {{call to deleted function 'fn'}}
  ov(1);
  ov(1.0); // expected-error {{call to deleted function 'ov'}}

  WithDel dd; // expected-error {{call to deleted constructor of 'dd'}}
  WithDel *d = 0;
  d->fn(); // expected-error {{attempt to use a deleted function}}
  int i = *d; // expected-error {{incompatible type initializing}}
}
