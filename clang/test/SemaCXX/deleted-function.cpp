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

void test() {
  fn(); // expected-error {{call to deleted function 'fn'}}
  ov(1);
  ov(1.0); // expected-error {{call to deleted function 'ov'}}
}
