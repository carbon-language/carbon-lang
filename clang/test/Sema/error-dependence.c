// RUN: %clang_cc1 -fsyntax-only -verify -frecovery-ast -fno-recovery-ast-type %s

int call(int); // expected-note3 {{'call' declared here}}

void test1(int s) {
  // verify "assigning to 'int' from incompatible type '<dependent type>'" is
  // not emitted.
  s = call(); // expected-error {{too few arguments to function call}}

  // verify diagnostic "operand of type '<dependent type>' where arithmetic or
  // pointer type is required" is not emitted.
  (float)call(); // expected-error {{too few arguments to function call}}
  // verify disgnostic "called object type '<dependent type>' is not a function
  // or function pointer" is not emitted.
  (*__builtin_classify_type)(1); // expected-error {{builtin functions must be directly called}}
}

void test2(int* ptr, float f) {
  // verify diagnostic "used type '<dependent type>' where arithmetic or pointer
  // type is required" is not emitted.
  (call() ? ptr : f); // expected-error {{too few arguments to function call}}
}
