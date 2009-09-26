// RUN: clang-cc %s -fsyntax-only -verify -fblocks -Wmissing-noreturn

int j;
void test1() { // expected-warning {{function could be attribute 'noreturn'}}
  ^ (void) { while (1) { } }(); // expected-warning {{block could be attribute 'noreturn'}}
  ^ (void) { if (j) while (1) { } }();
  while (1) { }
}

void test2() {
  if (j) while (1) { }
}

__attribute__((__noreturn__))
void test2_positive() {
  if (j) while (1) { }
} // expected-warning{{function declared 'noreturn' should not return}}


// This test case illustrates that we don't warn about the missing return
// because the function is marked noreturn and there is an infinite loop.
extern int foo_test_3();
__attribute__((__noreturn__)) void* test3(int arg) {
  while (1) foo_test_3();
}

__attribute__((__noreturn__)) void* test3_positive(int arg) {
  while (0) foo_test_3();
} // expected-warning{{function declared 'noreturn' should not return}}
