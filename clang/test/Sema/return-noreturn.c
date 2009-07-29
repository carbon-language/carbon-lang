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
