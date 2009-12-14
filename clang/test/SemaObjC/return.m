// RUN: clang -cc1 %s -fsyntax-only -verify -Wmissing-noreturn

int test1() {
  id a;
  @throw a;
}

// PR5286
void test2(int a) {
  while (1) {
    if (a)
      return;
  }
}

// PR5286
void test3(int a) {  // expected-warning {{function could be attribute 'noreturn'}}
  while (1) {
    if (a)
      @throw (id)0;
  }
}
