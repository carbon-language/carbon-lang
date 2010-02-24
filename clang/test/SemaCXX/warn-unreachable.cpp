// RUN: %clang %s -fsyntax-only -Xclang -verify -fblocks -Wunreachable-code -Wno-unused-value

int &halt() __attribute__((noreturn));
int &live();
int dead();
int liveti() throw(int);
int (*livetip)() throw(int);

int test1() {
  try {
    live();
  } catch (int i) {
    live();
  }
  return 1;
}

void test2() {
  try {
    live();
  } catch (int i) {
    live();
  }
  try {
    liveti();
  } catch (int i) {
    live();
  }
  try {
    livetip();
  } catch (int i) {
    live();
  }
  throw 1;
  dead();       // expected-warning {{will never be executed}}
}


void test3() {
  halt()
    --;         // expected-warning {{will never be executed}}
  halt()
    ?           // expected-warning {{will never be executed}}
    dead() : dead();
  live(),
    float       // expected-warning {{will never be executed}}
      (halt());
}

void test4() {
  struct S {
    int mem;
  } s;
  S &foor();
  halt(), foor()// expected-warning {{will never be executed}}
    .mem;       
}

void test5() {
  struct S {
    int mem;
  } s;
  S &foor() __attribute__((noreturn));
  foor()
    .mem;       // expected-warning {{will never be executed}}
}

void test6() {
  struct S {
    ~S() { }
    S(int i) { }
  };
  live(),
    S            // expected-warning {{will never be executed}}
      (halt());
}
