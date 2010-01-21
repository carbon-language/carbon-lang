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
}
