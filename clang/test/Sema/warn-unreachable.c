// RUN: %clang %s -fsyntax-only -Xclang -verify -fblocks -Wunreachable-code

int halt() __attribute__((noreturn));
int live();
int dead();

void test1() {
  goto c;
  d:
  goto e;       // expected-warning {{will never be executed}}
  c: ;
  int i;
  return;
  goto b;        // expected-warning {{will never be executed}}
  goto a;        // expected-warning {{will never be executed}}
  b:
  i = 1;
  a:
  i = 2;
  goto f;
  e:
  goto d;
  f: ;
}

void test2() {
  switch (live()) {
  case 1:
    halt(),
      dead();   // expected-warning {{will never be executed}}

  case 2:
    live(),
      halt(),
      dead();   // expected-warning {{will never be executed}}

  case 3:
    live(),
      halt();
    dead();     // expected-warning {{will never be executed}}

  case 4:
  a4:
    live(),
      halt();
    goto a4;    // expected-warning {{will never be executed}}

  case 5:
    goto a5;
  c5:
    dead();     // expected-warning {{will never be executed}}
    goto b5;
  a5:
    live(),
      halt();
  b5:
    goto c5;

  case 6:
    if (live())
      goto e6;
    live(),
      halt();
  d6:
    dead();     // expected-warning {{will never be executed}}
    goto b6;
  c6:
    dead();
    goto b6;
  e6:
    live(),
      halt();
  b6:
    goto c6;
  }
}
