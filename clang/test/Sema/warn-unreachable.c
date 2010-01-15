// RUN: %clang %s -fsyntax-only -Xclang -verify -fblocks -Wunreachable-code

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
