// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -fobjc-nonfragile-abi %s
// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -fobjc-nonfragile-abi -fobjc-arc %s

struct X {
  __unsafe_unretained id object;
  int (^ __unsafe_unretained block)(int, int);
};

void f(struct X x) {
  x.object = 0;
  x.block = ^(int x, int y) { return x + y; };
}
