// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks

int printf(const char *, ...);

int main(int argc, char **argv) {
  __block void(*bobTheFunction)(void);
  __block void(^bobTheBlock)(void);

  bobTheBlock = ^{;};

  __block int JJJJ;
  __attribute__((__blocks__(byref))) int III;

  int (^XXX)(void) = ^{ return III+JJJJ; };

  return 0;
}

