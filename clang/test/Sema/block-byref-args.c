// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks

int printf(const char *, ...);

int main(int argc, char **argv) {
  __block void(*bobTheFunction)(void);
  __block void(^bobTheBlock)(void);

  bobTheBlock = ^{;};

  __block int JJJJ;
  __attribute__((__blocks__(byref))) int III;

  int (^XXX)(void) = ^{ return III+JJJJ; };

   // rdar 7671883
   __block char array[10] = {'a', 'b', 'c', 'd'};
   char (^ch)() = ^{ array[1] = 'X'; return array[5]; };
   ch();

  return 0;
}
