// RUN: %clang_cc1 -triple i686 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify -DAMD64 %s

void I(int i, int j) {
  static const int BelowMin = -1;
  static const int AboveMax = 32;
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "I"(BelowMin)); // expected-error{{value '-1' out of range for constraint 'I'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "I"(AboveMax)); // expected-error{{value '32' out of range for constraint 'I'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "I"(16)); // expected-no-error
}

void J(int i, int j) {
  static const int BelowMin = -1;
  static const int AboveMax = 64;
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "J"(BelowMin)); // expected-error{{value '-1' out of range for constraint 'J'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "J"(AboveMax)); // expected-error{{value '64' out of range for constraint 'J'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "J"(32)); // expected-no-error
}

void K(int i, int j) {
  static const int BelowMin = -129;
  static const int AboveMax = 128;
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "K"(BelowMin)); // expected-error{{value '-129' out of range for constraint 'K'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "K"(AboveMax)); // expected-error{{value '128' out of range for constraint 'K'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "K"(96)); // expected-no-error
}

void L(int i, int j) {
  static const int Invalid1 = 1;
  static const int Invalid2 = 42;
  static const int Invalid3 = 0;
  static const long long Invalid4 = 0x1000000ff;
  static const int Valid1 = 0xff;
  static const int Valid2 = 0xffff;
  static const int Valid3 = 0xffffffff;
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "L"(Invalid1)); // expected-error{{value '1' out of range for constraint 'L'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "L"(Invalid2)); // expected-error{{value '42' out of range for constraint 'L'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "L"(Invalid3)); // expected-error{{value '0' out of range for constraint 'L'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "L"(Invalid4)); // expected-error{{value '4294967551' out of range for constraint 'L'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "L"(Valid1)); // expected-no-error
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "L"(Valid2)); // expected-no-error
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "L"(Valid3)); // expected-no-error
}

void M(int i, int j) {
  static const int BelowMin = -1;
  static const int AboveMax = 4;
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "M"(BelowMin)); // expected-error{{value '-1' out of range for constraint 'M'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "M"(AboveMax)); // expected-error{{value '4' out of range for constraint 'M'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "M"(2)); // expected-no-error
}

void N(int i, int j) {
  static const int BelowMin = -1;
  static const int AboveMax = 256;
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "N"(BelowMin)); // expected-error{{value '-1' out of range for constraint 'N'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "N"(AboveMax)); // expected-error{{value '256' out of range for constraint 'N'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "N"(128)); // expected-no-error
}

void O(int i, int j) {
  static const int BelowMin = -1;
  static const int AboveMax = 128;
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "O"(BelowMin)); // expected-error{{value '-1' out of range for constraint 'O'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "O"(AboveMax)); // expected-error{{value '128' out of range for constraint 'O'}}
  __asm__("xorl %0,%2"
          : "=r"(i)
          : "0"(i), "O"(64)); // expected-no-error
}

void pr40890(void) {
  struct s {
    int a, b;
  };
  static struct s s;
  // This null pointer can be used as an integer constant expression.
  __asm__ __volatile__("\n#define S_A abcd%0\n" : : "n"(&((struct s*)0)->a));
  // This offset-from-null pointer can be used as an integer constant expression.
  __asm__ __volatile__("\n#define S_B abcd%0\n" : : "n"(&((struct s*)0)->b));
#ifdef AMD64
  // This arbitrary pointer is fine.
  __asm__ __volatile__("\n#define BEEF abcd%0\n" : : "n"((int*)0xdeadbeeeeeef));
#endif
}
