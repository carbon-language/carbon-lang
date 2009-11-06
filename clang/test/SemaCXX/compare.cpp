// Force x86-64 because some of our heuristics are actually based
// on integer sizes.

// RUN: clang-cc -triple x86_64-apple-darwin -fsyntax-only -pedantic -verify -Wsign-compare %s

int test0(long a, unsigned long b) {
  enum EnumA {A};
  enum EnumB {B};
  enum EnumC {C = 0x10000};
  return
         // (a,b)
         (a == (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         (a == (unsigned int) b) +
         (a == (unsigned short) b) +
         (a == (unsigned char) b) +
         ((long) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a == (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a == (unsigned int) b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a == (unsigned short) b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a == (unsigned char) b) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) b) +
         (a < (unsigned short) b) +
         (a < (unsigned char) b) +
         ((long) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a < (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < (unsigned char) b) +  // expected-warning {{comparison of integers of different signs}}

         // (A,b)
         (A == (unsigned long) b) +
         (A == (unsigned int) b) +
         (A == (unsigned short) b) +
         (A == (unsigned char) b) +
         ((long) A == b) +
         ((int) A == b) +
         ((short) A == b) +
         ((signed char) A == b) +
         ((long) A == (unsigned long) b) +
         ((int) A == (unsigned int) b) +
         ((short) A == (unsigned short) b) +
         ((signed char) A == (unsigned char) b) +
         (A < (unsigned long) b) +
         (A < (unsigned int) b) +
         (A < (unsigned short) b) +  // expected-warning {{comparison of integers of different signs}}
         (A < (unsigned char) b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) A < b) +
         ((int) A < b) +
         ((short) A < b) +
         ((signed char) A < b) +
         ((long) A < (unsigned long) b) +
         ((int) A < (unsigned int) b) +
         ((short) A < (unsigned short) b) +
         ((signed char) A < (unsigned char) b) +

         // (a,B)
         (a == (unsigned long) B) +
         (a == (unsigned int) B) +
         (a == (unsigned short) B) +
         (a == (unsigned char) B) +
         ((long) a == B) +
         ((int) a == B) +
         ((short) a == B) +
         ((signed char) a == B) +
         ((long) a == (unsigned long) B) +
         ((int) a == (unsigned int) B) +
         ((short) a == (unsigned short) B) +
         ((signed char) a == (unsigned char) B) +
         (a < (unsigned long) B) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) B) +
         (a < (unsigned short) B) +
         (a < (unsigned char) B) +
         ((long) a < B) +
         ((int) a < B) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < B) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < B) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a < (unsigned long) B) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) B) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) B) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < (unsigned char) B) +  // expected-warning {{comparison of integers of different signs}}

         // (C,b)
         (C == (unsigned long) b) +
         (C == (unsigned int) b) +
         (C == (unsigned short) b) +
         (C == (unsigned char) b) +
         ((long) C == b) +
         ((int) C == b) +
         ((short) C == b) +
         ((signed char) C == b) +
         ((long) C == (unsigned long) b) +
         ((int) C == (unsigned int) b) +
         ((short) C == (unsigned short) b) +
         ((signed char) C == (unsigned char) b) +
         (C < (unsigned long) b) +
         (C < (unsigned int) b) +
         (C < (unsigned short) b) +  // expected-warning {{comparison of integers of different signs}}
         (C < (unsigned char) b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) C < b) +
         ((int) C < b) +
         ((short) C < b) +
         ((signed char) C < b) +
         ((long) C < (unsigned long) b) +
         ((int) C < (unsigned int) b) +
         ((short) C < (unsigned short) b) +
         ((signed char) C < (unsigned char) b) +

         // (a,C)
         (a == (unsigned long) C) +
         (a == (unsigned int) C) +
         (a == (unsigned short) C) +
         (a == (unsigned char) C) +
         ((long) a == C) +
         ((int) a == C) +
         ((short) a == C) +
         ((signed char) a == C) +
         ((long) a == (unsigned long) C) +
         ((int) a == (unsigned int) C) +
         ((short) a == (unsigned short) C) +
         ((signed char) a == (unsigned char) C) +
         (a < (unsigned long) C) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) C) +
         (a < (unsigned short) C) +
         (a < (unsigned char) C) +
         ((long) a < C) +
         ((int) a < C) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < C) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < C) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a < (unsigned long) C) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) C) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) C) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < (unsigned char) C) +  // expected-warning {{comparison of integers of different signs}}

         // (0x80000,b)
         (0x80000 == (unsigned long) b) +
         (0x80000 == (unsigned int) b) +
         (0x80000 == (unsigned short) b) +
         (0x80000 == (unsigned char) b) +
         ((long) 0x80000 == b) +
         ((int) 0x80000 == b) +
         ((short) 0x80000 == b) +
         ((signed char) 0x80000 == b) +
         ((long) 0x80000 == (unsigned long) b) +
         ((int) 0x80000 == (unsigned int) b) +
         ((short) 0x80000 == (unsigned short) b) +
         ((signed char) 0x80000 == (unsigned char) b) +
         (0x80000 < (unsigned long) b) +
         (0x80000 < (unsigned int) b) +
         (0x80000 < (unsigned short) b) +
         (0x80000 < (unsigned char) b) +
         ((long) 0x80000 < b) +
         ((int) 0x80000 < b) +
         ((short) 0x80000 < b) +
         ((signed char) 0x80000 < b) +
         ((long) 0x80000 < (unsigned long) b) +
         ((int) 0x80000 < (unsigned int) b) +
         ((short) 0x80000 < (unsigned short) b) +
         ((signed char) 0x80000 < (unsigned char) b) +

         // (a,0x80000)
         (a == (unsigned long) 0x80000) +
         (a == (unsigned int) 0x80000) +
         (a == (unsigned short) 0x80000) +
         (a == (unsigned char) 0x80000) +
         ((long) a == 0x80000) +
         ((int) a == 0x80000) +
         ((short) a == 0x80000) +
         ((signed char) a == 0x80000) +
         ((long) a == (unsigned long) 0x80000) +
         ((int) a == (unsigned int) 0x80000) +
         ((short) a == (unsigned short) 0x80000) +
         ((signed char) a == (unsigned char) 0x80000) +
         (a < (unsigned long) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) 0x80000) +
         (a < (unsigned short) 0x80000) +
         (a < (unsigned char) 0x80000) +
         ((long) a < 0x80000) +
         ((int) a < 0x80000) +
         ((short) a < 0x80000) +
         ((signed char) a < 0x80000) +
         ((long) a < (unsigned long) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < (unsigned char) 0x80000) +  // expected-warning {{comparison of integers of different signs}}

         10
    ;
}
