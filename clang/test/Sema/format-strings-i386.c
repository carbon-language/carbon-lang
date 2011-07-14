// RUN: %clang_cc1 -triple i386-apple-macosx10.7.0 -fsyntax-only -verify -Wformat-nonliteral %s

int printf(const char *restrict, ...);

// Test that 'long' is compatible with 'int' on 32-bit.
typedef unsigned int UInt32;
void test_rdar_9763999() {
 UInt32 x = 7;
 printf("x = %u\n", x); // no-warning
}

void test_positive() {
  printf("%d", "hello"); // expected-warning {{conversion specifies type 'int' but the argument has type 'char *'}}
}

