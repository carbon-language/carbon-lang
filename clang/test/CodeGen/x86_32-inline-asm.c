// RUN: %clang_cc1 -triple i386-apple-darwin9 -verify %s
// <rdar://problem/12415959>

typedef unsigned int u_int32_t;
typedef u_int32_t uint32_t;

typedef unsigned long long u_int64_t;
typedef u_int64_t uint64_t;

int func1() {
  // Error out if size is > 32-bits.
  uint32_t msr = 0x8b;
  uint64_t val = 0;
  __asm__ volatile("wrmsr"
                   :
                   : "c" (msr),
                     "a" ((val & 0xFFFFFFFFUL)), // expected-error {{invalid input size for constraint 'a'}}
                     "d" (((val >> 32) & 0xFFFFFFFFUL)));

  // Don't error out if the size of the destination is <= 32 bits.
  unsigned char data;
  unsigned int port;
  __asm__ volatile("outb %0, %w1" : : "a" (data), "Nd" (port)); // No error expected.
}

struct S;
void func2(struct S *s) {
  __asm__ volatile(""
                   :
                   : "a" (*s));
}
