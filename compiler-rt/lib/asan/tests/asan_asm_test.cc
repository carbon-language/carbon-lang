//===-- asan_asm_test.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//
#include "asan_test_utils.h"

#if defined(__linux__)

#if defined(__x86_64__) || (defined(__i386__) && defined(__SSE2__))

#include <emmintrin.h>

namespace {

template<typename T> void asm_write(T *ptr, T val);
template<typename T> T asm_read(T *ptr);

} // End of anonymous namespace

#endif // defined(__x86_64__) || (defined(__i386__) && defined(__SSE2__))

#if defined(__x86_64__)

namespace {

#define DECLARE_ASM_WRITE(Type, Size, Mov, Reg)        \
template<> void asm_write<Type>(Type *ptr, Type val) { \
  __asm__(                                             \
    Mov " %[val], (%[ptr])  \n\t"                      \
    :                                                  \
    : [ptr] "r" (ptr), [val] Reg (val)                 \
    : "memory"                                         \
  );                                                   \
}

#define DECLARE_ASM_READ(Type, Size, Mov, Reg)     \
template<> Type asm_read<Type>(Type *ptr) {        \
  Type res;                                        \
  __asm__(                                         \
    Mov " (%[ptr]), %[res]  \n\t"                  \
    : [res] Reg (res)                              \
    : [ptr] "r" (ptr)                              \
    : "memory"                                     \
  );                                               \
  return res;                                      \
}

DECLARE_ASM_WRITE(U8, "8", "movq", "r");
DECLARE_ASM_READ(U8, "8", "movq", "=r");

} // End of anonymous namespace

#endif // defined(__x86_64__)

#if defined(__i386__) && defined(__SSE2__)

namespace {

#define DECLARE_ASM_WRITE(Type, Size, Mov, Reg)        \
template<> void asm_write<Type>(Type *ptr, Type val) { \
  __asm__(                                             \
    Mov " %[val], (%[ptr])  \n\t"                      \
    :                                                  \
    : [ptr] "r" (ptr), [val] Reg (val)                 \
    : "memory"                                         \
  );                                                   \
}

#define DECLARE_ASM_READ(Type, Size, Mov, Reg)     \
template<> Type asm_read<Type>(Type *ptr) {        \
  Type res;                                        \
  __asm__(                                         \
    Mov " (%[ptr]), %[res]  \n\t"                  \
    : [res] Reg (res)                              \
    : [ptr] "r" (ptr)                              \
    : "memory"                                     \
  );                                               \
  return res;                                      \
}

} // End of anonymous namespace

#endif  // defined(__i386__) && defined(__SSE2__)

#if defined(__x86_64__) || (defined(__i386__) && defined(__SSE2__))

namespace {

DECLARE_ASM_WRITE(U1, "1", "movb", "r");
DECLARE_ASM_WRITE(U2, "2", "movw", "r");
DECLARE_ASM_WRITE(U4, "4", "movl", "r");
DECLARE_ASM_WRITE(__m128i, "16", "movaps", "x");

DECLARE_ASM_READ(U1, "1", "movb", "=r");
DECLARE_ASM_READ(U2, "2", "movw", "=r");
DECLARE_ASM_READ(U4, "4", "movl", "=r");
DECLARE_ASM_READ(__m128i, "16", "movaps", "=x");

template<typename T> void TestAsmWrite(const char *DeathPattern) {
  T *buf = new T;
  EXPECT_DEATH(asm_write(&buf[1], static_cast<T>(0)), DeathPattern);
  T var = 0x12;
  asm_write(&var, static_cast<T>(0x21));
  ASSERT_EQ(static_cast<T>(0x21), var);
  delete buf;
}

template<> void TestAsmWrite<__m128i>(const char *DeathPattern) {
  char *buf = new char[16];
  char *p = buf + 16;
  if (((uintptr_t) p % 16) != 0)
    p = buf + 8;
  assert(((uintptr_t) p % 16) == 0);
  __m128i val = _mm_set1_epi16(0x1234);
  EXPECT_DEATH(asm_write<__m128i>((__m128i*) p, val), DeathPattern);
  __m128i var = _mm_set1_epi16(0x4321);
  asm_write(&var, val);
  ASSERT_EQ(0x1234, _mm_extract_epi16(var, 0));
  delete [] buf;
}

template<typename T> void TestAsmRead(const char *DeathPattern) {
  T *buf = new T;
  EXPECT_DEATH(asm_read(&buf[1]), DeathPattern);
  T var = 0x12;
  ASSERT_EQ(static_cast<T>(0x12), asm_read(&var));
  delete buf;
}

template<> void TestAsmRead<__m128i>(const char *DeathPattern) {
  char *buf = new char[16];
  char *p = buf + 16;
  if (((uintptr_t) p % 16) != 0)
    p = buf + 8;
  assert(((uintptr_t) p % 16) == 0);
  EXPECT_DEATH(asm_read<__m128i>((__m128i*) p), DeathPattern);
  __m128i val = _mm_set1_epi16(0x1234);
  ASSERT_EQ(0x1234, _mm_extract_epi16(asm_read(&val), 0));
  delete [] buf;
}

U4 AsmLoad(U4 *a) {
  U4 r;
  __asm__("movl (%[a]), %[r]  \n\t" : [r] "=r" (r) : [a] "r" (a) : "memory");
  return r;
}

void AsmStore(U4 r, U4 *a) {
  __asm__("movl %[r], (%[a])  \n\t" : : [a] "r" (a), [r] "r" (r) : "memory");
}

} // End of anonymous namespace

TEST(AddressSanitizer, asm_load_store) {
  U4* buf = new U4[2];
  EXPECT_DEATH(AsmLoad(&buf[3]), "READ of size 4");
  EXPECT_DEATH(AsmStore(0x1234, &buf[3]), "WRITE of size 4");
  delete [] buf;
}

TEST(AddressSanitizer, asm_rw) {
  TestAsmWrite<U1>("WRITE of size 1");
  TestAsmWrite<U2>("WRITE of size 2");
  TestAsmWrite<U4>("WRITE of size 4");
#if defined(__x86_64__)
  TestAsmWrite<U8>("WRITE of size 8");
#endif // defined(__x86_64__)
  TestAsmWrite<__m128i>("WRITE of size 16");

  TestAsmRead<U1>("READ of size 1");
  TestAsmRead<U2>("READ of size 2");
  TestAsmRead<U4>("READ of size 4");
#if defined(__x86_64__)
  TestAsmRead<U8>("READ of size 8");
#endif // defined(__x86_64__)
  TestAsmRead<__m128i>("READ of size 16");
}

TEST(AddressSanitizer, asm_flags) {
  long magic = 0x1234;
  long r = 0x0;

#if defined(__x86_64__)
  __asm__("xorq %%rax, %%rax  \n\t"
          "movq (%[p]), %%rax \n\t"
          "sete %%al          \n\t"
          "movzbq %%al, %[r]  \n\t"
          : [r] "=r"(r)
          : [p] "r"(&magic)
          : "rax", "memory");
#else
  __asm__("xorl %%eax, %%eax  \n\t"
          "movl (%[p]), %%eax \n\t"
          "sete %%al          \n\t"
          "movzbl %%al, %[r]  \n\t"
          : [r] "=r"(r)
          : [p] "r"(&magic)
          : "eax", "memory");
#endif // defined(__x86_64__)

  ASSERT_EQ(0x1, r);
}

#endif // defined(__x86_64__) || (defined(__i386__) && defined(__SSE2__))

#endif // defined(__linux__)
