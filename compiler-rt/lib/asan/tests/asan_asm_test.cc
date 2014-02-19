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

// Tests for __sanitizer_sanitize_(store|load)N functions in compiler-rt.

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
    "leaq (%[ptr]), %%rdi  \n\t"                       \
    "movabsq $__sanitizer_sanitize_store" Size ", %%r11  \n\t" \
    "call *%%r11  \n\t"                                 \
    Mov " %[val], (%[ptr])  \n\t"                      \
    :                                                  \
    : [ptr] "r" (ptr), [val] Reg (val)                 \
    : "memory", "rdi", "r11"                           \
  );                                                   \
}

#define DECLARE_ASM_READ(Type, Size, Mov, Reg)     \
template<> Type asm_read<Type>(Type *ptr) {        \
  Type res;                                        \
  __asm__(                                         \
    "leaq (%[ptr]), %%rdi  \n\t"                   \
    "movabsq $__sanitizer_sanitize_load" Size ", %%r11  \n\t" \
    "callq *%%r11  \n\t"                           \
    Mov " (%[ptr]), %[res]  \n\t"                  \
    : [res] Reg (res)                              \
    : [ptr] "r" (ptr)                              \
    : "memory", "rdi", "r11"                       \
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
    "leal (%[ptr]), %%eax  \n\t"                       \
    "pushl %%eax  \n\t"                                \
    "call __sanitizer_sanitize_store" Size "  \n\t"    \
    "popl %%eax  \n\t"                                 \
    Mov " %[val], (%[ptr])  \n\t"                      \
    :                                                  \
    : [ptr] "r" (ptr), [val] Reg (val)                 \
    : "memory", "eax", "esp"                           \
  );                                                   \
}

#define DECLARE_ASM_READ(Type, Size, Mov, Reg)     \
template<> Type asm_read<Type>(Type *ptr) {        \
  Type res;                                        \
  __asm__(                                         \
    "leal (%[ptr]), %%eax  \n\t"                   \
    "pushl %%eax  \n\t"                            \
    "call __sanitizer_sanitize_load" Size "  \n\t" \
    "popl %%eax  \n\t"                             \
    Mov " (%[ptr]), %[res]  \n\t"                  \
    : [res] Reg (res)                              \
    : [ptr] "r" (ptr)                              \
    : "memory", "eax", "esp"                       \
  );                                               \
  return res;                                      \
}

template<> void asm_write<U8>(U8 *ptr, U8 val) {
  __asm__(
    "leal (%[ptr]), %%eax  \n\t"
    "pushl %%eax  \n\t"
    "call __sanitizer_sanitize_store8  \n\t"
    "popl %%eax  \n\t"
    "movl (%[val]), %%eax  \n\t"
    "movl %%eax, (%[ptr])  \n\t"
    "movl 0x4(%[val]), %%eax  \n\t"
    "movl %%eax, 0x4(%[ptr])  \n\t"
    :
    : [ptr] "r" (ptr), [val] "r" (&val)
    : "memory", "eax", "esp"
  );
}

template<> U8 asm_read(U8 *ptr) {
  U8 res;
  __asm__(
    "leal (%[ptr]), %%eax  \n\t"
    "pushl %%eax  \n\t"
    "call __sanitizer_sanitize_load8  \n\t"
    "popl  %%eax  \n\t"
    "movl (%[ptr]), %%eax  \n\t"
    "movl %%eax, (%[res])  \n\t"
    "movl 0x4(%[ptr]), %%eax  \n\t"
    "movl %%eax, 0x4(%[res])  \n\t"
    :
    : [ptr] "r" (ptr), [res] "r" (&res)
    : "memory", "eax", "esp"
  );
  return res;
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

} // End of anonymous namespace

TEST(AddressSanitizer, asm_rw) {
  TestAsmWrite<U1>("WRITE of size 1");
  TestAsmWrite<U2>("WRITE of size 2");
  TestAsmWrite<U4>("WRITE of size 4");
  TestAsmWrite<U8>("WRITE of size 8");
  TestAsmWrite<__m128i>("WRITE of size 16");

  TestAsmRead<U1>("READ of size 1");
  TestAsmRead<U2>("READ of size 2");
  TestAsmRead<U4>("READ of size 4");
  TestAsmRead<U8>("READ of size 8");
  TestAsmRead<__m128i>("READ of size 16");
}

#endif // defined(__x86_64__) || (defined(__i386__) && defined(__SSE2__))

#endif // defined(__linux__)
