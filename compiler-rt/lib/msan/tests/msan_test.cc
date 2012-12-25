//===-- msan_test.cc ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// MemorySanitizer unit tests.
//===----------------------------------------------------------------------===//

#include "sanitizer/msan_interface.h"
#include "msandr_test_so.h"
#include "gtest/gtest.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <assert.h>
#include <wchar.h>

#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/ioctl.h>
#include <sys/utsname.h>
#include <sys/mman.h>
#include <sys/vfs.h>

#if defined(__i386__) || defined(__x86_64__)
# include <emmintrin.h>
# define MSAN_HAS_M128 1
#else
# define MSAN_HAS_M128 0
#endif

typedef unsigned char      U1;
typedef unsigned short     U2;  // NOLINT
typedef unsigned int       U4;
typedef unsigned long long U8;  // NOLINT
typedef   signed char      S1;
typedef   signed short     S2;  // NOLINT
typedef   signed int       S4;
typedef   signed long long S8;  // NOLINT
#define NOINLINE      __attribute__((noinline))
#define INLINE      __attribute__((always_inline))


#define EXPECT_POISONED(action) \
    do {                        \
      __msan_set_expect_umr(1); \
      action;                   \
      __msan_set_expect_umr(0); \
    } while (0)

#define EXPECT_POISONED_O(action, origin) \
    do {                                            \
      __msan_set_expect_umr(1);                     \
      action;                                       \
      __msan_set_expect_umr(0);                     \
      if (TrackingOrigins())                        \
        EXPECT_EQ(origin, __msan_get_origin_tls()); \
    } while (0)

#define EXPECT_POISONED_S(action, stack_origin) \
    do {                                            \
      __msan_set_expect_umr(1);                     \
      action;                                       \
      __msan_set_expect_umr(0);                     \
      u32 id = __msan_get_origin_tls();             \
      const char *str = __msan_get_origin_descr_if_stack(id); \
      if (!str || strcmp(str, stack_origin)) {      \
        fprintf(stderr, "EXPECT_POISONED_S: id=%u %s, %s", \
                id, stack_origin, str);  \
        EXPECT_EQ(1, 0);                            \
      }                                             \
    } while (0)


static U8 poisoned_array[100];
template<class T>
T *GetPoisoned(int i = 0, T val = 0) {
  T *res = (T*)&poisoned_array[i];
  *res = val;
  __msan_poison(&poisoned_array[i], sizeof(T));
  return res;
}

template<class T>
T *GetPoisonedO(int i, u32 origin, T val = 0) {
  T *res = (T*)&poisoned_array[i];
  *res = val;
  __msan_poison(&poisoned_array[i], sizeof(T));
  __msan_set_origin(&poisoned_array[i], sizeof(T), origin);
  return res;
}

// This function returns its parameter but in such a way that compiler
// can not prove it.
template<class T>
NOINLINE
static T Ident(T t) {
  volatile T ret = t;
  return ret;
}

static bool TrackingOrigins() {
  S8 x;
  __msan_set_origin(&x, sizeof(x), 0x1234);
  u32 origin = __msan_get_origin(&x);
  __msan_set_origin(&x, sizeof(x), 0);
  return origin == 0x1234;
}

template<class T> NOINLINE T ReturnPoisoned() { return *GetPoisoned<T>(); }

static volatile S1 v_s1;
static volatile S2 v_s2;
static volatile S4 v_s4;
static volatile S8 v_s8;
static volatile U1 v_u1;
static volatile U2 v_u2;
static volatile U4 v_u4;
static volatile U8 v_u8;
static void* volatile v_p;
static volatile double v_d;
static volatile int g_one = 1;
static volatile int g_zero = 0;
static volatile int g_0 = 0;
static volatile int g_1 = 1;

#if MSAN_HAS_M128
static volatile __m128i v_m128;
#endif

S4 a_s4[100];
S8 a_s8[100];

TEST(MemorySanitizer, NegativeTest1) {
  S4 *x = GetPoisoned<S4>();
  if (g_one)
    *x = 0;
  v_s4 = *x;
}

TEST(MemorySanitizer, PositiveTest1) {
  // Load to store.
  EXPECT_POISONED(v_s1 = *GetPoisoned<S1>());
  EXPECT_POISONED(v_s2 = *GetPoisoned<S2>());
  EXPECT_POISONED(v_s4 = *GetPoisoned<S4>());
  EXPECT_POISONED(v_s8 = *GetPoisoned<S8>());

  // S->S conversions.
  EXPECT_POISONED(v_s2 = *GetPoisoned<S1>());
  EXPECT_POISONED(v_s4 = *GetPoisoned<S1>());
  EXPECT_POISONED(v_s8 = *GetPoisoned<S1>());

  EXPECT_POISONED(v_s1 = *GetPoisoned<S2>());
  EXPECT_POISONED(v_s4 = *GetPoisoned<S2>());
  EXPECT_POISONED(v_s8 = *GetPoisoned<S2>());

  EXPECT_POISONED(v_s1 = *GetPoisoned<S4>());
  EXPECT_POISONED(v_s2 = *GetPoisoned<S4>());
  EXPECT_POISONED(v_s8 = *GetPoisoned<S4>());

  EXPECT_POISONED(v_s1 = *GetPoisoned<S8>());
  EXPECT_POISONED(v_s2 = *GetPoisoned<S8>());
  EXPECT_POISONED(v_s4 = *GetPoisoned<S8>());

  // ZExt
  EXPECT_POISONED(v_s2 = *GetPoisoned<U1>());
  EXPECT_POISONED(v_s4 = *GetPoisoned<U1>());
  EXPECT_POISONED(v_s8 = *GetPoisoned<U1>());
  EXPECT_POISONED(v_s4 = *GetPoisoned<U2>());
  EXPECT_POISONED(v_s8 = *GetPoisoned<U2>());
  EXPECT_POISONED(v_s8 = *GetPoisoned<U4>());

  // Unary ops.
  EXPECT_POISONED(v_s4 = - *GetPoisoned<S4>());

  EXPECT_POISONED(a_s4[g_zero] = 100 / *GetPoisoned<S4>(0, 1));


  a_s4[g_zero] = 1 - *GetPoisoned<S4>();
  a_s4[g_zero] = 1 + *GetPoisoned<S4>();
}

TEST(MemorySanitizer, Phi1) {
  S4 c;
  if (g_one) {
    c = *GetPoisoned<S4>();
  } else {
    __msan_break_optimization(0);
    c = 0;
  }
  EXPECT_POISONED(v_s4 = c);
}

TEST(MemorySanitizer, Phi2) {
  S4 i = *GetPoisoned<S4>();
  S4 n = g_one;
  EXPECT_POISONED(for (; i < g_one; i++););
  EXPECT_POISONED(v_s4 = i);
}

NOINLINE void Arg1ExpectUMR(S4 a1) { EXPECT_POISONED(v_s4 = a1); }
NOINLINE void Arg2ExpectUMR(S4 a1, S4 a2) { EXPECT_POISONED(v_s4 = a2); }
NOINLINE void Arg3ExpectUMR(S1 a1, S4 a2, S8 a3) { EXPECT_POISONED(v_s8 = a3); }

TEST(MemorySanitizer, ArgTest) {
  Arg1ExpectUMR(*GetPoisoned<S4>());
  Arg2ExpectUMR(0, *GetPoisoned<S4>());
  Arg3ExpectUMR(0, 1, *GetPoisoned<S8>());
}


TEST(MemorySanitizer, CallAndRet) {
  if (!__msan_has_dynamic_component()) return;
  ReturnPoisoned<S1>();
  ReturnPoisoned<S2>();
  ReturnPoisoned<S4>();
  ReturnPoisoned<S8>();

  EXPECT_POISONED(v_s1 = ReturnPoisoned<S1>());
  EXPECT_POISONED(v_s2 = ReturnPoisoned<S2>());
  EXPECT_POISONED(v_s4 = ReturnPoisoned<S4>());
  EXPECT_POISONED(v_s8 = ReturnPoisoned<S8>());
}

// malloc() in the following test may be optimized to produce a compile-time
// undef value. Check that we trap on the volatile assignment anyway.
TEST(MemorySanitizer, DISABLED_MallocNoIdent) {
  S4 *x = (int*)malloc(sizeof(S4));
  EXPECT_POISONED(v_s4 = *x);
  free(x);
}

TEST(MemorySanitizer, Malloc) {
  S4 *x = (int*)Ident(malloc(sizeof(S4)));
  EXPECT_POISONED(v_s4 = *x);
  free(x);
}

TEST(MemorySanitizer, Realloc) {
  S4 *x = (int*)Ident(realloc(0, sizeof(S4)));
  EXPECT_POISONED(v_s4 = x[0]);
  x[0] = 1;
  x = (int*)Ident(realloc(x, 2 * sizeof(S4)));
  v_s4 = x[0];  // Ok, was inited before.
  EXPECT_POISONED(v_s4 = x[1]);
  x = (int*)Ident(realloc(x, 3 * sizeof(S4)));
  v_s4 = x[0];  // Ok, was inited before.
  EXPECT_POISONED(v_s4 = x[2]);
  EXPECT_POISONED(v_s4 = x[1]);
  x[2] = 1;  // Init this here. Check that after realloc it is poisoned again.
  x = (int*)Ident(realloc(x, 2 * sizeof(S4)));
  v_s4 = x[0];  // Ok, was inited before.
  EXPECT_POISONED(v_s4 = x[1]);
  x = (int*)Ident(realloc(x, 3 * sizeof(S4)));
  EXPECT_POISONED(v_s4 = x[1]);
  EXPECT_POISONED(v_s4 = x[2]);
  free(x);
}

TEST(MemorySanitizer, Calloc) {
  S4 *x = (int*)Ident(calloc(1, sizeof(S4)));
  v_s4 = *x;  // Should not be poisoned.
  // EXPECT_EQ(0, *x);
  free(x);
}

TEST(MemorySanitizer, AndOr) {
  U4 *p = GetPoisoned<U4>();
  // We poison two bytes in the midle of a 4-byte word to make the test
  // correct regardless of endianness.
  ((U1*)p)[1] = 0;
  ((U1*)p)[2] = 0xff;
  v_u4 = *p & 0x00ffff00;
  v_u4 = *p & 0x00ff0000;
  v_u4 = *p & 0x0000ff00;
  EXPECT_POISONED(v_u4 = *p & 0xff000000);
  EXPECT_POISONED(v_u4 = *p & 0x000000ff);
  EXPECT_POISONED(v_u4 = *p & 0x0000ffff);
  EXPECT_POISONED(v_u4 = *p & 0xffff0000);

  v_u4 = *p | 0xff0000ff;
  v_u4 = *p | 0xff00ffff;
  v_u4 = *p | 0xffff00ff;
  EXPECT_POISONED(v_u4 = *p | 0xff000000);
  EXPECT_POISONED(v_u4 = *p | 0x000000ff);
  EXPECT_POISONED(v_u4 = *p | 0x0000ffff);
  EXPECT_POISONED(v_u4 = *p | 0xffff0000);

  EXPECT_POISONED(v_u4 = *GetPoisoned<bool>() & *GetPoisoned<bool>());
}

template<class T>
static void testNot(T value, T shadow) {
  __msan_partial_poison(&value, &shadow, sizeof(T));
  volatile bool v_T = !value;
}

TEST(MemorySanitizer, Not) {
  testNot<U4>(0x0, 0x0);
  testNot<U4>(0xFFFFFFFF, 0x0);
  EXPECT_POISONED(testNot<U4>(0xFFFFFFFF, 0xFFFFFFFF));
  testNot<U4>(0xFF000000, 0x0FFFFFFF);
  testNot<U4>(0xFF000000, 0x00FFFFFF);
  testNot<U4>(0xFF000000, 0x0000FFFF);
  testNot<U4>(0xFF000000, 0x00000000);
  EXPECT_POISONED(testNot<U4>(0xFF000000, 0xFF000000));
  testNot<U4>(0xFF800000, 0xFF000000);
  EXPECT_POISONED(testNot<U4>(0x00008000, 0x00008000));

  testNot<U1>(0x0, 0x0);
  testNot<U1>(0xFF, 0xFE);
  testNot<U1>(0xFF, 0x0);
  EXPECT_POISONED(testNot<U1>(0xFF, 0xFF));

  EXPECT_POISONED(testNot<void*>((void*)0xFFFFFF, (void*)(-1)));
  testNot<void*>((void*)0xFFFFFF, (void*)(-2));
}

TEST(MemorySanitizer, Shift) {
  U4 *up = GetPoisoned<U4>();
  ((U1*)up)[0] = 0;
  ((U1*)up)[3] = 0xff;
  v_u4 = *up >> 30;
  v_u4 = *up >> 24;
  EXPECT_POISONED(v_u4 = *up >> 23);
  EXPECT_POISONED(v_u4 = *up >> 10);

  v_u4 = *up << 30;
  v_u4 = *up << 24;
  EXPECT_POISONED(v_u4 = *up << 23);
  EXPECT_POISONED(v_u4 = *up << 10);

  S4 *sp = (S4*)up;
  v_s4 = *sp >> 30;
  v_s4 = *sp >> 24;
  EXPECT_POISONED(v_s4 = *sp >> 23);
  EXPECT_POISONED(v_s4 = *sp >> 10);

  sp = GetPoisoned<S4>();
  ((S1*)sp)[1] = 0;
  ((S1*)sp)[2] = 0;
  EXPECT_POISONED(v_s4 = *sp >> 31);

  v_s4 = 100;
  EXPECT_POISONED(v_s4 = v_s4 >> *GetPoisoned<S4>());
  v_u4 = 100;
  EXPECT_POISONED(v_u4 = v_u4 >> *GetPoisoned<S4>());
  v_u4 = 100;
  EXPECT_POISONED(v_u4 = v_u4 << *GetPoisoned<S4>());
}

NOINLINE static int GetPoisonedZero() {
  int *zero = new int;
  *zero = 0;
  __msan_poison(zero, sizeof(*zero));
  int res = *zero;
  delete zero;
  return res;
}

TEST(MemorySanitizer, LoadFromDirtyAddress) {
  int *a = new int;
  *a = 0;
  EXPECT_POISONED(__msan_break_optimization((void*)(U8)a[GetPoisonedZero()]));
  delete a;
}

TEST(MemorySanitizer, StoreToDirtyAddress) {
  int *a = new int;
  EXPECT_POISONED(a[GetPoisonedZero()] = 0);
  __msan_break_optimization(a);
  delete a;
}


NOINLINE void StackTestFunc() {
  S4 p4;
  S4 ok4 = 1;
  S2 p2;
  S2 ok2 = 1;
  S1 p1;
  S1 ok1 = 1;
  __msan_break_optimization(&p4);
  __msan_break_optimization(&ok4);
  __msan_break_optimization(&p2);
  __msan_break_optimization(&ok2);
  __msan_break_optimization(&p1);
  __msan_break_optimization(&ok1);

  EXPECT_POISONED(v_s4 = p4);
  EXPECT_POISONED(v_s2 = p2);
  EXPECT_POISONED(v_s1 = p1);
  v_s1 = ok1;
  v_s2 = ok2;
  v_s4 = ok4;
}

TEST(MemorySanitizer, StackTest) {
  StackTestFunc();
}

NOINLINE void StackStressFunc() {
  int foo[10000];
  __msan_break_optimization(foo);
}

TEST(MemorySanitizer, DISABLED_StackStressTest) {
  for (int i = 0; i < 1000000; i++)
    StackStressFunc();
}

template<class T>
void TestFloatingPoint() {
  static volatile T v;
  static T g[100];
  __msan_break_optimization(&g);
  T *x = GetPoisoned<T>();
  T *y = GetPoisoned<T>(1);
  EXPECT_POISONED(v = *x);
  EXPECT_POISONED(v_s8 = *x);
  EXPECT_POISONED(v_s4 = *x);
  g[0] = *x;
  g[1] = *x + *y;
  g[2] = *x - *y;
  g[3] = *x * *y;
}

TEST(MemorySanitizer, FloatingPointTest) {
  TestFloatingPoint<float>();
  TestFloatingPoint<double>();
}

TEST(MemorySanitizer, DynMem) {
  S4 x = 0;
  S4 *y = GetPoisoned<S4>();
  memcpy(y, &x, g_one * sizeof(S4));
  v_s4 = *y;
}

static char *DynRetTestStr;

TEST(MemorySanitizer, DynRet) {
  if (!__msan_has_dynamic_component()) return;
  ReturnPoisoned<S8>();
  v_s4 = clearenv();
}


TEST(MemorySanitizer, DynRet1) {
  if (!__msan_has_dynamic_component()) return;
  ReturnPoisoned<S8>();
}

struct LargeStruct {
  S4 x[10];
};

NOINLINE
LargeStruct LargeRetTest() {
  LargeStruct res;
  res.x[0] = *GetPoisoned<S4>();
  res.x[1] = *GetPoisoned<S4>();
  res.x[2] = *GetPoisoned<S4>();
  res.x[3] = *GetPoisoned<S4>();
  res.x[4] = *GetPoisoned<S4>();
  res.x[5] = *GetPoisoned<S4>();
  res.x[6] = *GetPoisoned<S4>();
  res.x[7] = *GetPoisoned<S4>();
  res.x[8] = *GetPoisoned<S4>();
  res.x[9] = *GetPoisoned<S4>();
  return res;
}

TEST(MemorySanitizer, LargeRet) {
  LargeStruct a = LargeRetTest();
  EXPECT_POISONED(v_s4 = a.x[0]);
  EXPECT_POISONED(v_s4 = a.x[9]);
}

TEST(MemorySanitizer, fread) {
  char *x = new char[32];
  FILE *f = fopen("/proc/self/stat", "r");
  assert(f);
  fread(x, 1, 32, f);
  v_s1 = x[0];
  v_s1 = x[16];
  v_s1 = x[31];
  fclose(f);
  delete x;
}

TEST(MemorySanitizer, read) {
  char *x = new char[32];
  int fd = open("/proc/self/stat", O_RDONLY);
  assert(fd > 0);
  int sz = read(fd, x, 32);
  assert(sz == 32);
  v_s1 = x[0];
  v_s1 = x[16];
  v_s1 = x[31];
  close(fd);
  delete x;
}

TEST(MemorySanitizer, pread) {
  char *x = new char[32];
  int fd = open("/proc/self/stat", O_RDONLY);
  assert(fd > 0);
  int sz = pread(fd, x, 32, 0);
  assert(sz == 32);
  v_s1 = x[0];
  v_s1 = x[16];
  v_s1 = x[31];
  close(fd);
  delete x;
}

// FIXME: fails now.
TEST(MemorySanitizer, DISABLED_ioctl) {
  struct winsize ws;
  EXPECT_EQ(ioctl(2, TIOCGWINSZ, &ws), 0);
  v_s4 = ws.ws_col;
}

TEST(MemorySanitizer, readlink) {
  char *x = new char[1000];
  readlink("/proc/self/exe", x, 1000);
  v_s1 = x[0];
  delete [] x;
}


TEST(MemorySanitizer, stat) {
  struct stat* st = new struct stat;
  int res = stat("/proc/self/stat", st);
  assert(!res);
  v_u8 = st->st_dev;
  v_u8 = st->st_mode;
  v_u8 = st->st_size;
}

TEST(MemorySanitizer, statfs) {
  struct statfs* st = new struct statfs;
  int res = statfs("/", st);
  assert(!res);
  v_u8 = st->f_type;
  v_u8 = st->f_bfree;
  v_u8 = st->f_namelen;
}

TEST(MemorySanitizer, pipe) {
  int* pipefd = new int[2];
  int res = pipe(pipefd);
  assert(!res);
  v_u8 = pipefd[0];
  v_u8 = pipefd[1];
  close(pipefd[0]);
  close(pipefd[1]);
}

TEST(MemorySanitizer, getcwd) {
  char path[PATH_MAX + 1];
  char* res = getcwd(path, sizeof(path));
  assert(res);
  v_s1 = path[0];
}

TEST(MemorySanitizer, realpath) {
  const char* relpath = ".";
  char path[PATH_MAX + 1];
  char* res = realpath(relpath, path);
  assert(res);
  v_s1 = path[0];
}

TEST(MemorySanitizer, memcpy) {
  char* x = new char[2];
  char* y = new char[2];
  x[0] = 1;
  x[1] = *GetPoisoned<char>();
  memcpy(y, x, 2);
  v_s4 = y[0];
  EXPECT_POISONED(v_s4 = y[1]);
}

TEST(MemorySanitizer, memmove) {
  char* x = new char[2];
  char* y = new char[2];
  x[0] = 1;
  x[1] = *GetPoisoned<char>();
  memmove(y, x, 2);
  v_s4 = y[0];
  EXPECT_POISONED(v_s4 = y[1]);
}

TEST(MemorySanitizer, strdup) {
  char *x = strdup("zzz");
  v_s1 = *x;
  free(x);
}

template<class T, int size>
void TestOverlapMemmove() {
  T *x = new T[size];
  assert(size >= 3);
  x[2] = 0;
  memmove(x, x + 1, (size - 1) * sizeof(T));
  v_s8 = x[1];
  if (!__msan_has_dynamic_component()) {
    // FIXME: under DR we will lose this information
    // because accesses in memmove will unpoisin the shadow.
    // We need to use our own memove implementation instead of libc's.
    EXPECT_POISONED(v_s8 = x[0]);
    EXPECT_POISONED(v_s8 = x[2]);
  }
  delete [] x;
}

TEST(MemorySanitizer, overlap_memmove) {
  TestOverlapMemmove<U1, 10>();
  TestOverlapMemmove<U1, 1000>();
  TestOverlapMemmove<U8, 4>();
  TestOverlapMemmove<U8, 1000>();
}

TEST(MemorySanitizer, strcpy) {  // NOLINT
  char* x = new char[3];
  char* y = new char[3];
  x[0] = 'a';
  x[1] = *GetPoisoned<char>(1, 1);
  x[2] = 0;
  strcpy(y, x);  // NOLINT
  v_s4 = y[0];
  EXPECT_POISONED(v_s4 = y[1]);
  v_s4 = y[2];
}

TEST(MemorySanitizer, strncpy) {  // NOLINT
  char* x = new char[3];
  char* y = new char[3];
  x[0] = 'a';
  x[1] = *GetPoisoned<char>(1, 1);
  x[2] = 0;
  strncpy(y, x, 2);  // NOLINT
  v_s4 = y[0];
  EXPECT_POISONED(v_s4 = y[1]);
  EXPECT_POISONED(v_s4 = y[2]);
}

TEST(MemorySanitizer, strtol) {
  char *e;
  assert(1 == strtol("1", &e, 10));
  v_s8 = (S8) e;
}

TEST(MemorySanitizer, strtoll) {
  char *e;
  assert(1 == strtoll("1", &e, 10));
  v_s8 = (S8) e;
}

TEST(MemorySanitizer, strtoul) {
  char *e;
  assert(1 == strtoul("1", &e, 10));
  v_s8 = (S8) e;
}

TEST(MemorySanitizer, strtoull) {
  char *e;
  assert(1 == strtoull("1", &e, 10));
  v_s8 = (S8) e;
}

TEST(MemorySanitizer, sprintf) {  // NOLINT
  char buff[10];
  __msan_break_optimization(buff);
  EXPECT_POISONED(v_s1 = buff[0]);
  int res = sprintf(buff, "%d", 1234567);  // NOLINT
  assert(res == 7);
  assert(buff[0] == '1');
  assert(buff[1] == '2');
  assert(buff[2] == '3');
  assert(buff[6] == '7');
  assert(buff[7] == 0);
  EXPECT_POISONED(v_s1 = buff[8]);
}

TEST(MemorySanitizer, snprintf) {
  char buff[10];
  __msan_break_optimization(buff);
  EXPECT_POISONED(v_s1 = buff[0]);
  int res = snprintf(buff, sizeof(buff), "%d", 1234567);
  assert(res == 7);
  assert(buff[0] == '1');
  assert(buff[1] == '2');
  assert(buff[2] == '3');
  assert(buff[6] == '7');
  assert(buff[7] == 0);
  EXPECT_POISONED(v_s1 = buff[8]);
}

TEST(MemorySanitizer, swprintf) {
  wchar_t buff[10];
  assert(sizeof(wchar_t) == 4);
  __msan_break_optimization(buff);
  EXPECT_POISONED(v_s1 = buff[0]);
  int res = swprintf(buff, 9, L"%d", 1234567);
  assert(res == 7);
  assert(buff[0] == '1');
  assert(buff[1] == '2');
  assert(buff[2] == '3');
  assert(buff[6] == '7');
  assert(buff[7] == 0);
  EXPECT_POISONED(v_s4 = buff[8]);
}

TEST(MemorySanitizer, wcstombs) {
  const wchar_t *x = L"abc";
  char buff[10];
  int res = wcstombs(buff, x, 4);
  EXPECT_EQ(res, 3);
  EXPECT_EQ(buff[0], 'a');
  EXPECT_EQ(buff[1], 'b');
  EXPECT_EQ(buff[2], 'c');
}

TEST(MemorySanitizer, gettimeofday) {
  struct timeval tv;
  struct timezone tz;
  __msan_break_optimization(&tv);
  __msan_break_optimization(&tz);
  assert(sizeof(tv) == 16);
  assert(sizeof(tz) == 8);
  EXPECT_POISONED(v_s8 = tv.tv_sec);
  EXPECT_POISONED(v_s8 = tv.tv_usec);
  EXPECT_POISONED(v_s4 = tz.tz_minuteswest);
  EXPECT_POISONED(v_s4 = tz.tz_dsttime);
  assert(0 == gettimeofday(&tv, &tz));
  v_s8 = tv.tv_sec;
  v_s8 = tv.tv_usec;
  v_s4 = tz.tz_minuteswest;
  v_s4 = tz.tz_dsttime;
}

TEST(MemorySanitizer, mmap) {
  const int size = 4096;
  void *p1, *p2;
  p1 = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
  __msan_poison(p1, size);
  munmap(p1, size);
  for (int i = 0; i < 1000; i++) {
    p2 = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    if (p2 == p1)
      break;
    else
      munmap(p2, size);
  }
  if (p1 == p2) {
    v_s1 = *(char*)p2;
    munmap(p2, size);
  }
}

// FIXME: enable and add ecvt.
// FIXME: check why msandr does nt handle fcvt.
TEST(MemorySanitizer, fcvt) {
  int a, b;
  __msan_break_optimization(&a);
  __msan_break_optimization(&b);
  EXPECT_POISONED(v_s4 = a);
  EXPECT_POISONED(v_s4 = b);
  char *str = fcvt(12345.6789, 10, &a, &b);
  v_s4 = a;
  v_s4 = b;
}

TEST(MemorySanitizer, LoadUnpoisoned) {
  S8 s = *GetPoisoned<S8>();
  EXPECT_POISONED(v_s8 = s);
  S8 safe = *GetPoisoned<S8>();
  __msan_load_unpoisoned(&s, sizeof(s), &safe);
  v_s8 = safe;
}

struct StructWithDtor {
  ~StructWithDtor();
};

NOINLINE StructWithDtor::~StructWithDtor() {
  __msan_break_optimization(0);
}

NOINLINE void ExpectGood(int a) { v_s4 = a; }
NOINLINE void ExpectPoisoned(int a) {
  EXPECT_POISONED(v_s4 = a);
}

TEST(MemorySanitizer, Invoke) {
  StructWithDtor s;  // Will cause the calls to become invokes.
  ExpectGood(0);
  ExpectPoisoned(*GetPoisoned<int>());
  ExpectGood(0);
  ExpectPoisoned(*GetPoisoned<int>());
  EXPECT_POISONED(v_s4 = ReturnPoisoned<S4>());
}

TEST(MemorySanitizer, ptrtoint) {
  // Test that shadow is propagated through pointer-to-integer conversion.
  void* p = (void*)0xABCD;
  __msan_poison(((char*)&p) + 1, sizeof(p));
  v_u1 = (((uptr)p) & 0xFF) == 0;

  void* q = (void*)0xABCD;
  __msan_poison(&q, sizeof(q) - 1);
  EXPECT_POISONED(v_u1 = (((uptr)q) & 0xFF) == 0);
}

static void vaargsfn2(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_d = va_arg(vl, double));
  va_end(vl);
}

static void vaargsfn(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  // The following call will overwrite __msan_param_tls.
  // Checks after it test that arg shadow was somehow saved across the call.
  vaargsfn2(1, 2, 3, 4, *GetPoisoned<double>());
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  va_end(vl);
}

TEST(MemorySanitizer, VAArgTest) {
  int* x = GetPoisoned<int>();
  int* y = GetPoisoned<int>(4);
  vaargsfn(1, 13, *x, 42, *y);
}

static void vaargsfn_many(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  va_end(vl);
}

TEST(MemorySanitizer, VAArgManyTest) {
  int* x = GetPoisoned<int>();
  int* y = GetPoisoned<int>(4);
  vaargsfn_many(1, 2, *x, 3, 4, 5, 6, 7, 8, 9, *y);
}

static void vaargsfn_pass2(va_list vl) {
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
}

static void vaargsfn_pass(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  vaargsfn_pass2(vl);
  va_end(vl);
}

TEST(MemorySanitizer, VAArgPass) {
  int* x = GetPoisoned<int>();
  int* y = GetPoisoned<int>(4);
  vaargsfn_pass(1, *x, 2, 3, *y);
}

static void vaargsfn_copy2(va_list vl) {
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
}

static void vaargsfn_copy(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  va_list vl2;
  va_copy(vl2, vl);
  vaargsfn_copy2(vl2);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  va_end(vl);
}

TEST(MemorySanitizer, VAArgCopy) {
  int* x = GetPoisoned<int>();
  int* y = GetPoisoned<int>(4);
  vaargsfn_copy(1, 2, *x, 3, *y);
}

static void vaargsfn_ptr(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  v_p = va_arg(vl, int*);
  EXPECT_POISONED(v_p = va_arg(vl, int*));
  v_p = va_arg(vl, int*);
  EXPECT_POISONED(v_p = va_arg(vl, double*));
  va_end(vl);
}

TEST(MemorySanitizer, VAArgPtr) {
  int** x = GetPoisoned<int*>();
  double** y = GetPoisoned<double*>(8);
  int z;
  vaargsfn_ptr(1, &z, *x, &z, *y);
}

static void vaargsfn_overflow(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);
  v_s4 = va_arg(vl, int);

  v_d = va_arg(vl, double);
  v_d = va_arg(vl, double);
  v_d = va_arg(vl, double);
  EXPECT_POISONED(v_d = va_arg(vl, double));
  v_d = va_arg(vl, double);
  EXPECT_POISONED(v_p = va_arg(vl, int*));
  v_d = va_arg(vl, double);
  v_d = va_arg(vl, double);

  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  EXPECT_POISONED(v_d = va_arg(vl, double));
  EXPECT_POISONED(v_p = va_arg(vl, int*));

  v_s4 = va_arg(vl, int);
  v_d = va_arg(vl, double);
  v_p = va_arg(vl, int*);

  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  EXPECT_POISONED(v_d = va_arg(vl, double));
  EXPECT_POISONED(v_p = va_arg(vl, int*));

  va_end(vl);
}

TEST(MemorySanitizer, VAArgOverflow) {
  int* x = GetPoisoned<int>();
  double* y = GetPoisoned<double>(8);
  int** p = GetPoisoned<int*>(16);
  int z;
  vaargsfn_overflow(1,
      1, 2, *x, 4, 5, 6,
      1.1, 2.2, 3.3, *y, 5.5, *p, 7.7, 8.8,
      // the following args will overflow for sure
      *x, *y, *p,
      7, 9.9, &z,
      *x, *y, *p);
}

static void vaargsfn_tlsoverwrite2(int guard, ...) {
  va_list vl;
  va_start(vl, guard);
  v_s4 = va_arg(vl, int);
  va_end(vl);
}

static void vaargsfn_tlsoverwrite(int guard, ...) {
  // This call will overwrite TLS contents unless it's backed up somewhere.
  vaargsfn_tlsoverwrite2(2, 42);
  va_list vl;
  va_start(vl, guard);
  EXPECT_POISONED(v_s4 = va_arg(vl, int));
  va_end(vl);
}

TEST(MemorySanitizer, VAArgTLSOverwrite) {
  int* x = GetPoisoned<int>();
  vaargsfn_tlsoverwrite(1, *x);
}

struct StructByVal {
  int a, b, c, d, e, f;
};

NOINLINE void StructByValTestFunc(struct StructByVal s) {
  v_s4 = s.a;
  EXPECT_POISONED(v_s4 = s.b);
  v_s4 = s.c;
  EXPECT_POISONED(v_s4 = s.d);
  v_s4 = s.e;
  EXPECT_POISONED(v_s4 = s.f);
}

NOINLINE void StructByValTestFunc1(struct StructByVal s) {
  StructByValTestFunc(s);
}

NOINLINE void StructByValTestFunc2(int z, struct StructByVal s) {
  StructByValTestFunc(s);
}

TEST(MemorySanitizer, StructByVal) {
  // Large aggregates are passed as "byval" pointer argument in LLVM.
  struct StructByVal s;
  s.a = 1;
  s.b = *GetPoisoned<int>();
  s.c = 2;
  s.d = *GetPoisoned<int>();
  s.e = 3;
  s.f = *GetPoisoned<int>();
  StructByValTestFunc(s);
  StructByValTestFunc1(s);
  StructByValTestFunc2(0, s);
}


#if MSAN_HAS_M128
NOINLINE __m128i m128Eq(__m128i *a, __m128i *b) { return *a == *b; }
NOINLINE __m128i m128Lt(__m128i *a, __m128i *b) { return *a < *b; }
TEST(MemorySanitizer, m128) {
  __m128i a = _mm_set1_epi16(0x1234);
  __m128i b = _mm_set1_epi16(0x7890);
  v_m128 = m128Eq(&a, &b);
  v_m128 = m128Lt(&a, &b);
}
// FIXME: add more tests for __m128i.
#endif  // MSAN_HAS_M128

// We should not complain when copying this poisoned hole.
struct StructWithHole {
  U4  a;
  // 4-byte hole.
  U8  b;
};

NOINLINE StructWithHole ReturnStructWithHole() {
  StructWithHole res;
  __msan_poison(&res, sizeof(res));
  res.a = 1;
  res.b = 2;
  return res;
}

TEST(MemorySanitizer, StructWithHole) {
  StructWithHole a = ReturnStructWithHole();
  __msan_break_optimization(&a);
}

template <class T>
NOINLINE T ReturnStruct() {
  T res;
  __msan_poison(&res, sizeof(res));
  res.a = 1;
  return res;
}

template <class T>
NOINLINE void TestReturnStruct() {
  T s1 = ReturnStruct<T>();
  v_s4 = s1.a;
  EXPECT_POISONED(v_s4 = s1.b);
}

struct SSS1 {
  int a, b, c;
};
struct SSS2 {
  int b, a, c;
};
struct SSS3 {
  int b, c, a;
};
struct SSS4 {
  int c, b, a;
};

struct SSS5 {
  int a;
  float b;
};
struct SSS6 {
  int a;
  double b;
};
struct SSS7 {
  S8 b;
  int a;
};
struct SSS8 {
  S2 b;
  S8 a;
};

TEST(MemorySanitizer, IntStruct3) {
  TestReturnStruct<SSS1>();
  TestReturnStruct<SSS2>();
  TestReturnStruct<SSS3>();
  TestReturnStruct<SSS4>();
  TestReturnStruct<SSS5>();
  TestReturnStruct<SSS6>();
  TestReturnStruct<SSS7>();
  TestReturnStruct<SSS8>();
}

struct LongStruct {
  U1 a1, b1;
  U2 a2, b2;
  U4 a4, b4;
  U8 a8, b8;
};

NOINLINE LongStruct ReturnLongStruct1() {
  LongStruct res;
  __msan_poison(&res, sizeof(res));
  res.a1 = res.a2 = res.a4 = res.a8 = 111;
  // leaves b1, .., b8 poisoned.
  return res;
}

NOINLINE LongStruct ReturnLongStruct2() {
  LongStruct res;
  __msan_poison(&res, sizeof(res));
  res.b1 = res.b2 = res.b4 = res.b8 = 111;
  // leaves a1, .., a8 poisoned.
  return res;
}

TEST(MemorySanitizer, LongStruct) {
  LongStruct s1 = ReturnLongStruct1();
  __msan_print_shadow(&s1, sizeof(s1));
  v_u1 = s1.a1;
  v_u2 = s1.a2;
  v_u4 = s1.a4;
  v_u8 = s1.a8;

  EXPECT_POISONED(v_u1 = s1.b1);
  EXPECT_POISONED(v_u2 = s1.b2);
  EXPECT_POISONED(v_u4 = s1.b4);
  EXPECT_POISONED(v_u8 = s1.b8);

  LongStruct s2 = ReturnLongStruct2();
  __msan_print_shadow(&s2, sizeof(s2));
  v_u1 = s2.b1;
  v_u2 = s2.b2;
  v_u4 = s2.b4;
  v_u8 = s2.b8;

  EXPECT_POISONED(v_u1 = s2.a1);
  EXPECT_POISONED(v_u2 = s2.a2);
  EXPECT_POISONED(v_u4 = s2.a4);
  EXPECT_POISONED(v_u8 = s2.a8);
}

TEST(MemorySanitizer, getrlimit) {
  struct rlimit limit;
  __msan_poison(&limit, sizeof(limit));
  int result = getrlimit(RLIMIT_DATA, &limit);
  assert(result == 0);
  volatile rlim_t t;
  t = limit.rlim_cur;
  t = limit.rlim_max;
}

static void* SimpleThread_threadfn(void* data) {
  return new int;
}

TEST(MemorySanitizer, SimpleThread) {
  pthread_t t;
  void* p;
  int res = pthread_create(&t, NULL, SimpleThread_threadfn, NULL);
  assert(!res);
  res = pthread_join(t, &p);
  assert(!res);
  if (!__msan_has_dynamic_component())  // FIXME: intercept pthread_join (?).
    __msan_unpoison(&p, sizeof(p));
  delete (int*)p;
}

TEST(MemorySanitizer, uname) {
  struct utsname u;
  int res = uname(&u);
  assert(!res);
  v_u8 = strlen(u.sysname);
  v_u8 = strlen(u.nodename);
  v_u8 = strlen(u.release);
  v_u8 = strlen(u.version);
  v_u8 = strlen(u.machine);
}

template<class T>
static void testSlt(T value, T shadow) {
  __msan_partial_poison(&value, &shadow, sizeof(T));
  volatile bool zzz = true;
  // This "|| zzz" trick somehow makes LLVM emit "icmp slt" instead of
  // a shift-and-trunc to get at the highest bit.
  volatile bool v_T = value < 0 || zzz;
}

TEST(MemorySanitizer, SignedCompareWithZero) {
  testSlt<S4>(0xF, 0xF);
  testSlt<S4>(0xF, 0xFF);
  testSlt<S4>(0xF, 0xFFFFFF);
  testSlt<S4>(0xF, 0x7FFFFFF);
  EXPECT_POISONED(testSlt<S4>(0xF, 0x80FFFFFF));
  EXPECT_POISONED(testSlt<S4>(0xF, 0xFFFFFFFF));
}

extern "C" {
NOINLINE void ZZZZZZZZZZZZZZ() {
  __msan_break_optimization(0);

  // v_s1 = ReturnPoisoned<S1>();
  // a_s8[g_zero] = *GetPoisoned<S8>() - 1;
  // v_s4 = a_s4[g_zero];
  __msan_break_optimization(0);
}
}

TEST(MemorySanitizer, ZZZTest) {
  ZZZZZZZZZZZZZZ();
}

TEST(MemorySanitizerDr, StoreInDSOTest) {
  if (!__msan_has_dynamic_component()) return;
  char* s = new char[10];
  dso_memfill(s, 9);
  v_s1 = s[5];
  EXPECT_POISONED(v_s1 = s[9]);
}

int return_poisoned_int() {
  return ReturnPoisoned<U8>();
}

TEST(MemorySanitizerDr, ReturnFromDSOTest) {
  if (!__msan_has_dynamic_component()) return;
  v_u8 = dso_callfn(return_poisoned_int);
}

NOINLINE int TrashParamTLS(long long x, long long y, long long z) {  //NOLINT
  EXPECT_POISONED(v_s8 = x);
  EXPECT_POISONED(v_s8 = y);
  EXPECT_POISONED(v_s8 = z);
  return 0;
}

static int CheckParamTLS(long long x, long long y, long long z) {  //NOLINT
  v_s8 = x;
  v_s8 = y;
  v_s8 = z;
  return 0;
}

TEST(MemorySanitizerDr, CallFromDSOTest) {
  if (!__msan_has_dynamic_component()) return;
  S8* x = GetPoisoned<S8>();
  S8* y = GetPoisoned<S8>();
  S8* z = GetPoisoned<S8>();
  v_s4 = TrashParamTLS(*x, *y, *z);
  v_u8 = dso_callfn1(CheckParamTLS);
}

static void StackStoreInDSOFn(int* x, int* y) {
  v_s4 = *x;
  v_s4 = *y;
}

TEST(MemorySanitizerDr, StackStoreInDSOTest) {
  if (!__msan_has_dynamic_component()) return;
  dso_stack_store(StackStoreInDSOFn, 1);
}

TEST(MemorySanitizerOrigins, SetGet) {
  EXPECT_EQ(TrackingOrigins(), __msan_get_track_origins());
  if (!TrackingOrigins()) return;
  int x;
  __msan_set_origin(&x, sizeof(x), 1234);
  EXPECT_EQ(1234, __msan_get_origin(&x));
  __msan_set_origin(&x, sizeof(x), 5678);
  EXPECT_EQ(5678, __msan_get_origin(&x));
  __msan_set_origin(&x, sizeof(x), 0);
  EXPECT_EQ(0, __msan_get_origin(&x));
}

namespace {
struct S {
  U4 dummy;
  U2 a;
  U2 b;
};

// http://code.google.com/p/memory-sanitizer/issues/detail?id=6
TEST(MemorySanitizerOrigins, DISABLED_InitializedStoreDoesNotChangeOrigin) {
  if (!TrackingOrigins()) return;

  S s;
  u32 origin = rand();  // NOLINT
  s.a = *GetPoisonedO<U2>(0, origin);
  EXPECT_EQ(origin, __msan_get_origin(&s.a));
  EXPECT_EQ(origin, __msan_get_origin(&s.b));

  s.b = 42;
  EXPECT_EQ(origin, __msan_get_origin(&s.a));
  EXPECT_EQ(origin, __msan_get_origin(&s.b));
}
}  // namespace

template<class T, class BinaryOp>
INLINE
void BinaryOpOriginTest(BinaryOp op) {
  u32 ox = rand();  //NOLINT
  u32 oy = rand();  //NOLINT
  T *x = GetPoisonedO<T>(0, ox, 0);
  T *y = GetPoisonedO<T>(1, oy, 0);
  T *z = GetPoisonedO<T>(2, 0, 0);

  *z = op(*x, *y);
  u32 origin = __msan_get_origin(z);
  EXPECT_POISONED_O(v_s8 = *z, origin);
  EXPECT_EQ(true, origin == ox || origin == oy);

  // y is poisoned, x is not.
  *x = 10101;
  *y = *GetPoisonedO<T>(1, oy);
  __msan_break_optimization(x);
  __msan_set_origin(z, sizeof(*z), 0);
  *z = op(*x, *y);
  EXPECT_POISONED_O(v_s8 = *z, oy);
  EXPECT_EQ(__msan_get_origin(z), oy);

  // x is poisoned, y is not.
  *x = *GetPoisonedO<T>(0, ox);
  *y = 10101010;
  __msan_break_optimization(y);
  __msan_set_origin(z, sizeof(*z), 0);
  *z = op(*x, *y);
  EXPECT_POISONED_O(v_s8 = *z, ox);
  EXPECT_EQ(__msan_get_origin(z), ox);
}

template<class T> INLINE T XOR(const T &a, const T&b) { return a ^ b; }
template<class T> INLINE T ADD(const T &a, const T&b) { return a + b; }
template<class T> INLINE T SUB(const T &a, const T&b) { return a - b; }
template<class T> INLINE T MUL(const T &a, const T&b) { return a * b; }
template<class T> INLINE T AND(const T &a, const T&b) { return a & b; }
template<class T> INLINE T OR (const T &a, const T&b) { return a | b; }

TEST(MemorySanitizerOrigins, BinaryOp) {
  if (!TrackingOrigins()) return;
  BinaryOpOriginTest<S8>(XOR<S8>);
  BinaryOpOriginTest<U8>(ADD<U8>);
  BinaryOpOriginTest<S4>(SUB<S4>);
  BinaryOpOriginTest<S4>(MUL<S4>);
  BinaryOpOriginTest<U4>(OR<U4>);
  BinaryOpOriginTest<U4>(AND<U4>);
  BinaryOpOriginTest<double>(ADD<U4>);
  BinaryOpOriginTest<float>(ADD<S4>);
  BinaryOpOriginTest<double>(ADD<double>);
  BinaryOpOriginTest<float>(ADD<double>);
}

TEST(MemorySanitizerOrigins, Unary) {
  if (!TrackingOrigins()) return;
  EXPECT_POISONED_O(v_s8 = *GetPoisonedO<S8>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s4 = *GetPoisonedO<S8>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s2 = *GetPoisonedO<S8>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s1 = *GetPoisonedO<S8>(0, __LINE__), __LINE__);

  EXPECT_POISONED_O(v_s8 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s4 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s2 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s1 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);

  EXPECT_POISONED_O(v_s8 = *GetPoisonedO<U4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s4 = *GetPoisonedO<U4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s2 = *GetPoisonedO<U4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s1 = *GetPoisonedO<U4>(0, __LINE__), __LINE__);

  EXPECT_POISONED_O(v_u8 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_u4 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_u2 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_u1 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);

  EXPECT_POISONED_O(v_p = (void*)*GetPoisonedO<S8>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_u8 = (U8)*GetPoisonedO<void*>(0, __LINE__), __LINE__);
}

TEST(MemorySanitizerOrigins, EQ) {
  if (!TrackingOrigins()) return;
  EXPECT_POISONED_O(v_u1 = *GetPoisonedO<S4>(0, __LINE__) <= 11, __LINE__);
  EXPECT_POISONED_O(v_u1 = *GetPoisonedO<S4>(0, __LINE__) == 11, __LINE__);
  EXPECT_POISONED_O(v_u1 = *GetPoisonedO<float>(0, __LINE__) == 1.1, __LINE__);
}

TEST(MemorySanitizerOrigins, DIV) {
  if (!TrackingOrigins()) return;
  EXPECT_POISONED_O(v_u8 = *GetPoisonedO<U8>(0, __LINE__) / 100, __LINE__);
  EXPECT_POISONED_O(v_s4 = 100 / *GetPoisonedO<S4>(0, __LINE__, 1), __LINE__);
}

TEST(MemorySanitizerOrigins, SHIFT) {
  if (!TrackingOrigins()) return;
  EXPECT_POISONED_O(v_u8 = *GetPoisonedO<U8>(0, __LINE__) >> 10, __LINE__);
  EXPECT_POISONED_O(v_s8 = *GetPoisonedO<S8>(0, __LINE__) >> 10, __LINE__);
  EXPECT_POISONED_O(v_s8 = *GetPoisonedO<S8>(0, __LINE__) << 10, __LINE__);
  EXPECT_POISONED_O(v_u8 = 10U << *GetPoisonedO<U8>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s8 = -10 >> *GetPoisonedO<S8>(0, __LINE__), __LINE__);
  EXPECT_POISONED_O(v_s8 = -10 << *GetPoisonedO<S8>(0, __LINE__), __LINE__);
}

template<class T, int N>
void MemCpyTest() {
  int ox = __LINE__;
  T *x = new T[N];
  T *y = new T[N];
  T *z = new T[N];
  __msan_poison(x, N * sizeof(T));
  __msan_set_origin(x, N * sizeof(T), ox);
  __msan_set_origin(y, N * sizeof(T), 777777);
  __msan_set_origin(z, N * sizeof(T), 888888);
  v_p = x;
  memcpy(y, v_p, N * sizeof(T));
  EXPECT_POISONED_O(v_s1 = y[0], ox);
  EXPECT_POISONED_O(v_s1 = y[N/2], ox);
  EXPECT_POISONED_O(v_s1 = y[N-1], ox);
  v_p = x;
  memmove(z, v_p, N * sizeof(T));
  EXPECT_POISONED_O(v_s1 = z[0], ox);
  EXPECT_POISONED_O(v_s1 = z[N/2], ox);
  EXPECT_POISONED_O(v_s1 = z[N-1], ox);
}

TEST(MemorySanitizerOrigins, LargeMemCpy) {
  if (!TrackingOrigins()) return;
  MemCpyTest<U1, 10000>();
  MemCpyTest<U8, 10000>();
}

TEST(MemorySanitizerOrigins, SmallMemCpy) {
  if (!TrackingOrigins()) return;
  MemCpyTest<U8, 1>();
  MemCpyTest<U8, 2>();
  MemCpyTest<U8, 3>();
}

TEST(MemorySanitizerOrigins, Select) {
  if (!TrackingOrigins()) return;
  v_s8 = g_one ? 1 : *GetPoisonedO<S4>(0, __LINE__);
  EXPECT_POISONED_O(v_s8 = *GetPoisonedO<S4>(0, __LINE__), __LINE__);
  S4 x;
  __msan_break_optimization(&x);
  x = g_1 ? *GetPoisonedO<S4>(0, __LINE__) : 0;

  EXPECT_POISONED_O(v_s8 = g_1 ? *GetPoisonedO<S4>(0, __LINE__) : 1, __LINE__);
  EXPECT_POISONED_O(v_s8 = g_0 ? 1 : *GetPoisonedO<S4>(0, __LINE__), __LINE__);
}

extern "C"
NOINLINE void AllocaTOTest() {
  int ar[100];
  __msan_break_optimization(ar);
  v_s8 = ar[10];
  // fprintf(stderr, "Descr: %s\n",
  //        __msan_get_origin_descr_if_stack(__msan_get_origin_tls()));
}

TEST(MemorySanitizerOrigins, Alloca) {
  if (!TrackingOrigins()) return;
  EXPECT_POISONED_S(AllocaTOTest(), "ar@AllocaTOTest");
  EXPECT_POISONED_S(AllocaTOTest(), "ar@AllocaTOTest");
  EXPECT_POISONED_S(AllocaTOTest(), "ar@AllocaTOTest");
  EXPECT_POISONED_S(AllocaTOTest(), "ar@AllocaTOTest");
}

// FIXME: replace with a lit-like test.
TEST(MemorySanitizerOrigins, DISABLED_AllocaDeath) {
  if (!TrackingOrigins()) return;
  EXPECT_DEATH(AllocaTOTest(), "ORIGIN: stack allocation: ar@AllocaTOTest");
}

NOINLINE int RetvalOriginTest(u32 origin) {
  int *a = new int;
  __msan_break_optimization(a);
  __msan_set_origin(a, sizeof(*a), origin);
  int res = *a;
  delete a;
  return res;
}

TEST(MemorySanitizerOrigins, Retval) {
  if (!TrackingOrigins()) return;
  EXPECT_POISONED_O(v_s4 = RetvalOriginTest(__LINE__), __LINE__);
}

NOINLINE void ParamOriginTest(int param, u32 origin) {
  EXPECT_POISONED_O(v_s4 = param, origin);
}

TEST(MemorySanitizerOrigins, Param) {
  if (!TrackingOrigins()) return;
  int *a = new int;
  u32 origin = __LINE__;
  __msan_break_optimization(a);
  __msan_set_origin(a, sizeof(*a), origin);
  ParamOriginTest(*a, origin);
  delete a;
}

TEST(MemorySanitizerOrigins, Invoke) {
  if (!TrackingOrigins()) return;
  StructWithDtor s;  // Will cause the calls to become invokes.
  EXPECT_POISONED_O(v_s4 = RetvalOriginTest(__LINE__), __LINE__);
}

TEST(MemorySanitizerOrigins, strlen) {
  S8 alignment;
  __msan_break_optimization(&alignment);
  char x[4] = {'a', 'b', 0, 0};
  __msan_poison(&x[2], 1);
  u32 origin = __LINE__;
  __msan_set_origin(x, sizeof(x), origin);
  EXPECT_POISONED_O(v_s4 = strlen(x), origin);
}

TEST(MemorySanitizerOrigins, wcslen) {
  wchar_t w[3] = {'a', 'b', 0};
  u32 origin = __LINE__;
  __msan_set_origin(w, sizeof(w), origin);
  __msan_poison(&w[2], sizeof(wchar_t));
  EXPECT_POISONED_O(v_s4 = wcslen(w), origin);
}

#if MSAN_HAS_M128
TEST(MemorySanitizerOrigins, StoreIntrinsic) {
  __m128 x, y;
  u32 origin = __LINE__;
  __msan_set_origin(&x, sizeof(x), origin);
  __msan_poison(&x, sizeof(x));
  __builtin_ia32_storeups((float*)&y, x);
  EXPECT_POISONED_O(v_m128 = y, origin);
}
#endif

NOINLINE void RecursiveMalloc(int depth) {
  static int count;
  count++;
  if ((count % (1024 * 1024)) == 0)
    printf("RecursiveMalloc: %d\n", count);
  int *x1 = new int;
  int *x2 = new int;
  __msan_break_optimization(x1);
  __msan_break_optimization(x2);
  if (depth > 0) {
    RecursiveMalloc(depth-1);
    RecursiveMalloc(depth-1);
  }
  delete x1;
  delete x2;
}

TEST(MemorySanitizerStress, DISABLED_MallocStackTrace) {
  RecursiveMalloc(22);
}

int main(int argc, char **argv) {
  __msan_set_poison_in_malloc(1);
  testing::InitGoogleTest(&argc, argv);
  int res = RUN_ALL_TESTS();
  return res;
}
