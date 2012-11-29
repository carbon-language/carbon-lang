//===-- asan_test.cc ------------------------------------------------------===//
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
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <stdint.h>
#include <setjmp.h>
#include <assert.h>

#if defined(__i386__) || defined(__x86_64__)
#include <emmintrin.h>
#endif

#include "asan_test_utils.h"

#ifndef __APPLE__
#include <malloc.h>
#else
#include <malloc/malloc.h>
#include <AvailabilityMacros.h>  // For MAC_OS_X_VERSION_*
#include <CoreFoundation/CFString.h>
#endif  // __APPLE__

#if ASAN_HAS_EXCEPTIONS
# define ASAN_THROW(x) throw (x)
#else
# define ASAN_THROW(x)
#endif

#include <sys/mman.h>

typedef uint8_t   U1;
typedef uint16_t  U2;
typedef uint32_t  U4;
typedef uint64_t  U8;

static const int kPageSize = 4096;

// Simple stand-alone pseudorandom number generator.
// Current algorithm is ANSI C linear congruential PRNG.
static inline uint32_t my_rand(uint32_t* state) {
  return (*state = *state * 1103515245 + 12345) >> 16;
}

static uint32_t global_seed = 0;

const size_t kLargeMalloc = 1 << 24;

template<typename T>
NOINLINE void asan_write(T *a) {
  *a = 0;
}

NOINLINE void asan_write_sized_aligned(uint8_t *p, size_t size) {
  EXPECT_EQ(0U, ((uintptr_t)p % size));
  if      (size == 1) asan_write((uint8_t*)p);
  else if (size == 2) asan_write((uint16_t*)p);
  else if (size == 4) asan_write((uint32_t*)p);
  else if (size == 8) asan_write((uint64_t*)p);
}

NOINLINE void *malloc_fff(size_t size) {
  void *res = malloc/**/(size); break_optimization(0); return res;}
NOINLINE void *malloc_eee(size_t size) {
  void *res = malloc_fff(size); break_optimization(0); return res;}
NOINLINE void *malloc_ddd(size_t size) {
  void *res = malloc_eee(size); break_optimization(0); return res;}
NOINLINE void *malloc_ccc(size_t size) {
  void *res = malloc_ddd(size); break_optimization(0); return res;}
NOINLINE void *malloc_bbb(size_t size) {
  void *res = malloc_ccc(size); break_optimization(0); return res;}
NOINLINE void *malloc_aaa(size_t size) {
  void *res = malloc_bbb(size); break_optimization(0); return res;}

#ifndef __APPLE__
NOINLINE void *memalign_fff(size_t alignment, size_t size) {
  void *res = memalign/**/(alignment, size); break_optimization(0); return res;}
NOINLINE void *memalign_eee(size_t alignment, size_t size) {
  void *res = memalign_fff(alignment, size); break_optimization(0); return res;}
NOINLINE void *memalign_ddd(size_t alignment, size_t size) {
  void *res = memalign_eee(alignment, size); break_optimization(0); return res;}
NOINLINE void *memalign_ccc(size_t alignment, size_t size) {
  void *res = memalign_ddd(alignment, size); break_optimization(0); return res;}
NOINLINE void *memalign_bbb(size_t alignment, size_t size) {
  void *res = memalign_ccc(alignment, size); break_optimization(0); return res;}
NOINLINE void *memalign_aaa(size_t alignment, size_t size) {
  void *res = memalign_bbb(alignment, size); break_optimization(0); return res;}
#endif  // __APPLE__


NOINLINE void free_ccc(void *p) { free(p); break_optimization(0);}
NOINLINE void free_bbb(void *p) { free_ccc(p); break_optimization(0);}
NOINLINE void free_aaa(void *p) { free_bbb(p); break_optimization(0);}

template<typename T>
NOINLINE void oob_test(int size, int off) {
  char *p = (char*)malloc_aaa(size);
  // fprintf(stderr, "writing %d byte(s) into [%p,%p) with offset %d\n",
  //        sizeof(T), p, p + size, off);
  asan_write((T*)(p + off));
  free_aaa(p);
}


template<typename T>
NOINLINE void uaf_test(int size, int off) {
  char *p = (char *)malloc_aaa(size);
  free_aaa(p);
  for (int i = 1; i < 100; i++)
    free_aaa(malloc_aaa(i));
  fprintf(stderr, "writing %ld byte(s) at %p with offset %d\n",
          (long)sizeof(T), p, off);
  asan_write((T*)(p + off));
}

TEST(AddressSanitizer, HasFeatureAddressSanitizerTest) {
#if defined(__has_feature) && __has_feature(address_sanitizer)
  bool asan = 1;
#else
  bool asan = 0;
#endif
  EXPECT_EQ(true, asan);
}

TEST(AddressSanitizer, SimpleDeathTest) {
  EXPECT_DEATH(exit(1), "");
}

TEST(AddressSanitizer, VariousMallocsTest) {
  int *a = (int*)malloc(100 * sizeof(int));
  a[50] = 0;
  free(a);

  int *r = (int*)malloc(10);
  r = (int*)realloc(r, 2000 * sizeof(int));
  r[1000] = 0;
  free(r);

  int *b = new int[100];
  b[50] = 0;
  delete [] b;

  int *c = new int;
  *c = 0;
  delete c;

#if !defined(__APPLE__) && !defined(ANDROID) && !defined(__ANDROID__)
  int *pm;
  int pm_res = posix_memalign((void**)&pm, kPageSize, kPageSize);
  EXPECT_EQ(0, pm_res);
  free(pm);
#endif

#if !defined(__APPLE__)
  int *ma = (int*)memalign(kPageSize, kPageSize);
  EXPECT_EQ(0U, (uintptr_t)ma % kPageSize);
  ma[123] = 0;
  free(ma);
#endif  // __APPLE__
}

TEST(AddressSanitizer, CallocTest) {
  int *a = (int*)calloc(100, sizeof(int));
  EXPECT_EQ(0, a[10]);
  free(a);
}

TEST(AddressSanitizer, VallocTest) {
  void *a = valloc(100);
  EXPECT_EQ(0U, (uintptr_t)a % kPageSize);
  free(a);
}

#ifndef __APPLE__
TEST(AddressSanitizer, PvallocTest) {
  char *a = (char*)pvalloc(kPageSize + 100);
  EXPECT_EQ(0U, (uintptr_t)a % kPageSize);
  a[kPageSize + 101] = 1;  // we should not report an error here.
  free(a);

  a = (char*)pvalloc(0);  // pvalloc(0) should allocate at least one page.
  EXPECT_EQ(0U, (uintptr_t)a % kPageSize);
  a[101] = 1;  // we should not report an error here.
  free(a);
}
#endif  // __APPLE__

void *TSDWorker(void *test_key) {
  if (test_key) {
    pthread_setspecific(*(pthread_key_t*)test_key, (void*)0xfeedface);
  }
  return NULL;
}

void TSDDestructor(void *tsd) {
  // Spawning a thread will check that the current thread id is not -1.
  pthread_t th;
  pthread_create(&th, NULL, TSDWorker, NULL);
  pthread_join(th, NULL);
}

// This tests triggers the thread-specific data destruction fiasco which occurs
// if we don't manage the TSD destructors ourselves. We create a new pthread
// key with a non-NULL destructor which is likely to be put after the destructor
// of AsanThread in the list of destructors.
// In this case the TSD for AsanThread will be destroyed before TSDDestructor
// is called for the child thread, and a CHECK will fail when we call
// pthread_create() to spawn the grandchild.
TEST(AddressSanitizer, DISABLED_TSDTest) {
  pthread_t th;
  pthread_key_t test_key;
  pthread_key_create(&test_key, TSDDestructor);
  pthread_create(&th, NULL, TSDWorker, &test_key);
  pthread_join(th, NULL);
  pthread_key_delete(test_key);
}

template<typename T>
void OOBTest() {
  char expected_str[100];
  for (int size = sizeof(T); size < 20; size += 5) {
    for (int i = -5; i < 0; i++) {
      const char *str =
          "is located.*%d byte.*to the left";
      sprintf(expected_str, str, abs(i));
      EXPECT_DEATH(oob_test<T>(size, i), expected_str);
    }

    for (int i = 0; i < (int)(size - sizeof(T) + 1); i++)
      oob_test<T>(size, i);

    for (int i = size - sizeof(T) + 1; i <= (int)(size + 3 * sizeof(T)); i++) {
      const char *str =
          "is located.*%d byte.*to the right";
      int off = i >= size ? (i - size) : 0;
      // we don't catch unaligned partially OOB accesses.
      if (i % sizeof(T)) continue;
      sprintf(expected_str, str, off);
      EXPECT_DEATH(oob_test<T>(size, i), expected_str);
    }
  }

  EXPECT_DEATH(oob_test<T>(kLargeMalloc, -1),
          "is located.*1 byte.*to the left");
  EXPECT_DEATH(oob_test<T>(kLargeMalloc, kLargeMalloc),
          "is located.*0 byte.*to the right");
}

// TODO(glider): the following tests are EXTREMELY slow on Darwin:
//   AddressSanitizer.OOB_char (125503 ms)
//   AddressSanitizer.OOB_int (126890 ms)
//   AddressSanitizer.OOBRightTest (315605 ms)
//   AddressSanitizer.SimpleStackTest (366559 ms)

TEST(AddressSanitizer, OOB_char) {
  OOBTest<U1>();
}

TEST(AddressSanitizer, OOB_int) {
  OOBTest<U4>();
}

TEST(AddressSanitizer, OOBRightTest) {
  for (size_t access_size = 1; access_size <= 8; access_size *= 2) {
    for (size_t alloc_size = 1; alloc_size <= 8; alloc_size++) {
      for (size_t offset = 0; offset <= 8; offset += access_size) {
        void *p = malloc(alloc_size);
        // allocated: [p, p + alloc_size)
        // accessed:  [p + offset, p + offset + access_size)
        uint8_t *addr = (uint8_t*)p + offset;
        if (offset + access_size <= alloc_size) {
          asan_write_sized_aligned(addr, access_size);
        } else {
          int outside_bytes = offset > alloc_size ? (offset - alloc_size) : 0;
          const char *str =
              "is located.%d *byte.*to the right";
          char expected_str[100];
          sprintf(expected_str, str, outside_bytes);
          EXPECT_DEATH(asan_write_sized_aligned(addr, access_size),
                       expected_str);
        }
        free(p);
      }
    }
  }
}

TEST(AddressSanitizer, UAF_char) {
  const char *uaf_string = "AddressSanitizer:.*heap-use-after-free";
  EXPECT_DEATH(uaf_test<U1>(1, 0), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(10, 0), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(10, 10), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(kLargeMalloc, 0), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(kLargeMalloc, kLargeMalloc / 2), uaf_string);
}

#if ASAN_HAS_BLACKLIST
TEST(AddressSanitizer, IgnoreTest) {
  int *x = Ident(new int);
  delete Ident(x);
  *x = 0;
}
#endif  // ASAN_HAS_BLACKLIST

struct StructWithBitField {
  int bf1:1;
  int bf2:1;
  int bf3:1;
  int bf4:29;
};

TEST(AddressSanitizer, BitFieldPositiveTest) {
  StructWithBitField *x = new StructWithBitField;
  delete Ident(x);
  EXPECT_DEATH(x->bf1 = 0, "use-after-free");
  EXPECT_DEATH(x->bf2 = 0, "use-after-free");
  EXPECT_DEATH(x->bf3 = 0, "use-after-free");
  EXPECT_DEATH(x->bf4 = 0, "use-after-free");
}

struct StructWithBitFields_8_24 {
  int a:8;
  int b:24;
};

TEST(AddressSanitizer, BitFieldNegativeTest) {
  StructWithBitFields_8_24 *x = Ident(new StructWithBitFields_8_24);
  x->a = 0;
  x->b = 0;
  delete Ident(x);
}

TEST(AddressSanitizer, OutOfMemoryTest) {
  size_t size = SANITIZER_WORDSIZE == 64 ? (size_t)(1ULL << 48) : (0xf0000000);
  EXPECT_EQ(0, realloc(0, size));
  EXPECT_EQ(0, realloc(0, ~Ident(0)));
  EXPECT_EQ(0, malloc(size));
  EXPECT_EQ(0, malloc(~Ident(0)));
  EXPECT_EQ(0, calloc(1, size));
  EXPECT_EQ(0, calloc(1, ~Ident(0)));
}

#if ASAN_NEEDS_SEGV
namespace {

const char kUnknownCrash[] = "AddressSanitizer: SEGV on unknown address";
const char kOverriddenHandler[] = "ASan signal handler has been overridden\n";

TEST(AddressSanitizer, WildAddressTest) {
  char *c = (char*)0x123;
  EXPECT_DEATH(*c = 0, kUnknownCrash);
}

void my_sigaction_sighandler(int, siginfo_t*, void*) {
  fprintf(stderr, kOverriddenHandler);
  exit(1);
}

void my_signal_sighandler(int signum) {
  fprintf(stderr, kOverriddenHandler);
  exit(1);
}

TEST(AddressSanitizer, SignalTest) {
  struct sigaction sigact;
  memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = my_sigaction_sighandler;
  sigact.sa_flags = SA_SIGINFO;
  // ASan should silently ignore sigaction()...
  EXPECT_EQ(0, sigaction(SIGSEGV, &sigact, 0));
#ifdef __APPLE__
  EXPECT_EQ(0, sigaction(SIGBUS, &sigact, 0));
#endif
  char *c = (char*)0x123;
  EXPECT_DEATH(*c = 0, kUnknownCrash);
  // ... and signal().
  EXPECT_EQ(0, signal(SIGSEGV, my_signal_sighandler));
  EXPECT_DEATH(*c = 0, kUnknownCrash);
}
}  // namespace
#endif

static void MallocStress(size_t n) {
  uint32_t seed = my_rand(&global_seed);
  for (size_t iter = 0; iter < 10; iter++) {
    vector<void *> vec;
    for (size_t i = 0; i < n; i++) {
      if ((i % 3) == 0) {
        if (vec.empty()) continue;
        size_t idx = my_rand(&seed) % vec.size();
        void *ptr = vec[idx];
        vec[idx] = vec.back();
        vec.pop_back();
        free_aaa(ptr);
      } else {
        size_t size = my_rand(&seed) % 1000 + 1;
#ifndef __APPLE__
        size_t alignment = 1 << (my_rand(&seed) % 7 + 3);
        char *ptr = (char*)memalign_aaa(alignment, size);
#else
        char *ptr = (char*) malloc_aaa(size);
#endif
        vec.push_back(ptr);
        ptr[0] = 0;
        ptr[size-1] = 0;
        ptr[size/2] = 0;
      }
    }
    for (size_t i = 0; i < vec.size(); i++)
      free_aaa(vec[i]);
  }
}

TEST(AddressSanitizer, MallocStressTest) {
  MallocStress((ASAN_LOW_MEMORY) ? 20000 : 200000);
}

static void TestLargeMalloc(size_t size) {
  char buff[1024];
  sprintf(buff, "is located 1 bytes to the left of %lu-byte", (long)size);
  EXPECT_DEATH(Ident((char*)malloc(size))[-1] = 0, buff);
}

TEST(AddressSanitizer, LargeMallocTest) {
  for (int i = 113; i < (1 << 28); i = i * 2 + 13) {
    TestLargeMalloc(i);
  }
}

#if ASAN_LOW_MEMORY != 1
TEST(AddressSanitizer, HugeMallocTest) {
#ifdef __APPLE__
  // It was empirically found out that 1215 megabytes is the maximum amount of
  // memory available to the process under AddressSanitizer on 32-bit Mac 10.6.
  // 32-bit Mac 10.7 gives even less (< 1G).
  // (the libSystem malloc() allows allocating up to 2300 megabytes without
  // ASan).
  size_t n_megs = SANITIZER_WORDSIZE == 32 ? 500 : 4100;
#else
  size_t n_megs = SANITIZER_WORDSIZE == 32 ? 2600 : 4100;
#endif
  TestLargeMalloc(n_megs << 20);
}
#endif

TEST(AddressSanitizer, ThreadedMallocStressTest) {
  const int kNumThreads = 4;
  const int kNumIterations = (ASAN_LOW_MEMORY) ? 10000 : 100000;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, (void* (*)(void *x))MallocStress,
        (void*)kNumIterations);
  }
  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }
}

void *ManyThreadsWorker(void *a) {
  for (int iter = 0; iter < 100; iter++) {
    for (size_t size = 100; size < 2000; size *= 2) {
      free(Ident(malloc(size)));
    }
  }
  return 0;
}

TEST(AddressSanitizer, ManyThreadsTest) {
  const size_t kNumThreads = SANITIZER_WORDSIZE == 32 ? 30 : 1000;
  pthread_t t[kNumThreads];
  for (size_t i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, (void* (*)(void *x))ManyThreadsWorker, (void*)i);
  }
  for (size_t i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }
}

TEST(AddressSanitizer, ReallocTest) {
  const int kMinElem = 5;
  int *ptr = (int*)malloc(sizeof(int) * kMinElem);
  ptr[3] = 3;
  for (int i = 0; i < 10000; i++) {
    ptr = (int*)realloc(ptr,
        (my_rand(&global_seed) % 1000 + kMinElem) * sizeof(int));
    EXPECT_EQ(3, ptr[3]);
  }
}

#ifndef __APPLE__
static const char *kMallocUsableSizeErrorMsg =
  "AddressSanitizer: attempting to call malloc_usable_size()";

TEST(AddressSanitizer, MallocUsableSizeTest) {
  const size_t kArraySize = 100;
  char *array = Ident((char*)malloc(kArraySize));
  int *int_ptr = Ident(new int);
  EXPECT_EQ(0U, malloc_usable_size(NULL));
  EXPECT_EQ(kArraySize, malloc_usable_size(array));
  EXPECT_EQ(sizeof(int), malloc_usable_size(int_ptr));
  EXPECT_DEATH(malloc_usable_size((void*)0x123), kMallocUsableSizeErrorMsg);
  EXPECT_DEATH(malloc_usable_size(array + kArraySize / 2),
               kMallocUsableSizeErrorMsg);
  free(array);
  EXPECT_DEATH(malloc_usable_size(array), kMallocUsableSizeErrorMsg);
}
#endif

void WrongFree() {
  int *x = (int*)malloc(100 * sizeof(int));
  // Use the allocated memory, otherwise Clang will optimize it out.
  Ident(x);
  free(x + 1);
}

TEST(AddressSanitizer, WrongFreeTest) {
  EXPECT_DEATH(WrongFree(),
               "ERROR: AddressSanitizer: attempting free.*not malloc");
}

void DoubleFree() {
  int *x = (int*)malloc(100 * sizeof(int));
  fprintf(stderr, "DoubleFree: x=%p\n", x);
  free(x);
  free(x);
  fprintf(stderr, "should have failed in the second free(%p)\n", x);
  abort();
}

TEST(AddressSanitizer, DoubleFreeTest) {
  EXPECT_DEATH(DoubleFree(), ASAN_PCRE_DOTALL
               "ERROR: AddressSanitizer: attempting double-free"
               ".*is located 0 bytes inside of 400-byte region"
               ".*freed by thread T0 here"
               ".*previously allocated by thread T0 here");
}

template<int kSize>
NOINLINE void SizedStackTest() {
  char a[kSize];
  char  *A = Ident((char*)&a);
  for (size_t i = 0; i < kSize; i++)
    A[i] = i;
  EXPECT_DEATH(A[-1] = 0, "");
  EXPECT_DEATH(A[-20] = 0, "");
  EXPECT_DEATH(A[-31] = 0, "");
  EXPECT_DEATH(A[kSize] = 0, "");
  EXPECT_DEATH(A[kSize + 1] = 0, "");
  EXPECT_DEATH(A[kSize + 10] = 0, "");
  EXPECT_DEATH(A[kSize + 31] = 0, "");
}

TEST(AddressSanitizer, SimpleStackTest) {
  SizedStackTest<1>();
  SizedStackTest<2>();
  SizedStackTest<3>();
  SizedStackTest<4>();
  SizedStackTest<5>();
  SizedStackTest<6>();
  SizedStackTest<7>();
  SizedStackTest<16>();
  SizedStackTest<25>();
  SizedStackTest<34>();
  SizedStackTest<43>();
  SizedStackTest<51>();
  SizedStackTest<62>();
  SizedStackTest<64>();
  SizedStackTest<128>();
}

TEST(AddressSanitizer, ManyStackObjectsTest) {
  char XXX[10];
  char YYY[20];
  char ZZZ[30];
  Ident(XXX);
  Ident(YYY);
  EXPECT_DEATH(Ident(ZZZ)[-1] = 0, ASAN_PCRE_DOTALL "XXX.*YYY.*ZZZ");
}

NOINLINE static void Frame0(int frame, char *a, char *b, char *c) {
  char d[4] = {0};
  char *D = Ident(d);
  switch (frame) {
    case 3: a[5]++; break;
    case 2: b[5]++; break;
    case 1: c[5]++; break;
    case 0: D[5]++; break;
  }
}
NOINLINE static void Frame1(int frame, char *a, char *b) {
  char c[4] = {0}; Frame0(frame, a, b, c);
  break_optimization(0);
}
NOINLINE static void Frame2(int frame, char *a) {
  char b[4] = {0}; Frame1(frame, a, b);
  break_optimization(0);
}
NOINLINE static void Frame3(int frame) {
  char a[4] = {0}; Frame2(frame, a);
  break_optimization(0);
}

TEST(AddressSanitizer, GuiltyStackFrame0Test) {
  EXPECT_DEATH(Frame3(0), "located .*in frame <.*Frame0");
}
TEST(AddressSanitizer, GuiltyStackFrame1Test) {
  EXPECT_DEATH(Frame3(1), "located .*in frame <.*Frame1");
}
TEST(AddressSanitizer, GuiltyStackFrame2Test) {
  EXPECT_DEATH(Frame3(2), "located .*in frame <.*Frame2");
}
TEST(AddressSanitizer, GuiltyStackFrame3Test) {
  EXPECT_DEATH(Frame3(3), "located .*in frame <.*Frame3");
}

NOINLINE void LongJmpFunc1(jmp_buf buf) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  longjmp(buf, 1);
}

NOINLINE void BuiltinLongJmpFunc1(jmp_buf buf) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  __builtin_longjmp((void**)buf, 1);
}

NOINLINE void UnderscopeLongJmpFunc1(jmp_buf buf) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  _longjmp(buf, 1);
}

NOINLINE void SigLongJmpFunc1(sigjmp_buf buf) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  siglongjmp(buf, 1);
}


NOINLINE void TouchStackFunc() {
  int a[100];  // long array will intersect with redzones from LongJmpFunc1.
  int *A = Ident(a);
  for (int i = 0; i < 100; i++)
    A[i] = i*i;
}

// Test that we handle longjmp and do not report fals positives on stack.
TEST(AddressSanitizer, LongJmpTest) {
  static jmp_buf buf;
  if (!setjmp(buf)) {
    LongJmpFunc1(buf);
  } else {
    TouchStackFunc();
  }
}

TEST(AddressSanitizer, BuiltinLongJmpTest) {
  static jmp_buf buf;
  if (!__builtin_setjmp((void**)buf)) {
    BuiltinLongJmpFunc1(buf);
  } else {
    TouchStackFunc();
  }
}

TEST(AddressSanitizer, UnderscopeLongJmpTest) {
  static jmp_buf buf;
  if (!_setjmp(buf)) {
    UnderscopeLongJmpFunc1(buf);
  } else {
    TouchStackFunc();
  }
}

TEST(AddressSanitizer, SigLongJmpTest) {
  static sigjmp_buf buf;
  if (!sigsetjmp(buf, 1)) {
    SigLongJmpFunc1(buf);
  } else {
    TouchStackFunc();
  }
}

#ifdef __EXCEPTIONS
NOINLINE void ThrowFunc() {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  ASAN_THROW(1);
}

TEST(AddressSanitizer, CxxExceptionTest) {
  if (ASAN_UAR) return;
  // TODO(kcc): this test crashes on 32-bit for some reason...
  if (SANITIZER_WORDSIZE == 32) return;
  try {
    ThrowFunc();
  } catch(...) {}
  TouchStackFunc();
}
#endif

void *ThreadStackReuseFunc1(void *unused) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  pthread_exit(0);
  return 0;
}

void *ThreadStackReuseFunc2(void *unused) {
  TouchStackFunc();
  return 0;
}

TEST(AddressSanitizer, ThreadStackReuseTest) {
  pthread_t t;
  pthread_create(&t, 0, ThreadStackReuseFunc1, 0);
  pthread_join(t, 0);
  pthread_create(&t, 0, ThreadStackReuseFunc2, 0);
  pthread_join(t, 0);
}

#if defined(__i386__) || defined(__x86_64__)
TEST(AddressSanitizer, Store128Test) {
  char *a = Ident((char*)malloc(Ident(12)));
  char *p = a;
  if (((uintptr_t)a % 16) != 0)
    p = a + 8;
  assert(((uintptr_t)p % 16) == 0);
  __m128i value_wide = _mm_set1_epi16(0x1234);
  EXPECT_DEATH(_mm_store_si128((__m128i*)p, value_wide),
               "AddressSanitizer: heap-buffer-overflow");
  EXPECT_DEATH(_mm_store_si128((__m128i*)p, value_wide),
               "WRITE of size 16");
  EXPECT_DEATH(_mm_store_si128((__m128i*)p, value_wide),
               "located 0 bytes to the right of 12-byte");
  free(a);
}
#endif

static string RightOOBErrorMessage(int oob_distance) {
  assert(oob_distance >= 0);
  char expected_str[100];
  sprintf(expected_str, "located %d bytes to the right", oob_distance);
  return string(expected_str);
}

static string LeftOOBErrorMessage(int oob_distance) {
  assert(oob_distance > 0);
  char expected_str[100];
  sprintf(expected_str, "located %d bytes to the left", oob_distance);
  return string(expected_str);
}

template<typename T>
void MemSetOOBTestTemplate(size_t length) {
  if (length == 0) return;
  size_t size = Ident(sizeof(T) * length);
  T *array = Ident((T*)malloc(size));
  int element = Ident(42);
  int zero = Ident(0);
  // memset interval inside array
  memset(array, element, size);
  memset(array, element, size - 1);
  memset(array + length - 1, element, sizeof(T));
  memset(array, element, 1);

  // memset 0 bytes
  memset(array - 10, element, zero);
  memset(array - 1, element, zero);
  memset(array, element, zero);
  memset(array + length, 0, zero);
  memset(array + length + 1, 0, zero);

  // try to memset bytes to the right of array
  EXPECT_DEATH(memset(array, 0, size + 1),
               RightOOBErrorMessage(0));
  EXPECT_DEATH(memset((char*)(array + length) - 1, element, 6),
               RightOOBErrorMessage(4));
  EXPECT_DEATH(memset(array + 1, element, size + sizeof(T)),
               RightOOBErrorMessage(2 * sizeof(T) - 1));
  // whole interval is to the right
  EXPECT_DEATH(memset(array + length + 1, 0, 10),
               RightOOBErrorMessage(sizeof(T)));

  // try to memset bytes to the left of array
  EXPECT_DEATH(memset((char*)array - 1, element, size),
               LeftOOBErrorMessage(1));
  EXPECT_DEATH(memset((char*)array - 5, 0, 6),
               LeftOOBErrorMessage(5));
  EXPECT_DEATH(memset(array - 5, element, size + 5 * sizeof(T)),
               LeftOOBErrorMessage(5 * sizeof(T)));
  // whole interval is to the left
  EXPECT_DEATH(memset(array - 2, 0, sizeof(T)),
               LeftOOBErrorMessage(2 * sizeof(T)));

  // try to memset bytes both to the left & to the right
  EXPECT_DEATH(memset((char*)array - 2, element, size + 4),
               LeftOOBErrorMessage(2));

  free(array);
}

TEST(AddressSanitizer, MemSetOOBTest) {
  MemSetOOBTestTemplate<char>(100);
  MemSetOOBTestTemplate<int>(5);
  MemSetOOBTestTemplate<double>(256);
  // We can test arrays of structres/classes here, but what for?
}

// Same test for memcpy and memmove functions
template <typename T, class M>
void MemTransferOOBTestTemplate(size_t length) {
  if (length == 0) return;
  size_t size = Ident(sizeof(T) * length);
  T *src = Ident((T*)malloc(size));
  T *dest = Ident((T*)malloc(size));
  int zero = Ident(0);

  // valid transfer of bytes between arrays
  M::transfer(dest, src, size);
  M::transfer(dest + 1, src, size - sizeof(T));
  M::transfer(dest, src + length - 1, sizeof(T));
  M::transfer(dest, src, 1);

  // transfer zero bytes
  M::transfer(dest - 1, src, 0);
  M::transfer(dest + length, src, zero);
  M::transfer(dest, src - 1, zero);
  M::transfer(dest, src, zero);

  // try to change mem to the right of dest
  EXPECT_DEATH(M::transfer(dest + 1, src, size),
               RightOOBErrorMessage(sizeof(T) - 1));
  EXPECT_DEATH(M::transfer((char*)(dest + length) - 1, src, 5),
               RightOOBErrorMessage(3));

  // try to change mem to the left of dest
  EXPECT_DEATH(M::transfer(dest - 2, src, size),
               LeftOOBErrorMessage(2 * sizeof(T)));
  EXPECT_DEATH(M::transfer((char*)dest - 3, src, 4),
               LeftOOBErrorMessage(3));

  // try to access mem to the right of src
  EXPECT_DEATH(M::transfer(dest, src + 2, size),
               RightOOBErrorMessage(2 * sizeof(T) - 1));
  EXPECT_DEATH(M::transfer(dest, (char*)(src + length) - 3, 6),
               RightOOBErrorMessage(2));

  // try to access mem to the left of src
  EXPECT_DEATH(M::transfer(dest, src - 1, size),
               LeftOOBErrorMessage(sizeof(T)));
  EXPECT_DEATH(M::transfer(dest, (char*)src - 6, 7),
               LeftOOBErrorMessage(6));

  // Generally we don't need to test cases where both accessing src and writing
  // to dest address to poisoned memory.

  T *big_src = Ident((T*)malloc(size * 2));
  T *big_dest = Ident((T*)malloc(size * 2));
  // try to change mem to both sides of dest
  EXPECT_DEATH(M::transfer(dest - 1, big_src, size * 2),
               LeftOOBErrorMessage(sizeof(T)));
  // try to access mem to both sides of src
  EXPECT_DEATH(M::transfer(big_dest, src - 2, size * 2),
               LeftOOBErrorMessage(2 * sizeof(T)));

  free(src);
  free(dest);
  free(big_src);
  free(big_dest);
}

class MemCpyWrapper {
 public:
  static void* transfer(void *to, const void *from, size_t size) {
    return memcpy(to, from, size);
  }
};
TEST(AddressSanitizer, MemCpyOOBTest) {
  MemTransferOOBTestTemplate<char, MemCpyWrapper>(100);
  MemTransferOOBTestTemplate<int, MemCpyWrapper>(1024);
}

class MemMoveWrapper {
 public:
  static void* transfer(void *to, const void *from, size_t size) {
    return memmove(to, from, size);
  }
};
TEST(AddressSanitizer, MemMoveOOBTest) {
  MemTransferOOBTestTemplate<char, MemMoveWrapper>(100);
  MemTransferOOBTestTemplate<int, MemMoveWrapper>(1024);
}

// Tests for string functions

// Used for string functions tests
static char global_string[] = "global";
static size_t global_string_length = 6;

// Input to a test is a zero-terminated string str with given length
// Accesses to the bytes to the left and to the right of str
// are presumed to produce OOB errors
void StrLenOOBTestTemplate(char *str, size_t length, bool is_global) {
  // Normal strlen calls
  EXPECT_EQ(strlen(str), length);
  if (length > 0) {
    EXPECT_EQ(length - 1, strlen(str + 1));
    EXPECT_EQ(0U, strlen(str + length));
  }
  // Arg of strlen is not malloced, OOB access
  if (!is_global) {
    // We don't insert RedZones to the left of global variables
    EXPECT_DEATH(Ident(strlen(str - 1)), LeftOOBErrorMessage(1));
    EXPECT_DEATH(Ident(strlen(str - 5)), LeftOOBErrorMessage(5));
  }
  EXPECT_DEATH(Ident(strlen(str + length + 1)), RightOOBErrorMessage(0));
  // Overwrite terminator
  str[length] = 'a';
  // String is not zero-terminated, strlen will lead to OOB access
  EXPECT_DEATH(Ident(strlen(str)), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strlen(str + length)), RightOOBErrorMessage(0));
  // Restore terminator
  str[length] = 0;
}
TEST(AddressSanitizer, StrLenOOBTest) {
  // Check heap-allocated string
  size_t length = Ident(10);
  char *heap_string = Ident((char*)malloc(length + 1));
  char stack_string[10 + 1];
  for (size_t i = 0; i < length; i++) {
    heap_string[i] = 'a';
    stack_string[i] = 'b';
  }
  heap_string[length] = 0;
  stack_string[length] = 0;
  StrLenOOBTestTemplate(heap_string, length, false);
  // TODO(samsonov): Fix expected messages in StrLenOOBTestTemplate to
  //      make test for stack_string work. Or move it to output tests.
  // StrLenOOBTestTemplate(stack_string, length, false);
  StrLenOOBTestTemplate(global_string, global_string_length, true);
  free(heap_string);
}

static inline char* MallocAndMemsetString(size_t size, char ch) {
  char *s = Ident((char*)malloc(size));
  memset(s, ch, size);
  return s;
}
static inline char* MallocAndMemsetString(size_t size) {
  return MallocAndMemsetString(size, 'z');
}

#ifndef __APPLE__
TEST(AddressSanitizer, StrNLenOOBTest) {
  size_t size = Ident(123);
  char *str = MallocAndMemsetString(size);
  // Normal strnlen calls.
  Ident(strnlen(str - 1, 0));
  Ident(strnlen(str, size));
  Ident(strnlen(str + size - 1, 1));
  str[size - 1] = '\0';
  Ident(strnlen(str, 2 * size));
  // Argument points to not allocated memory.
  EXPECT_DEATH(Ident(strnlen(str - 1, 1)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strnlen(str + size, 1)), RightOOBErrorMessage(0));
  // Overwrite the terminating '\0' and hit unallocated memory.
  str[size - 1] = 'z';
  EXPECT_DEATH(Ident(strnlen(str, size + 1)), RightOOBErrorMessage(0));
  free(str);
}
#endif

TEST(AddressSanitizer, StrDupOOBTest) {
  size_t size = Ident(42);
  char *str = MallocAndMemsetString(size);
  char *new_str;
  // Normal strdup calls.
  str[size - 1] = '\0';
  new_str = strdup(str);
  free(new_str);
  new_str = strdup(str + size - 1);
  free(new_str);
  // Argument points to not allocated memory.
  EXPECT_DEATH(Ident(strdup(str - 1)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strdup(str + size)), RightOOBErrorMessage(0));
  // Overwrite the terminating '\0' and hit unallocated memory.
  str[size - 1] = 'z';
  EXPECT_DEATH(Ident(strdup(str)), RightOOBErrorMessage(0));
  free(str);
}

TEST(AddressSanitizer, StrCpyOOBTest) {
  size_t to_size = Ident(30);
  size_t from_size = Ident(6);  // less than to_size
  char *to = Ident((char*)malloc(to_size));
  char *from = Ident((char*)malloc(from_size));
  // Normal strcpy calls.
  strcpy(from, "hello");
  strcpy(to, from);
  strcpy(to + to_size - from_size, from);
  // Length of "from" is too small.
  EXPECT_DEATH(Ident(strcpy(from, "hello2")), RightOOBErrorMessage(0));
  // "to" or "from" points to not allocated memory.
  EXPECT_DEATH(Ident(strcpy(to - 1, from)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strcpy(to, from - 1)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strcpy(to, from + from_size)), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strcpy(to + to_size, from)), RightOOBErrorMessage(0));
  // Overwrite the terminating '\0' character and hit unallocated memory.
  from[from_size - 1] = '!';
  EXPECT_DEATH(Ident(strcpy(to, from)), RightOOBErrorMessage(0));
  free(to);
  free(from);
}

TEST(AddressSanitizer, StrNCpyOOBTest) {
  size_t to_size = Ident(20);
  size_t from_size = Ident(6);  // less than to_size
  char *to = Ident((char*)malloc(to_size));
  // From is a zero-terminated string "hello\0" of length 6
  char *from = Ident((char*)malloc(from_size));
  strcpy(from, "hello");
  // copy 0 bytes
  strncpy(to, from, 0);
  strncpy(to - 1, from - 1, 0);
  // normal strncpy calls
  strncpy(to, from, from_size);
  strncpy(to, from, to_size);
  strncpy(to, from + from_size - 1, to_size);
  strncpy(to + to_size - 1, from, 1);
  // One of {to, from} points to not allocated memory
  EXPECT_DEATH(Ident(strncpy(to, from - 1, from_size)),
               LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strncpy(to - 1, from, from_size)),
               LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strncpy(to, from + from_size, 1)),
               RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strncpy(to + to_size, from, 1)),
               RightOOBErrorMessage(0));
  // Length of "to" is too small
  EXPECT_DEATH(Ident(strncpy(to + to_size - from_size + 1, from, from_size)),
               RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strncpy(to + 1, from, to_size)),
               RightOOBErrorMessage(0));
  // Overwrite terminator in from
  from[from_size - 1] = '!';
  // normal strncpy call
  strncpy(to, from, from_size);
  // Length of "from" is too small
  EXPECT_DEATH(Ident(strncpy(to, from, to_size)),
               RightOOBErrorMessage(0));
  free(to);
  free(from);
}

// Users may have different definitions of "strchr" and "index", so provide
// function pointer typedefs and overload RunStrChrTest implementation.
// We can't use macro for RunStrChrTest body here, as this macro would
// confuse EXPECT_DEATH gtest macro.
typedef char*(*PointerToStrChr1)(const char*, int);
typedef char*(*PointerToStrChr2)(char*, int);

USED static void RunStrChrTest(PointerToStrChr1 StrChr) {
  size_t size = Ident(100);
  char *str = MallocAndMemsetString(size);
  str[10] = 'q';
  str[11] = '\0';
  EXPECT_EQ(str, StrChr(str, 'z'));
  EXPECT_EQ(str + 10, StrChr(str, 'q'));
  EXPECT_EQ(NULL, StrChr(str, 'a'));
  // StrChr argument points to not allocated memory.
  EXPECT_DEATH(Ident(StrChr(str - 1, 'z')), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(StrChr(str + size, 'z')), RightOOBErrorMessage(0));
  // Overwrite the terminator and hit not allocated memory.
  str[11] = 'z';
  EXPECT_DEATH(Ident(StrChr(str, 'a')), RightOOBErrorMessage(0));
  free(str);
}
USED static void RunStrChrTest(PointerToStrChr2 StrChr) {
  size_t size = Ident(100);
  char *str = MallocAndMemsetString(size);
  str[10] = 'q';
  str[11] = '\0';
  EXPECT_EQ(str, StrChr(str, 'z'));
  EXPECT_EQ(str + 10, StrChr(str, 'q'));
  EXPECT_EQ(NULL, StrChr(str, 'a'));
  // StrChr argument points to not allocated memory.
  EXPECT_DEATH(Ident(StrChr(str - 1, 'z')), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(StrChr(str + size, 'z')), RightOOBErrorMessage(0));
  // Overwrite the terminator and hit not allocated memory.
  str[11] = 'z';
  EXPECT_DEATH(Ident(StrChr(str, 'a')), RightOOBErrorMessage(0));
  free(str);
}

TEST(AddressSanitizer, StrChrAndIndexOOBTest) {
  RunStrChrTest(&strchr);
  RunStrChrTest(&index);
}

TEST(AddressSanitizer, StrCmpAndFriendsLogicTest) {
  // strcmp
  EXPECT_EQ(0, strcmp("", ""));
  EXPECT_EQ(0, strcmp("abcd", "abcd"));
  EXPECT_GT(0, strcmp("ab", "ac"));
  EXPECT_GT(0, strcmp("abc", "abcd"));
  EXPECT_LT(0, strcmp("acc", "abc"));
  EXPECT_LT(0, strcmp("abcd", "abc"));

  // strncmp
  EXPECT_EQ(0, strncmp("a", "b", 0));
  EXPECT_EQ(0, strncmp("abcd", "abcd", 10));
  EXPECT_EQ(0, strncmp("abcd", "abcef", 3));
  EXPECT_GT(0, strncmp("abcde", "abcfa", 4));
  EXPECT_GT(0, strncmp("a", "b", 5));
  EXPECT_GT(0, strncmp("bc", "bcde", 4));
  EXPECT_LT(0, strncmp("xyz", "xyy", 10));
  EXPECT_LT(0, strncmp("baa", "aaa", 1));
  EXPECT_LT(0, strncmp("zyx", "", 2));

  // strcasecmp
  EXPECT_EQ(0, strcasecmp("", ""));
  EXPECT_EQ(0, strcasecmp("zzz", "zzz"));
  EXPECT_EQ(0, strcasecmp("abCD", "ABcd"));
  EXPECT_GT(0, strcasecmp("aB", "Ac"));
  EXPECT_GT(0, strcasecmp("ABC", "ABCd"));
  EXPECT_LT(0, strcasecmp("acc", "abc"));
  EXPECT_LT(0, strcasecmp("ABCd", "abc"));

  // strncasecmp
  EXPECT_EQ(0, strncasecmp("a", "b", 0));
  EXPECT_EQ(0, strncasecmp("abCD", "ABcd", 10));
  EXPECT_EQ(0, strncasecmp("abCd", "ABcef", 3));
  EXPECT_GT(0, strncasecmp("abcde", "ABCfa", 4));
  EXPECT_GT(0, strncasecmp("a", "B", 5));
  EXPECT_GT(0, strncasecmp("bc", "BCde", 4));
  EXPECT_LT(0, strncasecmp("xyz", "xyy", 10));
  EXPECT_LT(0, strncasecmp("Baa", "aaa", 1));
  EXPECT_LT(0, strncasecmp("zyx", "", 2));

  // memcmp
  EXPECT_EQ(0, memcmp("a", "b", 0));
  EXPECT_EQ(0, memcmp("ab\0c", "ab\0c", 4));
  EXPECT_GT(0, memcmp("\0ab", "\0ac", 3));
  EXPECT_GT(0, memcmp("abb\0", "abba", 4));
  EXPECT_LT(0, memcmp("ab\0cd", "ab\0c\0", 5));
  EXPECT_LT(0, memcmp("zza", "zyx", 3));
}

typedef int(*PointerToStrCmp)(const char*, const char*);
void RunStrCmpTest(PointerToStrCmp StrCmp) {
  size_t size = Ident(100);
  char *s1 = MallocAndMemsetString(size);
  char *s2 = MallocAndMemsetString(size);
  s1[size - 1] = '\0';
  s2[size - 1] = '\0';
  // Normal StrCmp calls
  Ident(StrCmp(s1, s2));
  Ident(StrCmp(s1, s2 + size - 1));
  Ident(StrCmp(s1 + size - 1, s2 + size - 1));
  s1[size - 1] = 'z';
  s2[size - 1] = 'x';
  Ident(StrCmp(s1, s2));
  // One of arguments points to not allocated memory.
  EXPECT_DEATH(Ident(StrCmp)(s1 - 1, s2), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(StrCmp)(s1, s2 - 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(StrCmp)(s1 + size, s2), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(StrCmp)(s1, s2 + size), RightOOBErrorMessage(0));
  // Hit unallocated memory and die.
  s2[size - 1] = 'z';
  EXPECT_DEATH(Ident(StrCmp)(s1, s1), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(StrCmp)(s1 + size - 1, s2), RightOOBErrorMessage(0));
  free(s1);
  free(s2);
}

TEST(AddressSanitizer, StrCmpOOBTest) {
  RunStrCmpTest(&strcmp);
}

TEST(AddressSanitizer, StrCaseCmpOOBTest) {
  RunStrCmpTest(&strcasecmp);
}

typedef int(*PointerToStrNCmp)(const char*, const char*, size_t);
void RunStrNCmpTest(PointerToStrNCmp StrNCmp) {
  size_t size = Ident(100);
  char *s1 = MallocAndMemsetString(size);
  char *s2 = MallocAndMemsetString(size);
  s1[size - 1] = '\0';
  s2[size - 1] = '\0';
  // Normal StrNCmp calls
  Ident(StrNCmp(s1, s2, size + 2));
  s1[size - 1] = 'z';
  s2[size - 1] = 'x';
  Ident(StrNCmp(s1 + size - 2, s2 + size - 2, size));
  s2[size - 1] = 'z';
  Ident(StrNCmp(s1 - 1, s2 - 1, 0));
  Ident(StrNCmp(s1 + size - 1, s2 + size - 1, 1));
  // One of arguments points to not allocated memory.
  EXPECT_DEATH(Ident(StrNCmp)(s1 - 1, s2, 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(StrNCmp)(s1, s2 - 1, 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(StrNCmp)(s1 + size, s2, 1), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(StrNCmp)(s1, s2 + size, 1), RightOOBErrorMessage(0));
  // Hit unallocated memory and die.
  EXPECT_DEATH(Ident(StrNCmp)(s1 + 1, s2 + 1, size), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(StrNCmp)(s1 + size - 1, s2, 2), RightOOBErrorMessage(0));
  free(s1);
  free(s2);
}

TEST(AddressSanitizer, StrNCmpOOBTest) {
  RunStrNCmpTest(&strncmp);
}

TEST(AddressSanitizer, StrNCaseCmpOOBTest) {
  RunStrNCmpTest(&strncasecmp);
}

TEST(AddressSanitizer, MemCmpOOBTest) {
  size_t size = Ident(100);
  char *s1 = MallocAndMemsetString(size);
  char *s2 = MallocAndMemsetString(size);
  // Normal memcmp calls.
  Ident(memcmp(s1, s2, size));
  Ident(memcmp(s1 + size - 1, s2 + size - 1, 1));
  Ident(memcmp(s1 - 1, s2 - 1, 0));
  // One of arguments points to not allocated memory.
  EXPECT_DEATH(Ident(memcmp)(s1 - 1, s2, 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(memcmp)(s1, s2 - 1, 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(memcmp)(s1 + size, s2, 1), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(memcmp)(s1, s2 + size, 1), RightOOBErrorMessage(0));
  // Hit unallocated memory and die.
  EXPECT_DEATH(Ident(memcmp)(s1 + 1, s2 + 1, size), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(memcmp)(s1 + size - 1, s2, 2), RightOOBErrorMessage(0));
  // Zero bytes are not terminators and don't prevent from OOB.
  s1[size - 1] = '\0';
  s2[size - 1] = '\0';
  EXPECT_DEATH(Ident(memcmp)(s1, s2, size + 1), RightOOBErrorMessage(0));
  free(s1);
  free(s2);
}

TEST(AddressSanitizer, StrCatOOBTest) {
  size_t to_size = Ident(100);
  char *to = MallocAndMemsetString(to_size);
  to[0] = '\0';
  size_t from_size = Ident(20);
  char *from = MallocAndMemsetString(from_size);
  from[from_size - 1] = '\0';
  // Normal strcat calls.
  strcat(to, from);
  strcat(to, from);
  strcat(to + from_size, from + from_size - 2);
  // Passing an invalid pointer is an error even when concatenating an empty
  // string.
  EXPECT_DEATH(strcat(to - 1, from + from_size - 1), LeftOOBErrorMessage(1));
  // One of arguments points to not allocated memory.
  EXPECT_DEATH(strcat(to - 1, from), LeftOOBErrorMessage(1));
  EXPECT_DEATH(strcat(to, from - 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(strcat(to + to_size, from), RightOOBErrorMessage(0));
  EXPECT_DEATH(strcat(to, from + from_size), RightOOBErrorMessage(0));

  // "from" is not zero-terminated.
  from[from_size - 1] = 'z';
  EXPECT_DEATH(strcat(to, from), RightOOBErrorMessage(0));
  from[from_size - 1] = '\0';
  // "to" is not zero-terminated.
  memset(to, 'z', to_size);
  EXPECT_DEATH(strcat(to, from), RightOOBErrorMessage(0));
  // "to" is too short to fit "from".
  to[to_size - from_size + 1] = '\0';
  EXPECT_DEATH(strcat(to, from), RightOOBErrorMessage(0));
  // length of "to" is just enough.
  strcat(to, from + 1);

  free(to);
  free(from);
}

TEST(AddressSanitizer, StrNCatOOBTest) {
  size_t to_size = Ident(100);
  char *to = MallocAndMemsetString(to_size);
  to[0] = '\0';
  size_t from_size = Ident(20);
  char *from = MallocAndMemsetString(from_size);
  // Normal strncat calls.
  strncat(to, from, 0);
  strncat(to, from, from_size);
  from[from_size - 1] = '\0';
  strncat(to, from, 2 * from_size);
  // Catenating empty string with an invalid string is still an error.
  EXPECT_DEATH(strncat(to - 1, from, 0), LeftOOBErrorMessage(1));
  strncat(to, from + from_size - 1, 10);
  // One of arguments points to not allocated memory.
  EXPECT_DEATH(strncat(to - 1, from, 2), LeftOOBErrorMessage(1));
  EXPECT_DEATH(strncat(to, from - 1, 2), LeftOOBErrorMessage(1));
  EXPECT_DEATH(strncat(to + to_size, from, 2), RightOOBErrorMessage(0));
  EXPECT_DEATH(strncat(to, from + from_size, 2), RightOOBErrorMessage(0));

  memset(from, 'z', from_size);
  memset(to, 'z', to_size);
  to[0] = '\0';
  // "from" is too short.
  EXPECT_DEATH(strncat(to, from, from_size + 1), RightOOBErrorMessage(0));
  // "to" is not zero-terminated.
  EXPECT_DEATH(strncat(to + 1, from, 1), RightOOBErrorMessage(0));
  // "to" is too short to fit "from".
  to[0] = 'z';
  to[to_size - from_size + 1] = '\0';
  EXPECT_DEATH(strncat(to, from, from_size - 1), RightOOBErrorMessage(0));
  // "to" is just enough.
  strncat(to, from, from_size - 2);

  free(to);
  free(from);
}

static string OverlapErrorMessage(const string &func) {
  return func + "-param-overlap";
}

TEST(AddressSanitizer, StrArgsOverlapTest) {
  size_t size = Ident(100);
  char *str = Ident((char*)malloc(size));

// Do not check memcpy() on OS X 10.7 and later, where it actually aliases
// memmove().
#if !defined(__APPLE__) || !defined(MAC_OS_X_VERSION_10_7) || \
    (MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_7)
  // Check "memcpy". Use Ident() to avoid inlining.
  memset(str, 'z', size);
  Ident(memcpy)(str + 1, str + 11, 10);
  Ident(memcpy)(str, str, 0);
  EXPECT_DEATH(Ident(memcpy)(str, str + 14, 15), OverlapErrorMessage("memcpy"));
  EXPECT_DEATH(Ident(memcpy)(str + 14, str, 15), OverlapErrorMessage("memcpy"));
#endif

  // We do not treat memcpy with to==from as a bug.
  // See http://llvm.org/bugs/show_bug.cgi?id=11763.
  // EXPECT_DEATH(Ident(memcpy)(str + 20, str + 20, 1),
  //              OverlapErrorMessage("memcpy"));

  // Check "strcpy".
  memset(str, 'z', size);
  str[9] = '\0';
  strcpy(str + 10, str);
  EXPECT_DEATH(strcpy(str + 9, str), OverlapErrorMessage("strcpy"));
  EXPECT_DEATH(strcpy(str, str + 4), OverlapErrorMessage("strcpy"));
  strcpy(str, str + 5);

  // Check "strncpy".
  memset(str, 'z', size);
  strncpy(str, str + 10, 10);
  EXPECT_DEATH(strncpy(str, str + 9, 10), OverlapErrorMessage("strncpy"));
  EXPECT_DEATH(strncpy(str + 9, str, 10), OverlapErrorMessage("strncpy"));
  str[10] = '\0';
  strncpy(str + 11, str, 20);
  EXPECT_DEATH(strncpy(str + 10, str, 20), OverlapErrorMessage("strncpy"));

  // Check "strcat".
  memset(str, 'z', size);
  str[10] = '\0';
  str[20] = '\0';
  strcat(str, str + 10);
  EXPECT_DEATH(strcat(str, str + 11), OverlapErrorMessage("strcat"));
  str[10] = '\0';
  strcat(str + 11, str);
  EXPECT_DEATH(strcat(str, str + 9), OverlapErrorMessage("strcat"));
  EXPECT_DEATH(strcat(str + 9, str), OverlapErrorMessage("strcat"));
  EXPECT_DEATH(strcat(str + 10, str), OverlapErrorMessage("strcat"));

  // Check "strncat".
  memset(str, 'z', size);
  str[10] = '\0';
  strncat(str, str + 10, 10);  // from is empty
  EXPECT_DEATH(strncat(str, str + 11, 10), OverlapErrorMessage("strncat"));
  str[10] = '\0';
  str[20] = '\0';
  strncat(str + 5, str, 5);
  str[10] = '\0';
  EXPECT_DEATH(strncat(str + 5, str, 6), OverlapErrorMessage("strncat"));
  EXPECT_DEATH(strncat(str, str + 9, 10), OverlapErrorMessage("strncat"));

  free(str);
}

void CallAtoi(const char *nptr) {
  Ident(atoi(nptr));
}
void CallAtol(const char *nptr) {
  Ident(atol(nptr));
}
void CallAtoll(const char *nptr) {
  Ident(atoll(nptr));
}
typedef void(*PointerToCallAtoi)(const char*);

void RunAtoiOOBTest(PointerToCallAtoi Atoi) {
  char *array = MallocAndMemsetString(10, '1');
  // Invalid pointer to the string.
  EXPECT_DEATH(Atoi(array + 11), RightOOBErrorMessage(1));
  EXPECT_DEATH(Atoi(array - 1), LeftOOBErrorMessage(1));
  // Die if a buffer doesn't have terminating NULL.
  EXPECT_DEATH(Atoi(array), RightOOBErrorMessage(0));
  // Make last symbol a terminating NULL or other non-digit.
  array[9] = '\0';
  Atoi(array);
  array[9] = 'a';
  Atoi(array);
  Atoi(array + 9);
  // Sometimes we need to detect overflow if no digits are found.
  memset(array, ' ', 10);
  EXPECT_DEATH(Atoi(array), RightOOBErrorMessage(0));
  array[9] = '-';
  EXPECT_DEATH(Atoi(array), RightOOBErrorMessage(0));
  EXPECT_DEATH(Atoi(array + 9), RightOOBErrorMessage(0));
  array[8] = '-';
  Atoi(array);
  delete array;
}

TEST(AddressSanitizer, AtoiAndFriendsOOBTest) {
  RunAtoiOOBTest(&CallAtoi);
  RunAtoiOOBTest(&CallAtol);
  RunAtoiOOBTest(&CallAtoll);
}

void CallStrtol(const char *nptr, char **endptr, int base) {
  Ident(strtol(nptr, endptr, base));
}
void CallStrtoll(const char *nptr, char **endptr, int base) {
  Ident(strtoll(nptr, endptr, base));
}
typedef void(*PointerToCallStrtol)(const char*, char**, int);

void RunStrtolOOBTest(PointerToCallStrtol Strtol) {
  char *array = MallocAndMemsetString(3);
  char *endptr = NULL;
  array[0] = '1';
  array[1] = '2';
  array[2] = '3';
  // Invalid pointer to the string.
  EXPECT_DEATH(Strtol(array + 3, NULL, 0), RightOOBErrorMessage(0));
  EXPECT_DEATH(Strtol(array - 1, NULL, 0), LeftOOBErrorMessage(1));
  // Buffer overflow if there is no terminating null (depends on base).
  Strtol(array, &endptr, 3);
  EXPECT_EQ(array + 2, endptr);
  EXPECT_DEATH(Strtol(array, NULL, 0), RightOOBErrorMessage(0));
  array[2] = 'z';
  Strtol(array, &endptr, 35);
  EXPECT_EQ(array + 2, endptr);
  EXPECT_DEATH(Strtol(array, NULL, 36), RightOOBErrorMessage(0));
  // Add terminating zero to get rid of overflow.
  array[2] = '\0';
  Strtol(array, NULL, 36);
  // Don't check for overflow if base is invalid.
  Strtol(array - 1, NULL, -1);
  Strtol(array + 3, NULL, 1);
  // Sometimes we need to detect overflow if no digits are found.
  array[0] = array[1] = array[2] = ' ';
  EXPECT_DEATH(Strtol(array, NULL, 0), RightOOBErrorMessage(0));
  array[2] = '+';
  EXPECT_DEATH(Strtol(array, NULL, 0), RightOOBErrorMessage(0));
  array[2] = '-';
  EXPECT_DEATH(Strtol(array, NULL, 0), RightOOBErrorMessage(0));
  array[1] = '+';
  Strtol(array, NULL, 0);
  array[1] = array[2] = 'z';
  Strtol(array, &endptr, 0);
  EXPECT_EQ(array, endptr);
  Strtol(array + 2, NULL, 0);
  EXPECT_EQ(array, endptr);
  delete array;
}

TEST(AddressSanitizer, StrtollOOBTest) {
  RunStrtolOOBTest(&CallStrtoll);
}
TEST(AddressSanitizer, StrtolOOBTest) {
  RunStrtolOOBTest(&CallStrtol);
}

// At the moment we instrument memcpy/memove/memset calls at compile time so we
// can't handle OOB error if these functions are called by pointer, see disabled
// MemIntrinsicCallByPointerTest below
typedef void*(*PointerToMemTransfer)(void*, const void*, size_t);
typedef void*(*PointerToMemSet)(void*, int, size_t);

void CallMemSetByPointer(PointerToMemSet MemSet) {
  size_t size = Ident(100);
  char *array = Ident((char*)malloc(size));
  EXPECT_DEATH(MemSet(array, 0, 101), RightOOBErrorMessage(0));
  free(array);
}

void CallMemTransferByPointer(PointerToMemTransfer MemTransfer) {
  size_t size = Ident(100);
  char *src = Ident((char*)malloc(size));
  char *dst = Ident((char*)malloc(size));
  EXPECT_DEATH(MemTransfer(dst, src, 101), RightOOBErrorMessage(0));
  free(src);
  free(dst);
}

TEST(AddressSanitizer, DISABLED_MemIntrinsicCallByPointerTest) {
  CallMemSetByPointer(&memset);
  CallMemTransferByPointer(&memcpy);
  CallMemTransferByPointer(&memmove);
}

// This test case fails
// Clang optimizes memcpy/memset calls which lead to unaligned access
TEST(AddressSanitizer, DISABLED_MemIntrinsicUnalignedAccessTest) {
  int size = Ident(4096);
  char *s = Ident((char*)malloc(size));
  EXPECT_DEATH(memset(s + size - 1, 0, 2), RightOOBErrorMessage(0));
  free(s);
}

// TODO(samsonov): Add a test with malloc(0)
// TODO(samsonov): Add tests for str* and mem* functions.

NOINLINE static int LargeFunction(bool do_bad_access) {
  int *x = new int[100];
  x[0]++;
  x[1]++;
  x[2]++;
  x[3]++;
  x[4]++;
  x[5]++;
  x[6]++;
  x[7]++;
  x[8]++;
  x[9]++;

  x[do_bad_access ? 100 : 0]++; int res = __LINE__;

  x[10]++;
  x[11]++;
  x[12]++;
  x[13]++;
  x[14]++;
  x[15]++;
  x[16]++;
  x[17]++;
  x[18]++;
  x[19]++;

  delete x;
  return res;
}

// Test the we have correct debug info for the failing instruction.
// This test requires the in-process symbolizer to be enabled by default.
TEST(AddressSanitizer, DISABLED_LargeFunctionSymbolizeTest) {
  int failing_line = LargeFunction(false);
  char expected_warning[128];
  sprintf(expected_warning, "LargeFunction.*asan_test.cc:%d", failing_line);
  EXPECT_DEATH(LargeFunction(true), expected_warning);
}

// Check that we unwind and symbolize correctly.
TEST(AddressSanitizer, DISABLED_MallocFreeUnwindAndSymbolizeTest) {
  int *a = (int*)malloc_aaa(sizeof(int));
  *a = 1;
  free_aaa(a);
  EXPECT_DEATH(*a = 1, "free_ccc.*free_bbb.*free_aaa.*"
               "malloc_fff.*malloc_eee.*malloc_ddd");
}

void *ThreadedTestAlloc(void *a) {
  int **p = (int**)a;
  *p = new int;
  return 0;
}

void *ThreadedTestFree(void *a) {
  int **p = (int**)a;
  delete *p;
  return 0;
}

void *ThreadedTestUse(void *a) {
  int **p = (int**)a;
  **p = 1;
  return 0;
}

void ThreadedTestSpawn() {
  pthread_t t;
  int *x;
  pthread_create(&t, 0, ThreadedTestAlloc, &x);
  pthread_join(t, 0);
  pthread_create(&t, 0, ThreadedTestFree, &x);
  pthread_join(t, 0);
  pthread_create(&t, 0, ThreadedTestUse, &x);
  pthread_join(t, 0);
}

TEST(AddressSanitizer, ThreadedTest) {
  EXPECT_DEATH(ThreadedTestSpawn(),
               ASAN_PCRE_DOTALL
               "Thread T.*created"
               ".*Thread T.*created"
               ".*Thread T.*created");
}

#if ASAN_NEEDS_SEGV
TEST(AddressSanitizer, ShadowGapTest) {
#if SANITIZER_WORDSIZE == 32
  char *addr = (char*)0x22000000;
#else
  char *addr = (char*)0x0000100000080000;
#endif
  EXPECT_DEATH(*addr = 1, "AddressSanitizer: SEGV on unknown");
}
#endif  // ASAN_NEEDS_SEGV

extern "C" {
NOINLINE static void UseThenFreeThenUse() {
  char *x = Ident((char*)malloc(8));
  *x = 1;
  free_aaa(x);
  *x = 2;
}
}

TEST(AddressSanitizer, UseThenFreeThenUseTest) {
  EXPECT_DEATH(UseThenFreeThenUse(), "freed by thread");
}

TEST(AddressSanitizer, StrDupTest) {
  free(strdup(Ident("123")));
}

// Currently we create and poison redzone at right of global variables.
char glob5[5];
static char static110[110];
const char ConstGlob[7] = {1, 2, 3, 4, 5, 6, 7};
static const char StaticConstGlob[3] = {9, 8, 7};
extern int GlobalsTest(int x);

TEST(AddressSanitizer, GlobalTest) {
  static char func_static15[15];

  static char fs1[10];
  static char fs2[10];
  static char fs3[10];

  glob5[Ident(0)] = 0;
  glob5[Ident(1)] = 0;
  glob5[Ident(2)] = 0;
  glob5[Ident(3)] = 0;
  glob5[Ident(4)] = 0;

  EXPECT_DEATH(glob5[Ident(5)] = 0,
               "0 bytes to the right of global variable.*glob5.* size 5");
  EXPECT_DEATH(glob5[Ident(5+6)] = 0,
               "6 bytes to the right of global variable.*glob5.* size 5");
  Ident(static110);  // avoid optimizations
  static110[Ident(0)] = 0;
  static110[Ident(109)] = 0;
  EXPECT_DEATH(static110[Ident(110)] = 0,
               "0 bytes to the right of global variable");
  EXPECT_DEATH(static110[Ident(110+7)] = 0,
               "7 bytes to the right of global variable");

  Ident(func_static15);  // avoid optimizations
  func_static15[Ident(0)] = 0;
  EXPECT_DEATH(func_static15[Ident(15)] = 0,
               "0 bytes to the right of global variable");
  EXPECT_DEATH(func_static15[Ident(15 + 9)] = 0,
               "9 bytes to the right of global variable");

  Ident(fs1);
  Ident(fs2);
  Ident(fs3);

  // We don't create left redzones, so this is not 100% guaranteed to fail.
  // But most likely will.
  EXPECT_DEATH(fs2[Ident(-1)] = 0, "is located.*of global variable");

  EXPECT_DEATH(Ident(Ident(ConstGlob)[8]),
               "is located 1 bytes to the right of .*ConstGlob");
  EXPECT_DEATH(Ident(Ident(StaticConstGlob)[5]),
               "is located 2 bytes to the right of .*StaticConstGlob");

  // call stuff from another file.
  GlobalsTest(0);
}

TEST(AddressSanitizer, GlobalStringConstTest) {
  static const char *zoo = "FOOBAR123";
  const char *p = Ident(zoo);
  EXPECT_DEATH(Ident(p[15]), "is ascii string 'FOOBAR123'");
}

TEST(AddressSanitizer, FileNameInGlobalReportTest) {
  static char zoo[10];
  const char *p = Ident(zoo);
  // The file name should be present in the report.
  EXPECT_DEATH(Ident(p[15]), "zoo.*asan_test.cc");
}

int *ReturnsPointerToALocalObject() {
  int a = 0;
  return Ident(&a);
}

#if ASAN_UAR == 1
TEST(AddressSanitizer, LocalReferenceReturnTest) {
  int *(*f)() = Ident(ReturnsPointerToALocalObject);
  int *p = f();
  // Call 'f' a few more times, 'p' should still be poisoned.
  for (int i = 0; i < 32; i++)
    f();
  EXPECT_DEATH(*p = 1, "AddressSanitizer: stack-use-after-return");
  EXPECT_DEATH(*p = 1, "is located.*in frame .*ReturnsPointerToALocal");
}
#endif

template <int kSize>
NOINLINE static void FuncWithStack() {
  char x[kSize];
  Ident(x)[0] = 0;
  Ident(x)[kSize-1] = 0;
}

static void LotsOfStackReuse() {
  int LargeStack[10000];
  Ident(LargeStack)[0] = 0;
  for (int i = 0; i < 10000; i++) {
    FuncWithStack<128 * 1>();
    FuncWithStack<128 * 2>();
    FuncWithStack<128 * 4>();
    FuncWithStack<128 * 8>();
    FuncWithStack<128 * 16>();
    FuncWithStack<128 * 32>();
    FuncWithStack<128 * 64>();
    FuncWithStack<128 * 128>();
    FuncWithStack<128 * 256>();
    FuncWithStack<128 * 512>();
    Ident(LargeStack)[0] = 0;
  }
}

TEST(AddressSanitizer, StressStackReuseTest) {
  LotsOfStackReuse();
}

TEST(AddressSanitizer, ThreadedStressStackReuseTest) {
  const int kNumThreads = 20;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, (void* (*)(void *x))LotsOfStackReuse, 0);
  }
  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }
}

static void *PthreadExit(void *a) {
  pthread_exit(0);
  return 0;
}

TEST(AddressSanitizer, PthreadExitTest) {
  pthread_t t;
  for (int i = 0; i < 1000; i++) {
    pthread_create(&t, 0, PthreadExit, 0);
    pthread_join(t, 0);
  }
}

#ifdef __EXCEPTIONS
NOINLINE static void StackReuseAndException() {
  int large_stack[1000];
  Ident(large_stack);
  ASAN_THROW(1);
}

// TODO(kcc): support exceptions with use-after-return.
TEST(AddressSanitizer, DISABLED_StressStackReuseAndExceptionsTest) {
  for (int i = 0; i < 10000; i++) {
    try {
    StackReuseAndException();
    } catch(...) {
    }
  }
}
#endif

TEST(AddressSanitizer, MlockTest) {
  EXPECT_EQ(0, mlockall(MCL_CURRENT));
  EXPECT_EQ(0, mlock((void*)0x12345, 0x5678));
  EXPECT_EQ(0, munlockall());
  EXPECT_EQ(0, munlock((void*)0x987, 0x654));
}

struct LargeStruct {
  int foo[100];
};

// Test for bug http://llvm.org/bugs/show_bug.cgi?id=11763.
// Struct copy should not cause asan warning even if lhs == rhs.
TEST(AddressSanitizer, LargeStructCopyTest) {
  LargeStruct a;
  *Ident(&a) = *Ident(&a);
}

ATTRIBUTE_NO_ADDRESS_SAFETY_ANALYSIS
static void NoAddressSafety() {
  char *foo = new char[10];
  Ident(foo)[10] = 0;
  delete [] foo;
}

TEST(AddressSanitizer, AttributeNoAddressSafetyTest) {
  Ident(NoAddressSafety)();
}

// ------------------ demo tests; run each one-by-one -------------
// e.g. --gtest_filter=*DemoOOBLeftHigh --gtest_also_run_disabled_tests
TEST(AddressSanitizer, DISABLED_DemoThreadedTest) {
  ThreadedTestSpawn();
}

void *SimpleBugOnSTack(void *x = 0) {
  char a[20];
  Ident(a)[20] = 0;
  return 0;
}

TEST(AddressSanitizer, DISABLED_DemoStackTest) {
  SimpleBugOnSTack();
}

TEST(AddressSanitizer, DISABLED_DemoThreadStackTest) {
  pthread_t t;
  pthread_create(&t, 0, SimpleBugOnSTack, 0);
  pthread_join(t, 0);
}

TEST(AddressSanitizer, DISABLED_DemoUAFLowIn) {
  uaf_test<U1>(10, 0);
}
TEST(AddressSanitizer, DISABLED_DemoUAFLowLeft) {
  uaf_test<U1>(10, -2);
}
TEST(AddressSanitizer, DISABLED_DemoUAFLowRight) {
  uaf_test<U1>(10, 10);
}

TEST(AddressSanitizer, DISABLED_DemoUAFHigh) {
  uaf_test<U1>(kLargeMalloc, 0);
}

TEST(AddressSanitizer, DISABLED_DemoOOBLeftLow) {
  oob_test<U1>(10, -1);
}

TEST(AddressSanitizer, DISABLED_DemoOOBLeftHigh) {
  oob_test<U1>(kLargeMalloc, -1);
}

TEST(AddressSanitizer, DISABLED_DemoOOBRightLow) {
  oob_test<U1>(10, 10);
}

TEST(AddressSanitizer, DISABLED_DemoOOBRightHigh) {
  oob_test<U1>(kLargeMalloc, kLargeMalloc);
}

TEST(AddressSanitizer, DISABLED_DemoOOM) {
  size_t size = SANITIZER_WORDSIZE == 64 ? (size_t)(1ULL << 40) : (0xf0000000);
  printf("%p\n", malloc(size));
}

TEST(AddressSanitizer, DISABLED_DemoDoubleFreeTest) {
  DoubleFree();
}

TEST(AddressSanitizer, DISABLED_DemoNullDerefTest) {
  int *a = 0;
  Ident(a)[10] = 0;
}

TEST(AddressSanitizer, DISABLED_DemoFunctionStaticTest) {
  static char a[100];
  static char b[100];
  static char c[100];
  Ident(a);
  Ident(b);
  Ident(c);
  Ident(a)[5] = 0;
  Ident(b)[105] = 0;
  Ident(a)[5] = 0;
}

TEST(AddressSanitizer, DISABLED_DemoTooMuchMemoryTest) {
  const size_t kAllocSize = (1 << 28) - 1024;
  size_t total_size = 0;
  while (true) {
    char *x = (char*)malloc(kAllocSize);
    memset(x, 0, kAllocSize);
    total_size += kAllocSize;
    fprintf(stderr, "total: %ldM %p\n", (long)total_size >> 20, x);
  }
}

// http://code.google.com/p/address-sanitizer/issues/detail?id=66
TEST(AddressSanitizer, BufferOverflowAfterManyFrees) {
  for (int i = 0; i < 1000000; i++) {
    delete [] (Ident(new char [8644]));
  }
  char *x = new char[8192];
  EXPECT_DEATH(x[Ident(8192)] = 0, "AddressSanitizer: heap-buffer-overflow");
  delete [] Ident(x);
}

#ifdef __APPLE__
#include "asan_mac_test.h"
TEST(AddressSanitizerMac, CFAllocatorDefaultDoubleFree) {
  EXPECT_DEATH(
      CFAllocatorDefaultDoubleFree(NULL),
      "attempting double-free");
}

void CFAllocator_DoubleFreeOnPthread() {
  pthread_t child;
  pthread_create(&child, NULL, CFAllocatorDefaultDoubleFree, NULL);
  pthread_join(child, NULL);  // Shouldn't be reached.
}

TEST(AddressSanitizerMac, CFAllocatorDefaultDoubleFree_ChildPhread) {
  EXPECT_DEATH(CFAllocator_DoubleFreeOnPthread(), "attempting double-free");
}

namespace {

void *GLOB;

void *CFAllocatorAllocateToGlob(void *unused) {
  GLOB = CFAllocatorAllocate(NULL, 100, /*hint*/0);
  return NULL;
}

void *CFAllocatorDeallocateFromGlob(void *unused) {
  char *p = (char*)GLOB;
  p[100] = 'A';  // ASan should report an error here.
  CFAllocatorDeallocate(NULL, GLOB);
  return NULL;
}

void CFAllocator_PassMemoryToAnotherThread() {
  pthread_t th1, th2;
  pthread_create(&th1, NULL, CFAllocatorAllocateToGlob, NULL);
  pthread_join(th1, NULL);
  pthread_create(&th2, NULL, CFAllocatorDeallocateFromGlob, NULL);
  pthread_join(th2, NULL);
}

TEST(AddressSanitizerMac, CFAllocator_PassMemoryToAnotherThread) {
  EXPECT_DEATH(CFAllocator_PassMemoryToAnotherThread(),
               "heap-buffer-overflow");
}

}  // namespace

// TODO(glider): figure out whether we still need these tests. Is it correct
// to intercept the non-default CFAllocators?
TEST(AddressSanitizerMac, DISABLED_CFAllocatorSystemDefaultDoubleFree) {
  EXPECT_DEATH(
      CFAllocatorSystemDefaultDoubleFree(),
      "attempting double-free");
}

// We're intercepting malloc, so kCFAllocatorMalloc is routed to ASan.
TEST(AddressSanitizerMac, CFAllocatorMallocDoubleFree) {
  EXPECT_DEATH(CFAllocatorMallocDoubleFree(), "attempting double-free");
}

TEST(AddressSanitizerMac, DISABLED_CFAllocatorMallocZoneDoubleFree) {
  EXPECT_DEATH(CFAllocatorMallocZoneDoubleFree(), "attempting double-free");
}

TEST(AddressSanitizerMac, GCDDispatchAsync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDDispatchAsync(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, GCDDispatchSync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDDispatchSync(), "Shadow byte and word");
}


TEST(AddressSanitizerMac, GCDReuseWqthreadsAsync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDReuseWqthreadsAsync(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, GCDReuseWqthreadsSync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDReuseWqthreadsSync(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, GCDDispatchAfter) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDDispatchAfter(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, GCDSourceEvent) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDSourceEvent(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, GCDSourceCancel) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDSourceCancel(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, GCDGroupAsync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDGroupAsync(), "Shadow byte and word");
}

void *MallocIntrospectionLockWorker(void *_) {
  const int kNumPointers = 100;
  int i;
  void *pointers[kNumPointers];
  for (i = 0; i < kNumPointers; i++) {
    pointers[i] = malloc(i + 1);
  }
  for (i = 0; i < kNumPointers; i++) {
    free(pointers[i]);
  }

  return NULL;
}

void *MallocIntrospectionLockForker(void *_) {
  pid_t result = fork();
  if (result == -1) {
    perror("fork");
  }
  assert(result != -1);
  if (result == 0) {
    // Call malloc in the child process to make sure we won't deadlock.
    void *ptr = malloc(42);
    free(ptr);
    exit(0);
  } else {
    // Return in the parent process.
    return NULL;
  }
}

TEST(AddressSanitizerMac, MallocIntrospectionLock) {
  // Incorrect implementation of force_lock and force_unlock in our malloc zone
  // will cause forked processes to deadlock.
  // TODO(glider): need to detect that none of the child processes deadlocked.
  const int kNumWorkers = 5, kNumIterations = 100;
  int i, iter;
  for (iter = 0; iter < kNumIterations; iter++) {
    pthread_t workers[kNumWorkers], forker;
    for (i = 0; i < kNumWorkers; i++) {
      pthread_create(&workers[i], 0, MallocIntrospectionLockWorker, 0);
    }
    pthread_create(&forker, 0, MallocIntrospectionLockForker, 0);
    for (i = 0; i < kNumWorkers; i++) {
      pthread_join(workers[i], 0);
    }
    pthread_join(forker, 0);
  }
}

void *TSDAllocWorker(void *test_key) {
  if (test_key) {
    void *mem = malloc(10);
    pthread_setspecific(*(pthread_key_t*)test_key, mem);
  }
  return NULL;
}

TEST(AddressSanitizerMac, DISABLED_TSDWorkqueueTest) {
  pthread_t th;
  pthread_key_t test_key;
  pthread_key_create(&test_key, CallFreeOnWorkqueue);
  pthread_create(&th, NULL, TSDAllocWorker, &test_key);
  pthread_join(th, NULL);
  pthread_key_delete(test_key);
}

// Test that CFStringCreateCopy does not copy constant strings.
TEST(AddressSanitizerMac, CFStringCreateCopy) {
  CFStringRef str = CFSTR("Hello world!\n");
  CFStringRef str2 = CFStringCreateCopy(0, str);
  EXPECT_EQ(str, str2);
}

TEST(AddressSanitizerMac, NSObjectOOB) {
  // Make sure that our allocators are used for NSObjects.
  EXPECT_DEATH(TestOOBNSObjects(), "heap-buffer-overflow");
}

// Make sure that correct pointer is passed to free() when deallocating a
// NSURL object.
// See http://code.google.com/p/address-sanitizer/issues/detail?id=70.
TEST(AddressSanitizerMac, NSURLDeallocation) {
  TestNSURLDeallocation();
}

// See http://code.google.com/p/address-sanitizer/issues/detail?id=109.
TEST(AddressSanitizerMac, Mstats) {
  malloc_statistics_t stats1, stats2;
  malloc_zone_statistics(/*all zones*/NULL, &stats1);
  const int kMallocSize = 100000;
  void *alloc = Ident(malloc(kMallocSize));
  malloc_zone_statistics(/*all zones*/NULL, &stats2);
  EXPECT_GT(stats2.blocks_in_use, stats1.blocks_in_use);
  EXPECT_GE(stats2.size_in_use - stats1.size_in_use, kMallocSize);
  free(alloc);
  // Even the default OSX allocator may not change the stats after free().
}
#endif  // __APPLE__

// Test that instrumentation of stack allocations takes into account
// AllocSize of a type, and not its StoreSize (16 vs 10 bytes for long double).
// See http://llvm.org/bugs/show_bug.cgi?id=12047 for more details.
TEST(AddressSanitizer, LongDoubleNegativeTest) {
  long double a, b;
  static long double c;
  memcpy(Ident(&a), Ident(&b), sizeof(long double));
  memcpy(Ident(&c), Ident(&b), sizeof(long double));
}
