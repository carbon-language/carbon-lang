//===-- asan_noinst_test.cc ----------------------===//
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
// This test file should be compiled w/o asan instrumentation.
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_test_utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "gtest/gtest.h"

// Simple stand-alone pseudorandom number generator.
// Current algorithm is ANSI C linear congruential PRNG.
static inline u32 my_rand(u32* state) {
  return (*state = *state * 1103515245 + 12345) >> 16;
}

static u32 global_seed = 0;


TEST(AddressSanitizer, InternalSimpleDeathTest) {
  EXPECT_DEATH(exit(1), "");
}

static void MallocStress(size_t n) {
  u32 seed = my_rand(&global_seed);
  __asan::AsanStackTrace stack1;
  stack1.trace[0] = 0xa123;
  stack1.trace[1] = 0xa456;
  stack1.size = 2;

  __asan::AsanStackTrace stack2;
  stack2.trace[0] = 0xb123;
  stack2.trace[1] = 0xb456;
  stack2.size = 2;

  __asan::AsanStackTrace stack3;
  stack3.trace[0] = 0xc123;
  stack3.trace[1] = 0xc456;
  stack3.size = 2;

  std::vector<void *> vec;
  for (size_t i = 0; i < n; i++) {
    if ((i % 3) == 0) {
      if (vec.empty()) continue;
      size_t idx = my_rand(&seed) % vec.size();
      void *ptr = vec[idx];
      vec[idx] = vec.back();
      vec.pop_back();
      __asan::asan_free(ptr, &stack1);
    } else {
      size_t size = my_rand(&seed) % 1000 + 1;
      switch ((my_rand(&seed) % 128)) {
        case 0: size += 1024; break;
        case 1: size += 2048; break;
        case 2: size += 4096; break;
      }
      size_t alignment = 1 << (my_rand(&seed) % 10 + 1);
      char *ptr = (char*)__asan::asan_memalign(alignment, size, &stack2);
      vec.push_back(ptr);
      ptr[0] = 0;
      ptr[size-1] = 0;
      ptr[size/2] = 0;
    }
  }
  for (size_t i = 0; i < vec.size(); i++)
    __asan::asan_free(vec[i], &stack3);
}


TEST(AddressSanitizer, NoInstMallocTest) {
#ifdef __arm__
  MallocStress(300000);
#else
  MallocStress(1000000);
#endif
}

static void PrintShadow(const char *tag, uptr ptr, size_t size) {
  fprintf(stderr, "%s shadow: %lx size % 3ld: ", tag, (long)ptr, (long)size);
  uptr prev_shadow = 0;
  for (sptr i = -32; i < (sptr)size + 32; i++) {
    uptr shadow = __asan::MemToShadow(ptr + i);
    if (i == 0 || i == (sptr)size)
      fprintf(stderr, ".");
    if (shadow != prev_shadow) {
      prev_shadow = shadow;
      fprintf(stderr, "%02x", (int)*(u8*)shadow);
    }
  }
  fprintf(stderr, "\n");
}

TEST(AddressSanitizer, DISABLED_InternalPrintShadow) {
  for (size_t size = 1; size <= 513; size++) {
    char *ptr = new char[size];
    PrintShadow("m", (uptr)ptr, size);
    delete [] ptr;
    PrintShadow("f", (uptr)ptr, size);
  }
}

static uptr pc_array[] = {
#if __WORDSIZE == 64
  0x7effbf756068ULL,
  0x7effbf75e5abULL,
  0x7effc0625b7cULL,
  0x7effc05b8997ULL,
  0x7effbf990577ULL,
  0x7effbf990c56ULL,
  0x7effbf992f3cULL,
  0x7effbf950c22ULL,
  0x7effc036dba0ULL,
  0x7effc03638a3ULL,
  0x7effc035be4aULL,
  0x7effc0539c45ULL,
  0x7effc0539a65ULL,
  0x7effc03db9b3ULL,
  0x7effc03db100ULL,
  0x7effc037c7b8ULL,
  0x7effc037bfffULL,
  0x7effc038b777ULL,
  0x7effc038021cULL,
  0x7effc037c7d1ULL,
  0x7effc037bfffULL,
  0x7effc038b777ULL,
  0x7effc038021cULL,
  0x7effc037c7d1ULL,
  0x7effc037bfffULL,
  0x7effc038b777ULL,
  0x7effc038021cULL,
  0x7effc037c7d1ULL,
  0x7effc037bfffULL,
  0x7effc0520d26ULL,
  0x7effc009ddffULL,
  0x7effbf90bb50ULL,
  0x7effbdddfa69ULL,
  0x7effbdde1fe2ULL,
  0x7effbdde2424ULL,
  0x7effbdde27b3ULL,
  0x7effbddee53bULL,
  0x7effbdde1988ULL,
  0x7effbdde0904ULL,
  0x7effc106ce0dULL,
  0x7effbcc3fa04ULL,
  0x7effbcc3f6a4ULL,
  0x7effbcc3e726ULL,
  0x7effbcc40852ULL,
  0x7effb681ec4dULL,
#endif  // __WORDSIZE
  0xB0B5E768,
  0x7B682EC1,
  0x367F9918,
  0xAE34E13,
  0xBA0C6C6,
  0x13250F46,
  0xA0D6A8AB,
  0x2B07C1A8,
  0x6C844F4A,
  0x2321B53,
  0x1F3D4F8F,
  0x3FE2924B,
  0xB7A2F568,
  0xBD23950A,
  0x61020930,
  0x33E7970C,
  0x405998A1,
  0x59F3551D,
  0x350E3028,
  0xBC55A28D,
  0x361F3AED,
  0xBEAD0F73,
  0xAEF28479,
  0x757E971F,
  0xAEBA450,
  0x43AD22F5,
  0x8C2C50C4,
  0x7AD8A2E1,
  0x69EE4EE8,
  0xC08DFF,
  0x4BA6538,
  0x3708AB2,
  0xC24B6475,
  0x7C8890D7,
  0x6662495F,
  0x9B641689,
  0xD3596B,
  0xA1049569,
  0x44CBC16,
  0x4D39C39F
};

void CompressStackTraceTest(size_t n_iter) {
  u32 seed = my_rand(&global_seed);
  const size_t kNumPcs = ASAN_ARRAY_SIZE(pc_array);
  u32 compressed[2 * kNumPcs];

  for (size_t iter = 0; iter < n_iter; iter++) {
    std::random_shuffle(pc_array, pc_array + kNumPcs);
    __asan::AsanStackTrace stack0, stack1;
    stack0.CopyFrom(pc_array, kNumPcs);
    stack0.size = std::max((size_t)1, (size_t)(my_rand(&seed) % stack0.size));
    size_t compress_size =
      std::max((size_t)2, (size_t)my_rand(&seed) % (2 * kNumPcs));
    size_t n_frames =
      __asan::AsanStackTrace::CompressStack(&stack0, compressed, compress_size);
    assert(n_frames <= stack0.size);
    __asan::AsanStackTrace::UncompressStack(&stack1, compressed, compress_size);
    assert(stack1.size == n_frames);
    for (size_t i = 0; i < stack1.size; i++) {
      assert(stack0.trace[i] == stack1.trace[i]);
    }
  }
}

TEST(AddressSanitizer, CompressStackTraceTest) {
  CompressStackTraceTest(10000);
}

void CompressStackTraceBenchmark(size_t n_iter) {
  const size_t kNumPcs = ASAN_ARRAY_SIZE(pc_array);
  u32 compressed[2 * kNumPcs];
  std::random_shuffle(pc_array, pc_array + kNumPcs);

  __asan::AsanStackTrace stack0;
  stack0.CopyFrom(pc_array, kNumPcs);
  stack0.size = kNumPcs;
  for (size_t iter = 0; iter < n_iter; iter++) {
    size_t compress_size = kNumPcs;
    size_t n_frames =
      __asan::AsanStackTrace::CompressStack(&stack0, compressed, compress_size);
    Ident(n_frames);
  }
}

TEST(AddressSanitizer, CompressStackTraceBenchmark) {
  CompressStackTraceBenchmark(1 << 24);
}

TEST(AddressSanitizer, QuarantineTest) {
  __asan::AsanStackTrace stack;
  stack.trace[0] = 0x890;
  stack.size = 1;

  const int size = 32;
  void *p = __asan::asan_malloc(size, &stack);
  __asan::asan_free(p, &stack);
  size_t i;
  size_t max_i = 1 << 30;
  for (i = 0; i < max_i; i++) {
    void *p1 = __asan::asan_malloc(size, &stack);
    __asan::asan_free(p1, &stack);
    if (p1 == p) break;
  }
  // fprintf(stderr, "i=%ld\n", i);
  EXPECT_GE(i, 100000U);
  EXPECT_LT(i, max_i);
}

void *ThreadedQuarantineTestWorker(void *unused) {
  u32 seed = my_rand(&global_seed);
  __asan::AsanStackTrace stack;
  stack.trace[0] = 0x890;
  stack.size = 1;

  for (size_t i = 0; i < 1000; i++) {
    void *p = __asan::asan_malloc(1 + (my_rand(&seed) % 4000), &stack);
    __asan::asan_free(p, &stack);
  }
  return NULL;
}

// Check that the thread local allocators are flushed when threads are
// destroyed.
TEST(AddressSanitizer, ThreadedQuarantineTest) {
  const int n_threads = 3000;
  size_t mmaped1 = __asan_get_heap_size();
  for (int i = 0; i < n_threads; i++) {
    pthread_t t;
    pthread_create(&t, NULL, ThreadedQuarantineTestWorker, 0);
    pthread_join(t, 0);
    size_t mmaped2 = __asan_get_heap_size();
    EXPECT_LT(mmaped2 - mmaped1, 320U * (1 << 20));
  }
}

void *ThreadedOneSizeMallocStress(void *unused) {
  __asan::AsanStackTrace stack;
  stack.trace[0] = 0x890;
  stack.size = 1;
  const size_t kNumMallocs = 1000;
  for (int iter = 0; iter < 1000; iter++) {
    void *p[kNumMallocs];
    for (size_t i = 0; i < kNumMallocs; i++) {
      p[i] = __asan::asan_malloc(32, &stack);
    }
    for (size_t i = 0; i < kNumMallocs; i++) {
      __asan::asan_free(p[i], &stack);
    }
  }
  return NULL;
}

TEST(AddressSanitizer, ThreadedOneSizeMallocStressTest) {
  const int kNumThreads = 4;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, ThreadedOneSizeMallocStress, 0);
  }
  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }
}
