//===-- asan_noinst_test.cc -----------------------------------------------===//
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
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_test_utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // for memset()
#include <algorithm>
#include <vector>
#include <limits>


TEST(AddressSanitizer, InternalSimpleDeathTest) {
  EXPECT_DEATH(exit(1), "");
}

static void MallocStress(size_t n) {
  u32 seed = my_rand();
  __asan::StackTrace stack1;
  stack1.trace[0] = 0xa123;
  stack1.trace[1] = 0xa456;
  stack1.size = 2;

  __asan::StackTrace stack2;
  stack2.trace[0] = 0xb123;
  stack2.trace[1] = 0xb456;
  stack2.size = 2;

  __asan::StackTrace stack3;
  stack3.trace[0] = 0xc123;
  stack3.trace[1] = 0xc456;
  stack3.size = 2;

  std::vector<void *> vec;
  for (size_t i = 0; i < n; i++) {
    if ((i % 3) == 0) {
      if (vec.empty()) continue;
      size_t idx = my_rand_r(&seed) % vec.size();
      void *ptr = vec[idx];
      vec[idx] = vec.back();
      vec.pop_back();
      __asan::asan_free(ptr, &stack1, __asan::FROM_MALLOC);
    } else {
      size_t size = my_rand_r(&seed) % 1000 + 1;
      switch ((my_rand_r(&seed) % 128)) {
        case 0: size += 1024; break;
        case 1: size += 2048; break;
        case 2: size += 4096; break;
      }
      size_t alignment = 1 << (my_rand_r(&seed) % 10 + 1);
      char *ptr = (char*)__asan::asan_memalign(alignment, size,
                                               &stack2, __asan::FROM_MALLOC);
      vec.push_back(ptr);
      ptr[0] = 0;
      ptr[size-1] = 0;
      ptr[size/2] = 0;
    }
  }
  for (size_t i = 0; i < vec.size(); i++)
    __asan::asan_free(vec[i], &stack3, __asan::FROM_MALLOC);
}


TEST(AddressSanitizer, NoInstMallocTest) {
  MallocStress(ASAN_LOW_MEMORY ? 300000 : 1000000);
}

TEST(AddressSanitizer, ThreadedMallocStressTest) {
  const int kNumThreads = 4;
  const int kNumIterations = (ASAN_LOW_MEMORY) ? 10000 : 100000;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_CREATE(&t[i], 0, (void* (*)(void *x))MallocStress,
        (void*)kNumIterations);
  }
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_JOIN(t[i], 0);
  }
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
#if SANITIZER_WORDSIZE == 64
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
#endif  // SANITIZER_WORDSIZE
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
  u32 seed = my_rand();
  const size_t kNumPcs = ARRAY_SIZE(pc_array);
  u32 compressed[2 * kNumPcs];

  for (size_t iter = 0; iter < n_iter; iter++) {
    std::random_shuffle(pc_array, pc_array + kNumPcs);
    __asan::StackTrace stack0, stack1;
    stack0.CopyFrom(pc_array, kNumPcs);
    stack0.size = std::max((size_t)1, (size_t)(my_rand_r(&seed) % stack0.size));
    size_t compress_size =
      std::max((size_t)2, (size_t)my_rand_r(&seed) % (2 * kNumPcs));
    size_t n_frames =
      __asan::StackTrace::CompressStack(&stack0, compressed, compress_size);
    Ident(n_frames);
    assert(n_frames <= stack0.size);
    __asan::StackTrace::UncompressStack(&stack1, compressed, compress_size);
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
  const size_t kNumPcs = ARRAY_SIZE(pc_array);
  u32 compressed[2 * kNumPcs];
  std::random_shuffle(pc_array, pc_array + kNumPcs);

  __asan::StackTrace stack0;
  stack0.CopyFrom(pc_array, kNumPcs);
  stack0.size = kNumPcs;
  for (size_t iter = 0; iter < n_iter; iter++) {
    size_t compress_size = kNumPcs;
    size_t n_frames =
      __asan::StackTrace::CompressStack(&stack0, compressed, compress_size);
    Ident(n_frames);
  }
}

TEST(AddressSanitizer, CompressStackTraceBenchmark) {
  CompressStackTraceBenchmark(1 << 24);
}

TEST(AddressSanitizer, QuarantineTest) {
  __asan::StackTrace stack;
  stack.trace[0] = 0x890;
  stack.size = 1;

  const int size = 1024;
  void *p = __asan::asan_malloc(size, &stack);
  __asan::asan_free(p, &stack, __asan::FROM_MALLOC);
  size_t i;
  size_t max_i = 1 << 30;
  for (i = 0; i < max_i; i++) {
    void *p1 = __asan::asan_malloc(size, &stack);
    __asan::asan_free(p1, &stack, __asan::FROM_MALLOC);
    if (p1 == p) break;
  }
  EXPECT_GE(i, 10000U);
  EXPECT_LT(i, max_i);
}

void *ThreadedQuarantineTestWorker(void *unused) {
  (void)unused;
  u32 seed = my_rand();
  __asan::StackTrace stack;
  stack.trace[0] = 0x890;
  stack.size = 1;

  for (size_t i = 0; i < 1000; i++) {
    void *p = __asan::asan_malloc(1 + (my_rand_r(&seed) % 4000), &stack);
    __asan::asan_free(p, &stack, __asan::FROM_MALLOC);
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
    PTHREAD_CREATE(&t, NULL, ThreadedQuarantineTestWorker, 0);
    PTHREAD_JOIN(t, 0);
    size_t mmaped2 = __asan_get_heap_size();
    EXPECT_LT(mmaped2 - mmaped1, 320U * (1 << 20));
  }
}

void *ThreadedOneSizeMallocStress(void *unused) {
  (void)unused;
  __asan::StackTrace stack;
  stack.trace[0] = 0x890;
  stack.size = 1;
  const size_t kNumMallocs = 1000;
  for (int iter = 0; iter < 1000; iter++) {
    void *p[kNumMallocs];
    for (size_t i = 0; i < kNumMallocs; i++) {
      p[i] = __asan::asan_malloc(32, &stack);
    }
    for (size_t i = 0; i < kNumMallocs; i++) {
      __asan::asan_free(p[i], &stack, __asan::FROM_MALLOC);
    }
  }
  return NULL;
}

TEST(AddressSanitizer, ThreadedOneSizeMallocStressTest) {
  const int kNumThreads = 4;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_CREATE(&t[i], 0, ThreadedOneSizeMallocStress, 0);
  }
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_JOIN(t[i], 0);
  }
}

TEST(AddressSanitizer, MemsetWildAddressTest) {
  using __asan::kHighMemEnd;
  typedef void*(*memset_p)(void*, int, size_t);
  // Prevent inlining of memset().
  volatile memset_p libc_memset = (memset_p)memset;
  EXPECT_DEATH(libc_memset((void*)(kLowShadowBeg + 200), 0, 100),
               (kLowShadowEnd == 0) ? "unknown-crash.*shadow gap"
                                    : "unknown-crash.*low shadow");
  EXPECT_DEATH(libc_memset((void*)(kShadowGapBeg + 200), 0, 100),
               "unknown-crash.*shadow gap");
  EXPECT_DEATH(libc_memset((void*)(kHighShadowBeg + 200), 0, 100),
               "unknown-crash.*high shadow");
}

TEST(AddressSanitizerInterface, GetEstimatedAllocatedSize) {
  EXPECT_EQ(0U, __asan_get_estimated_allocated_size(0));
  const size_t sizes[] = { 1, 30, 1<<30 };
  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(sizes[i], __asan_get_estimated_allocated_size(sizes[i]));
  }
}

static const char* kGetAllocatedSizeErrorMsg =
  "attempting to call __asan_get_allocated_size()";

TEST(AddressSanitizerInterface, GetAllocatedSizeAndOwnershipTest) {
  const size_t kArraySize = 100;
  char *array = Ident((char*)malloc(kArraySize));
  int *int_ptr = Ident(new int);

  // Allocated memory is owned by allocator. Allocated size should be
  // equal to requested size.
  EXPECT_EQ(true, __asan_get_ownership(array));
  EXPECT_EQ(kArraySize, __asan_get_allocated_size(array));
  EXPECT_EQ(true, __asan_get_ownership(int_ptr));
  EXPECT_EQ(sizeof(int), __asan_get_allocated_size(int_ptr));

  // We cannot call GetAllocatedSize from the memory we didn't map,
  // and from the interior pointers (not returned by previous malloc).
  void *wild_addr = (void*)0x1;
  EXPECT_FALSE(__asan_get_ownership(wild_addr));
  EXPECT_DEATH(__asan_get_allocated_size(wild_addr), kGetAllocatedSizeErrorMsg);
  EXPECT_FALSE(__asan_get_ownership(array + kArraySize / 2));
  EXPECT_DEATH(__asan_get_allocated_size(array + kArraySize / 2),
               kGetAllocatedSizeErrorMsg);

  // NULL is not owned, but is a valid argument for __asan_get_allocated_size().
  EXPECT_FALSE(__asan_get_ownership(NULL));
  EXPECT_EQ(0U, __asan_get_allocated_size(NULL));

  // When memory is freed, it's not owned, and call to GetAllocatedSize
  // is forbidden.
  free(array);
  EXPECT_FALSE(__asan_get_ownership(array));
  EXPECT_DEATH(__asan_get_allocated_size(array), kGetAllocatedSizeErrorMsg);
  delete int_ptr;

  void *zero_alloc = Ident(malloc(0));
  if (zero_alloc != 0) {
    // If malloc(0) is not null, this pointer is owned and should have valid
    // allocated size.
    EXPECT_TRUE(__asan_get_ownership(zero_alloc));
    // Allocated size is 0 or 1 depending on the allocator used.
    EXPECT_LT(__asan_get_allocated_size(zero_alloc), 2U);
  }
  free(zero_alloc);
}

TEST(AddressSanitizerInterface, GetCurrentAllocatedBytesTest) {
  size_t before_malloc, after_malloc, after_free;
  char *array;
  const size_t kMallocSize = 100;
  before_malloc = __asan_get_current_allocated_bytes();

  array = Ident((char*)malloc(kMallocSize));
  after_malloc = __asan_get_current_allocated_bytes();
  EXPECT_EQ(before_malloc + kMallocSize, after_malloc);

  free(array);
  after_free = __asan_get_current_allocated_bytes();
  EXPECT_EQ(before_malloc, after_free);
}

static void DoDoubleFree() {
  int *x = Ident(new int);
  delete Ident(x);
  delete Ident(x);
}

TEST(AddressSanitizerInterface, GetHeapSizeTest) {
  // asan_allocator2 does not keep huge chunks in free list, but unmaps them.
  // The chunk should be greater than the quarantine size,
  // otherwise it will be stuck in quarantine instead of being unmaped.
  static const size_t kLargeMallocSize = (1 << 28) + 1;  // 256M
  free(Ident(malloc(kLargeMallocSize)));  // Drain quarantine.
  uptr old_heap_size = __asan_get_heap_size();
  for (int i = 0; i < 3; i++) {
    // fprintf(stderr, "allocating %zu bytes:\n", kLargeMallocSize);
    free(Ident(malloc(kLargeMallocSize)));
    EXPECT_EQ(old_heap_size, __asan_get_heap_size());
  }
}

static const size_t kManyThreadsMallocSizes[] = {5, 1UL<<10, 1UL<<14, 357};
static const size_t kManyThreadsIterations = 250;
static const size_t kManyThreadsNumThreads =
  (SANITIZER_WORDSIZE == 32) ? 40 : 200;

void *ManyThreadsWithStatsWorker(void *arg) {
  (void)arg;
  for (size_t iter = 0; iter < kManyThreadsIterations; iter++) {
    for (size_t size_index = 0; size_index < 4; size_index++) {
      free(Ident(malloc(kManyThreadsMallocSizes[size_index])));
    }
  }
  // Just one large allocation.
  free(Ident(malloc(1 << 20)));
  return 0;
}

TEST(AddressSanitizerInterface, ManyThreadsWithStatsStressTest) {
  size_t before_test, after_test, i;
  pthread_t threads[kManyThreadsNumThreads];
  before_test = __asan_get_current_allocated_bytes();
  for (i = 0; i < kManyThreadsNumThreads; i++) {
    PTHREAD_CREATE(&threads[i], 0,
                   (void* (*)(void *x))ManyThreadsWithStatsWorker, (void*)i);
  }
  for (i = 0; i < kManyThreadsNumThreads; i++) {
    PTHREAD_JOIN(threads[i], 0);
  }
  after_test = __asan_get_current_allocated_bytes();
  // ASan stats also reflect memory usage of internal ASan RTL structs,
  // so we can't check for equality here.
  EXPECT_LT(after_test, before_test + (1UL<<20));
}

TEST(AddressSanitizerInterface, ExitCode) {
  int original_exit_code = __asan_set_error_exit_code(7);
  EXPECT_EXIT(DoDoubleFree(), ::testing::ExitedWithCode(7), "");
  EXPECT_EQ(7, __asan_set_error_exit_code(8));
  EXPECT_EXIT(DoDoubleFree(), ::testing::ExitedWithCode(8), "");
  EXPECT_EQ(8, __asan_set_error_exit_code(original_exit_code));
  EXPECT_EXIT(DoDoubleFree(),
              ::testing::ExitedWithCode(original_exit_code), "");
}

static void MyDeathCallback() {
  fprintf(stderr, "MyDeathCallback\n");
}

TEST(AddressSanitizerInterface, DeathCallbackTest) {
  __asan_set_death_callback(MyDeathCallback);
  EXPECT_DEATH(DoDoubleFree(), "MyDeathCallback");
  __asan_set_death_callback(NULL);
}

static const char* kUseAfterPoisonErrorMessage = "use-after-poison";

#define GOOD_ACCESS(ptr, offset)  \
    EXPECT_FALSE(__asan::AddressIsPoisoned((uptr)(ptr + offset)))

#define BAD_ACCESS(ptr, offset) \
    EXPECT_TRUE(__asan::AddressIsPoisoned((uptr)(ptr + offset)))

TEST(AddressSanitizerInterface, SimplePoisonMemoryRegionTest) {
  char *array = Ident((char*)malloc(120));
  // poison array[40..80)
  __asan_poison_memory_region(array + 40, 40);
  GOOD_ACCESS(array, 39);
  GOOD_ACCESS(array, 80);
  BAD_ACCESS(array, 40);
  BAD_ACCESS(array, 60);
  BAD_ACCESS(array, 79);
  EXPECT_DEATH(__asan_report_error(0, 0, 0, (uptr)(array + 40), true, 1),
               kUseAfterPoisonErrorMessage);
  __asan_unpoison_memory_region(array + 40, 40);
  // access previously poisoned memory.
  GOOD_ACCESS(array, 40);
  GOOD_ACCESS(array, 79);
  free(array);
}

TEST(AddressSanitizerInterface, OverlappingPoisonMemoryRegionTest) {
  char *array = Ident((char*)malloc(120));
  // Poison [0..40) and [80..120)
  __asan_poison_memory_region(array, 40);
  __asan_poison_memory_region(array + 80, 40);
  BAD_ACCESS(array, 20);
  GOOD_ACCESS(array, 60);
  BAD_ACCESS(array, 100);
  // Poison whole array - [0..120)
  __asan_poison_memory_region(array, 120);
  BAD_ACCESS(array, 60);
  // Unpoison [24..96)
  __asan_unpoison_memory_region(array + 24, 72);
  BAD_ACCESS(array, 23);
  GOOD_ACCESS(array, 24);
  GOOD_ACCESS(array, 60);
  GOOD_ACCESS(array, 95);
  BAD_ACCESS(array, 96);
  free(array);
}

TEST(AddressSanitizerInterface, PushAndPopWithPoisoningTest) {
  // Vector of capacity 20
  char *vec = Ident((char*)malloc(20));
  __asan_poison_memory_region(vec, 20);
  for (size_t i = 0; i < 7; i++) {
    // Simulate push_back.
    __asan_unpoison_memory_region(vec + i, 1);
    GOOD_ACCESS(vec, i);
    BAD_ACCESS(vec, i + 1);
  }
  for (size_t i = 7; i > 0; i--) {
    // Simulate pop_back.
    __asan_poison_memory_region(vec + i - 1, 1);
    BAD_ACCESS(vec, i - 1);
    if (i > 1) GOOD_ACCESS(vec, i - 2);
  }
  free(vec);
}

TEST(AddressSanitizerInterface, GlobalRedzones) {
  GOOD_ACCESS(glob1, 1 - 1);
  GOOD_ACCESS(glob2, 2 - 1);
  GOOD_ACCESS(glob3, 3 - 1);
  GOOD_ACCESS(glob4, 4 - 1);
  GOOD_ACCESS(glob5, 5 - 1);
  GOOD_ACCESS(glob6, 6 - 1);
  GOOD_ACCESS(glob7, 7 - 1);
  GOOD_ACCESS(glob8, 8 - 1);
  GOOD_ACCESS(glob9, 9 - 1);
  GOOD_ACCESS(glob10, 10 - 1);
  GOOD_ACCESS(glob11, 11 - 1);
  GOOD_ACCESS(glob12, 12 - 1);
  GOOD_ACCESS(glob13, 13 - 1);
  GOOD_ACCESS(glob14, 14 - 1);
  GOOD_ACCESS(glob15, 15 - 1);
  GOOD_ACCESS(glob16, 16 - 1);
  GOOD_ACCESS(glob17, 17 - 1);
  GOOD_ACCESS(glob1000, 1000 - 1);
  GOOD_ACCESS(glob10000, 10000 - 1);
  GOOD_ACCESS(glob100000, 100000 - 1);

  BAD_ACCESS(glob1, 1);
  BAD_ACCESS(glob2, 2);
  BAD_ACCESS(glob3, 3);
  BAD_ACCESS(glob4, 4);
  BAD_ACCESS(glob5, 5);
  BAD_ACCESS(glob6, 6);
  BAD_ACCESS(glob7, 7);
  BAD_ACCESS(glob8, 8);
  BAD_ACCESS(glob9, 9);
  BAD_ACCESS(glob10, 10);
  BAD_ACCESS(glob11, 11);
  BAD_ACCESS(glob12, 12);
  BAD_ACCESS(glob13, 13);
  BAD_ACCESS(glob14, 14);
  BAD_ACCESS(glob15, 15);
  BAD_ACCESS(glob16, 16);
  BAD_ACCESS(glob17, 17);
  BAD_ACCESS(glob1000, 1000);
  BAD_ACCESS(glob1000, 1100);  // Redzone is at least 101 bytes.
  BAD_ACCESS(glob10000, 10000);
  BAD_ACCESS(glob10000, 11000);  // Redzone is at least 1001 bytes.
  BAD_ACCESS(glob100000, 100000);
  BAD_ACCESS(glob100000, 110000);  // Redzone is at least 10001 bytes.
}

// Make sure that each aligned block of size "2^granularity" doesn't have
// "true" value before "false" value.
static void MakeShadowValid(bool *shadow, int length, int granularity) {
  bool can_be_poisoned = true;
  for (int i = length - 1; i >= 0; i--) {
    if (!shadow[i])
      can_be_poisoned = false;
    if (!can_be_poisoned)
      shadow[i] = false;
    if (i % (1 << granularity) == 0) {
      can_be_poisoned = true;
    }
  }
}

TEST(AddressSanitizerInterface, PoisoningStressTest) {
  const size_t kSize = 24;
  bool expected[kSize];
  char *arr = Ident((char*)malloc(kSize));
  for (size_t l1 = 0; l1 < kSize; l1++) {
    for (size_t s1 = 1; l1 + s1 <= kSize; s1++) {
      for (size_t l2 = 0; l2 < kSize; l2++) {
        for (size_t s2 = 1; l2 + s2 <= kSize; s2++) {
          // Poison [l1, l1+s1), [l2, l2+s2) and check result.
          __asan_unpoison_memory_region(arr, kSize);
          __asan_poison_memory_region(arr + l1, s1);
          __asan_poison_memory_region(arr + l2, s2);
          memset(expected, false, kSize);
          memset(expected + l1, true, s1);
          MakeShadowValid(expected, kSize, /*granularity*/ 3);
          memset(expected + l2, true, s2);
          MakeShadowValid(expected, kSize, /*granularity*/ 3);
          for (size_t i = 0; i < kSize; i++) {
            ASSERT_EQ(expected[i], __asan_address_is_poisoned(arr + i));
          }
          // Unpoison [l1, l1+s1) and [l2, l2+s2) and check result.
          __asan_poison_memory_region(arr, kSize);
          __asan_unpoison_memory_region(arr + l1, s1);
          __asan_unpoison_memory_region(arr + l2, s2);
          memset(expected, true, kSize);
          memset(expected + l1, false, s1);
          MakeShadowValid(expected, kSize, /*granularity*/ 3);
          memset(expected + l2, false, s2);
          MakeShadowValid(expected, kSize, /*granularity*/ 3);
          for (size_t i = 0; i < kSize; i++) {
            ASSERT_EQ(expected[i], __asan_address_is_poisoned(arr + i));
          }
        }
      }
    }
  }
}

TEST(AddressSanitizerInterface, PoisonedRegion) {
  size_t rz = 16;
  for (size_t size = 1; size <= 64; size++) {
    char *p = new char[size];
    uptr x = reinterpret_cast<uptr>(p);
    for (size_t beg = 0; beg < size + rz; beg++) {
      for (size_t end = beg; end < size + rz; end++) {
        uptr first_poisoned = __asan_region_is_poisoned(x + beg, end - beg);
        if (beg == end) {
          EXPECT_FALSE(first_poisoned);
        } else if (beg < size && end <= size) {
          EXPECT_FALSE(first_poisoned);
        } else if (beg >= size) {
          EXPECT_EQ(x + beg, first_poisoned);
        } else {
          EXPECT_GT(end, size);
          EXPECT_EQ(x + size, first_poisoned);
        }
      }
    }
    delete [] p;
  }
}

// This is a performance benchmark for manual runs.
// asan's memset interceptor calls mem_is_zero for the entire shadow region.
// the profile should look like this:
//     89.10%   [.] __memset_sse2
//     10.50%   [.] __sanitizer::mem_is_zero
// I.e. mem_is_zero should consume ~ SHADOW_GRANULARITY less CPU cycles
// than memset itself.
TEST(AddressSanitizerInterface, DISABLED_StressLargeMemset) {
  size_t size = 1 << 20;
  char *x = new char[size];
  for (int i = 0; i < 100000; i++)
    Ident(memset)(x, 0, size);
  delete [] x;
}

// Same here, but we run memset with small sizes.
TEST(AddressSanitizerInterface, DISABLED_StressSmallMemset) {
  size_t size = 32;
  char *x = new char[size];
  for (int i = 0; i < 100000000; i++)
    Ident(memset)(x, 0, size);
  delete [] x;
}

static const char *kInvalidPoisonMessage = "invalid-poison-memory-range";
static const char *kInvalidUnpoisonMessage = "invalid-unpoison-memory-range";

TEST(AddressSanitizerInterface, DISABLED_InvalidPoisonAndUnpoisonCallsTest) {
  char *array = Ident((char*)malloc(120));
  __asan_unpoison_memory_region(array, 120);
  // Try to unpoison not owned memory
  EXPECT_DEATH(__asan_unpoison_memory_region(array, 121),
               kInvalidUnpoisonMessage);
  EXPECT_DEATH(__asan_unpoison_memory_region(array - 1, 120),
               kInvalidUnpoisonMessage);

  __asan_poison_memory_region(array, 120);
  // Try to poison not owned memory.
  EXPECT_DEATH(__asan_poison_memory_region(array, 121), kInvalidPoisonMessage);
  EXPECT_DEATH(__asan_poison_memory_region(array - 1, 120),
               kInvalidPoisonMessage);
  free(array);
}

static void ErrorReportCallbackOneToZ(const char *report) {
  int report_len = strlen(report);
  ASSERT_EQ(6, write(2, "ABCDEF", 6));
  ASSERT_EQ(report_len, write(2, report, report_len));
  ASSERT_EQ(6, write(2, "ABCDEF", 6));
  _exit(1);
}

TEST(AddressSanitizerInterface, SetErrorReportCallbackTest) {
  __asan_set_error_report_callback(ErrorReportCallbackOneToZ);
  EXPECT_DEATH(__asan_report_error(0, 0, 0, 0, true, 1),
               ASAN_PCRE_DOTALL "ABCDEF.*AddressSanitizer.*WRITE.*ABCDEF");
  __asan_set_error_report_callback(NULL);
}

TEST(AddressSanitizerInterface, GetOwnershipStressTest) {
  std::vector<char *> pointers;
  std::vector<size_t> sizes;
  const size_t kNumMallocs = 1 << 9;
  for (size_t i = 0; i < kNumMallocs; i++) {
    size_t size = i * 100 + 1;
    pointers.push_back((char*)malloc(size));
    sizes.push_back(size);
  }
  for (size_t i = 0; i < 4000000; i++) {
    EXPECT_FALSE(__asan_get_ownership(&pointers));
    EXPECT_FALSE(__asan_get_ownership((void*)0x1234));
    size_t idx = i % kNumMallocs;
    EXPECT_TRUE(__asan_get_ownership(pointers[idx]));
    EXPECT_EQ(sizes[idx], __asan_get_allocated_size(pointers[idx]));
  }
  for (size_t i = 0, n = pointers.size(); i < n; i++)
    free(pointers[i]);
}

TEST(AddressSanitizerInterface, CallocOverflow) {
  size_t kArraySize = 4096;
  volatile size_t kMaxSizeT = std::numeric_limits<size_t>::max();
  volatile size_t kArraySize2 = kMaxSizeT / kArraySize + 10;
  void *p = calloc(kArraySize, kArraySize2);  // Should return 0.
  EXPECT_EQ(0L, Ident(p));
}

TEST(AddressSanitizerInterface, CallocOverflow2) {
#if SANITIZER_WORDSIZE == 32
  size_t kArraySize = 112;
  volatile size_t kArraySize2 = 43878406;
  void *p = calloc(kArraySize, kArraySize2);  // Should return 0.
  EXPECT_EQ(0L, Ident(p));
#endif
}

TEST(AddressSanitizerInterface, CallocReturnsZeroMem) {
  size_t sizes[] = {16, 1000, 10000, 100000, 2100000};
  for (size_t s = 0; s < ARRAY_SIZE(sizes); s++) {
    size_t size = sizes[s];
    for (size_t iter = 0; iter < 5; iter++) {
      char *x = Ident((char*)calloc(1, size));
      EXPECT_EQ(x[0], 0);
      EXPECT_EQ(x[size - 1], 0);
      EXPECT_EQ(x[size / 2], 0);
      EXPECT_EQ(x[size / 3], 0);
      EXPECT_EQ(x[size / 4], 0);
      memset(x, 0x42, size);
      free(Ident(x));
      free(Ident(malloc(Ident(1 << 27))));  // Try to drain the quarantine.
    }
  }
}
