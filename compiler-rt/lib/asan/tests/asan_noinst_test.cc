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
#include "sanitizer/asan_interface.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // for memset()
#include <algorithm>
#include <vector>

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
  const size_t kNumPcs = ARRAY_SIZE(pc_array);
  u32 compressed[2 * kNumPcs];

  for (size_t iter = 0; iter < n_iter; iter++) {
    std::random_shuffle(pc_array, pc_array + kNumPcs);
    __asan::StackTrace stack0, stack1;
    stack0.CopyFrom(pc_array, kNumPcs);
    stack0.size = std::max((size_t)1, (size_t)(my_rand(&seed) % stack0.size));
    size_t compress_size =
      std::max((size_t)2, (size_t)my_rand(&seed) % (2 * kNumPcs));
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
  (void)unused;
  u32 seed = my_rand(&global_seed);
  __asan::StackTrace stack;
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

TEST(AddressSanitizer, MemsetWildAddressTest) {
  typedef void*(*memset_p)(void*, int, size_t);
  // Prevent inlining of memset().
  volatile memset_p libc_memset = (memset_p)memset;
  EXPECT_DEATH(libc_memset((void*)(kLowShadowBeg + kPageSize), 0, 100),
               "unknown-crash.*low shadow");
  EXPECT_DEATH(libc_memset((void*)(kShadowGapBeg + kPageSize), 0, 100),
               "unknown-crash.*shadow gap");
  EXPECT_DEATH(libc_memset((void*)(kHighShadowBeg + kPageSize), 0, 100),
               "unknown-crash.*high shadow");
}

TEST(AddressSanitizerInterface, GetEstimatedAllocatedSize) {
  EXPECT_EQ(1U, __asan_get_estimated_allocated_size(0));
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
  EXPECT_EQ(false, __asan_get_ownership(wild_addr));
  EXPECT_DEATH(__asan_get_allocated_size(wild_addr), kGetAllocatedSizeErrorMsg);
  EXPECT_EQ(false, __asan_get_ownership(array + kArraySize / 2));
  EXPECT_DEATH(__asan_get_allocated_size(array + kArraySize / 2),
               kGetAllocatedSizeErrorMsg);

  // NULL is not owned, but is a valid argument for __asan_get_allocated_size().
  EXPECT_EQ(false, __asan_get_ownership(NULL));
  EXPECT_EQ(0U, __asan_get_allocated_size(NULL));

  // When memory is freed, it's not owned, and call to GetAllocatedSize
  // is forbidden.
  free(array);
  EXPECT_EQ(false, __asan_get_ownership(array));
  EXPECT_DEATH(__asan_get_allocated_size(array), kGetAllocatedSizeErrorMsg);

  delete int_ptr;
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

// This test is run in a separate process, so that large malloced
// chunk won't remain in the free lists after the test.
// Note: use ASSERT_* instead of EXPECT_* here.
static void RunGetHeapSizeTestAndDie() {
  size_t old_heap_size, new_heap_size, heap_growth;
  // We unlikely have have chunk of this size in free list.
  static const size_t kLargeMallocSize = 1 << 29;  // 512M
  old_heap_size = __asan_get_heap_size();
  fprintf(stderr, "allocating %zu bytes:\n", kLargeMallocSize);
  free(Ident(malloc(kLargeMallocSize)));
  new_heap_size = __asan_get_heap_size();
  heap_growth = new_heap_size - old_heap_size;
  fprintf(stderr, "heap growth after first malloc: %zu\n", heap_growth);
  ASSERT_GE(heap_growth, kLargeMallocSize);
  ASSERT_LE(heap_growth, 2 * kLargeMallocSize);

  // Now large chunk should fall into free list, and can be
  // allocated without increasing heap size.
  old_heap_size = new_heap_size;
  free(Ident(malloc(kLargeMallocSize)));
  heap_growth = __asan_get_heap_size() - old_heap_size;
  fprintf(stderr, "heap growth after second malloc: %zu\n", heap_growth);
  ASSERT_LT(heap_growth, kLargeMallocSize);

  // Test passed. Now die with expected double-free.
  DoDoubleFree();
}

TEST(AddressSanitizerInterface, GetHeapSizeTest) {
  EXPECT_DEATH(RunGetHeapSizeTestAndDie(), "double-free");
}

// Note: use ASSERT_* instead of EXPECT_* here.
static void DoLargeMallocForGetFreeBytesTestAndDie() {
  size_t old_free_bytes, new_free_bytes;
  static const size_t kLargeMallocSize = 1 << 29;  // 512M
  // If we malloc and free a large memory chunk, it will not fall
  // into quarantine and will be available for future requests.
  old_free_bytes = __asan_get_free_bytes();
  fprintf(stderr, "allocating %zu bytes:\n", kLargeMallocSize);
  fprintf(stderr, "free bytes before malloc: %zu\n", old_free_bytes);
  free(Ident(malloc(kLargeMallocSize)));
  new_free_bytes = __asan_get_free_bytes();
  fprintf(stderr, "free bytes after malloc and free: %zu\n", new_free_bytes);
  ASSERT_GE(new_free_bytes, old_free_bytes + kLargeMallocSize);
  // Test passed.
  DoDoubleFree();
}

TEST(AddressSanitizerInterface, GetFreeBytesTest) {
  static const size_t kNumOfChunks = 100;
  static const size_t kChunkSize = 100;
  char *chunks[kNumOfChunks];
  size_t i;
  size_t old_free_bytes, new_free_bytes;
  // Allocate a small chunk. Now allocator probably has a lot of these
  // chunks to fulfill future requests. So, future requests will decrease
  // the number of free bytes.
  chunks[0] = Ident((char*)malloc(kChunkSize));
  old_free_bytes = __asan_get_free_bytes();
  for (i = 1; i < kNumOfChunks; i++) {
    chunks[i] = Ident((char*)malloc(kChunkSize));
    new_free_bytes = __asan_get_free_bytes();
    EXPECT_LT(new_free_bytes, old_free_bytes);
    old_free_bytes = new_free_bytes;
  }
  EXPECT_DEATH(DoLargeMallocForGetFreeBytesTestAndDie(), "double-free");
}

static const size_t kManyThreadsMallocSizes[] = {5, 1UL<<10, 1UL<<20, 357};
static const size_t kManyThreadsIterations = 250;
static const size_t kManyThreadsNumThreads = (__WORDSIZE == 32) ? 40 : 200;

void *ManyThreadsWithStatsWorker(void *arg) {
  (void)arg;
  for (size_t iter = 0; iter < kManyThreadsIterations; iter++) {
    for (size_t size_index = 0; size_index < 4; size_index++) {
      free(Ident(malloc(kManyThreadsMallocSizes[size_index])));
    }
  }
  return 0;
}

TEST(AddressSanitizerInterface, ManyThreadsWithStatsStressTest) {
  size_t before_test, after_test, i;
  pthread_t threads[kManyThreadsNumThreads];
  before_test = __asan_get_current_allocated_bytes();
  for (i = 0; i < kManyThreadsNumThreads; i++) {
    pthread_create(&threads[i], 0,
                   (void* (*)(void *x))ManyThreadsWithStatsWorker, (void*)i);
  }
  for (i = 0; i < kManyThreadsNumThreads; i++) {
    pthread_join(threads[i], 0);
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
  const size_t kNumMallocs =
      (__WORDSIZE <= 32 || ASAN_LOW_MEMORY) ? 1 << 10 : 1 << 14;
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
