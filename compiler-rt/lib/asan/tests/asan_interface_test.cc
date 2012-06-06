//===-- asan_interface_test.cc ----------------------===//
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
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "asan_test_config.h"
#include "asan_test_utils.h"
#include "asan_interface.h"

TEST(AddressSanitizerInterface, GetEstimatedAllocatedSize) {
  EXPECT_EQ(1, __asan_get_estimated_allocated_size(0));
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
  EXPECT_EQ(0, __asan_get_allocated_size(NULL));

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
static const size_t kManyThreadsNumThreads = 200;

void *ManyThreadsWithStatsWorker(void *arg) {
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

#define ACCESS(ptr, offset) Ident(*(ptr + offset))

#define DIE_ON_ACCESS(ptr, offset) \
    EXPECT_DEATH(Ident(*(ptr + offset)), kUseAfterPoisonErrorMessage)

TEST(AddressSanitizerInterface, SimplePoisonMemoryRegionTest) {
  char *array = Ident((char*)malloc(120));
  // poison array[40..80)
  ASAN_POISON_MEMORY_REGION(array + 40, 40);
  ACCESS(array, 39);
  ACCESS(array, 80);
  DIE_ON_ACCESS(array, 40);
  DIE_ON_ACCESS(array, 60);
  DIE_ON_ACCESS(array, 79);
  ASAN_UNPOISON_MEMORY_REGION(array + 40, 40);
  // access previously poisoned memory.
  ACCESS(array, 40);
  ACCESS(array, 79);
  free(array);
}

TEST(AddressSanitizerInterface, OverlappingPoisonMemoryRegionTest) {
  char *array = Ident((char*)malloc(120));
  // Poison [0..40) and [80..120)
  ASAN_POISON_MEMORY_REGION(array, 40);
  ASAN_POISON_MEMORY_REGION(array + 80, 40);
  DIE_ON_ACCESS(array, 20);
  ACCESS(array, 60);
  DIE_ON_ACCESS(array, 100);
  // Poison whole array - [0..120)
  ASAN_POISON_MEMORY_REGION(array, 120);
  DIE_ON_ACCESS(array, 60);
  // Unpoison [24..96)
  ASAN_UNPOISON_MEMORY_REGION(array + 24, 72);
  DIE_ON_ACCESS(array, 23);
  ACCESS(array, 24);
  ACCESS(array, 60);
  ACCESS(array, 95);
  DIE_ON_ACCESS(array, 96);
  free(array);
}

TEST(AddressSanitizerInterface, PushAndPopWithPoisoningTest) {
  // Vector of capacity 20
  char *vec = Ident((char*)malloc(20));
  ASAN_POISON_MEMORY_REGION(vec, 20);
  for (size_t i = 0; i < 7; i++) {
    // Simulate push_back.
    ASAN_UNPOISON_MEMORY_REGION(vec + i, 1);
    ACCESS(vec, i);
    DIE_ON_ACCESS(vec, i + 1);
  }
  for (size_t i = 7; i > 0; i--) {
    // Simulate pop_back.
    ASAN_POISON_MEMORY_REGION(vec + i - 1, 1);
    DIE_ON_ACCESS(vec, i - 1);
    if (i > 1) ACCESS(vec, i - 2);
  }
  free(vec);
}

// Make sure that each aligned block of size "2^granularity" doesn't have
// "true" value before "false" value.
static void MakeShadowValid(bool *shadow, int length, int granularity) {
  bool can_be_poisoned = true;
  for (int i = length - 1; i >= 0; i--) {
    can_be_poisoned &= shadow[i];
    shadow[i] &= can_be_poisoned;
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
          ASAN_UNPOISON_MEMORY_REGION(arr, kSize);
          ASAN_POISON_MEMORY_REGION(arr + l1, s1);
          ASAN_POISON_MEMORY_REGION(arr + l2, s2);
          memset(expected, false, kSize);
          memset(expected + l1, true, s1);
          MakeShadowValid(expected, 24, /*granularity*/ 3);
          memset(expected + l2, true, s2);
          MakeShadowValid(expected, 24, /*granularity*/ 3);
          for (size_t i = 0; i < kSize; i++) {
            ASSERT_EQ(expected[i], __asan_address_is_poisoned(arr + i));
          }
          // Unpoison [l1, l1+s1) and [l2, l2+s2) and check result.
          ASAN_POISON_MEMORY_REGION(arr, kSize);
          ASAN_UNPOISON_MEMORY_REGION(arr + l1, s1);
          ASAN_UNPOISON_MEMORY_REGION(arr + l2, s2);
          memset(expected, true, kSize);
          memset(expected + l1, false, s1);
          MakeShadowValid(expected, 24, /*granularity*/ 3);
          memset(expected + l2, false, s2);
          MakeShadowValid(expected, 24, /*granularity*/ 3);
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
  ASAN_UNPOISON_MEMORY_REGION(array, 120);
  // Try to unpoison not owned memory
  EXPECT_DEATH(ASAN_UNPOISON_MEMORY_REGION(array, 121),
               kInvalidUnpoisonMessage);
  EXPECT_DEATH(ASAN_UNPOISON_MEMORY_REGION(array - 1, 120),
               kInvalidUnpoisonMessage);

  ASAN_POISON_MEMORY_REGION(array, 120);
  // Try to poison not owned memory.
  EXPECT_DEATH(ASAN_POISON_MEMORY_REGION(array, 121), kInvalidPoisonMessage);
  EXPECT_DEATH(ASAN_POISON_MEMORY_REGION(array - 1, 120),
               kInvalidPoisonMessage);
  free(array);
}

static void ErrorReportCallbackOneToZ(const char *report) {
  int len = strlen(report);
  char *dup = (char*)malloc(len);
  strcpy(dup, report);
  for (int i = 0; i < len; i++) {
    if (dup[i] == '1') dup[i] = 'Z';
  }
  write(2, dup, len);
  free(dup);
}

TEST(AddressSanitizerInterface, SetErrorReportCallbackTest) {
  __asan_set_error_report_callback(ErrorReportCallbackOneToZ);
  char *array = Ident((char*)malloc(120));
  EXPECT_DEATH(ACCESS(array, 120), "size Z");
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
