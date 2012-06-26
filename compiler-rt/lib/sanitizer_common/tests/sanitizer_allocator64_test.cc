//===-- sanitizer_allocator64_test.cc -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Tests for sanitizer_allocator64.h.
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_allocator64.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

TEST(SanitizerCommon, DefaultSizeClassMap) {
  typedef DefaultSizeClassMap SCMap;

#if 0
  for (uptr i = 0; i < SCMap::kNumClasses; i++) {
    // printf("% 3ld: % 5ld (%4lx);   ", i, SCMap::Size(i), SCMap::Size(i));
    printf("c%ld => %ld  ", i, SCMap::Size(i));
    if ((i % 8) == 7)
      printf("\n");
  }
  printf("\n");
#endif

  for (uptr c = 0; c < SCMap::kNumClasses; c++) {
    uptr s = SCMap::Size(c);
    CHECK_EQ(SCMap::ClassID(s), c);
    if (c != SCMap::kNumClasses - 1)
      CHECK_EQ(SCMap::ClassID(s + 1), c + 1);
    CHECK_EQ(SCMap::ClassID(s - 1), c);
    if (c)
      CHECK_GT(SCMap::Size(c), SCMap::Size(c-1));
  }
  CHECK_EQ(SCMap::ClassID(SCMap::kMaxSize + 1), 0);

  for (uptr s = 1; s <= SCMap::kMaxSize; s++) {
    uptr c = SCMap::ClassID(s);
    CHECK_LT(c, SCMap::kNumClasses);
    CHECK_GE(SCMap::Size(c), s);
    if (c > 0)
      CHECK_LT(SCMap::Size(c-1), s);
  }
}

static const uptr kAllocatorSpace = 0x600000000000ULL;
static const uptr kAllocatorSize = 0x10000000000;  // 1T.

TEST(SanitizerCommon, SizeClassAllocator64) {
  typedef DefaultSizeClassMap SCMap;
  typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize,
                               16, SCMap> Allocator;

  Allocator a;
  a.Init();

  static const uptr sizes[] = {1, 16, 30, 40, 100, 1000, 10000,
    50000, 60000, 100000, 300000, 500000, 1000000, 2000000};

  std::vector<void *> allocated;

  uptr last_total_allocated = 0;
  for (int i = 0; i < 5; i++) {
    // Allocate a bunch of chunks.
    for (uptr s = 0; s < sizeof(sizes) /sizeof(sizes[0]); s++) {
      uptr size = sizes[s];
      // printf("s = %ld\n", size);
      uptr n_iter = std::max((uptr)2, 1000000 / size);
      for (uptr i = 0; i < n_iter; i++) {
        void *x = a.Allocate(size);
        allocated.push_back(x);
        CHECK(a.PointerIsMine(x));
        uptr class_id = a.GetSizeClass(x);
        CHECK_EQ(class_id, SCMap::ClassID(size));
        uptr *metadata = reinterpret_cast<uptr*>(a.GetMetaData(x));
        metadata[0] = reinterpret_cast<uptr>(x) + 1;
        metadata[1] = 0xABCD;
      }
    }
    // Deallocate all.
    for (uptr i = 0; i < allocated.size(); i++) {
      void *x = allocated[i];
      uptr *metadata = reinterpret_cast<uptr*>(a.GetMetaData(x));
      CHECK_EQ(metadata[0], reinterpret_cast<uptr>(x) + 1);
      CHECK_EQ(metadata[1], 0xABCD);
      a.Deallocate(x);
    }
    allocated.clear();
    uptr total_allocated = a.TotalMemoryUsed();
    if (last_total_allocated == 0)
      last_total_allocated = total_allocated;
    CHECK_EQ(last_total_allocated, total_allocated);
  }

  a.TestOnlyUnmap();
}


TEST(SanitizerCommon, SizeClassAllocator64MetadataStress) {
  typedef DefaultSizeClassMap SCMap;
  typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize,
          16, SCMap> Allocator;
  Allocator a;
  a.Init();
  static volatile void *sink;

  const uptr kNumAllocs = 10000;
  void *allocated[kNumAllocs];
  for (uptr i = 0; i < kNumAllocs; i++) {
    uptr size = (i % 4096) + 1;
    void *x = a.Allocate(size);
    allocated[i] = x;
  }
  // Get Metadata kNumAllocs^2 times.
  for (uptr i = 0; i < kNumAllocs * kNumAllocs; i++) {
    sink = a.GetMetaData(allocated[i % kNumAllocs]);
  }
  for (uptr i = 0; i < kNumAllocs; i++) {
    a.Deallocate(allocated[i]);
  }

  a.TestOnlyUnmap();
}

void FailInAssertionOnOOM() {
  typedef DefaultSizeClassMap SCMap;
  typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize,
          16, SCMap> Allocator;
  Allocator a;
  a.Init();
  const uptr size = 1 << 20;
  for (int i = 0; i < 1000000; i++) {
    a.Allocate(size);
  }

  a.TestOnlyUnmap();
}

TEST(SanitizerCommon, SizeClassAllocator64Overflow) {
  EXPECT_DEATH(FailInAssertionOnOOM(),
               "allocated_user.*allocated_meta.*kRegionSize");
}

TEST(SanitizerCommon, LargeMmapAllocator) {
  LargeMmapAllocator a;
  a.Init();

  static const int kNumAllocs = 100;
  void *allocated[kNumAllocs];
  static const uptr size = 1000;
  // Allocate some.
  for (int i = 0; i < kNumAllocs; i++) {
    allocated[i] = a.Allocate(size);
  }
  // Deallocate all.
  CHECK_GT(a.TotalMemoryUsed(), size * kNumAllocs);
  for (int i = 0; i < kNumAllocs; i++) {
    void *p = allocated[i];
    CHECK(a.PointerIsMine(p));
    a.Deallocate(p);
  }
  // Check that non left.
  CHECK_EQ(a.TotalMemoryUsed(), 0);

  // Allocate some more, also add metadata.
  for (int i = 0; i < kNumAllocs; i++) {
    void *x = a.Allocate(size);
    uptr *meta = reinterpret_cast<uptr*>(a.GetMetaData(x));
    *meta = i;
    allocated[i] = x;
  }
  CHECK_GT(a.TotalMemoryUsed(), size * kNumAllocs);
  // Deallocate all in reverse order.
  for (int i = 0; i < kNumAllocs; i++) {
    int idx = kNumAllocs - i - 1;
    void *p = allocated[idx];
    uptr *meta = reinterpret_cast<uptr*>(a.GetMetaData(p));
    CHECK_EQ(*meta, idx);
    CHECK(a.PointerIsMine(p));
    a.Deallocate(p);
  }
  CHECK_EQ(a.TotalMemoryUsed(), 0);
}
