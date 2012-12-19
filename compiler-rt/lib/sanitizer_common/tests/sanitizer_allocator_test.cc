//===-- sanitizer_allocator_test.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Tests for sanitizer_allocator.h.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_common.h"

#include "gtest/gtest.h"

#include <stdlib.h>
#include <pthread.h>
#include <algorithm>
#include <vector>

// Too slow for debug build
#if TSAN_DEBUG == 0

#if SANITIZER_WORDSIZE == 64
static const uptr kAllocatorSpace = 0x700000000000ULL;
static const uptr kAllocatorSize  = 0x010000000000ULL;  // 1T.
static const u64 kAddressSpaceSize = 1ULL << 47;

typedef SizeClassAllocator64<
  kAllocatorSpace, kAllocatorSize, 16, DefaultSizeClassMap> Allocator64;

typedef SizeClassAllocator64<
  kAllocatorSpace, kAllocatorSize, 16, CompactSizeClassMap> Allocator64Compact;
#else
static const u64 kAddressSpaceSize = 1ULL << 32;
#endif

typedef SizeClassAllocator32<
  0, kAddressSpaceSize, 16, CompactSizeClassMap> Allocator32Compact;

template <class SizeClassMap>
void TestSizeClassMap() {
  typedef SizeClassMap SCMap;
#if 0
  for (uptr i = 0; i < SCMap::kNumClasses; i++) {
    printf("c%ld => %ld (%lx) cached=%ld(%ld)\n",
        i, SCMap::Size(i), SCMap::Size(i), SCMap::MaxCached(i) * SCMap::Size(i),
        SCMap::MaxCached(i));
  }
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

TEST(SanitizerCommon, DefaultSizeClassMap) {
  TestSizeClassMap<DefaultSizeClassMap>();
}

TEST(SanitizerCommon, CompactSizeClassMap) {
  TestSizeClassMap<CompactSizeClassMap>();
}

template <class Allocator>
void TestSizeClassAllocator() {
  Allocator *a = new Allocator;
  a->Init();

  static const uptr sizes[] = {1, 16, 30, 40, 100, 1000, 10000,
    50000, 60000, 100000, 120000, 300000, 500000, 1000000, 2000000};

  std::vector<void *> allocated;

  uptr last_total_allocated = 0;
  for (int i = 0; i < 3; i++) {
    // Allocate a bunch of chunks.
    for (uptr s = 0; s < ARRAY_SIZE(sizes); s++) {
      uptr size = sizes[s];
      if (!a->CanAllocate(size, 1)) continue;
      // printf("s = %ld\n", size);
      uptr n_iter = std::max((uptr)6, 10000000 / size);
      // fprintf(stderr, "size: %ld iter: %ld\n", size, n_iter);
      for (uptr i = 0; i < n_iter; i++) {
        char *x = (char*)a->Allocate(size, 1);
        x[0] = 0;
        x[size - 1] = 0;
        x[size / 2] = 0;
        allocated.push_back(x);
        CHECK_EQ(x, a->GetBlockBegin(x));
        CHECK_EQ(x, a->GetBlockBegin(x + size - 1));
        CHECK(a->PointerIsMine(x));
        CHECK(a->PointerIsMine(x + size - 1));
        CHECK(a->PointerIsMine(x + size / 2));
        CHECK_GE(a->GetActuallyAllocatedSize(x), size);
        uptr class_id = a->GetSizeClass(x);
        CHECK_EQ(class_id, Allocator::SizeClassMapT::ClassID(size));
        uptr *metadata = reinterpret_cast<uptr*>(a->GetMetaData(x));
        metadata[0] = reinterpret_cast<uptr>(x) + 1;
        metadata[1] = 0xABCD;
      }
    }
    // Deallocate all.
    for (uptr i = 0; i < allocated.size(); i++) {
      void *x = allocated[i];
      uptr *metadata = reinterpret_cast<uptr*>(a->GetMetaData(x));
      CHECK_EQ(metadata[0], reinterpret_cast<uptr>(x) + 1);
      CHECK_EQ(metadata[1], 0xABCD);
      a->Deallocate(x);
    }
    allocated.clear();
    uptr total_allocated = a->TotalMemoryUsed();
    if (last_total_allocated == 0)
      last_total_allocated = total_allocated;
    CHECK_EQ(last_total_allocated, total_allocated);
  }

  a->TestOnlyUnmap();
  delete a;
}

#if SANITIZER_WORDSIZE == 64
TEST(SanitizerCommon, SizeClassAllocator64) {
  TestSizeClassAllocator<Allocator64>();
}

TEST(SanitizerCommon, SizeClassAllocator64Compact) {
  TestSizeClassAllocator<Allocator64Compact>();
}
#endif

TEST(SanitizerCommon, SizeClassAllocator32Compact) {
  TestSizeClassAllocator<Allocator32Compact>();
}

template <class Allocator>
void SizeClassAllocatorMetadataStress() {
  Allocator *a = new Allocator;
  a->Init();
  static volatile void *sink;

  const uptr kNumAllocs = 10000;
  void *allocated[kNumAllocs];
  for (uptr i = 0; i < kNumAllocs; i++) {
    uptr size = (i % 4096) + 1;
    void *x = a->Allocate(size, 1);
    allocated[i] = x;
  }
  // Get Metadata kNumAllocs^2 times.
  for (uptr i = 0; i < kNumAllocs * kNumAllocs; i++) {
    sink = a->GetMetaData(allocated[i % kNumAllocs]);
  }
  for (uptr i = 0; i < kNumAllocs; i++) {
    a->Deallocate(allocated[i]);
  }

  a->TestOnlyUnmap();
  (void)sink;
  delete a;
}

#if SANITIZER_WORDSIZE == 64
TEST(SanitizerCommon, SizeClassAllocator64MetadataStress) {
  SizeClassAllocatorMetadataStress<Allocator64>();
}

TEST(SanitizerCommon, SizeClassAllocator64CompactMetadataStress) {
  SizeClassAllocatorMetadataStress<Allocator64Compact>();
}
#endif
TEST(SanitizerCommon, SizeClassAllocator32CompactMetadataStress) {
  SizeClassAllocatorMetadataStress<Allocator32Compact>();
}

struct TestMapUnmapCallback {
  static int map_count, unmap_count;
  void OnMap(uptr p, uptr size) const { map_count++; }
  void OnUnmap(uptr p, uptr size) const { unmap_count++; }
};
int TestMapUnmapCallback::map_count;
int TestMapUnmapCallback::unmap_count;

#if SANITIZER_WORDSIZE == 64
TEST(SanitizerCommon, SizeClassAllocator64MapUnmapCallback) {
  TestMapUnmapCallback::map_count = 0;
  TestMapUnmapCallback::unmap_count = 0;
  typedef SizeClassAllocator64<
      kAllocatorSpace, kAllocatorSize, 16, DefaultSizeClassMap,
      TestMapUnmapCallback> Allocator64WithCallBack;
  Allocator64WithCallBack *a = new Allocator64WithCallBack;
  a->Init();
  EXPECT_EQ(TestMapUnmapCallback::map_count, 1);  // Allocator state.
  a->Allocate(100, 1);
  EXPECT_EQ(TestMapUnmapCallback::map_count, 3);  // State + alloc + metadata.
  a->TestOnlyUnmap();
  EXPECT_EQ(TestMapUnmapCallback::unmap_count, 1);  // The whole thing.
  delete a;
}
#endif

TEST(SanitizerCommon, SizeClassAllocator32MapUnmapCallback) {
  TestMapUnmapCallback::map_count = 0;
  TestMapUnmapCallback::unmap_count = 0;
  typedef SizeClassAllocator32<
      0, kAddressSpaceSize, 16, CompactSizeClassMap,
      TestMapUnmapCallback> Allocator32WithCallBack;
  Allocator32WithCallBack *a = new Allocator32WithCallBack;
  a->Init();
  EXPECT_EQ(TestMapUnmapCallback::map_count, 1);  // Allocator state.
  a->Allocate(100, 1);
  EXPECT_EQ(TestMapUnmapCallback::map_count, 2);  // alloc.
  a->TestOnlyUnmap();
  EXPECT_EQ(TestMapUnmapCallback::unmap_count, 2);  // The whole thing + alloc.
  delete a;
  // fprintf(stderr, "Map: %d Unmap: %d\n",
  //         TestMapUnmapCallback::map_count,
  //         TestMapUnmapCallback::unmap_count);
}

TEST(SanitizerCommon, LargeMmapAllocatorMapUnmapCallback) {
  TestMapUnmapCallback::map_count = 0;
  TestMapUnmapCallback::unmap_count = 0;
  LargeMmapAllocator<TestMapUnmapCallback> a;
  a.Init();
  void *x = a.Allocate(1 << 20, 1);
  EXPECT_EQ(TestMapUnmapCallback::map_count, 1);
  a.Deallocate(x);
  EXPECT_EQ(TestMapUnmapCallback::unmap_count, 1);
}

template<class Allocator>
void FailInAssertionOnOOM() {
  Allocator a;
  a.Init();
  const uptr size = 1 << 20;
  for (int i = 0; i < 1000000; i++) {
    a.Allocate(size, 1);
  }

  a.TestOnlyUnmap();
}

#if SANITIZER_WORDSIZE == 64
TEST(SanitizerCommon, SizeClassAllocator64Overflow) {
  EXPECT_DEATH(FailInAssertionOnOOM<Allocator64>(), "Out of memory");
}
#endif

TEST(SanitizerCommon, LargeMmapAllocator) {
  LargeMmapAllocator<> a;
  a.Init();

  static const int kNumAllocs = 100;
  char *allocated[kNumAllocs];
  static const uptr size = 1000;
  // Allocate some.
  for (int i = 0; i < kNumAllocs; i++) {
    allocated[i] = (char *)a.Allocate(size, 1);
  }
  // Deallocate all.
  CHECK_GT(a.TotalMemoryUsed(), size * kNumAllocs);
  for (int i = 0; i < kNumAllocs; i++) {
    char *p = allocated[i];
    CHECK(a.PointerIsMine(p));
    a.Deallocate(p);
  }
  // Check that non left.
  CHECK_EQ(a.TotalMemoryUsed(), 0);

  // Allocate some more, also add metadata.
  for (int i = 0; i < kNumAllocs; i++) {
    char *x = (char *)a.Allocate(size, 1);
    CHECK_GE(a.GetActuallyAllocatedSize(x), size);
    uptr *meta = reinterpret_cast<uptr*>(a.GetMetaData(x));
    *meta = i;
    allocated[i] = x;
  }
  CHECK_GT(a.TotalMemoryUsed(), size * kNumAllocs);
  // Deallocate all in reverse order.
  for (int i = 0; i < kNumAllocs; i++) {
    int idx = kNumAllocs - i - 1;
    char *p = allocated[idx];
    uptr *meta = reinterpret_cast<uptr*>(a.GetMetaData(p));
    CHECK_EQ(*meta, idx);
    CHECK(a.PointerIsMine(p));
    a.Deallocate(p);
  }
  CHECK_EQ(a.TotalMemoryUsed(), 0);
  uptr max_alignment = SANITIZER_WORDSIZE == 64 ? (1 << 28) : (1 << 24);
  for (uptr alignment = 8; alignment <= max_alignment; alignment *= 2) {
    for (int i = 0; i < kNumAllocs; i++) {
      uptr size = ((i % 10) + 1) * 4096;
      char *p = allocated[i] = (char *)a.Allocate(size, alignment);
      CHECK_EQ(p, a.GetBlockBegin(p));
      CHECK_EQ(p, a.GetBlockBegin(p + size - 1));
      CHECK_EQ(p, a.GetBlockBegin(p + size / 2));
      CHECK_EQ(0, (uptr)allocated[i] % alignment);
      p[0] = p[size - 1] = 0;
    }
    for (int i = 0; i < kNumAllocs; i++) {
      a.Deallocate(allocated[i]);
    }
  }
}

template
<class PrimaryAllocator, class SecondaryAllocator, class AllocatorCache>
void TestCombinedAllocator() {
  typedef
      CombinedAllocator<PrimaryAllocator, AllocatorCache, SecondaryAllocator>
      Allocator;
  Allocator *a = new Allocator;
  a->Init();

  AllocatorCache cache;
  cache.Init();

  EXPECT_EQ(a->Allocate(&cache, -1, 1), (void*)0);
  EXPECT_EQ(a->Allocate(&cache, -1, 1024), (void*)0);
  EXPECT_EQ(a->Allocate(&cache, (uptr)-1 - 1024, 1), (void*)0);
  EXPECT_EQ(a->Allocate(&cache, (uptr)-1 - 1024, 1024), (void*)0);
  EXPECT_EQ(a->Allocate(&cache, (uptr)-1 - 1023, 1024), (void*)0);

  const uptr kNumAllocs = 100000;
  const uptr kNumIter = 10;
  for (uptr iter = 0; iter < kNumIter; iter++) {
    std::vector<void*> allocated;
    for (uptr i = 0; i < kNumAllocs; i++) {
      uptr size = (i % (1 << 14)) + 1;
      if ((i % 1024) == 0)
        size = 1 << (10 + (i % 14));
      void *x = a->Allocate(&cache, size, 1);
      uptr *meta = reinterpret_cast<uptr*>(a->GetMetaData(x));
      CHECK_EQ(*meta, 0);
      *meta = size;
      allocated.push_back(x);
    }

    random_shuffle(allocated.begin(), allocated.end());

    for (uptr i = 0; i < kNumAllocs; i++) {
      void *x = allocated[i];
      uptr *meta = reinterpret_cast<uptr*>(a->GetMetaData(x));
      CHECK_NE(*meta, 0);
      CHECK(a->PointerIsMine(x));
      *meta = 0;
      a->Deallocate(&cache, x);
    }
    allocated.clear();
    a->SwallowCache(&cache);
  }
  a->TestOnlyUnmap();
}

#if SANITIZER_WORDSIZE == 64
TEST(SanitizerCommon, CombinedAllocator64) {
  TestCombinedAllocator<Allocator64,
      LargeMmapAllocator<>,
      SizeClassAllocatorLocalCache<Allocator64> > ();
}

TEST(SanitizerCommon, CombinedAllocator64Compact) {
  TestCombinedAllocator<Allocator64Compact,
      LargeMmapAllocator<>,
      SizeClassAllocatorLocalCache<Allocator64Compact> > ();
}
#endif

TEST(SanitizerCommon, CombinedAllocator32Compact) {
  TestCombinedAllocator<Allocator32Compact,
      LargeMmapAllocator<>,
      SizeClassAllocatorLocalCache<Allocator32Compact> > ();
}

template <class AllocatorCache>
void TestSizeClassAllocatorLocalCache() {
  static THREADLOCAL AllocatorCache static_allocator_cache;
  static_allocator_cache.Init();
  AllocatorCache cache;
  typedef typename AllocatorCache::Allocator Allocator;
  Allocator *a = new Allocator();

  a->Init();
  cache.Init();

  const uptr kNumAllocs = 10000;
  const int kNumIter = 100;
  uptr saved_total = 0;
  for (int i = 0; i < kNumIter; i++) {
    void *allocated[kNumAllocs];
    for (uptr i = 0; i < kNumAllocs; i++) {
      allocated[i] = cache.Allocate(a, 0);
    }
    for (uptr i = 0; i < kNumAllocs; i++) {
      cache.Deallocate(a, 0, allocated[i]);
    }
    cache.Drain(a);
    uptr total_allocated = a->TotalMemoryUsed();
    if (saved_total)
      CHECK_EQ(saved_total, total_allocated);
    saved_total = total_allocated;
  }

  a->TestOnlyUnmap();
  delete a;
}

#if SANITIZER_WORDSIZE == 64
TEST(SanitizerCommon, SizeClassAllocator64LocalCache) {
  TestSizeClassAllocatorLocalCache<
      SizeClassAllocatorLocalCache<Allocator64> >();
}

TEST(SanitizerCommon, SizeClassAllocator64CompactLocalCache) {
  TestSizeClassAllocatorLocalCache<
      SizeClassAllocatorLocalCache<Allocator64Compact> >();
}
#endif

TEST(SanitizerCommon, SizeClassAllocator32CompactLocalCache) {
  TestSizeClassAllocatorLocalCache<
      SizeClassAllocatorLocalCache<Allocator32Compact> >();
}

#if SANITIZER_WORDSIZE == 64
typedef SizeClassAllocatorLocalCache<Allocator64> AllocatorCache;
static THREADLOCAL AllocatorCache static_allocator_cache;

void *AllocatorLeakTestWorker(void *arg) {
  typedef AllocatorCache::Allocator Allocator;
  Allocator *a = (Allocator*)(arg);
  static_allocator_cache.Allocate(a, 10);
  static_allocator_cache.Drain(a);
  return 0;
}

TEST(SanitizerCommon, AllocatorLeakTest) {
  typedef AllocatorCache::Allocator Allocator;
  Allocator a;
  a.Init();
  uptr total_used_memory = 0;
  for (int i = 0; i < 100; i++) {
    pthread_t t;
    EXPECT_EQ(0, pthread_create(&t, 0, AllocatorLeakTestWorker, &a));
    EXPECT_EQ(0, pthread_join(t, 0));
    if (i == 0)
      total_used_memory = a.TotalMemoryUsed();
    EXPECT_EQ(a.TotalMemoryUsed(), total_used_memory);
  }

  a.TestOnlyUnmap();
}
#endif

TEST(Allocator, Basic) {
  char *p = (char*)InternalAlloc(10);
  EXPECT_NE(p, (char*)0);
  char *p2 = (char*)InternalAlloc(20);
  EXPECT_NE(p2, (char*)0);
  EXPECT_NE(p2, p);
  InternalFree(p);
  InternalFree(p2);
}

TEST(Allocator, Stress) {
  const int kCount = 1000;
  char *ptrs[kCount];
  unsigned rnd = 42;
  for (int i = 0; i < kCount; i++) {
    uptr sz = rand_r(&rnd) % 1000;
    char *p = (char*)InternalAlloc(sz);
    EXPECT_NE(p, (char*)0);
    ptrs[i] = p;
  }
  for (int i = 0; i < kCount; i++) {
    InternalFree(ptrs[i]);
  }
}

TEST(Allocator, ScopedBuffer) {
  const int kSize = 512;
  {
    InternalScopedBuffer<int> int_buf(kSize);
    EXPECT_EQ(sizeof(int) * kSize, int_buf.size());  // NOLINT
  }
  InternalScopedBuffer<char> char_buf(kSize);
  EXPECT_EQ(sizeof(char) * kSize, char_buf.size());  // NOLINT
  internal_memset(char_buf.data(), 'c', kSize);
  for (int i = 0; i < kSize; i++) {
    EXPECT_EQ('c', char_buf[i]);
  }
}

#endif  // #if TSAN_DEBUG==0
