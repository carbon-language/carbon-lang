//===-- tsan_clock_test.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_clock.h"
#include "tsan_rtl.h"
#include "gtest/gtest.h"
#include <sys/time.h>
#include <time.h>

namespace __tsan {

ClockCache cache;

TEST(Clock, VectorBasic) {
  ThreadClock clk(0);
  ASSERT_EQ(clk.size(), 1U);
  clk.tick();
  ASSERT_EQ(clk.size(), 1U);
  ASSERT_EQ(clk.get(0), 1U);
  clk.set(&cache, 3, clk.get(3) + 1);
  ASSERT_EQ(clk.size(), 4U);
  ASSERT_EQ(clk.get(0), 1U);
  ASSERT_EQ(clk.get(1), 0U);
  ASSERT_EQ(clk.get(2), 0U);
  ASSERT_EQ(clk.get(3), 1U);
  clk.set(&cache, 3, clk.get(3) + 1);
  ASSERT_EQ(clk.get(3), 2U);
}

TEST(Clock, ChunkedBasic) {
  ThreadClock vector(0);
  SyncClock chunked;
  ASSERT_EQ(vector.size(), 1U);
  ASSERT_EQ(chunked.size(), 0U);
  vector.acquire(&cache, &chunked);
  ASSERT_EQ(vector.size(), 1U);
  ASSERT_EQ(chunked.size(), 0U);
  vector.release(&cache, &chunked);
  ASSERT_EQ(vector.size(), 1U);
  ASSERT_EQ(chunked.size(), 1U);
  vector.acq_rel(&cache, &chunked);
  ASSERT_EQ(vector.size(), 1U);
  ASSERT_EQ(chunked.size(), 1U);
  chunked.Reset(&cache);
}

static const uptr interesting_sizes[] = {0, 1, 2, 30, 61, 62, 63, 64, 65, 66,
    100, 124, 125, 126, 127, 128, 129, 130, 188, 189, 190, 191, 192, 193, 254,
    255};

TEST(Clock, Iter) {
  const uptr n = ARRAY_SIZE(interesting_sizes);
  for (uptr fi = 0; fi < n; fi++) {
    const uptr size = interesting_sizes[fi];
    SyncClock sync;
    ThreadClock vector(0);
    for (uptr i = 0; i < size; i++)
      vector.set(&cache, i, i + 1);
    if (size != 0)
      vector.release(&cache, &sync);
    uptr i = 0;
    for (ClockElem &ce : sync) {
      ASSERT_LT(i, size);
      ASSERT_EQ(sync.get_clean(i), ce.epoch);
      i++;
    }
    ASSERT_EQ(i, size);
    sync.Reset(&cache);
  }
}

TEST(Clock, AcquireRelease) {
  ThreadClock vector1(100);
  vector1.tick();
  SyncClock chunked;
  vector1.release(&cache, &chunked);
  ASSERT_EQ(chunked.size(), 101U);
  ThreadClock vector2(0);
  vector2.acquire(&cache, &chunked);
  ASSERT_EQ(vector2.size(), 101U);
  ASSERT_EQ(vector2.get(0), 0U);
  ASSERT_EQ(vector2.get(1), 0U);
  ASSERT_EQ(vector2.get(99), 0U);
  ASSERT_EQ(vector2.get(100), 1U);
  chunked.Reset(&cache);
}

TEST(Clock, RepeatedAcquire) {
  ThreadClock thr1(1);
  thr1.tick();
  ThreadClock thr2(2);
  thr2.tick();

  SyncClock sync;
  thr1.ReleaseStore(&cache, &sync);

  thr2.acquire(&cache, &sync);
  thr2.acquire(&cache, &sync);

  sync.Reset(&cache);
}

TEST(Clock, releaseStoreAcquire) {
  ThreadClock thr0(0);
  thr0.tick();
  ThreadClock thr1(1);
  thr1.tick();
  SyncClock syncA;
  SyncClock syncB;
  ASSERT_EQ(syncA.size(), 0U);
  ASSERT_EQ(syncB.size(), 0U);
  thr1.releaseStoreAcquire(&cache, &syncB);
  ASSERT_EQ(syncB.size(), 2U); // T0 and T1
  // releaseStoreAcquire to an empty SyncClock
  thr0.releaseStoreAcquire(&cache, &syncA);
  ASSERT_EQ(syncA.size(), 1U);
  // releaseStoreAcquire from a non-empty SyncClock
  // T0 learns about T1
  thr0.releaseStoreAcquire(&cache, &syncB);
  // releaseStoreAcquire to the originally empty SyncClock
  // T0 deposits info about T1 into syncA
  thr0.releaseStoreAcquire(&cache, &syncA);
  ASSERT_EQ(syncA.size(), 2U);
  syncA.Reset(&cache);
  syncB.Reset(&cache);
}

TEST(Clock, ManyThreads) {
  SyncClock chunked;
  for (unsigned i = 0; i < 200; i++) {
    ThreadClock vector(0);
    vector.tick();
    vector.set(&cache, i, i + 1);
    vector.release(&cache, &chunked);
    ASSERT_EQ(i + 1, chunked.size());
    vector.acquire(&cache, &chunked);
    ASSERT_EQ(i + 1, vector.size());
  }

  for (unsigned i = 0; i < 200; i++) {
    printf("i=%d\n", i);
    ASSERT_EQ(i + 1, chunked.get(i));
  }

  ThreadClock vector(1);
  vector.acquire(&cache, &chunked);
  ASSERT_EQ(200U, vector.size());
  for (unsigned i = 0; i < 200; i++)
    ASSERT_EQ(i + 1, vector.get(i));

  chunked.Reset(&cache);
}

TEST(Clock, DifferentSizes) {
  {
    ThreadClock vector1(10);
    vector1.tick();
    ThreadClock vector2(20);
    vector2.tick();
    {
      SyncClock chunked;
      vector1.release(&cache, &chunked);
      ASSERT_EQ(chunked.size(), 11U);
      vector2.release(&cache, &chunked);
      ASSERT_EQ(chunked.size(), 21U);
      chunked.Reset(&cache);
    }
    {
      SyncClock chunked;
      vector2.release(&cache, &chunked);
      ASSERT_EQ(chunked.size(), 21U);
      vector1.release(&cache, &chunked);
      ASSERT_EQ(chunked.size(), 21U);
      chunked.Reset(&cache);
    }
    {
      SyncClock chunked;
      vector1.release(&cache, &chunked);
      vector2.acquire(&cache, &chunked);
      ASSERT_EQ(vector2.size(), 21U);
      chunked.Reset(&cache);
    }
    {
      SyncClock chunked;
      vector2.release(&cache, &chunked);
      vector1.acquire(&cache, &chunked);
      ASSERT_EQ(vector1.size(), 21U);
      chunked.Reset(&cache);
    }
  }
}

TEST(Clock, Growth) {
  {
    ThreadClock vector(10);
    vector.tick();
    vector.set(&cache, 5, 42);
    SyncClock sync;
    vector.release(&cache, &sync);
    ASSERT_EQ(sync.size(), 11U);
    ASSERT_EQ(sync.get(0), 0ULL);
    ASSERT_EQ(sync.get(1), 0ULL);
    ASSERT_EQ(sync.get(5), 42ULL);
    ASSERT_EQ(sync.get(9), 0ULL);
    ASSERT_EQ(sync.get(10), 1ULL);
    sync.Reset(&cache);
  }
  {
    ThreadClock vector1(10);
    vector1.tick();
    ThreadClock vector2(20);
    vector2.tick();
    SyncClock sync;
    vector1.release(&cache, &sync);
    vector2.release(&cache, &sync);
    ASSERT_EQ(sync.size(), 21U);
    ASSERT_EQ(sync.get(0), 0ULL);
    ASSERT_EQ(sync.get(10), 1ULL);
    ASSERT_EQ(sync.get(19), 0ULL);
    ASSERT_EQ(sync.get(20), 1ULL);
    sync.Reset(&cache);
  }
  {
    ThreadClock vector(100);
    vector.tick();
    vector.set(&cache, 5, 42);
    vector.set(&cache, 90, 84);
    SyncClock sync;
    vector.release(&cache, &sync);
    ASSERT_EQ(sync.size(), 101U);
    ASSERT_EQ(sync.get(0), 0ULL);
    ASSERT_EQ(sync.get(1), 0ULL);
    ASSERT_EQ(sync.get(5), 42ULL);
    ASSERT_EQ(sync.get(60), 0ULL);
    ASSERT_EQ(sync.get(70), 0ULL);
    ASSERT_EQ(sync.get(90), 84ULL);
    ASSERT_EQ(sync.get(99), 0ULL);
    ASSERT_EQ(sync.get(100), 1ULL);
    sync.Reset(&cache);
  }
  {
    ThreadClock vector1(10);
    vector1.tick();
    ThreadClock vector2(100);
    vector2.tick();
    SyncClock sync;
    vector1.release(&cache, &sync);
    vector2.release(&cache, &sync);
    ASSERT_EQ(sync.size(), 101U);
    ASSERT_EQ(sync.get(0), 0ULL);
    ASSERT_EQ(sync.get(10), 1ULL);
    ASSERT_EQ(sync.get(99), 0ULL);
    ASSERT_EQ(sync.get(100), 1ULL);
    sync.Reset(&cache);
  }
}

TEST(Clock, Growth2) {
  // Test clock growth for every pair of sizes:
  const uptr n = ARRAY_SIZE(interesting_sizes);
  for (uptr fi = 0; fi < n; fi++) {
    for (uptr ti = fi + 1; ti < n; ti++) {
      const uptr from = interesting_sizes[fi];
      const uptr to = interesting_sizes[ti];
      SyncClock sync;
      ThreadClock vector(0);
      for (uptr i = 0; i < from; i++)
        vector.set(&cache, i, i + 1);
      if (from != 0)
        vector.release(&cache, &sync);
      ASSERT_EQ(sync.size(), from);
      for (uptr i = 0; i < from; i++)
        ASSERT_EQ(sync.get(i), i + 1);
      for (uptr i = 0; i < to; i++)
        vector.set(&cache, i, i + 1);
      vector.release(&cache, &sync);
      ASSERT_EQ(sync.size(), to);
      for (uptr i = 0; i < to; i++)
        ASSERT_EQ(sync.get(i), i + 1);
      vector.set(&cache, to + 1, to + 1);
      vector.release(&cache, &sync);
      ASSERT_EQ(sync.size(), to + 2);
      for (uptr i = 0; i < to; i++)
        ASSERT_EQ(sync.get(i), i + 1);
      ASSERT_EQ(sync.get(to), 0U);
      ASSERT_EQ(sync.get(to + 1), to + 1);
      sync.Reset(&cache);
    }
  }
}

const uptr kThreads = 4;
const uptr kClocks = 4;

// SimpleSyncClock and SimpleThreadClock implement the same thing as
// SyncClock and ThreadClock, but in a very simple way.
struct SimpleSyncClock {
  u64 clock[kThreads];
  uptr size;

  SimpleSyncClock() {
    Reset();
  }

  void Reset() {
    size = 0;
    for (uptr i = 0; i < kThreads; i++)
      clock[i] = 0;
  }

  bool verify(const SyncClock *other) const {
    for (uptr i = 0; i < min(size, other->size()); i++) {
      if (clock[i] != other->get(i))
        return false;
    }
    for (uptr i = min(size, other->size()); i < max(size, other->size()); i++) {
      if (i < size && clock[i] != 0)
        return false;
      if (i < other->size() && other->get(i) != 0)
        return false;
    }
    return true;
  }
};

struct SimpleThreadClock {
  u64 clock[kThreads];
  uptr size;
  unsigned tid;

  explicit SimpleThreadClock(unsigned tid) {
    this->tid = tid;
    size = tid + 1;
    for (uptr i = 0; i < kThreads; i++)
      clock[i] = 0;
  }

  void tick() {
    clock[tid]++;
  }

  void acquire(const SimpleSyncClock *src) {
    if (size < src->size)
      size = src->size;
    for (uptr i = 0; i < kThreads; i++)
      clock[i] = max(clock[i], src->clock[i]);
  }

  void release(SimpleSyncClock *dst) const {
    if (dst->size < size)
      dst->size = size;
    for (uptr i = 0; i < kThreads; i++)
      dst->clock[i] = max(dst->clock[i], clock[i]);
  }

  void releaseStoreAcquire(SimpleSyncClock *sc) {
    if (sc->size < size)
      sc->size = size;
    else
      size = sc->size;
    for (uptr i = 0; i < kThreads; i++) {
      uptr tmp = clock[i];
      clock[i] = max(sc->clock[i], clock[i]);
      sc->clock[i] = tmp;
    }
  }

  void acq_rel(SimpleSyncClock *dst) {
    acquire(dst);
    release(dst);
  }

  void ReleaseStore(SimpleSyncClock *dst) const {
    if (dst->size < size)
      dst->size = size;
    for (uptr i = 0; i < kThreads; i++)
      dst->clock[i] = clock[i];
  }

  bool verify(const ThreadClock *other) const {
    for (uptr i = 0; i < min(size, other->size()); i++) {
      if (clock[i] != other->get(i))
        return false;
    }
    for (uptr i = min(size, other->size()); i < max(size, other->size()); i++) {
      if (i < size && clock[i] != 0)
        return false;
      if (i < other->size() && other->get(i) != 0)
        return false;
    }
    return true;
  }
};

static bool ClockFuzzer(bool printing) {
  // Create kThreads thread clocks.
  SimpleThreadClock *thr0[kThreads];
  ThreadClock *thr1[kThreads];
  unsigned reused[kThreads];
  for (unsigned i = 0; i < kThreads; i++) {
    reused[i] = 0;
    thr0[i] = new SimpleThreadClock(i);
    thr1[i] = new ThreadClock(i, reused[i]);
  }

  // Create kClocks sync clocks.
  SimpleSyncClock *sync0[kClocks];
  SyncClock *sync1[kClocks];
  for (unsigned i = 0; i < kClocks; i++) {
    sync0[i] = new SimpleSyncClock();
    sync1[i] = new SyncClock();
  }

  // Do N random operations (acquire, release, etc) and compare results
  // for SimpleThread/SyncClock and real Thread/SyncClock.
  for (int i = 0; i < 10000; i++) {
    unsigned tid = rand() % kThreads;
    unsigned cid = rand() % kClocks;
    thr0[tid]->tick();
    thr1[tid]->tick();

    switch (rand() % 7) {
    case 0:
      if (printing)
        printf("acquire thr%d <- clk%d\n", tid, cid);
      thr0[tid]->acquire(sync0[cid]);
      thr1[tid]->acquire(&cache, sync1[cid]);
      break;
    case 1:
      if (printing)
        printf("release thr%d -> clk%d\n", tid, cid);
      thr0[tid]->release(sync0[cid]);
      thr1[tid]->release(&cache, sync1[cid]);
      break;
    case 2:
      if (printing)
        printf("acq_rel thr%d <> clk%d\n", tid, cid);
      thr0[tid]->acq_rel(sync0[cid]);
      thr1[tid]->acq_rel(&cache, sync1[cid]);
      break;
    case 3:
      if (printing)
        printf("rel_str thr%d >> clk%d\n", tid, cid);
      thr0[tid]->ReleaseStore(sync0[cid]);
      thr1[tid]->ReleaseStore(&cache, sync1[cid]);
      break;
    case 4:
      if (printing)
        printf("reset clk%d\n", cid);
      sync0[cid]->Reset();
      sync1[cid]->Reset(&cache);
      break;
    case 5:
      if (printing)
        printf("releaseStoreAcquire thr%d -> clk%d\n", tid, cid);
      thr0[tid]->releaseStoreAcquire(sync0[cid]);
      thr1[tid]->releaseStoreAcquire(&cache, sync1[cid]);
      break;
    case 6:
      if (printing)
        printf("reset thr%d\n", tid);
      u64 epoch = thr0[tid]->clock[tid] + 1;
      reused[tid]++;
      delete thr0[tid];
      thr0[tid] = new SimpleThreadClock(tid);
      thr0[tid]->clock[tid] = epoch;
      delete thr1[tid];
      thr1[tid] = new ThreadClock(tid, reused[tid]);
      thr1[tid]->set(epoch);
      break;
    }

    if (printing) {
      for (unsigned i = 0; i < kThreads; i++) {
        printf("thr%d: ", i);
        thr1[i]->DebugDump(printf);
        printf("\n");
      }
      for (unsigned i = 0; i < kClocks; i++) {
        printf("clk%d: ", i);
        sync1[i]->DebugDump(printf);
        printf("\n");
      }

      printf("\n");
    }

    if (!thr0[tid]->verify(thr1[tid]) || !sync0[cid]->verify(sync1[cid])) {
      if (!printing)
        return false;
      printf("differs with model:\n");
      for (unsigned i = 0; i < kThreads; i++) {
        printf("thr%d: clock=[", i);
        for (uptr j = 0; j < thr0[i]->size; j++)
          printf("%s%llu", j == 0 ? "" : ",", thr0[i]->clock[j]);
        printf("]\n");
      }
      for (unsigned i = 0; i < kClocks; i++) {
        printf("clk%d: clock=[", i);
        for (uptr j = 0; j < sync0[i]->size; j++)
          printf("%s%llu", j == 0 ? "" : ",", sync0[i]->clock[j]);
        printf("]\n");
      }
      return false;
    }
  }

  for (unsigned i = 0; i < kClocks; i++) {
    sync1[i]->Reset(&cache);
  }
  return true;
}

TEST(Clock, Fuzzer) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  int seed = tv.tv_sec + tv.tv_usec;
  printf("seed=%d\n", seed);
  srand(seed);
  if (!ClockFuzzer(false)) {
    // Redo the test with the same seed, but logging operations.
    srand(seed);
    ClockFuzzer(true);
    ASSERT_TRUE(false);
  }
}

}  // namespace __tsan
