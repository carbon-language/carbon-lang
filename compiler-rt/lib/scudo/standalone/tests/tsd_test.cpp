//===-- tsd_test.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "tsd_exclusive.h"
#include "tsd_shared.h"

#include <stdlib.h>

#include <condition_variable>
#include <mutex>
#include <set>
#include <thread>

// We mock out an allocator with a TSD registry, mostly using empty stubs. The
// cache contains a single volatile uptr, to be able to test that several
// concurrent threads will not access or modify the same cache at the same time.
template <class Config> class MockAllocator {
public:
  using ThisT = MockAllocator<Config>;
  using TSDRegistryT = typename Config::template TSDRegistryT<ThisT>;
  using CacheT = struct MockCache { volatile scudo::uptr Canary; };
  using QuarantineCacheT = struct MockQuarantine {};

  void init() {
    // This should only be called once by the registry.
    EXPECT_FALSE(Initialized);
    Initialized = true;
  }

  void unmapTestOnly() { TSDRegistry.unmapTestOnly(this); }
  void initCache(CacheT *Cache) { *Cache = {}; }
  void commitBack(scudo::TSD<MockAllocator> *TSD) {}
  TSDRegistryT *getTSDRegistry() { return &TSDRegistry; }
  void callPostInitCallback() {}

  bool isInitialized() { return Initialized; }

  void *operator new(size_t Size) {
    void *P = nullptr;
    EXPECT_EQ(0, posix_memalign(&P, alignof(ThisT), Size));
    return P;
  }
  void operator delete(void *P) { free(P); }

private:
  bool Initialized = false;
  TSDRegistryT TSDRegistry;
};

struct OneCache {
  template <class Allocator>
  using TSDRegistryT = scudo::TSDRegistrySharedT<Allocator, 1U, 1U>;
};

struct SharedCaches {
  template <class Allocator>
  using TSDRegistryT = scudo::TSDRegistrySharedT<Allocator, 16U, 8U>;
};

struct ExclusiveCaches {
  template <class Allocator>
  using TSDRegistryT = scudo::TSDRegistryExT<Allocator>;
};

TEST(ScudoTSDTest, TSDRegistryInit) {
  using AllocatorT = MockAllocator<OneCache>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  EXPECT_FALSE(Allocator->isInitialized());

  auto Registry = Allocator->getTSDRegistry();
  Registry->init(Allocator.get());
  EXPECT_TRUE(Allocator->isInitialized());
}

template <class AllocatorT> static void testRegistry() {
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  EXPECT_FALSE(Allocator->isInitialized());

  auto Registry = Allocator->getTSDRegistry();
  Registry->initThreadMaybe(Allocator.get(), /*MinimalInit=*/true);
  EXPECT_TRUE(Allocator->isInitialized());

  bool UnlockRequired;
  auto TSD = Registry->getTSDAndLock(&UnlockRequired);
  EXPECT_NE(TSD, nullptr);
  EXPECT_EQ(TSD->Cache.Canary, 0U);
  if (UnlockRequired)
    TSD->unlock();

  Registry->initThreadMaybe(Allocator.get(), /*MinimalInit=*/false);
  TSD = Registry->getTSDAndLock(&UnlockRequired);
  EXPECT_NE(TSD, nullptr);
  EXPECT_EQ(TSD->Cache.Canary, 0U);
  memset(&TSD->Cache, 0x42, sizeof(TSD->Cache));
  if (UnlockRequired)
    TSD->unlock();
}

TEST(ScudoTSDTest, TSDRegistryBasic) {
  testRegistry<MockAllocator<OneCache>>();
  testRegistry<MockAllocator<SharedCaches>>();
#if !SCUDO_FUCHSIA
  testRegistry<MockAllocator<ExclusiveCaches>>();
#endif
}

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready;

template <typename AllocatorT> static void stressCache(AllocatorT *Allocator) {
  auto Registry = Allocator->getTSDRegistry();
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  Registry->initThreadMaybe(Allocator, /*MinimalInit=*/false);
  bool UnlockRequired;
  auto TSD = Registry->getTSDAndLock(&UnlockRequired);
  EXPECT_NE(TSD, nullptr);
  // For an exclusive TSD, the cache should be empty. We cannot guarantee the
  // same for a shared TSD.
  if (!UnlockRequired)
    EXPECT_EQ(TSD->Cache.Canary, 0U);
  // Transform the thread id to a uptr to use it as canary.
  const scudo::uptr Canary = static_cast<scudo::uptr>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  TSD->Cache.Canary = Canary;
  // Loop a few times to make sure that a concurrent thread isn't modifying it.
  for (scudo::uptr I = 0; I < 4096U; I++)
    EXPECT_EQ(TSD->Cache.Canary, Canary);
  if (UnlockRequired)
    TSD->unlock();
}

template <class AllocatorT> static void testRegistryThreaded() {
  Ready = false;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  std::thread Threads[32];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread(stressCache<AllocatorT>, Allocator.get());
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
}

TEST(ScudoTSDTest, TSDRegistryThreaded) {
  testRegistryThreaded<MockAllocator<OneCache>>();
  testRegistryThreaded<MockAllocator<SharedCaches>>();
#if !SCUDO_FUCHSIA
  testRegistryThreaded<MockAllocator<ExclusiveCaches>>();
#endif
}

static std::set<void *> Pointers;

static void stressSharedRegistry(MockAllocator<SharedCaches> *Allocator) {
  std::set<void *> Set;
  auto Registry = Allocator->getTSDRegistry();
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  Registry->initThreadMaybe(Allocator, /*MinimalInit=*/false);
  bool UnlockRequired;
  for (scudo::uptr I = 0; I < 4096U; I++) {
    auto TSD = Registry->getTSDAndLock(&UnlockRequired);
    EXPECT_NE(TSD, nullptr);
    Set.insert(reinterpret_cast<void *>(TSD));
    if (UnlockRequired)
      TSD->unlock();
  }
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Pointers.insert(Set.begin(), Set.end());
  }
}

TEST(ScudoTSDTest, TSDRegistryTSDsCount) {
  Ready = false;
  Pointers.clear();
  using AllocatorT = MockAllocator<SharedCaches>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  // We attempt to use as many TSDs as the shared cache offers by creating a
  // decent amount of threads that will be run concurrently and attempt to get
  // and lock TSDs. We put them all in a set and count the number of entries
  // after we are done.
  std::thread Threads[32];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread(stressSharedRegistry, Allocator.get());
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
  // The initial number of TSDs we get will be the minimum of the default count
  // and the number of CPUs.
  EXPECT_LE(Pointers.size(), 8U);
  Pointers.clear();
  auto Registry = Allocator->getTSDRegistry();
  // Increase the number of TSDs to 16.
  Registry->setOption(scudo::Option::MaxTSDsCount, 16);
  Ready = false;
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread(stressSharedRegistry, Allocator.get());
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
  // We should get 16 distinct TSDs back.
  EXPECT_EQ(Pointers.size(), 16U);
}
