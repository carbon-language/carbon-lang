//===-- asan_thread_registry.cc ---------------------------------*- C++ -*-===//
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
// AsanThreadRegistry-related code. AsanThreadRegistry is a container
// for summaries of all created threads.
//===----------------------------------------------------------------------===//

#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"

namespace __asan {

static AsanThreadRegistry asan_thread_registry(__asan::LINKER_INITIALIZED);

AsanThreadRegistry &asanThreadRegistry() {
  return asan_thread_registry;
}

AsanThreadRegistry::AsanThreadRegistry(LinkerInitialized x)
    : main_thread_(x),
      main_thread_summary_(x),
      accumulated_stats_(x),
      mu_(x) { }

void AsanThreadRegistry::Init() {
  AsanTSDInit(AsanThreadSummary::TSDDtor);
  main_thread_.set_summary(&main_thread_summary_);
  main_thread_summary_.set_thread(&main_thread_);
  RegisterThread(&main_thread_);
  SetCurrent(&main_thread_);
  // At this point only one thread exists.
  inited_ = true;
}

void AsanThreadRegistry::RegisterThread(AsanThread *thread) {
  ScopedLock lock(&mu_);
  int tid = n_threads_;
  n_threads_++;
  CHECK(n_threads_ < kMaxNumberOfThreads);

  AsanThreadSummary *summary = thread->summary();
  CHECK(summary != NULL);
  summary->set_tid(tid);
  thread_summaries_[tid] = summary;
}

void AsanThreadRegistry::UnregisterThread(AsanThread *thread) {
  ScopedLock lock(&mu_);
  FlushToAccumulatedStatsUnlocked(&thread->stats());
  AsanThreadSummary *summary = thread->summary();
  CHECK(summary);
  summary->set_thread(NULL);
}

AsanThread *AsanThreadRegistry::GetMain() {
  return &main_thread_;
}

AsanThread *AsanThreadRegistry::GetCurrent() {
  AsanThreadSummary *summary = (AsanThreadSummary *)AsanTSDGet();
  if (!summary) {
#ifdef ANDROID
    // On Android, libc constructor is called _after_ asan_init, and cleans up
    // TSD. Try to figure out if this is still the main thread by the stack
    // address. We are not entirely sure that we have correct main thread
    // limits, so only do this magic on Android, and only if the found thread is
    // the main thread.
    AsanThread* thread = FindThreadByStackAddress((uintptr_t)&summary);
    if (thread && thread->tid() == 0) {
      SetCurrent(thread);
      return thread;
    }
#endif
    return 0;
  }
  return summary->thread();
}

void AsanThreadRegistry::SetCurrent(AsanThread *t) {
  CHECK(t->summary());
  if (FLAG_v >= 2) {
    Report("SetCurrent: %p for thread %p\n", t->summary(), GetThreadSelf());
  }
  // Make sure we do not reset the current AsanThread.
  CHECK(AsanTSDGet() == 0);
  AsanTSDSet(t->summary());
  CHECK(AsanTSDGet() == t->summary());
}

AsanStats &AsanThreadRegistry::GetCurrentThreadStats() {
  AsanThread *t = GetCurrent();
  return (t) ? t->stats() : main_thread_.stats();
}

AsanStats AsanThreadRegistry::GetAccumulatedStats() {
  ScopedLock lock(&mu_);
  UpdateAccumulatedStatsUnlocked();
  return accumulated_stats_;
}

size_t AsanThreadRegistry::GetCurrentAllocatedBytes() {
  ScopedLock lock(&mu_);
  UpdateAccumulatedStatsUnlocked();
  return accumulated_stats_.malloced - accumulated_stats_.freed;
}

size_t AsanThreadRegistry::GetHeapSize() {
  ScopedLock lock(&mu_);
  UpdateAccumulatedStatsUnlocked();
  return accumulated_stats_.mmaped;
}

size_t AsanThreadRegistry::GetFreeBytes() {
  ScopedLock lock(&mu_);
  UpdateAccumulatedStatsUnlocked();
  return accumulated_stats_.mmaped
         - accumulated_stats_.malloced
         - accumulated_stats_.malloced_redzones
         + accumulated_stats_.really_freed
         + accumulated_stats_.really_freed_redzones;
}

AsanThreadSummary *AsanThreadRegistry::FindByTid(int tid) {
  CHECK(tid >= 0);
  CHECK(tid < n_threads_);
  CHECK(thread_summaries_[tid]);
  return thread_summaries_[tid];
}

AsanThread *AsanThreadRegistry::FindThreadByStackAddress(uintptr_t addr) {
  ScopedLock lock(&mu_);
  // Main thread (tid = 0) stack limits are pretty much guessed; for the other
  // threads we ask libpthread, so their limits must be correct.
  // Scanning the thread list backwards makes this function more reliable.
  for (int tid = n_threads_ - 1; tid >= 0; tid--) {
    AsanThread *t = thread_summaries_[tid]->thread();
    if (!t || !(t->fake_stack().StackSize())) continue;
    if (t->fake_stack().AddrIsInFakeStack(addr) || t->AddrIsInStack(addr)) {
      return t;
    }
  }
  return 0;
}

void AsanThreadRegistry::UpdateAccumulatedStatsUnlocked() {
  for (int tid = 0; tid < n_threads_; tid++) {
    AsanThread *t = thread_summaries_[tid]->thread();
    if (t != NULL) {
      FlushToAccumulatedStatsUnlocked(&t->stats());
    }
  }
}

void AsanThreadRegistry::FlushToAccumulatedStatsUnlocked(AsanStats *stats) {
  // AsanStats consists of variables of type size_t only.
  size_t *dst = (size_t*)&accumulated_stats_;
  size_t *src = (size_t*)stats;
  size_t num_fields = sizeof(AsanStats) / sizeof(size_t);
  for (size_t i = 0; i < num_fields; i++) {
    dst[i] += src[i];
    src[i] = 0;
  }
}

}  // namespace __asan
