//===-- asan_thread_registry.cc -------------------------------------------===//
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
#include "sanitizer_common/sanitizer_common.h"

namespace __asan {

static AsanThreadRegistry asan_thread_registry(LINKER_INITIALIZED);

AsanThreadRegistry &asanThreadRegistry() {
  return asan_thread_registry;
}

AsanThreadRegistry::AsanThreadRegistry(LinkerInitialized x)
    : main_thread_(x),
      main_thread_summary_(x),
      mu_(x) { }

void AsanThreadRegistry::Init() {
  AsanTSDInit(AsanThreadSummary::TSDDtor);
  main_thread_.set_summary(&main_thread_summary_);
  main_thread_summary_.set_thread(&main_thread_);
  RegisterThread(&main_thread_);
  SetCurrentThread(&main_thread_);
  // At this point only one thread exists.
  inited_ = true;
}

void AsanThreadRegistry::RegisterThread(AsanThread *thread) {
  BlockingMutexLock lock(&mu_);
  u32 tid = n_threads_;
  n_threads_++;
  CHECK(n_threads_ < kMaxNumberOfThreads);

  AsanThreadSummary *summary = thread->summary();
  CHECK(summary != 0);
  summary->set_tid(tid);
  thread_summaries_[tid] = summary;
}

void AsanThreadRegistry::UnregisterThread(AsanThread *thread) {
  BlockingMutexLock lock(&mu_);
  FlushToAccumulatedStats(&thread->stats());
  AsanThreadSummary *summary = thread->summary();
  CHECK(summary);
  summary->set_thread(0);
}

AsanThread *AsanThreadRegistry::GetMain() {
  return &main_thread_;
}

void AsanThreadRegistry::FlushAllStats() {
  BlockingMutexLock lock(&mu_);
  for (u32 tid = 0; tid < n_threads_; tid++) {
    AsanThread *t = thread_summaries_[tid]->thread();
    if (t != 0) {
      FlushToAccumulatedStatsUnlocked(&t->stats());
    }
  }
}

AsanThreadSummary *AsanThreadRegistry::FindByTid(u32 tid) {
  CHECK(tid < n_threads_);
  CHECK(thread_summaries_[tid]);
  return thread_summaries_[tid];
}

AsanThread *AsanThreadRegistry::FindThreadByStackAddress(uptr addr) {
  BlockingMutexLock lock(&mu_);
  for (u32 tid = 0; tid < n_threads_; tid++) {
    AsanThread *t = thread_summaries_[tid]->thread();
    if (!t || !(t->fake_stack().StackSize())) continue;
    if (t->fake_stack().AddrIsInFakeStack(addr) || t->AddrIsInStack(addr)) {
      return t;
    }
  }
  return 0;
}

}  // namespace __asan
