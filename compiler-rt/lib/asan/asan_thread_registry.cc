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

#include <limits.h>
#include <pthread.h>

namespace __asan {

static AsanThreadRegistry asan_thread_registry(__asan::LINKER_INITIALIZED);

AsanThreadRegistry &asanThreadRegistry() {
  return asan_thread_registry;
}

#ifdef ANDROID
#ifndef PTHREAD_DESTRUCTOR_ITERATIONS
#define PTHREAD_DESTRUCTOR_ITERATIONS 4
#endif
#endif

// Dark magic below. In order to be able to notice that we're not handling
// some thread creation routines (e.g. on Mac OS) we want to distinguish the
// thread that used to have a corresponding AsanThread object from the thread
// that never had one. That's why upon AsanThread destruction we set the
// pthread_key value to some odd number (that's not a valid pointer), instead
// of NULL.
// Because the TSD destructor for a non-NULL key value is called iteratively,
// we increase the value by two, keeping it an invalid pointer.
// Because the TSD implementations are allowed to call such a destructor
// infinitely (see
// http://pubs.opengroup.org/onlinepubs/009604499/functions/pthread_key_create.html
// ), we exit the program after a certain number of iterations.
static void DestroyAsanTsd(void *tsd) {
  intptr_t iter = (intptr_t)tsd;
  if (iter % 2 == 0) {
    // The pointer is valid.
    AsanThread *t = (AsanThread*)tsd;
    if (t != asanThreadRegistry().GetMain()) {
      asanThreadRegistry().UnregisterThread(t);
      t->Destroy();
    }
    iter = 1;
  } else {
    // The pointer is invalid -- we've already destroyed the TSD before.
    // If |iter| is too big, we're in the infinite loop. This should be
    // impossible on the systems AddressSanitizer was tested on.
    CHECK(iter < 4 * PTHREAD_DESTRUCTOR_ITERATIONS);
    iter += 2;
  }
  CHECK(0 == pthread_setspecific(asanThreadRegistry().GetTlsKey(),
                                 (void*)iter));
  if (FLAG_v >= 2) {
    Report("DestroyAsanTsd: writing %p to the TSD slot of thread %p\n",
           (void*)iter, pthread_self());
  }
}

AsanThreadRegistry::AsanThreadRegistry(LinkerInitialized x)
    : main_thread_(x),
      main_thread_summary_(x),
      accumulated_stats_(x),
      mu_(x) { }

void AsanThreadRegistry::Init() {
  CHECK(0 == pthread_key_create(&tls_key_, DestroyAsanTsd));
  tls_key_created_ = true;
  SetCurrent(&main_thread_);
  main_thread_.set_summary(&main_thread_summary_);
  main_thread_summary_.set_thread(&main_thread_);
  thread_summaries_[0] = &main_thread_summary_;
  n_threads_ = 1;
}

void AsanThreadRegistry::RegisterThread(AsanThread *thread, int parent_tid,
                                        AsanStackTrace *stack) {
  ScopedLock lock(&mu_);
  CHECK(n_threads_ > 0);
  int tid = n_threads_;
  n_threads_++;
  CHECK(n_threads_ < kMaxNumberOfThreads);
  AsanThreadSummary *summary = new AsanThreadSummary(tid, parent_tid, stack);
  summary->set_thread(thread);
  thread_summaries_[tid] = summary;
  thread->set_summary(summary);
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
  CHECK(tls_key_created_);
  AsanThread *thread = (AsanThread*)pthread_getspecific(tls_key_);
  if ((!thread || (intptr_t)thread % 2) && FLAG_v >= 2) {
    Report("GetCurrent: %p for thread %p\n", thread, pthread_self());
  }
  if ((intptr_t)thread % 2) {
    // Invalid pointer -- we've deleted the AsanThread already. Return NULL as
    // if the TSD was empty.
    // TODO(glider): if the code in the client TSD destructor calls
    // pthread_create(), we'll set the parent tid of the spawned thread to NULL,
    // although the creation stack will belong to the current thread. This may
    // confuse the user, but is quite unlikely.
    return NULL;
  } else {
    // NULL or valid pointer to AsanThread.
    return thread;
  }
}

void AsanThreadRegistry::SetCurrent(AsanThread *t) {
  if (FLAG_v >=2) {
    Report("SetCurrent: %p for thread %p\n", t, pthread_self());
  }
  // Make sure we do not reset the current AsanThread.
  intptr_t old_key = (intptr_t)pthread_getspecific(tls_key_);
  CHECK(!old_key || old_key % 2);
  CHECK(0 == pthread_setspecific(tls_key_, t));
  CHECK(pthread_getspecific(tls_key_) == t);
}

pthread_key_t AsanThreadRegistry::GetTlsKey() {
  return tls_key_;
}

// Returns true iff DestroyAsanTsd() was already called for this thread.
bool AsanThreadRegistry::IsCurrentThreadDying() {
  CHECK(tls_key_created_);
  intptr_t thread = (intptr_t)pthread_getspecific(tls_key_);
  return (bool)(thread % 2);
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
  for (int tid = 0; tid < n_threads_; tid++) {
    AsanThread *t = thread_summaries_[tid]->thread();
    if (!t) continue;
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
