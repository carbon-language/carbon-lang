//===-- sanitizer_linux_test.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for sanitizer_linux.h
//
//===----------------------------------------------------------------------===//

#ifdef __linux__

#include "sanitizer_common/sanitizer_linux.h"
#include "gtest/gtest.h"

#include "sanitizer_common/sanitizer_common.h"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <set>

namespace __sanitizer {
static pthread_cond_t thread_exit_cond;
static pthread_mutex_t thread_exit_mutex;
static pthread_cond_t tid_reported_cond;
static pthread_mutex_t tid_reported_mutex;
static bool thread_exit;

void *TIDReporterThread(void *tid_storage) {
  pthread_mutex_lock(&tid_reported_mutex);
  *(pid_t *)tid_storage = GetTid();
  pthread_cond_broadcast(&tid_reported_cond);
  pthread_mutex_unlock(&tid_reported_mutex);

  pthread_mutex_lock(&thread_exit_mutex);
  while (!thread_exit)
    pthread_cond_wait(&thread_exit_cond, &thread_exit_mutex);
  pthread_mutex_unlock(&thread_exit_mutex);
  return NULL;
}

// The set of TIDs produced by ThreadLister should include the TID of every
// thread we spawn here. The two sets may differ if there are other threads
// running in the current process that we are not aware of.
// Calling ThreadLister::Reset() should not change this.
TEST(SanitizerLinux, ThreadListerMultiThreaded) {
  pthread_mutex_init(&thread_exit_mutex, NULL);
  pthread_mutex_init(&tid_reported_mutex, NULL);
  pthread_cond_init(&thread_exit_cond, NULL);
  pthread_cond_init(&tid_reported_cond, NULL);
  const uptr kThreadCount = 20; // does not include the main thread
  pthread_t thread_ids[kThreadCount];
  pid_t  thread_tids[kThreadCount];
  pid_t pid = getpid();
  pid_t self_tid = GetTid();
  thread_exit = false;
  pthread_mutex_lock(&tid_reported_mutex);
  for (uptr i = 0; i < kThreadCount; i++) {
    int pthread_create_result;
    thread_tids[i] = -1;
    pthread_create_result = pthread_create(&thread_ids[i], NULL,
                                           TIDReporterThread,
                                           &thread_tids[i]);
    ASSERT_EQ(pthread_create_result, 0);
    while (thread_tids[i] == -1)
      pthread_cond_wait(&tid_reported_cond, &tid_reported_mutex);
  }
  pthread_mutex_unlock(&tid_reported_mutex);
  std::set<pid_t> reported_tids(thread_tids, thread_tids + kThreadCount);
  reported_tids.insert(self_tid);

  ThreadLister thread_lister(pid);
  // There's a Reset() call between the first and second iteration.
  for (uptr i = 0; i < 2; i++) {
    std::set<pid_t> listed_tids;

    EXPECT_FALSE(thread_lister.error());
    for (uptr i = 0; i < kThreadCount + 1; i++) {
      pid_t tid = thread_lister.GetNextTID();
      EXPECT_GE(tid, 0);
      EXPECT_FALSE(thread_lister.error());
      listed_tids.insert(tid);
    }
    pid_t tid = thread_lister.GetNextTID();
    EXPECT_LT(tid, 0);
    EXPECT_FALSE(thread_lister.error());

    std::set<pid_t> intersection;
    std::set_intersection(reported_tids.begin(), reported_tids.end(),
                          listed_tids.begin(), listed_tids.end(),
                          std::inserter(intersection, intersection.begin()));
    EXPECT_EQ(intersection, reported_tids);
    thread_lister.Reset();
  }

  pthread_mutex_lock(&thread_exit_mutex);
  thread_exit = true;
  pthread_cond_broadcast(&thread_exit_cond);
  pthread_mutex_unlock(&thread_exit_mutex);
  for (uptr i = 0; i < kThreadCount; i++)
    pthread_join(thread_ids[i], NULL);
  pthread_mutex_destroy(&thread_exit_mutex);
  pthread_mutex_destroy(&tid_reported_mutex);
  pthread_cond_destroy(&thread_exit_cond);
  pthread_cond_destroy(&tid_reported_cond);
}
}  // namespace __sanitizer

#endif  // __linux__
