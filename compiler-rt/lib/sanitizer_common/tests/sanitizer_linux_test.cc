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
#include <vector>

namespace __sanitizer {

struct TidReporterArgument {
  TidReporterArgument() {
    pthread_mutex_init(&terminate_thread_mutex, NULL);
    pthread_mutex_init(&tid_reported_mutex, NULL);
    pthread_cond_init(&terminate_thread_cond, NULL);
    pthread_cond_init(&tid_reported_cond, NULL);
    terminate_thread = false;
  }

  ~TidReporterArgument() {
    pthread_mutex_destroy(&terminate_thread_mutex);
    pthread_mutex_destroy(&tid_reported_mutex);
    pthread_cond_destroy(&terminate_thread_cond);
    pthread_cond_destroy(&tid_reported_cond);
  }

  pid_t reported_tid;
  // For signaling to spawned threads that they should terminate.
  pthread_cond_t terminate_thread_cond;
  pthread_mutex_t terminate_thread_mutex;
  bool terminate_thread;
  // For signaling to main thread that a child thread has reported its tid.
  pthread_cond_t tid_reported_cond;
  pthread_mutex_t tid_reported_mutex;

 private:
  // Disallow evil constructors
  TidReporterArgument(const TidReporterArgument &);
  void operator=(const TidReporterArgument &);
};

class ThreadListerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    pthread_t pthread_id;
    pid_t tid;
    for (uptr i = 0; i < kThreadCount; i++) {
      SpawnTidReporter(&pthread_id, &tid);
      pthread_ids_.push_back(pthread_id);
      tids_.push_back(tid);
    }
  }

  virtual void TearDown() {
    pthread_mutex_lock(&thread_arg.terminate_thread_mutex);
    thread_arg.terminate_thread = true;
    pthread_cond_broadcast(&thread_arg.terminate_thread_cond);
    pthread_mutex_unlock(&thread_arg.terminate_thread_mutex);
    for (uptr i = 0; i < pthread_ids_.size(); i++)
      pthread_join(pthread_ids_[i], NULL);
  }

  void SpawnTidReporter(pthread_t *pthread_id, pid_t *tid);

  static const uptr kThreadCount = 20;

  std::vector<pthread_t> pthread_ids_;
  std::vector<pid_t> tids_;

  TidReporterArgument thread_arg;
};

// Writes its TID once to reported_tid and waits until signaled to terminate.
void *TidReporterThread(void *argument) {
  TidReporterArgument *arg = reinterpret_cast<TidReporterArgument *>(argument);
  pthread_mutex_lock(&arg->tid_reported_mutex);
  arg->reported_tid = GetTid();
  pthread_cond_broadcast(&arg->tid_reported_cond);
  pthread_mutex_unlock(&arg->tid_reported_mutex);

  pthread_mutex_lock(&arg->terminate_thread_mutex);
  while (!arg->terminate_thread)
    pthread_cond_wait(&arg->terminate_thread_cond,
                      &arg->terminate_thread_mutex);
  pthread_mutex_unlock(&arg->terminate_thread_mutex);
  return NULL;
}

void ThreadListerTest::SpawnTidReporter(pthread_t *pthread_id,
                                        pid_t *tid) {
  pthread_mutex_lock(&thread_arg.tid_reported_mutex);
  thread_arg.reported_tid = -1;
  ASSERT_EQ(0, pthread_create(pthread_id, NULL,
                              TidReporterThread,
                              &thread_arg));
  while (thread_arg.reported_tid == -1)
    pthread_cond_wait(&thread_arg.tid_reported_cond,
                      &thread_arg.tid_reported_mutex);
  pthread_mutex_unlock(&thread_arg.tid_reported_mutex);
  *tid = thread_arg.reported_tid;
}

std::vector<pid_t> ReadTidsToVector(ThreadLister *thread_lister) {
  std::vector<pid_t> listed_tids;
  pid_t tid;
  while ((tid = thread_lister->GetNextTID()) >= 0)
    listed_tids.push_back(tid);
  EXPECT_FALSE(thread_lister->error());
  return listed_tids;
}

static bool Includes(std::vector<pid_t> first, std::vector<pid_t> second) {
  std::sort(first.begin(), first.end());
  std::sort(second.begin(), second.end());
  return std::includes(first.begin(), first.end(),
                       second.begin(), second.end());
}

static bool HasElement(std::vector<pid_t> vector, pid_t element) {
  return std::find(vector.begin(), vector.end(), element) != vector.end();
}

// ThreadLister's output should include the current thread's TID and the TID of
// every thread we spawned.
TEST_F(ThreadListerTest, ThreadListerSeesAllSpawnedThreads) {
  pid_t self_tid = GetTid();
  ThreadLister thread_lister(getpid());
  std::vector<pid_t> listed_tids = ReadTidsToVector(&thread_lister);
  ASSERT_TRUE(HasElement(listed_tids, self_tid));
  ASSERT_TRUE(Includes(listed_tids, tids_));
}

// Calling Reset() should not cause ThreadLister to forget any threads it's
// supposed to know about.
TEST_F(ThreadListerTest, ResetDoesNotForgetThreads) {
  ThreadLister thread_lister(getpid());

  // Run the loop body twice, because Reset() might behave differently if called
  // on a freshly created object.
  for (uptr i = 0; i < 2; i++) {
    thread_lister.Reset();
    std::vector<pid_t> listed_tids = ReadTidsToVector(&thread_lister);
    ASSERT_TRUE(Includes(listed_tids, tids_));
  }
}

// If new threads have spawned during ThreadLister object's lifetime, calling
// Reset() should cause ThreadLister to recognize their existence.
TEST_F(ThreadListerTest, ResetMakesNewThreadsKnown) {
  ThreadLister thread_lister(getpid());
  std::vector<pid_t> threads_before_extra = ReadTidsToVector(&thread_lister);

  pthread_t extra_pthread_id;
  pid_t extra_tid;
  SpawnTidReporter(&extra_pthread_id, &extra_tid);
  // Register the new thread so it gets terminated in TearDown().
  pthread_ids_.push_back(extra_pthread_id);

  // It would be very bizarre if the new TID had been listed before we even
  // spawned that thread, but it would also cause a false success in this test,
  // so better check for that.
  ASSERT_FALSE(HasElement(threads_before_extra, extra_tid));

  thread_lister.Reset();

  std::vector<pid_t> threads_after_extra = ReadTidsToVector(&thread_lister);
  ASSERT_TRUE(HasElement(threads_after_extra, extra_tid));
}

}  // namespace __sanitizer

#endif  // __linux__
