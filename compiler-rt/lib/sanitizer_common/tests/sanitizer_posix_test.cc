//===-- sanitizer_posix_test.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for POSIX-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "sanitizer_common/sanitizer_common.h"
#include "gtest/gtest.h"

#include <pthread.h>

namespace __sanitizer {

static pthread_key_t key;
static bool destructor_executed;

extern "C"
void destructor(void *arg) {
  uptr iter = reinterpret_cast<uptr>(arg);
  if (iter > 1) {
    ASSERT_EQ(0, pthread_setspecific(key, reinterpret_cast<void *>(iter - 1)));
    return;
  }
  destructor_executed = true;
}

extern "C"
void *thread_func(void *arg) {
  return reinterpret_cast<void*>(pthread_setspecific(key, arg));
}

static void SpawnThread(uptr iteration) {
  destructor_executed = false;
  pthread_t tid;
  ASSERT_EQ(0, pthread_create(&tid, 0, &thread_func,
                              reinterpret_cast<void *>(iteration)));
  void *retval;
  ASSERT_EQ(0, pthread_join(tid, &retval));
  ASSERT_EQ(0, retval);
}

TEST(SanitizerCommon, PthreadDestructorIterations) {
  ASSERT_EQ(0, pthread_key_create(&key, &destructor));
  SpawnThread(kPthreadDestructorIterations);
  EXPECT_TRUE(destructor_executed);
  SpawnThread(kPthreadDestructorIterations + 1);
  EXPECT_FALSE(destructor_executed);
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
