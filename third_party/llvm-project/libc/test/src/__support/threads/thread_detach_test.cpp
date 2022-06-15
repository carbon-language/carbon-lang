//===-- Unittests for thrd_t ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/mutex.h"
#include "src/__support/threads/thread.h"
#include "src/__support/threads/thread_attrib.h"
#include "utils/UnitTest/Test.h"

__llvm_libc::Mutex mutex(false, false, false);

int func(void *) {
  mutex.lock();
  mutex.unlock();
  return 0;
}

TEST(LlvmLibcTestThreadTest, DetachSimpleTest) {
  mutex.lock();
  __llvm_libc::Thread<int> th;
  th.run(func, nullptr, nullptr, 0);

  // Since |mutex| is held by the current thread, we guarantee that
  // th is running and hence it is safe to detach. Since the thread is
  // still running, it should be simple detach.
  ASSERT_EQ(th.detach(), int(__llvm_libc::DetachType::SIMPLE));

  // We will release |mutex| now to let the thread finish an cleanup itself.
  mutex.unlock();
}

TEST(LlvmLibcTestThreadTest, DetachCleanupTest) {
  mutex.lock();
  __llvm_libc::Thread<int> th;
  ASSERT_EQ(0, th.run(func, nullptr, nullptr, 0));

  // Since |mutex| is held by the current thread, we will release it
  // to let |th| run.
  mutex.unlock();

  // We will wait for |th| to finish. Since it is a joinable thread,
  // we can wait on it safely.
  th.wait();

  // Since |th| is now finished, detaching should cleanup the thread
  // resources.
  ASSERT_EQ(th.detach(), int(__llvm_libc::DetachType::CLEANUP));
}
