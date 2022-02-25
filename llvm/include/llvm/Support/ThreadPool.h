//===-- llvm/Support/ThreadPool.h - A ThreadPool implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a crude C++11 based thread pool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREADPOOL_H
#define LLVM_SUPPORT_THREADPOOL_H

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/thread.h"

#include <future>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

namespace llvm {

/// A ThreadPool for asynchronous parallel execution on a defined number of
/// threads.
///
/// The pool keeps a vector of threads alive, waiting on a condition variable
/// for some work to become available.
class ThreadPool {
public:
  using TaskTy = std::function<void()>;
  using PackagedTaskTy = std::packaged_task<void()>;

  /// Construct a pool using the hardware strategy \p S for mapping hardware
  /// execution resources (threads, cores, CPUs)
  /// Defaults to using the maximum execution resources in the system, but
  /// accounting for the affinity mask.
  ThreadPool(ThreadPoolStrategy S = hardware_concurrency());

  /// Blocking destructor: the pool will wait for all the threads to complete.
  ~ThreadPool();

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Function, typename... Args>
  inline std::shared_future<void> async(Function &&F, Args &&... ArgList) {
    auto Task =
        std::bind(std::forward<Function>(F), std::forward<Args>(ArgList)...);
    return asyncImpl(std::move(Task));
  }

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Function>
  inline std::shared_future<void> async(Function &&F) {
    return asyncImpl(std::forward<Function>(F));
  }

  /// Blocking wait for all the threads to complete and the queue to be empty.
  /// It is an error to try to add new tasks while blocking on this call.
  void wait();

  unsigned getThreadCount() const { return ThreadCount; }

  /// Returns true if the current thread is a worker thread of this thread pool.
  bool isWorkerThread() const;

private:
  bool workCompletedUnlocked() { return !ActiveThreads && Tasks.empty(); }

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  std::shared_future<void> asyncImpl(TaskTy F);

  /// Threads in flight
  std::vector<llvm::thread> Threads;

  /// Tasks waiting for execution in the pool.
  std::queue<PackagedTaskTy> Tasks;

  /// Locking and signaling for accessing the Tasks queue.
  std::mutex QueueLock;
  std::condition_variable QueueCondition;

  /// Signaling for job completion
  std::condition_variable CompletionCondition;

  /// Keep track of the number of thread actually busy
  unsigned ActiveThreads = 0;

#if LLVM_ENABLE_THREADS // avoids warning for unused variable
  /// Signal for the destruction of the pool, asking thread to exit.
  bool EnableFlag = true;
#endif

  unsigned ThreadCount;
};
}

#endif // LLVM_SUPPORT_THREADPOOL_H
