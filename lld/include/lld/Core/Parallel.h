//===- lld/Core/Parallel.h - Parallel utilities ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_PARALLEL_H
#define LLD_CORE_PARALLEL_H

#include "lld/Core/Instrumentation.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/range.h"

#include "llvm/Support/MathExtras.h"

#ifdef _MSC_VER
// Exceptions are disabled so this isn't defined, but concrt assumes it is.
namespace {
void *__uncaught_exception() { return nullptr; }
}
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stack>

#ifdef _MSC_VER
#include <concrt.h>
#include <ppl.h>
#endif

namespace lld {
/// \brief Allows one or more threads to wait on a potentially unknown number of
///   events.
///
/// A latch starts at \p count. inc() increments this, and dec() decrements it.
/// All calls to sync() will block while the count is not 0.
///
/// Calling dec() on a Latch with a count of 0 has undefined behaivor.
class Latch {
  std::atomic<uint32_t> _count;
  mutable std::mutex _condMut;
  mutable std::condition_variable _cond;

public:
  Latch(uint32_t count = 0) : _count(count) {}
  ~Latch() { sync(); }
  void inc() { ++_count; }

  void dec() {
    if (--_count == 0)
      _cond.notify_all();
  }

  void sync() const {
    std::unique_lock<std::mutex> lock(_condMut);
    _cond.wait(lock, [&] {
      return _count == 0;
    });
  }
};

/// \brief An abstract class that takes closures and runs them asynchronously.
class Executor {
public:
  virtual ~Executor() {}
  virtual void add(std::function<void()> func) = 0;
};

/// \brief An implementation of an Executor that runs closures on a thread pool
///   in filo order.
class ThreadPoolExecutor : public Executor {
public:
  ThreadPoolExecutor(unsigned threadCount =
                         std::thread::hardware_concurrency())
      : _stop(false), _done(threadCount) {
    // Spawn all but one of the threads in another thread as spawning threads
    // can take a while.
    std::thread([&, threadCount] {
      for (std::size_t i = 1; i < threadCount; ++i) {
        std::thread([=] {
          work();
        }).detach();
      }
      work();
    }).detach();
  }

  ~ThreadPoolExecutor() {
    _stop = true;
    _cond.notify_all();
    // Wait for ~Latch.
  }

  virtual void add(std::function<void()> f) {
    std::unique_lock<std::mutex> lock(_mutex);
    _workStack.push(f);
    lock.unlock();
    _cond.notify_one();
  }

private:
  void work() {
    while (true) {
      std::unique_lock<std::mutex> lock(_mutex);
      _cond.wait(lock, [&] {
        return _stop || !_workStack.empty();
      });
      if (_stop)
        break;
      auto task = _workStack.top();
      _workStack.pop();
      lock.unlock();
      task();
    }
    _done.dec();
  }

  std::atomic<bool> _stop;
  std::stack<std::function<void()>> _workStack;
  std::mutex _mutex;
  std::condition_variable _cond;
  Latch _done;
};

#ifdef _MSC_VER
/// \brief An Executor that runs tasks via ConcRT.
class ConcRTExecutor : public Executor {
  struct Taskish {
    Taskish(std::function<void()> task) : _task(task) {}

    std::function<void()> _task;

    static void run(void *p) {
      Taskish *self = static_cast<Taskish *>(p);
      self->_task();
      delete self;
    }
  };

public:
  virtual void add(std::function<void()> func) {
    Concurrency::CurrentScheduler::ScheduleTask(Taskish::run,
                                                new Taskish(func));
  }
};

inline Executor *getDefaultExecutor() {
  static ConcRTExecutor exec;
  return &exec;
}
#else
inline Executor *getDefaultExecutor() {
  static ThreadPoolExecutor exec;
  return &exec;
}
#endif

/// \brief Allows launching a number of tasks and waiting for them to finish
///   either explicitly via sync() or implicitly on destruction.
class TaskGroup {
  Latch _latch;

public:
  void spawn(std::function<void()> f) {
    _latch.inc();
    getDefaultExecutor()->add([&, f] {
      f();
      _latch.dec();
    });
  }

  void sync() const { _latch.sync(); }
};

#ifdef _MSC_VER
// Use ppl parallel_sort on Windows.
template <class RandomAccessIterator, class Comp>
void parallel_sort(
    RandomAccessIterator start, RandomAccessIterator end,
    const Comp &comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  concurrency::parallel_sort(start, end, comp);
}
#else
namespace detail {
const ptrdiff_t minParallelSize = 1024;

/// \brief Inclusive median.
template <class RandomAccessIterator, class Comp>
RandomAccessIterator medianOf3(RandomAccessIterator start,
                               RandomAccessIterator end, const Comp &comp) {
  RandomAccessIterator mid = start + (std::distance(start, end) / 2);
  return comp(*start, *(end - 1))
         ? (comp(*mid, *(end - 1)) ? (comp(*start, *mid) ? mid : start)
                                   : end - 1)
         : (comp(*mid, *start) ? (comp(*(end - 1), *mid) ? mid : end - 1)
                               : start);
}

template <class RandomAccessIterator, class Comp>
void parallel_quick_sort(RandomAccessIterator start, RandomAccessIterator end,
                         const Comp &comp, TaskGroup &tg, size_t depth) {
  // Do a sequential sort for small inputs.
  if (std::distance(start, end) < detail::minParallelSize || depth == 0) {
    std::sort(start, end, comp);
    return;
  }

  // Partition.
  auto pivot = medianOf3(start, end, comp);
  // Move pivot to end.
  std::swap(*(end - 1), *pivot);
  pivot = std::partition(start, end - 1, [end](decltype(*start) v) {
    return v < *(end - 1);
  });
  // Move pivot to middle of partition.
  std::swap(*pivot, *(end - 1));

  // Recurse.
  tg.spawn([=, &tg] {
    parallel_quick_sort(start, pivot, comp, tg, depth - 1);
  });
  parallel_quick_sort(pivot + 1, end, comp, tg, depth - 1);
}
}

template <class RandomAccessIterator, class Comp>
void parallel_sort(
    RandomAccessIterator start, RandomAccessIterator end,
    const Comp &comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  TaskGroup tg;
  detail::parallel_quick_sort(start, end, comp, tg,
                              llvm::Log2_64(std::distance(start, end)) + 1);
}
#endif

template <class T> void parallel_sort(T *start, T *end) {
  parallel_sort(start, end, std::less<T>());
}
} // end namespace lld

#endif
