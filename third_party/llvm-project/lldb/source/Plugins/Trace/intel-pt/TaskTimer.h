//===-- TaskTimer.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TASKTIMER_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TASKTIMER_H

#include "lldb/lldb-types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <functional>
#include <unordered_map>

namespace lldb_private {
namespace trace_intel_pt {

/// Class used to track the duration of long running tasks related to a single
/// thread for reporting.
class ThreadTaskTimer {
public:
  /// Execute the given \p task and record its duration.
  ///
  /// \param[in] name
  ///     The name used to identify this task for reporting.
  ///
  /// \param[in] task
  ///     The task function.
  ///
  /// \return
  ///     The return value of the task.
  template <class R> R TimeTask(llvm::StringRef name, std::function<R()> task) {
    auto start = std::chrono::steady_clock::now();
    R result = task();
    auto end = std::chrono::steady_clock::now();
    std::chrono::milliseconds duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    m_timed_tasks.insert({name.str(), duration});
    return result;
  }

  /// Executive the given \p callback on each recorded task.
  ///
  /// \param[in] callback
  ///     The first parameter of the callback is the name of the recorded task,
  ///     and the second parameter is the duration of that task.
  void ForEachTimedTask(std::function<void(const std::string &name,
                                           std::chrono::milliseconds duration)>
                            callback);

private:
  std::unordered_map<std::string, std::chrono::milliseconds> m_timed_tasks;
};

/// Class used to track the duration of long running tasks for reporting.
class TaskTimer {
public:
  /// \return
  ///     The timer object for the given thread.
  ThreadTaskTimer &ForThread(lldb::tid_t tid);

private:
  llvm::DenseMap<lldb::tid_t, ThreadTaskTimer> m_thread_timers;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TASKTIMER_H
