//===-- sanitizer_stoptheworld.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the StopTheWorld function which suspends the execution of the current
// process and runs the user-supplied callback in the same address space.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_STOPTHEWORLD_H
#define SANITIZER_STOPTHEWORLD_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_common.h"

namespace __sanitizer {
typedef int SuspendedThreadID;

// Holds the list of suspended threads. Also provides register dumping
// functionality (to be implemented).
class SuspendedThreadsList {
 public:
  SuspendedThreadsList()
    : thread_ids_(1024) {}
  SuspendedThreadID GetThreadID(uptr index) {
    CHECK_LT(index, thread_ids_.size());
    return thread_ids_[index];
  }
  void DumpRegisters(uptr index) const {
    UNIMPLEMENTED();
  }
  uptr thread_count() { return thread_ids_.size(); }
  bool Contains(SuspendedThreadID thread_id) {
    for (uptr i = 0; i < thread_ids_.size(); i++) {
      if (thread_ids_[i] == thread_id)
        return true;
    }
    return false;
  }
  void Append(SuspendedThreadID thread_id) {
    thread_ids_.push_back(thread_id);
  }

 private:
  InternalVector<SuspendedThreadID> thread_ids_;

  // Prohibit copy and assign.
  SuspendedThreadsList(const SuspendedThreadsList&);
  void operator=(const SuspendedThreadsList&);
};

typedef void (*StopTheWorldCallback)(
    const SuspendedThreadsList &suspended_threads_list,
    void *argument);

// Suspend all threads in the current process and run the callback on the list
// of suspended threads. This function will resume the threads before returning.
// The callback should not call any libc functions.
// This function should NOT be called from multiple threads simultaneously.
void StopTheWorld(StopTheWorldCallback callback, void *argument);

}  // namespace __sanitizer

#endif  // SANITIZER_STOPTHEWORLD_H
