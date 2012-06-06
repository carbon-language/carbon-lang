//===-- asan_thread.h -------------------------------------------*- C++ -*-===//
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
// ASan-private header for asan_thread.cc.
//===----------------------------------------------------------------------===//
#ifndef ASAN_THREAD_H
#define ASAN_THREAD_H

#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_stack.h"
#include "asan_stats.h"

namespace __asan {

class AsanThread;

// These objects are created for every thread and are never deleted,
// so we can find them by tid even if the thread is long dead.
class AsanThreadSummary {
 public:
  explicit AsanThreadSummary(LinkerInitialized) { }  // for T0.
  AsanThreadSummary(int parent_tid, AsanStackTrace *stack)
      : parent_tid_(parent_tid),
        announced_(false) {
    tid_ = -1;
    if (stack) {
      stack_ = *stack;
    }
    thread_ = 0;
  }
  void Announce() {
    if (tid_ == 0) return;  // no need to announce the main thread.
    if (!announced_) {
      announced_ = true;
      AsanPrintf("Thread T%d created by T%d here:\n", tid_, parent_tid_);
      stack_.PrintStack();
    }
  }
  int tid() { return tid_; }
  void set_tid(int tid) { tid_ = tid; }
  AsanThread *thread() { return thread_; }
  void set_thread(AsanThread *thread) { thread_ = thread; }
  static void TSDDtor(void *tsd);

 private:
  int tid_;
  int parent_tid_;
  bool announced_;
  AsanStackTrace stack_;
  AsanThread *thread_;
};

// AsanThread are stored in TSD and destroyed when the thread dies.
class AsanThread {
 public:
  explicit AsanThread(LinkerInitialized);  // for T0.
  static AsanThread *Create(int parent_tid, thread_callback_t start_routine,
                            void *arg, AsanStackTrace *stack);
  void Destroy();

  void Init();  // Should be called from the thread itself.
  thread_return_t ThreadStart();

  uptr stack_top() { return stack_top_; }
  uptr stack_bottom() { return stack_bottom_; }
  uptr stack_size() { return stack_top_ - stack_bottom_; }
  int tid() { return summary_->tid(); }
  AsanThreadSummary *summary() { return summary_; }
  void set_summary(AsanThreadSummary *summary) { summary_ = summary; }

  const char *GetFrameNameByAddr(uptr addr, uptr *offset);

  bool AddrIsInStack(uptr addr) {
    return addr >= stack_bottom_ && addr < stack_top_;
  }

  FakeStack &fake_stack() { return fake_stack_; }
  AsanThreadLocalMallocStorage &malloc_storage() { return malloc_storage_; }
  AsanStats &stats() { return stats_; }

  static const int kInvalidTid = -1;

 private:

  void SetThreadStackTopAndBottom();
  void ClearShadowForThreadStack();
  AsanThreadSummary *summary_;
  thread_callback_t start_routine_;
  void *arg_;
  uptr  stack_top_;
  uptr  stack_bottom_;

  FakeStack fake_stack_;
  AsanThreadLocalMallocStorage malloc_storage_;
  AsanStats stats_;
};

}  // namespace __asan

#endif  // ASAN_THREAD_H
