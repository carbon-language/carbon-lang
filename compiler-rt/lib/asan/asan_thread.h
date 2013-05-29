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
#include "asan_fake_stack.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_thread_registry.h"

namespace __asan {

const u32 kInvalidTid = 0xffffff;  // Must fit into 24 bits.
const u32 kMaxNumberOfThreads = (1 << 22);  // 4M

class AsanThread;

// These objects are created for every thread and are never deleted,
// so we can find them by tid even if the thread is long dead.
class AsanThreadContext : public ThreadContextBase {
 public:
  explicit AsanThreadContext(int tid)
      : ThreadContextBase(tid),
        announced(false),
        thread(0) {
    internal_memset(&stack, 0, sizeof(stack));
  }
  bool announced;
  StackTrace stack;
  AsanThread *thread;

  void OnCreated(void *arg);
  void OnFinished();
};

// AsanThreadContext objects are never freed, so we need many of them.
COMPILER_CHECK(sizeof(AsanThreadContext) <= 4096);

// AsanThread are stored in TSD and destroyed when the thread dies.
class AsanThread {
 public:
  static AsanThread *Create(thread_callback_t start_routine, void *arg);
  static void TSDDtor(void *tsd);
  void Destroy();

  void Init();  // Should be called from the thread itself.
  thread_return_t ThreadStart(uptr os_id);

  uptr stack_top() { return stack_top_; }
  uptr stack_bottom() { return stack_bottom_; }
  uptr stack_size() { return stack_top_ - stack_bottom_; }
  uptr tls_begin() { return tls_begin_; }
  uptr tls_end() { return tls_end_; }
  u32 tid() { return context_->tid; }
  AsanThreadContext *context() { return context_; }
  void set_context(AsanThreadContext *context) { context_ = context; }

  const char *GetFrameNameByAddr(uptr addr, uptr *offset, uptr *frame_pc);

  bool AddrIsInStack(uptr addr) {
    return addr >= stack_bottom_ && addr < stack_top_;
  }

  FakeStack &fake_stack() { return fake_stack_; }
  AsanThreadLocalMallocStorage &malloc_storage() { return malloc_storage_; }
  AsanStats &stats() { return stats_; }

 private:
  AsanThread() {}
  void SetThreadStackAndTls();
  void ClearShadowForThreadStackAndTLS();
  AsanThreadContext *context_;
  thread_callback_t start_routine_;
  void *arg_;
  uptr  stack_top_;
  uptr  stack_bottom_;
  uptr tls_begin_;
  uptr tls_end_;

  FakeStack fake_stack_;
  AsanThreadLocalMallocStorage malloc_storage_;
  AsanStats stats_;
};

struct CreateThreadContextArgs {
  AsanThread *thread;
  StackTrace *stack;
};

// Returns a single instance of registry.
ThreadRegistry &asanThreadRegistry();

// Must be called under ThreadRegistryLock.
AsanThreadContext *GetThreadContextByTidLocked(u32 tid);

// Get the current thread. May return 0.
AsanThread *GetCurrentThread();
void SetCurrentThread(AsanThread *t);
u32 GetCurrentTidOrInvalid();
AsanThread *FindThreadByStackAddress(uptr addr);

}  // namespace __asan

#endif  // ASAN_THREAD_H
