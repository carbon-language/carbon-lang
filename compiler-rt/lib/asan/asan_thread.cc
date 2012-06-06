//===-- asan_thread.cc ----------------------------------------------------===//
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
// Thread-related code.
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"
#include "asan_interceptors.h"
#include "asan_procmaps.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "asan_mapping.h"

namespace __asan {

AsanThread::AsanThread(LinkerInitialized x)
    : fake_stack_(x),
      malloc_storage_(x),
      stats_(x) { }

AsanThread *AsanThread::Create(int parent_tid, thread_callback_t start_routine,
                               void *arg, AsanStackTrace *stack) {
  uptr size = RoundUpTo(sizeof(AsanThread), kPageSize);
  AsanThread *thread = (AsanThread*)AsanMmapSomewhereOrDie(size, __FUNCTION__);
  thread->start_routine_ = start_routine;
  thread->arg_ = arg;

  AsanThreadSummary *summary = new AsanThreadSummary(parent_tid, stack);
  summary->set_thread(thread);
  thread->set_summary(summary);

  return thread;
}

void AsanThreadSummary::TSDDtor(void *tsd) {
  AsanThreadSummary *summary = (AsanThreadSummary*)tsd;
  if (FLAG_v >= 1) {
    Report("T%d TSDDtor\n", summary->tid());
  }
  if (summary->thread()) {
    summary->thread()->Destroy();
  }
}

void AsanThread::Destroy() {
  if (FLAG_v >= 1) {
    Report("T%d exited\n", tid());
  }

  asanThreadRegistry().UnregisterThread(this);
  CHECK(summary()->thread() == 0);
  // We also clear the shadow on thread destruction because
  // some code may still be executing in later TSD destructors
  // and we don't want it to have any poisoned stack.
  ClearShadowForThreadStack();
  fake_stack().Cleanup();
  uptr size = RoundUpTo(sizeof(AsanThread), kPageSize);
  AsanUnmapOrDie(this, size);
}

void AsanThread::Init() {
  SetThreadStackTopAndBottom();
  CHECK(AddrIsInMem(stack_bottom_));
  CHECK(AddrIsInMem(stack_top_));
  ClearShadowForThreadStack();
  if (FLAG_v >= 1) {
    int local = 0;
    Report("T%d: stack [%p,%p) size 0x%zx; local=%p\n",
           tid(), (void*)stack_bottom_, (void*)stack_top_,
           stack_top_ - stack_bottom_, &local);
  }
  fake_stack_.Init(stack_size());
}

thread_return_t AsanThread::ThreadStart() {
  Init();
  if (FLAG_use_sigaltstack) SetAlternateSignalStack();

  if (!start_routine_) {
    // start_routine_ == 0 if we're on the main thread or on one of the
    // OS X libdispatch worker threads. But nobody is supposed to call
    // ThreadStart() for the worker threads.
    CHECK(tid() == 0);
    return 0;
  }

  thread_return_t res = start_routine_(arg_);
  malloc_storage().CommitBack();
  if (FLAG_use_sigaltstack) UnsetAlternateSignalStack();

  this->Destroy();

  return res;
}

void AsanThread::ClearShadowForThreadStack() {
  PoisonShadow(stack_bottom_, stack_top_ - stack_bottom_, 0);
}

const char *AsanThread::GetFrameNameByAddr(uptr addr, uptr *offset) {
  uptr bottom = 0;
  bool is_fake_stack = false;
  if (AddrIsInStack(addr)) {
    bottom = stack_bottom();
  } else {
    bottom = fake_stack().AddrIsInFakeStack(addr);
    CHECK(bottom);
    is_fake_stack = true;
  }
  uptr aligned_addr = addr & ~(__WORDSIZE/8 - 1);  // align addr.
  u8 *shadow_ptr = (u8*)MemToShadow(aligned_addr);
  u8 *shadow_bottom = (u8*)MemToShadow(bottom);

  while (shadow_ptr >= shadow_bottom &&
      *shadow_ptr != kAsanStackLeftRedzoneMagic) {
    shadow_ptr--;
  }

  while (shadow_ptr >= shadow_bottom &&
      *shadow_ptr == kAsanStackLeftRedzoneMagic) {
    shadow_ptr--;
  }

  if (shadow_ptr < shadow_bottom) {
    *offset = 0;
    return "UNKNOWN";
  }

  uptr* ptr = (uptr*)SHADOW_TO_MEM((uptr)(shadow_ptr + 1));
  CHECK((ptr[0] == kCurrentStackFrameMagic) ||
      (is_fake_stack && ptr[0] == kRetiredStackFrameMagic));
  *offset = addr - (uptr)ptr;
  return (const char*)ptr[1];
}

}  // namespace __asan
