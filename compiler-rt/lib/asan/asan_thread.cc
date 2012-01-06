//===-- asan_thread.cc ------------------------------------------*- C++ -*-===//
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
#include "asan_thread.h"
#include "asan_mapping.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

namespace __asan {

AsanThread::AsanThread(LinkerInitialized x)
    : fake_stack_(x),
      malloc_storage_(x),
      stats_(x) { }

AsanThread *AsanThread::Create(int parent_tid, void *(*start_routine) (void *),
                               void *arg) {
  size_t size = RoundUpTo(sizeof(AsanThread), kPageSize);
  AsanThread *res = (AsanThread*)AsanMmapSomewhereOrDie(size, __FUNCTION__);
  res->start_routine_ = start_routine;
  res->arg_ = arg;
  return res;
}

void AsanThread::Destroy() {
  fake_stack().Cleanup();
  // We also clear the shadow on thread destruction because
  // some code may still be executing in later TSD destructors
  // and we don't want it to have any poisoned stack.
  ClearShadowForThreadStack();
  size_t size = RoundUpTo(sizeof(AsanThread), kPageSize);
  AsanUnmapOrDie(this, size);
}

void AsanThread::ClearShadowForThreadStack() {
  uintptr_t shadow_bot = MemToShadow(stack_bottom_);
  uintptr_t shadow_top = MemToShadow(stack_top_);
  real_memset((void*)shadow_bot, 0, shadow_top - shadow_bot);
}

void AsanThread::Init() {
  SetThreadStackTopAndBottom();
  fake_stack_.Init(stack_size());
  if (FLAG_v >= 1) {
    int local = 0;
    Report("T%d: stack [%p,%p) size 0x%lx; local=%p, pthread_self=%p\n",
           tid(), stack_bottom_, stack_top_,
           stack_top_ - stack_bottom_, &local, pthread_self());
  }

  CHECK(AddrIsInMem(stack_bottom_));
  CHECK(AddrIsInMem(stack_top_));

  ClearShadowForThreadStack();
}

void *AsanThread::ThreadStart() {
  Init();

  if (!start_routine_) {
    // start_routine_ == NULL if we're on the main thread or on one of the
    // OS X libdispatch worker threads. But nobody is supposed to call
    // ThreadStart() for the worker threads.
    CHECK(tid() == 0);
    return 0;
  }

  void *res = start_routine_(arg_);
  malloc_storage().CommitBack();

  if (FLAG_v >= 1) {
    Report("T%d exited\n", tid());
  }

  return res;
}

const char *AsanThread::GetFrameNameByAddr(uintptr_t addr, uintptr_t *offset) {
  uintptr_t bottom = 0;
  bool is_fake_stack = false;
  if (AddrIsInStack(addr)) {
    bottom = stack_bottom();
  } else {
    bottom = fake_stack().AddrIsInFakeStack(addr);
    CHECK(bottom);
    is_fake_stack = true;
  }
  uintptr_t aligned_addr = addr & ~(__WORDSIZE/8 - 1);  // align addr.
  uintptr_t *ptr = (uintptr_t*)aligned_addr;
  while (ptr >= (uintptr_t*)bottom) {
    if (ptr[0] == kCurrentStackFrameMagic ||
        (is_fake_stack && ptr[0] == kRetiredStackFrameMagic)) {
      *offset = addr - (uintptr_t)ptr;
      return (const char*)ptr[1];
    }
    ptr--;
  }
  *offset = 0;
  return "UNKNOWN";
}

}  // namespace __asan
