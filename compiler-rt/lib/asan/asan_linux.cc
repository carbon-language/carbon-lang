//===-- asan_linux.cc -----------------------------------------------------===//
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
// Linux-specific details.
//===----------------------------------------------------------------------===//
#ifdef __linux__

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <unwind.h>

#if !ASAN_ANDROID
// FIXME: where to get ucontext on Android?
#include <sys/ucontext.h>
#endif

extern "C" void* _DYNAMIC;

namespace __asan {

void MaybeReexec() {
  // No need to re-exec on Linux.
}

void *AsanDoesNotSupportStaticLinkage() {
  // This will fail to link with -static.
  return &_DYNAMIC;  // defined in link.h
}

void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp) {
#if ASAN_ANDROID
  *pc = *sp = *bp = 0;
#elif defined(__arm__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.arm_pc;
  *bp = ucontext->uc_mcontext.arm_fp;
  *sp = ucontext->uc_mcontext.arm_sp;
# elif defined(__x86_64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_RIP];
  *bp = ucontext->uc_mcontext.gregs[REG_RBP];
  *sp = ucontext->uc_mcontext.gregs[REG_RSP];
# elif defined(__i386__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_EIP];
  *bp = ucontext->uc_mcontext.gregs[REG_EBP];
  *sp = ucontext->uc_mcontext.gregs[REG_ESP];
# elif defined(__powerpc__) || defined(__powerpc64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.regs->nip;
  *sp = ucontext->uc_mcontext.regs->gpr[PT_R1];
  // The powerpc{,64}-linux ABIs do not specify r31 as the frame
  // pointer, but GCC always uses r31 when we need a frame pointer.
  *bp = ucontext->uc_mcontext.regs->gpr[PT_R31];
# elif defined(__sparc__)
  ucontext_t *ucontext = (ucontext_t*)context;
  uptr *stk_ptr;
# if defined (__arch64__)
  *pc = ucontext->uc_mcontext.mc_gregs[MC_PC];
  *sp = ucontext->uc_mcontext.mc_gregs[MC_O6];
  stk_ptr = (uptr *) (*sp + 2047);
  *bp = stk_ptr[15];
# else
  *pc = ucontext->uc_mcontext.gregs[REG_PC];
  *sp = ucontext->uc_mcontext.gregs[REG_O6];
  stk_ptr = (uptr *) *sp;
  *bp = stk_ptr[15];
# endif
#else
# error "Unsupported arch"
#endif
}

bool AsanInterceptsSignal(int signum) {
  return signum == SIGSEGV && flags()->handle_segv;
}

void AsanPlatformThreadInit() {
  // Nothing here for now.
}

AsanLock::AsanLock(LinkerInitialized) {
  // We assume that pthread_mutex_t initialized to all zeroes is a valid
  // unlocked mutex. We can not use PTHREAD_MUTEX_INITIALIZER as it triggers
  // a gcc warning:
  // extended initializer lists only available with -std=c++0x or -std=gnu++0x
}

void AsanLock::Lock() {
  CHECK(sizeof(pthread_mutex_t) <= sizeof(opaque_storage_));
  pthread_mutex_lock((pthread_mutex_t*)&opaque_storage_);
  CHECK(!owner_);
  owner_ = (uptr)pthread_self();
}

void AsanLock::Unlock() {
  CHECK(owner_ == (uptr)pthread_self());
  owner_ = 0;
  pthread_mutex_unlock((pthread_mutex_t*)&opaque_storage_);
}

#ifdef __arm__
#define UNWIND_STOP _URC_END_OF_STACK
#define UNWIND_CONTINUE _URC_NO_REASON
#else
#define UNWIND_STOP _URC_NORMAL_STOP
#define UNWIND_CONTINUE _URC_NO_REASON
#endif

uptr Unwind_GetIP(struct _Unwind_Context *ctx) {
#ifdef __arm__
  uptr val;
  _Unwind_VRS_Result res = _Unwind_VRS_Get(ctx, _UVRSC_CORE,
      15 /* r15 = PC */, _UVRSD_UINT32, &val);
  CHECK(res == _UVRSR_OK && "_Unwind_VRS_Get failed");
  // Clear the Thumb bit.
  return val & ~(uptr)1;
#else
  return _Unwind_GetIP(ctx);
#endif
}

_Unwind_Reason_Code Unwind_Trace(struct _Unwind_Context *ctx,
    void *param) {
  StackTrace *b = (StackTrace*)param;
  CHECK(b->size < b->max_size);
  uptr pc = Unwind_GetIP(ctx);
  b->trace[b->size++] = pc;
  if (b->size == b->max_size) return UNWIND_STOP;
  return UNWIND_CONTINUE;
}

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp) {
  stack->size = 0;
  stack->trace[0] = pc;
  if ((max_s) > 1) {
    stack->max_size = max_s;
#if defined(__arm__) || defined(__powerpc__) || defined(__powerpc64__)
    _Unwind_Backtrace(Unwind_Trace, stack);
    // Pop off the two ASAN functions from the backtrace.
    stack->PopStackFrames(2);
#else
    if (!asan_inited) return;
    if (AsanThread *t = asanThreadRegistry().GetCurrent())
      stack->FastUnwindStack(pc, bp, t->stack_top(), t->stack_bottom());
#endif
  }
}

#if !ASAN_ANDROID
void ClearShadowMemoryForContext(void *context) {
  ucontext_t *ucp = (ucontext_t*)context;
  uptr sp = (uptr)ucp->uc_stack.ss_sp;
  uptr size = ucp->uc_stack.ss_size;
  // Align to page size.
  uptr bottom = sp & ~(kPageSize - 1);
  size += sp - bottom;
  size = RoundUpTo(size, kPageSize);
  PoisonShadow(bottom, size, 0);
}
#else
void ClearShadowMemoryForContext(void *context) {
  UNIMPLEMENTED();
}
#endif

}  // namespace __asan

#endif  // __linux__
