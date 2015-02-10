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

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_FREEBSD || SANITIZER_LINUX

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_freebsd.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <unwind.h>

#if SANITIZER_FREEBSD
#include <sys/link_elf.h>
#endif

#if SANITIZER_ANDROID || SANITIZER_FREEBSD
#include <ucontext.h>
extern "C" void* _DYNAMIC;
#else
#include <sys/ucontext.h>
#include <link.h>
#endif

// x86-64 FreeBSD 9.2 and older define 'ucontext_t' incorrectly in
// 32-bit mode.
#if SANITIZER_FREEBSD && (SANITIZER_WORDSIZE == 32) && \
  __FreeBSD_version <= 902001  // v9.2
#define ucontext_t xucontext_t
#endif

typedef enum {
  ASAN_RT_VERSION_UNDEFINED = 0,
  ASAN_RT_VERSION_DYNAMIC,
  ASAN_RT_VERSION_STATIC,
} asan_rt_version_t;

// FIXME: perhaps also store abi version here?
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
asan_rt_version_t  __asan_rt_version;
}

namespace __asan {

void DisableReexec() {
  // No need to re-exec on Linux.
}

void MaybeReexec() {
  // No need to re-exec on Linux.
}

void *AsanDoesNotSupportStaticLinkage() {
  // This will fail to link with -static.
  return &_DYNAMIC;  // defined in link.h
}

#if SANITIZER_ANDROID
// FIXME: should we do anything for Android?
void AsanCheckDynamicRTPrereqs() {}
void AsanCheckIncompatibleRT() {}
#else
static int FindFirstDSOCallback(struct dl_phdr_info *info, size_t size,
                                void *data) {
  // Continue until the first dynamic library is found
  if (!info->dlpi_name || info->dlpi_name[0] == 0)
    return 0;

  // Ignore vDSO
  if (internal_strncmp(info->dlpi_name, "linux-", sizeof("linux-") - 1) == 0)
    return 0;

  *(const char **)data = info->dlpi_name;
  return 1;
}

static bool IsDynamicRTName(const char *libname) {
  return internal_strstr(libname, "libclang_rt.asan") ||
    internal_strstr(libname, "libasan.so");
}

static void ReportIncompatibleRT() {
  Report("Your application is linked against incompatible ASan runtimes.\n");
  Die();
}

void AsanCheckDynamicRTPrereqs() {
  // Ensure that dynamic RT is the first DSO in the list
  const char *first_dso_name = 0;
  dl_iterate_phdr(FindFirstDSOCallback, &first_dso_name);
  if (first_dso_name && !IsDynamicRTName(first_dso_name)) {
    Report("ASan runtime does not come first in initial library list; "
           "you should either link runtime to your application or "
           "manually preload it with LD_PRELOAD.\n");
    Die();
  }
}

void AsanCheckIncompatibleRT() {
  if (ASAN_DYNAMIC) {
    if (__asan_rt_version == ASAN_RT_VERSION_UNDEFINED) {
      __asan_rt_version = ASAN_RT_VERSION_DYNAMIC;
    } else if (__asan_rt_version != ASAN_RT_VERSION_DYNAMIC) {
      ReportIncompatibleRT();
    }
  } else {
    if (__asan_rt_version == ASAN_RT_VERSION_UNDEFINED) {
      // Ensure that dynamic runtime is not present. We should detect it
      // as early as possible, otherwise ASan interceptors could bind to
      // the functions in dynamic ASan runtime instead of the functions in
      // system libraries, causing crashes later in ASan initialization.
      MemoryMappingLayout proc_maps(/*cache_enabled*/true);
      char filename[128];
      while (proc_maps.Next(0, 0, 0, filename, sizeof(filename), 0)) {
        if (IsDynamicRTName(filename)) {
          Report("Your application is linked against "
                 "incompatible ASan runtimes.\n");
          Die();
        }
      }
      __asan_rt_version = ASAN_RT_VERSION_STATIC;
    } else if (__asan_rt_version != ASAN_RT_VERSION_STATIC) {
      ReportIncompatibleRT();
    }
  }
}
#endif  // SANITIZER_ANDROID

void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp) {
#if defined(__arm__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.arm_pc;
  *bp = ucontext->uc_mcontext.arm_fp;
  *sp = ucontext->uc_mcontext.arm_sp;
#elif defined(__aarch64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.pc;
  *bp = ucontext->uc_mcontext.regs[29];
  *sp = ucontext->uc_mcontext.sp;
#elif defined(__hppa__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.sc_iaoq[0];
  /* GCC uses %r3 whenever a frame pointer is needed.  */
  *bp = ucontext->uc_mcontext.sc_gr[3];
  *sp = ucontext->uc_mcontext.sc_gr[30];
#elif defined(__x86_64__)
# if SANITIZER_FREEBSD
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.mc_rip;
  *bp = ucontext->uc_mcontext.mc_rbp;
  *sp = ucontext->uc_mcontext.mc_rsp;
# else
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_RIP];
  *bp = ucontext->uc_mcontext.gregs[REG_RBP];
  *sp = ucontext->uc_mcontext.gregs[REG_RSP];
# endif
#elif defined(__i386__)
# if SANITIZER_FREEBSD
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.mc_eip;
  *bp = ucontext->uc_mcontext.mc_ebp;
  *sp = ucontext->uc_mcontext.mc_esp;
# else
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_EIP];
  *bp = ucontext->uc_mcontext.gregs[REG_EBP];
  *sp = ucontext->uc_mcontext.gregs[REG_ESP];
# endif
#elif defined(__powerpc__) || defined(__powerpc64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.regs->nip;
  *sp = ucontext->uc_mcontext.regs->gpr[PT_R1];
  // The powerpc{,64}-linux ABIs do not specify r31 as the frame
  // pointer, but GCC always uses r31 when we need a frame pointer.
  *bp = ucontext->uc_mcontext.regs->gpr[PT_R31];
#elif defined(__sparc__)
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
#elif defined(__mips__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[31];
  *bp = ucontext->uc_mcontext.gregs[30];
  *sp = ucontext->uc_mcontext.gregs[29];
#else
# error "Unsupported arch"
#endif
}

void AsanPlatformThreadInit() {
  // Nothing here for now.
}

#if !SANITIZER_ANDROID
void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  ucontext_t *ucp = (ucontext_t*)context;
  *stack = (uptr)ucp->uc_stack.ss_sp;
  *ssize = ucp->uc_stack.ss_size;
}
#else
void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  UNIMPLEMENTED();
}
#endif

void *AsanDlSymNext(const char *sym) {
  return dlsym(RTLD_NEXT, sym);
}

}  // namespace __asan

#endif  // SANITIZER_FREEBSD || SANITIZER_LINUX
