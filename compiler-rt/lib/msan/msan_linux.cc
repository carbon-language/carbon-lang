//===-- msan_linux.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Linux-specific code.
//===----------------------------------------------------------------------===//

#ifdef __linux__

#include "msan.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <unwind.h>
#include <execinfo.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_procmaps.h"

namespace __msan {

static const uptr kMemBeg     = 0x600000000000;
static const uptr kMemEnd     = 0x7fffffffffff;
static const uptr kShadowBeg  = MEM_TO_SHADOW(kMemBeg);
static const uptr kShadowEnd  = MEM_TO_SHADOW(kMemEnd);
static const uptr kBad1Beg    = 0x100000000;  // 4G
static const uptr kBad1End    = kShadowBeg - 1;
static const uptr kBad2Beg    = kShadowEnd + 1;
static const uptr kBad2End    = kMemBeg - 1;
static const uptr kOriginsBeg = kBad2Beg;
static const uptr kOriginsEnd = kBad2End;

bool InitShadow(bool prot1, bool prot2, bool map_shadow, bool init_origins) {
  if (flags()->verbosity) {
    Printf("__msan_init %p\n", &__msan_init);
    Printf("Memory   : %p %p\n", kMemBeg, kMemEnd);
    Printf("Bad2     : %p %p\n", kBad2Beg, kBad2End);
    Printf("Origins  : %p %p\n", kOriginsBeg, kOriginsEnd);
    Printf("Shadow   : %p %p\n", kShadowBeg, kShadowEnd);
    Printf("Bad1     : %p %p\n", kBad1Beg, kBad1End);
  }

  if (prot1 && !Mprotect(kBad1Beg, kBad1End - kBad1Beg))
    return false;
  if (prot2 && !Mprotect(kBad2Beg, kBad2End - kBad2Beg))
    return false;
  if (map_shadow) {
    void *shadow = MmapFixedNoReserve(kShadowBeg, kShadowEnd - kShadowBeg);
    if (shadow != (void*)kShadowBeg) return false;
  }
  if (init_origins) {
    void *origins = MmapFixedNoReserve(kOriginsBeg, kOriginsEnd - kOriginsBeg);
    if (origins != (void*)kOriginsBeg) return false;
  }
  return true;
}

static void MsanTrap(int, siginfo_t *siginfo, void *context) {
  ucontext_t *ucontext = (ucontext_t*)context;
  uptr pc = ucontext->uc_mcontext.gregs[REG_RIP];
  uptr bp = ucontext->uc_mcontext.gregs[REG_RBP];
  PrintWarning(pc + 1 /*1 will be subtracted in StackTrace::Print */, bp);
  ucontext->uc_mcontext.gregs[REG_RIP] += 2;
}

void InstallTrapHandler() {
  struct sigaction sigact;
  internal_memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = MsanTrap;
  sigact.sa_flags = SA_SIGINFO;
  CHECK_EQ(0, sigaction(SIGILL, &sigact, 0));
}

void MsanDie() {
  _exit(flags()->exit_code);
}
}

#endif  // __linux__
