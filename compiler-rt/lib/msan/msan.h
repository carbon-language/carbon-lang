//===-- msan.h ------------------------------------------------------------===//
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
// Private MSan header.
//===----------------------------------------------------------------------===//

#ifndef MSAN_H
#define MSAN_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer/msan_interface.h"
#include "msan_flags.h"

#define MEM_TO_SHADOW(mem) (((uptr)mem)       & ~0x400000000000ULL)
#define MEM_TO_ORIGIN(mem) (MEM_TO_SHADOW(mem) + 0x200000000000ULL)
#define MEM_IS_APP(mem)    ((uptr)mem >=         0x600000000000ULL)
#define MEM_IS_SHADOW(mem) ((uptr)mem >=         0x200000000000ULL && \
                            (uptr)mem <=         0x400000000000ULL)

const int kMsanParamTlsSizeInWords = 100;
const int kMsanRetvalTlsSizeInWords = 100;

namespace __msan {
extern int msan_inited;
extern bool msan_init_is_running;

bool ProtectRange(uptr beg, uptr end);
bool InitShadow(bool prot1, bool prot2, bool map_shadow, bool init_origins);
char *GetProcSelfMaps();
void InitializeInterceptors();

void *MsanReallocate(StackTrace *stack, void *oldp, uptr size,
                     uptr alignment, bool zeroise);
void MsanDeallocate(void *ptr);
void InstallTrapHandler();
void ReplaceOperatorsNewAndDelete();

void MsanDie();
void PrintWarning(uptr pc, uptr bp);
void PrintWarningWithOrigin(uptr pc, uptr bp, u32 origin);

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp);

#define GET_MALLOC_STACK_TRACE                                     \
  StackTrace stack;                                                \
  stack.size = 0;                                                  \
  if (__msan_get_track_origins() && msan_inited)                   \
    GetStackTrace(&stack, flags()->num_callers,                    \
      StackTrace::GetCurrentPc(), GET_CURRENT_FRAME())

}  // namespace __msan

#endif  // MSAN_H
