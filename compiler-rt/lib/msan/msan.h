//===-- msan.h --------------------------------------------------*- C++ -*-===//
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

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "msan_interface_internal.h"
#include "msan_flags.h"

#ifndef MSAN_REPLACE_OPERATORS_NEW_AND_DELETE
# define MSAN_REPLACE_OPERATORS_NEW_AND_DELETE 1
#endif

#define MEM_TO_SHADOW(mem)       (((uptr)mem) & ~0x400000000000ULL)
#define SHADOW_TO_ORIGIN(shadow) (((uptr)shadow) + 0x200000000000ULL)
#define MEM_TO_ORIGIN(mem)       (SHADOW_TO_ORIGIN(MEM_TO_SHADOW(mem)))
#define MEM_IS_APP(mem)          ((uptr)mem >= 0x600000000000ULL)
#define MEM_IS_SHADOW(mem) \
  ((uptr)mem >= 0x200000000000ULL && (uptr)mem <= 0x400000000000ULL)

const int kMsanParamTlsSizeInWords = 100;
const int kMsanRetvalTlsSizeInWords = 100;

namespace __msan {
extern int msan_inited;
extern bool msan_init_is_running;
extern int msan_report_count;

bool ProtectRange(uptr beg, uptr end);
bool InitShadow(bool prot1, bool prot2, bool map_shadow, bool init_origins);
char *GetProcSelfMaps();
void InitializeInterceptors();

void MsanAllocatorThreadFinish();
void *MsanReallocate(StackTrace *stack, void *oldp, uptr size,
                     uptr alignment, bool zeroise);
void MsanDeallocate(StackTrace *stack, void *ptr);
void InstallTrapHandler();
void InstallAtExitHandler();
void ReplaceOperatorsNewAndDelete();

const char *GetStackOriginDescr(u32 id, uptr *pc);

void EnterSymbolizer();
void ExitSymbolizer();
bool IsInSymbolizer();

struct SymbolizerScope {
  SymbolizerScope() { EnterSymbolizer(); }
  ~SymbolizerScope() { ExitSymbolizer(); }
};

void EnterLoader();
void ExitLoader();

void MsanDie();
void PrintWarning(uptr pc, uptr bp);
void PrintWarningWithOrigin(uptr pc, uptr bp, u32 origin);

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp,
                   bool request_fast_unwind);

void ReportUMR(StackTrace *stack, u32 origin);
void ReportExpectedUMRNotFound(StackTrace *stack);
void ReportStats();
void ReportAtExitStatistics();
void DescribeMemoryRange(const void *x, uptr size);
void ReportUMRInsideAddressRange(const char *what, const void *start, uptr size,
                                 uptr offset);

// Unpoison first n function arguments.
void UnpoisonParam(uptr n);
void UnpoisonThreadLocalState();

u32 GetOriginIfPoisoned(uptr a, uptr size);
void SetOriginIfPoisoned(uptr addr, uptr src_shadow, uptr size, u32 src_origin);
void CopyOrigin(void *dst, const void *src, uptr size, StackTrace *stack);
void MovePoison(void *dst, const void *src, uptr size, StackTrace *stack);
void CopyPoison(void *dst, const void *src, uptr size, StackTrace *stack);

// Returns a "chained" origin id, pointing to the given stack trace followed by
// the previous origin id.
u32 ChainOrigin(u32 id, StackTrace *stack);

#define GET_MALLOC_STACK_TRACE                                     \
  StackTrace stack;                                                \
  stack.size = 0;                                                  \
  if (__msan_get_track_origins() && msan_inited)                   \
    GetStackTrace(&stack, common_flags()->malloc_context_size,     \
        StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),           \
        common_flags()->fast_unwind_on_malloc)

#define GET_STORE_STACK_TRACE_PC_BP(pc, bp)                            \
  StackTrace stack;                                                    \
  stack.size = 0;                                                      \
  if (__msan_get_track_origins() > 1 && msan_inited)                   \
    GetStackTrace(&stack, common_flags()->malloc_context_size, pc, bp, \
                  common_flags()->fast_unwind_on_malloc)

#define GET_FATAL_STACK_TRACE_PC_BP(pc, bp)       \
  StackTrace stack;                               \
  stack.size = 0;                                 \
  if (msan_inited)                                \
    GetStackTrace(&stack, kStackTraceMax, pc, bp, \
                  common_flags()->fast_unwind_on_fatal)

#define GET_STORE_STACK_TRACE \
  GET_STORE_STACK_TRACE_PC_BP(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME())

class ScopedThreadLocalStateBackup {
 public:
  ScopedThreadLocalStateBackup() { Backup(); }
  ~ScopedThreadLocalStateBackup() { Restore(); }
  void Backup();
  void Restore();
 private:
  u64 va_arg_overflow_size_tls;
};

extern void (*death_callback)(void);

void MsanTSDInit(void (*destructor)(void *tsd));
void *MsanTSDGet();
void MsanTSDSet(void *tsd);
void MsanTSDDtor(void *tsd);

}  // namespace __msan

#define MSAN_MALLOC_HOOK(ptr, size) \
  if (&__msan_malloc_hook) __msan_malloc_hook(ptr, size)
#define MSAN_FREE_HOOK(ptr) \
  if (&__msan_free_hook) __msan_free_hook(ptr)

#endif  // MSAN_H
