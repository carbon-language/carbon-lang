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

/*
C/C++ on FreeBSD
0000 0000 0000 - 00ff ffff ffff: Low memory: main binary, MAP_32BIT mappings and modules
0100 0000 0000 - 0fff ffff ffff: Bad1
1000 0000 0000 - 30ff ffff ffff: Shadow
3100 0000 0000 - 37ff ffff ffff: Bad2
3800 0000 0000 - 58ff ffff ffff: Origins
5900 0000 0000 - 5fff ffff ffff: Bad3
6000 0000 0000 - 7fff ffff ffff: High memory: heap, modules and main thread stack

C/C++ on Linux/PIE
0000 0000 0000 - 1fff ffff ffff: Bad1
2000 0000 0000 - 3fff ffff ffff: Shadow
4000 0000 0000 - 5fff ffff ffff: Origins
6000 0000 0000 - 7fff ffff ffff: Main memory

C/C++ on Mips
0000 0000 0000 - 009f ffff ffff: Bad1
00a0 0000 0000 - 00bf ffff ffff: Shadow
00c0 0000 0000 - 00df ffff ffff: Origins
00e0 0000 0000 - 00ff ffff ffff: Main memory
*/

#if SANITIZER_LINUX && defined(__mips64)
const uptr kLowMemBeg   = 0;
const uptr kLowMemSize  = 0;
const uptr kHighMemBeg  = 0x00e000000000;
const uptr kHighMemSize = 0x002000000000;
const uptr kShadowBeg   = 0x00a000000000;
const uptr kShadowSize  = 0x002000000000;
const uptr kOriginsBeg  = 0x00c000000000;
# define MEM_TO_SHADOW(mem) (((uptr)(mem)) & ~0x4000000000ULL)
#elif SANITIZER_FREEBSD && SANITIZER_WORDSIZE == 64
const uptr kLowMemBeg   = 0x000000000000;
const uptr kLowMemSize  = 0x010000000000;
const uptr kHighMemBeg  = 0x600000000000;
const uptr kHighMemSize = 0x200000000000;
const uptr kShadowBeg   = 0x100000000000;
const uptr kShadowSize  = 0x210000000000;
const uptr kOriginsBeg  = 0x380000000000;
// Maps low and high app ranges to contiguous space with zero base:
//   Low:  0000 0000 0000 - 00ff ffff ffff  ->  2000 0000 0000 - 20ff ffff ffff
//   High: 6000 0000 0000 - 7fff ffff ffff  ->  0000 0000 0000 - 1fff ffff ffff
# define LINEARIZE_MEM(mem) \
    (((uptr)(mem) & ~0xc00000000000ULL) ^ 0x200000000000ULL)
# define MEM_TO_SHADOW(mem) (LINEARIZE_MEM((mem)) + 0x100000000000ULL)
#elif SANITIZER_LINUX && SANITIZER_WORDSIZE == 64
const uptr kLowMemBeg   = 0;
const uptr kLowMemSize  = 0;
const uptr kHighMemBeg  = 0x600000000000;
const uptr kHighMemSize = 0x200000000000;
const uptr kShadowBeg   = 0x200000000000;
const uptr kShadowSize  = 0x200000000000;
const uptr kOriginsBeg  = 0x400000000000;
# define MEM_TO_SHADOW(mem) (((uptr)(mem)) & ~0x400000000000ULL)
#else
#error "Unsupported platform"
#endif

const uptr kBad1Beg  = kLowMemBeg + kLowMemSize;
const uptr kBad1Size = kShadowBeg - kBad1Beg;

const uptr kBad2Beg  = kShadowBeg + kShadowSize;
const uptr kBad2Size = kOriginsBeg - kBad2Beg;

const uptr kOriginsSize = kShadowSize;

const uptr kBad3Beg  = kOriginsBeg + kOriginsSize;
const uptr kBad3Size = kHighMemBeg - kBad3Beg;

#define SHADOW_TO_ORIGIN(shadow) \
  (((uptr)(shadow)) + (kOriginsBeg - kShadowBeg))

#define MEM_TO_ORIGIN(mem) (SHADOW_TO_ORIGIN(MEM_TO_SHADOW((mem))))

#define MEM_IS_APP(mem) \
  ((kLowMemSize > 0 && (uptr)(mem) < kLowMemSize) || \
    (uptr)(mem) >= kHighMemBeg)

#define MEM_IS_SHADOW(mem) \
  ((uptr)(mem) >= kShadowBeg && (uptr)(mem) < kShadowBeg + kShadowSize)

#define MEM_IS_ORIGIN(mem) \
  ((uptr)(mem) >= kOriginsBeg && (uptr)(mem) < kOriginsBeg + kOriginsSize)

// These constants must be kept in sync with the ones in MemorySanitizer.cc.
const int kMsanParamTlsSize = 800;
const int kMsanRetvalTlsSize = 800;

namespace __msan {
extern int msan_inited;
extern bool msan_init_is_running;
extern int msan_report_count;

bool ProtectRange(uptr beg, uptr end);
bool InitShadow(bool map_shadow, bool init_origins);
char *GetProcSelfMaps();
void InitializeInterceptors();

void MsanAllocatorThreadFinish();
void *MsanCalloc(StackTrace *stack, uptr nmemb, uptr size);
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

void MsanDie();
void PrintWarning(uptr pc, uptr bp);
void PrintWarningWithOrigin(uptr pc, uptr bp, u32 origin);

void GetStackTrace(BufferedStackTrace *stack, uptr max_s, uptr pc, uptr bp,
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

#define GET_MALLOC_STACK_TRACE                                                 \
  BufferedStackTrace stack;                                                    \
  if (__msan_get_track_origins() && msan_inited)                               \
  GetStackTrace(&stack, common_flags()->malloc_context_size,                   \
                StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),               \
                common_flags()->fast_unwind_on_malloc)

#define GET_STORE_STACK_TRACE_PC_BP(pc, bp)                                    \
  BufferedStackTrace stack;                                                    \
  if (__msan_get_track_origins() > 1 && msan_inited)                           \
  GetStackTrace(&stack, flags()->store_context_size, pc, bp,                   \
                common_flags()->fast_unwind_on_malloc)

#define GET_FATAL_STACK_TRACE_PC_BP(pc, bp)                                    \
  BufferedStackTrace stack;                                                    \
  if (msan_inited)                                                             \
  GetStackTrace(&stack, kStackTraceMax, pc, bp,                                \
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
  if (&__sanitizer_malloc_hook) __sanitizer_malloc_hook(ptr, size)
#define MSAN_FREE_HOOK(ptr) \
  if (&__sanitizer_free_hook) __sanitizer_free_hook(ptr)

#endif  // MSAN_H
