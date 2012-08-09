//===-- asan_report.cc ----------------------------------------------------===//
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
// This file contains error reporting code.
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_thread_registry.h"

namespace __asan {

void ReportSIGSEGV(uptr pc, uptr sp, uptr bp, uptr addr) {
  AsanReport("ERROR: AddressSanitizer crashed on unknown address %p"
             " (pc %p sp %p bp %p T%d)\n",
             (void*)addr, (void*)pc, (void*)sp, (void*)bp,
             asanThreadRegistry().GetCurrentTidOrInvalid());
  AsanPrintf("AddressSanitizer can not provide additional info. ABORTING\n");
  GET_STACK_TRACE_WITH_PC_AND_BP(kStackTraceMax, pc, bp);
  stack.PrintStack();
  ShowStatsAndAbort();
}

void ReportDoubleFree(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting double-free on %p:\n", addr);
  stack->PrintStack();
  DescribeHeapAddress(addr, 1);
  ShowStatsAndAbort();
}

void ReportFreeNotMalloced(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting free on address "
             "which was not malloc()-ed: %p\n", addr);
  stack->PrintStack();
  ShowStatsAndAbort();
}

void ReportMallocUsableSizeNotOwned(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting to call "
             "malloc_usable_size() for pointer which is "
             "not owned: %p\n", addr);
  stack->PrintStack();
  DescribeHeapAddress(addr, 1);
  ShowStatsAndAbort();
}

void ReportAsanGetAllocatedSizeNotOwned(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting to call "
             "__asan_get_allocated_size() for pointer which is "
             "not owned: %p\n", addr);
  stack->PrintStack();
  DescribeHeapAddress(addr, 1);
  ShowStatsAndAbort();
}

void ReportStringFunctionMemoryRangesOverlap(
    const char *function, const char *offset1, uptr length1,
    const char *offset2, uptr length2, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer %s-param-overlap: "
             "memory ranges [%p,%p) and [%p, %p) overlap\n", \
             function, offset1, offset1 + length1, offset2, offset2 + length2);
  stack->PrintStack();
  ShowStatsAndAbort();
}

}  // namespace __asan
