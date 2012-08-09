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
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_thread_registry.h"

namespace __asan {

// ---------------------- Address Descriptions ------------------- {{{1

bool DescribeAddressIfShadow(uptr addr) {
  if (AddrIsInMem(addr))
    return false;
  static const char kAddrInShadowReport[] =
      "Address %p is located in the %s.\n";
  if (AddrIsInShadowGap(addr)) {
    AsanPrintf(kAddrInShadowReport, addr, "shadow gap area");
    return true;
  }
  if (AddrIsInHighShadow(addr)) {
    AsanPrintf(kAddrInShadowReport, addr, "high shadow area");
    return true;
  }
  if (AddrIsInLowShadow(addr)) {
    AsanPrintf(kAddrInShadowReport, addr, "low shadow area");
    return true;
  }
  CHECK(0 && "Address is not in memory and not in shadow?");
  return false;
}

bool DescribeAddressIfStack(uptr addr, uptr access_size) {
  AsanThread *t = asanThreadRegistry().FindThreadByStackAddress(addr);
  if (!t) return false;
  const sptr kBufSize = 4095;
  char buf[kBufSize];
  uptr offset = 0;
  const char *frame_descr = t->GetFrameNameByAddr(addr, &offset);
  // This string is created by the compiler and has the following form:
  // "FunctioName n alloc_1 alloc_2 ... alloc_n"
  // where alloc_i looks like "offset size len ObjectName ".
  CHECK(frame_descr);
  // Report the function name and the offset.
  const char *name_end = internal_strchr(frame_descr, ' ');
  CHECK(name_end);
  buf[0] = 0;
  internal_strncat(buf, frame_descr,
                   Min(kBufSize,
                       static_cast<sptr>(name_end - frame_descr)));
  AsanPrintf("Address %p is located at offset %zu "
             "in frame <%s> of T%d's stack:\n",
             (void*)addr, offset, buf, t->tid());
  // Report the number of stack objects.
  char *p;
  uptr n_objects = internal_simple_strtoll(name_end, &p, 10);
  CHECK(n_objects > 0);
  AsanPrintf("  This frame has %zu object(s):\n", n_objects);
  // Report all objects in this frame.
  for (uptr i = 0; i < n_objects; i++) {
    uptr beg, size;
    sptr len;
    beg  = internal_simple_strtoll(p, &p, 10);
    size = internal_simple_strtoll(p, &p, 10);
    len  = internal_simple_strtoll(p, &p, 10);
    if (beg <= 0 || size <= 0 || len < 0 || *p != ' ') {
      AsanPrintf("AddressSanitizer can't parse the stack frame "
                 "descriptor: |%s|\n", frame_descr);
      break;
    }
    p++;
    buf[0] = 0;
    internal_strncat(buf, p, Min(kBufSize, len));
    p += len;
    AsanPrintf("    [%zu, %zu) '%s'\n", beg, beg + size, buf);
  }
  AsanPrintf("HINT: this may be a false positive if your program uses "
             "some custom stack unwind mechanism\n"
             "      (longjmp and C++ exceptions *are* supported)\n");
  t->summary()->Announce();
  return true;
}

void DescribeAddress(uptr addr, uptr access_size) {
  // Check if this is shadow or shadow gap.
  if (DescribeAddressIfShadow(addr))
    return;
  CHECK(AddrIsInMem(addr));
  if (DescribeAddressIfGlobal(addr))
    return;
  if (DescribeAddressIfStack(addr, access_size))
    return;
  // Assume it is a heap address.
  DescribeHeapAddress(addr, access_size);
}

// -------------------- Different kinds of reports ----------------- {{{1

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
