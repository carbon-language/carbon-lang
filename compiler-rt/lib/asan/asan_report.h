//===-- asan_report.h -------------------------------------------*- C++ -*-===//
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
// ASan-private header for error reporting functions.
//===----------------------------------------------------------------------===//

#include "asan_internal.h"

namespace __asan {

// The following functions prints address description depending
// on the memory type (shadow/heap/stack/global).
void DescribeHeapAddress(uptr addr, uptr access_size);
bool DescribeAddressIfGlobal(uptr addr);
bool DescribeAddressIfShadow(uptr addr);
bool DescribeAddressIfStack(uptr addr, uptr access_size);
// Determines memory type on its own.
void DescribeAddress(uptr addr, uptr access_size);

// Different kinds of error reports.
void NORETURN ReportSIGSEGV(uptr pc, uptr sp, uptr bp, uptr addr);

void NORETURN ReportDoubleFree(uptr addr, AsanStackTrace *stack);
void NORETURN ReportFreeNotMalloced(uptr addr, AsanStackTrace *stack);
void NORETURN ReportMallocUsableSizeNotOwned(uptr addr,
                                             AsanStackTrace *stack);
void NORETURN ReportAsanGetAllocatedSizeNotOwned(uptr addr,
                                                 AsanStackTrace *stack);
void NORETURN ReportStringFunctionMemoryRangesOverlap(
    const char *function, const char *offset1, uptr length1,
    const char *offset2, uptr length2, AsanStackTrace *stack);

}  // namespace __asan
