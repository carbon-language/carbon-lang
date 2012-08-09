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

void NORETURN ReportSIGSEGV(uptr pc, uptr sp, uptr bp, uptr addr);

void NORETURN ReportDoubleFree(uptr addr, AsanStackTrace *stack);
void NORETURN ReportFreeNotMalloced(uptr addr, AsanStackTrace *stack);
void NORETURN ReportMallocUsableSizeNotOwned(uptr addr,
                                             AsanStackTrace *stack);
void NORETURN ReportAsanGetAllocatedSizeNotOwned(uptr addr,
                                                 AsanStackTrace *stack);

}  // namespace __asan
