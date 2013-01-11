//===-- tsan_symbolize.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_SYMBOLIZE_H
#define TSAN_SYMBOLIZE_H

#include "tsan_defs.h"
#include "tsan_report.h"

namespace __tsan {

ReportStack *SymbolizeCode(uptr addr);
ReportLocation *SymbolizeData(uptr addr);

ReportStack *SymbolizeCodeAddr2Line(uptr addr);

ReportStack *NewReportStackEntry(uptr addr);

}  // namespace __tsan

#endif  // TSAN_SYMBOLIZE_H
