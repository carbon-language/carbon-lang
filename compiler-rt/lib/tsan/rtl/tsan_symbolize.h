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

// Denotes fake PC values that come from JIT/JAVA/etc.
// For such PC values __tsan_symbolize_external() will be called.
const uptr kExternalPCBit = 1ULL << 60;

void EnterSymbolizer();
void ExitSymbolizer();
SymbolizedStack *SymbolizeCode(uptr addr);
ReportLocation *SymbolizeData(uptr addr);
void SymbolizeFlush();

ReportStack *NewReportStackEntry(uptr addr);

}  // namespace __tsan

#endif  // TSAN_SYMBOLIZE_H
