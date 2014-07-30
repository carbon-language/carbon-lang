//===-- tsan_suppressions.h -------------------------------------*- C++ -*-===//
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
#ifndef TSAN_SUPPRESSIONS_H
#define TSAN_SUPPRESSIONS_H

#include "sanitizer_common/sanitizer_suppressions.h"
#include "tsan_report.h"

namespace __tsan {

void InitializeSuppressions();
void PrintMatchedSuppressions();
uptr IsSuppressed(ReportType typ, const ReportStack *stack, Suppression **sp);
uptr IsSuppressed(ReportType typ, const ReportLocation *loc, Suppression **sp);

}  // namespace __tsan

#endif  // TSAN_SUPPRESSIONS_H
