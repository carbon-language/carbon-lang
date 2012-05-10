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

#include "tsan_report.h"

namespace __tsan {

void InitializeSuppressions();
void FinalizeSuppressions();
bool IsSuppressed(ReportType typ, const ReportStack *stack);

// Exposed for testing.
enum SuppressionType {
  SuppressionRace,
  SuppressionMutex,
  SuppressionThread,
  SuppressionSignal,
};

struct Suppression {
  Suppression *next;
  SuppressionType type;
  char *func;
};
Suppression *SuppressionParse(const char* supp);
bool SuppressionMatch(char *templ, const char *str);
void SuppressionFree(Suppression *supp);

}  // namespace __tsan

#endif  // TSAN_SUPPRESSIONS_H
