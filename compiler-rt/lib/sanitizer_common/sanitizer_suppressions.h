//===-- sanitizer_suppressions.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Suppression parsing/matching code shared between TSan and LSan.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_SUPPRESSIONS_H
#define SANITIZER_SUPPRESSIONS_H

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"

namespace __sanitizer {

enum SuppressionType {
  SuppressionNone,
  SuppressionRace,
  SuppressionMutex,
  SuppressionThread,
  SuppressionSignal,
  SuppressionLeak,
  SuppressionLib,
  SuppressionDeadlock,
  SuppressionTypeCount
};

struct Suppression {
  SuppressionType type;
  char *templ;
  unsigned hit_count;
  uptr weight;
};

class SuppressionContext {
 public:
  SuppressionContext() : suppressions_(1), can_parse_(true) {}
  void Parse(const char *str);
  bool Match(const char* str, SuppressionType type, Suppression **s);
  uptr SuppressionCount() const;
  const Suppression *SuppressionAt(uptr i) const;
  void GetMatched(InternalMmapVector<Suppression *> *matched);

 private:
  InternalMmapVector<Suppression> suppressions_;
  bool can_parse_;

  friend class SuppressionContextTest;
};

const char *SuppressionTypeString(SuppressionType t);

bool TemplateMatch(char *templ, const char *str);

}  // namespace __sanitizer

#endif  // SANITIZER_SUPPRESSIONS_H
