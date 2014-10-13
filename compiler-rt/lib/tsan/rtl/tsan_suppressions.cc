//===-- tsan_suppressions.cc ----------------------------------------------===//
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

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "tsan_suppressions.h"
#include "tsan_rtl.h"
#include "tsan_flags.h"
#include "tsan_mman.h"
#include "tsan_platform.h"

// Suppressions for true/false positives in standard libraries.
static const char *const std_suppressions =
// Libstdc++ 4.4 has data races in std::string.
// See http://crbug.com/181502 for an example.
"race:^_M_rep$\n"
"race:^_M_is_leaked$\n"
// False positive when using std <thread>.
// Happens because we miss atomic synchronization in libstdc++.
// See http://llvm.org/bugs/show_bug.cgi?id=17066 for details.
"race:std::_Sp_counted_ptr_inplace<std::thread::_Impl\n";

// Can be overriden in frontend.
#ifndef TSAN_GO
extern "C" const char *WEAK __tsan_default_suppressions() {
  return 0;
}
#endif

namespace __tsan {

static bool suppressions_inited = false;

void InitializeSuppressions() {
  CHECK(!suppressions_inited);
  SuppressionContext::InitIfNecessary();
#ifndef TSAN_GO
  SuppressionContext::Get()->Parse(__tsan_default_suppressions());
  SuppressionContext::Get()->Parse(std_suppressions);
#endif
  suppressions_inited = true;
}

SuppressionType conv(ReportType typ) {
  if (typ == ReportTypeRace)
    return SuppressionRace;
  else if (typ == ReportTypeVptrRace)
    return SuppressionRace;
  else if (typ == ReportTypeUseAfterFree)
    return SuppressionRace;
  else if (typ == ReportTypeVptrUseAfterFree)
    return SuppressionRace;
  else if (typ == ReportTypeThreadLeak)
    return SuppressionThread;
  else if (typ == ReportTypeMutexDestroyLocked)
    return SuppressionMutex;
  else if (typ == ReportTypeMutexDoubleLock)
    return SuppressionMutex;
  else if (typ == ReportTypeMutexBadUnlock)
    return SuppressionMutex;
  else if (typ == ReportTypeMutexBadReadLock)
    return SuppressionMutex;
  else if (typ == ReportTypeMutexBadReadUnlock)
    return SuppressionMutex;
  else if (typ == ReportTypeSignalUnsafe)
    return SuppressionSignal;
  else if (typ == ReportTypeErrnoInSignal)
    return SuppressionNone;
  else if (typ == ReportTypeDeadlock)
    return SuppressionDeadlock;
  Printf("ThreadSanitizer: unknown report type %d\n", typ),
  Die();
}

uptr IsSuppressed(ReportType typ, const ReportStack *stack, Suppression **sp) {
  if (!SuppressionContext::Get()->SuppressionCount() || stack == 0 ||
      !stack->suppressable)
    return 0;
  SuppressionType stype = conv(typ);
  if (stype == SuppressionNone)
    return 0;
  Suppression *s;
  for (const ReportStack *frame = stack; frame; frame = frame->next) {
    if (SuppressionContext::Get()->Match(frame->func, stype, &s) ||
        SuppressionContext::Get()->Match(frame->file, stype, &s) ||
        SuppressionContext::Get()->Match(frame->module, stype, &s)) {
      DPrintf("ThreadSanitizer: matched suppression '%s'\n", s->templ);
      s->hit_count++;
      *sp = s;
      return frame->pc;
    }
  }
  return 0;
}

uptr IsSuppressed(ReportType typ, const ReportLocation *loc, Suppression **sp) {
  if (!SuppressionContext::Get()->SuppressionCount() || loc == 0 ||
      loc->type != ReportLocationGlobal || !loc->suppressable)
    return 0;
  SuppressionType stype = conv(typ);
  if (stype == SuppressionNone)
    return 0;
  Suppression *s;
  if (SuppressionContext::Get()->Match(loc->name, stype, &s) ||
      SuppressionContext::Get()->Match(loc->file, stype, &s) ||
      SuppressionContext::Get()->Match(loc->module, stype, &s)) {
      DPrintf("ThreadSanitizer: matched suppression '%s'\n", s->templ);
      s->hit_count++;
      *sp = s;
      return loc->addr;
  }
  return 0;
}

void PrintMatchedSuppressions() {
  InternalMmapVector<Suppression *> matched(1);
  SuppressionContext::Get()->GetMatched(&matched);
  if (!matched.size())
    return;
  int hit_count = 0;
  for (uptr i = 0; i < matched.size(); i++)
    hit_count += matched[i]->hit_count;
  Printf("ThreadSanitizer: Matched %d suppressions (pid=%d):\n", hit_count,
         (int)internal_getpid());
  for (uptr i = 0; i < matched.size(); i++) {
    Printf("%d %s:%s\n", matched[i]->hit_count,
           SuppressionTypeString(matched[i]->type), matched[i]->templ);
  }
}
}  // namespace __tsan
