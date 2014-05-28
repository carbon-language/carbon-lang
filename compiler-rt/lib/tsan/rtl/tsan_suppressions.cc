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

static SuppressionContext* g_ctx;

static char *ReadFile(const char *filename) {
  if (filename == 0 || filename[0] == 0)
    return 0;
  InternalScopedBuffer<char> tmp(4*1024);
  if (filename[0] == '/' || GetPwd() == 0)
    internal_snprintf(tmp.data(), tmp.size(), "%s", filename);
  else
    internal_snprintf(tmp.data(), tmp.size(), "%s/%s", GetPwd(), filename);
  uptr openrv = OpenFile(tmp.data(), false);
  if (internal_iserror(openrv)) {
    Printf("ThreadSanitizer: failed to open suppressions file '%s'\n",
               tmp.data());
    Die();
  }
  fd_t fd = openrv;
  const uptr fsize = internal_filesize(fd);
  if (fsize == (uptr)-1) {
    Printf("ThreadSanitizer: failed to stat suppressions file '%s'\n",
               tmp.data());
    Die();
  }
  char *buf = (char*)internal_alloc(MBlockSuppression, fsize + 1);
  if (fsize != internal_read(fd, buf, fsize)) {
    Printf("ThreadSanitizer: failed to read suppressions file '%s'\n",
               tmp.data());
    Die();
  }
  internal_close(fd);
  buf[fsize] = 0;
  return buf;
}

void InitializeSuppressions() {
  ALIGNED(64) static char placeholder_[sizeof(SuppressionContext)];
  g_ctx = new(placeholder_) SuppressionContext;
  const char *supp = ReadFile(flags()->suppressions);
  g_ctx->Parse(supp);
#ifndef TSAN_GO
  supp = __tsan_default_suppressions();
  g_ctx->Parse(supp);
  g_ctx->Parse(std_suppressions);
#endif
}

SuppressionContext *GetSuppressionContext() {
  CHECK_NE(g_ctx, 0);
  return g_ctx;
}

SuppressionType conv(ReportType typ) {
  if (typ == ReportTypeRace)
    return SuppressionRace;
  else if (typ == ReportTypeVptrRace)
    return SuppressionRace;
  else if (typ == ReportTypeUseAfterFree)
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
  CHECK(g_ctx);
  if (!g_ctx->SuppressionCount() || stack == 0 || !stack->suppressable)
    return 0;
  SuppressionType stype = conv(typ);
  if (stype == SuppressionNone)
    return 0;
  Suppression *s;
  for (const ReportStack *frame = stack; frame; frame = frame->next) {
    if (g_ctx->Match(frame->func, stype, &s) ||
        g_ctx->Match(frame->file, stype, &s) ||
        g_ctx->Match(frame->module, stype, &s)) {
      DPrintf("ThreadSanitizer: matched suppression '%s'\n", s->templ);
      s->hit_count++;
      *sp = s;
      return frame->pc;
    }
  }
  return 0;
}

uptr IsSuppressed(ReportType typ, const ReportLocation *loc, Suppression **sp) {
  CHECK(g_ctx);
  if (!g_ctx->SuppressionCount() || loc == 0 ||
      loc->type != ReportLocationGlobal || !loc->suppressable)
    return 0;
  SuppressionType stype = conv(typ);
  if (stype == SuppressionNone)
    return 0;
  Suppression *s;
  if (g_ctx->Match(loc->name, stype, &s) ||
      g_ctx->Match(loc->file, stype, &s) ||
      g_ctx->Match(loc->module, stype, &s)) {
      DPrintf("ThreadSanitizer: matched suppression '%s'\n", s->templ);
      s->hit_count++;
      *sp = s;
      return loc->addr;
  }
  return 0;
}

void PrintMatchedSuppressions() {
  CHECK(g_ctx);
  InternalMmapVector<Suppression *> matched(1);
  g_ctx->GetMatched(&matched);
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
