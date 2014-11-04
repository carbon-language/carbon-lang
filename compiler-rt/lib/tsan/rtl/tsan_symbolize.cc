//===-- tsan_symbolize.cc -------------------------------------------------===//
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

#include "tsan_symbolize.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "tsan_flags.h"
#include "tsan_report.h"
#include "tsan_rtl.h"

namespace __tsan {

void EnterSymbolizer() {
  ThreadState *thr = cur_thread();
  CHECK(!thr->in_symbolizer);
  thr->in_symbolizer = true;
  thr->ignore_interceptors++;
}

void ExitSymbolizer() {
  ThreadState *thr = cur_thread();
  CHECK(thr->in_symbolizer);
  thr->in_symbolizer = false;
  thr->ignore_interceptors--;
}

// Denotes fake PC values that come from JIT/JAVA/etc.
// For such PC values __tsan_symbolize_external() will be called.
const uptr kExternalPCBit = 1ULL << 60;

// May be overriden by JIT/JAVA/etc,
// whatever produces PCs marked with kExternalPCBit.
extern "C" bool __tsan_symbolize_external(uptr pc,
                               char *func_buf, uptr func_siz,
                               char *file_buf, uptr file_siz,
                               int *line, int *col)
                               SANITIZER_WEAK_ATTRIBUTE;

bool __tsan_symbolize_external(uptr pc,
                               char *func_buf, uptr func_siz,
                               char *file_buf, uptr file_siz,
                               int *line, int *col) {
  return false;
}

ReportStack *SymbolizeCode(uptr addr) {
  // Check if PC comes from non-native land.
  if (addr & kExternalPCBit) {
    // Declare static to not consume too much stack space.
    // We symbolize reports in a single thread, so this is fine.
    static char func_buf[1024];
    static char file_buf[1024];
    int line, col;
    ReportStack *ent = ReportStack::New(addr);
    if (!__tsan_symbolize_external(addr, func_buf, sizeof(func_buf),
                                  file_buf, sizeof(file_buf), &line, &col))
      return ent;
    ent->info.function = internal_strdup(func_buf);
    ent->info.file = internal_strdup(file_buf);
    ent->info.line = line;
    ent->info.column = col;
    return ent;
  }
  static const uptr kMaxAddrFrames = 16;
  InternalScopedBuffer<AddressInfo> addr_frames(kMaxAddrFrames);
  for (uptr i = 0; i < kMaxAddrFrames; i++)
    new(&addr_frames[i]) AddressInfo();
  uptr addr_frames_num = Symbolizer::GetOrInit()->SymbolizePC(
      addr, addr_frames.data(), kMaxAddrFrames);
  if (addr_frames_num == 0)
    return ReportStack::New(addr);
  ReportStack *top = 0;
  ReportStack *bottom = 0;
  for (uptr i = 0; i < addr_frames_num; i++) {
    ReportStack *cur_entry = ReportStack::New(addr);
    cur_entry->info = addr_frames[i];
    if (i == 0)
      top = cur_entry;
    else
      bottom->next = cur_entry;
    bottom = cur_entry;
  }
  return top;
}

ReportLocation *SymbolizeData(uptr addr) {
  DataInfo info;
  if (!Symbolizer::GetOrInit()->SymbolizeData(addr, &info))
    return 0;
  ReportLocation *ent = ReportLocation::New(ReportLocationGlobal);
  ent->global = info;
  return ent;
}

void SymbolizeFlush() {
  Symbolizer::GetOrInit()->Flush();
}

}  // namespace __tsan
