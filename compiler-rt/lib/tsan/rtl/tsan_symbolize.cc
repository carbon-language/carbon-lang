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

ReportStack *NewReportStackEntry(uptr addr) {
  ReportStack *ent = (ReportStack*)internal_alloc(MBlockReportStack,
                                                  sizeof(ReportStack));
  internal_memset(ent, 0, sizeof(*ent));
  ent->pc = addr;
  return ent;
}

static ReportStack *NewReportStackEntry(const AddressInfo &info) {
  ReportStack *ent = NewReportStackEntry(info.address);
  ent->module = StripModuleName(info.module);
  ent->offset = info.module_offset;
  if (info.function)
    ent->func = internal_strdup(info.function);
  if (info.file)
    ent->file = internal_strdup(info.file);
  ent->line = info.line;
  ent->col = info.column;
  return ent;
}


  ReportStack *next;
  char *module;
  uptr offset;
  uptr pc;
  char *func;
  char *file;
  int line;
  int col;


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
    if (!__tsan_symbolize_external(addr, func_buf, sizeof(func_buf),
                                  file_buf, sizeof(file_buf), &line, &col))
      return NewReportStackEntry(addr);
    ReportStack *ent = NewReportStackEntry(addr);
    ent->module = 0;
    ent->offset = 0;
    ent->func = internal_strdup(func_buf);
    ent->file = internal_strdup(file_buf);
    ent->line = line;
    ent->col = col;
    return ent;
  }
  static const uptr kMaxAddrFrames = 16;
  InternalScopedBuffer<AddressInfo> addr_frames(kMaxAddrFrames);
  for (uptr i = 0; i < kMaxAddrFrames; i++)
    new(&addr_frames[i]) AddressInfo();
  uptr addr_frames_num = Symbolizer::Get()->SymbolizePC(
      addr, addr_frames.data(), kMaxAddrFrames);
  if (addr_frames_num == 0)
    return NewReportStackEntry(addr);
  ReportStack *top = 0;
  ReportStack *bottom = 0;
  for (uptr i = 0; i < addr_frames_num; i++) {
    ReportStack *cur_entry = NewReportStackEntry(addr_frames[i]);
    CHECK(cur_entry);
    addr_frames[i].Clear();
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
  if (!Symbolizer::Get()->SymbolizeData(addr, &info))
    return 0;
  ReportLocation *ent = (ReportLocation*)internal_alloc(MBlockReportStack,
                                                        sizeof(ReportLocation));
  internal_memset(ent, 0, sizeof(*ent));
  ent->type = ReportLocationGlobal;
  ent->module = StripModuleName(info.module);
  ent->offset = info.module_offset;
  if (info.name)
    ent->name = internal_strdup(info.name);
  ent->addr = info.start;
  ent->size = info.size;
  return ent;
}

void SymbolizeFlush() {
  Symbolizer::Get()->Flush();
}

}  // namespace __tsan
