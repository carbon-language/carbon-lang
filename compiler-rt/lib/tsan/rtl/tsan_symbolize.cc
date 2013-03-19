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

struct ScopedInSymbolizer {
  ScopedInSymbolizer() {
    ThreadState *thr = cur_thread();
    CHECK(!thr->in_symbolizer);
    thr->in_symbolizer = true;
  }

  ~ScopedInSymbolizer() {
    ThreadState *thr = cur_thread();
    CHECK(thr->in_symbolizer);
    thr->in_symbolizer = false;
  }
};

ReportStack *NewReportStackEntry(uptr addr) {
  ReportStack *ent = (ReportStack*)internal_alloc(MBlockReportStack,
                                                  sizeof(ReportStack));
  internal_memset(ent, 0, sizeof(*ent));
  ent->pc = addr;
  return ent;
}

// Strip module path to make output shorter.
static char *StripModuleName(const char *module) {
  if (module == 0)
    return 0;
  const char *short_module_name = internal_strrchr(module, '/');
  if (short_module_name)
    short_module_name += 1;
  else
    short_module_name = module;
  return internal_strdup(short_module_name);
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

ReportStack *SymbolizeCode(uptr addr) {
  if (!IsSymbolizerAvailable())
    return SymbolizeCodeAddr2Line(addr);
  ScopedInSymbolizer in_symbolizer;
  static const uptr kMaxAddrFrames = 16;
  InternalScopedBuffer<AddressInfo> addr_frames(kMaxAddrFrames);
  for (uptr i = 0; i < kMaxAddrFrames; i++)
    new(&addr_frames[i]) AddressInfo();
  uptr addr_frames_num = __sanitizer::SymbolizeCode(addr, addr_frames.data(),
                                                    kMaxAddrFrames);
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
  if (!IsSymbolizerAvailable())
    return 0;
  ScopedInSymbolizer in_symbolizer;
  DataInfo info;
  if (!__sanitizer::SymbolizeData(addr, &info))
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
  if (!IsSymbolizerAvailable())
    return;
  ScopedInSymbolizer in_symbolizer;
  __sanitizer::FlushSymbolizer();
}

}  // namespace __tsan
