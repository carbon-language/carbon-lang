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

namespace __tsan {

ReportStack *NewReportStackEntry(uptr addr) {
  ReportStack *ent = (ReportStack*)internal_alloc(MBlockReportStack,
                                                  sizeof(ReportStack));
  internal_memset(ent, 0, sizeof(*ent));
  ent->pc = addr;
  return ent;
}

static ReportStack *NewReportStackEntry(const AddressInfo &info) {
  ReportStack *ent = NewReportStackEntry(info.address);
  if (info.module) {
    // Strip module path to make output shorter.
    const char *short_module_name = internal_strrchr(info.module, '/');
    if (short_module_name)
      short_module_name += 1;
    else
      short_module_name = info.module;
    ent->module = internal_strdup(short_module_name);
  }
  ent->offset = info.module_offset;
  if (info.function) {
    ent->func = internal_strdup(info.function);
  }
  if (info.file)
    ent->file = internal_strdup(info.file);
  ent->line = info.line;
  ent->col = info.column;
  return ent;
}

ReportStack *SymbolizeCode(uptr addr) {
  if (flags()->external_symbolizer_path[0]) {
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
  return SymbolizeCodeAddr2Line(addr);
}

ReportStack *SymbolizeData(uptr addr) {
  if (flags()->external_symbolizer_path[0]) {
    AddressInfo frame;
    if (!__sanitizer::SymbolizeData(addr, &frame))
      return 0;
    return NewReportStackEntry(frame);
  }
  return SymbolizeDataAddr2Line(addr);
}

}  // namespace __tsan
