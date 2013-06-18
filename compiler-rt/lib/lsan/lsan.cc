//=-- lsan.cc -------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Standalone LSan RTL.
//
//===----------------------------------------------------------------------===//

#include "lsan.h"

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "lsan_allocator.h"
#include "lsan_common.h"
#include "lsan_thread.h"

namespace __lsan {

static void InitializeCommonFlags() {
  CommonFlags *cf = common_flags();
  cf->external_symbolizer_path = GetEnv("LSAN_SYMBOLIZER_PATH");
  cf->symbolize = (cf->external_symbolizer_path &&
      cf->external_symbolizer_path[0]);
  cf->strip_path_prefix = "";
  cf->fast_unwind_on_malloc = true;
  cf->malloc_context_size = 30;

  ParseCommonFlagsFromString(GetEnv("LSAN_OPTIONS"));
}

void Init() {
  static bool inited;
  if (inited)
    return;
  inited = true;
  SanitizerToolName = "LeakSanitizer";
  InitializeCommonFlags();
  InitializeAllocator();
  InitTlsSize();
  InitializeInterceptors();
  InitializeThreadRegistry();
  u32 tid = ThreadCreate(0, 0, true);
  CHECK_EQ(tid, 0);
  ThreadStart(tid, GetTid());
  SetCurrentThread(tid);

  // Start symbolizer process if necessary.
  const char* external_symbolizer = common_flags()->external_symbolizer_path;
  if (common_flags()->symbolize && external_symbolizer &&
      external_symbolizer[0]) {
    InitializeExternalSymbolizer(external_symbolizer);
  }

  InitCommonLsan();
  Atexit(DoLeakCheck);
}

}  // namespace __lsan
