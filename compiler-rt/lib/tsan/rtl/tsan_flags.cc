//===-- tsan_flags.cc -----------------------------------------------------===//
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

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_flags.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

Flags *flags() {
  return &ctx->flags;
}

// Can be overriden in frontend.
#ifdef TSAN_EXTERNAL_HOOKS
extern "C" const char* __tsan_default_options();
#else
extern "C" SANITIZER_INTERFACE_ATTRIBUTE
const char *WEAK __tsan_default_options() {
  return "";
}
#endif

void Flags::SetDefaults() {
#define TSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "tsan_flags.inc"
#undef TSAN_FLAG
  // DDFlags
  second_deadlock_stack = false;
}

void RegisterTsanFlags(FlagParser *parser, Flags *f) {
#define TSAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "tsan_flags.inc"
#undef TSAN_FLAG
}

void InitializeFlags(Flags *f, const char *env) {
  FlagParser parser;
  RegisterTsanFlags(&parser, f);
  RegisterCommonFlags(&parser);

  f->SetDefaults();

  SetCommonFlagsDefaults();
  {
    // Override some common flags defaults.
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.allow_addr2line = true;
    cf.detect_deadlocks = true;
    cf.print_suppressions = false;
    cf.stack_trace_format = "    #%n %f %S %M";
    OverrideCommonFlags(cf);
  }

  // Let a frontend override.
  parser.ParseString(__tsan_default_options());
  // Override from command line.
  parser.ParseString(env);

  // Sanity check.
  if (!f->report_bugs) {
    f->report_thread_leaks = false;
    f->report_destroy_locked = false;
    f->report_signal_unsafe = false;
  }

  SetVerbosity(common_flags()->verbosity);

  if (Verbosity()) ReportUnrecognizedFlags();

  if (common_flags()->help) parser.PrintFlagDescriptions();

  if (f->history_size < 0 || f->history_size > 7) {
    Printf("ThreadSanitizer: incorrect value for history_size"
           " (must be [0..7])\n");
    Die();
  }

  if (f->io_sync < 0 || f->io_sync > 2) {
    Printf("ThreadSanitizer: incorrect value for io_sync"
           " (must be [0..2])\n");
    Die();
  }
}

}  // namespace __tsan
