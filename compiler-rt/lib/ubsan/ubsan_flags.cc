//===-- ubsan_flags.cc ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Runtime flags for UndefinedBehaviorSanitizer.
//
//===----------------------------------------------------------------------===//

#include "ubsan_flags.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_flag_parser.h"

namespace __ubsan {

static const char *MaybeCallUbsanDefaultOptions() {
  return (&__ubsan_default_options) ? __ubsan_default_options() : "";
}

Flags ubsan_flags;

void Flags::SetDefaults() {
#define UBSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "ubsan_flags.inc"
#undef UBSAN_FLAG
}

void RegisterUbsanFlags(FlagParser *parser, Flags *f) {
#define UBSAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "ubsan_flags.inc"
#undef UBSAN_FLAG
}

void InitializeFlags(bool standalone) {
  Flags *f = flags();
  FlagParser parser;
  RegisterUbsanFlags(&parser, f);

  if (standalone) {
    RegisterCommonFlags(&parser);

    SetCommonFlagsDefaults();
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.print_summary = false;
    OverrideCommonFlags(cf);
  } else {
    // Ignore common flags if not standalone.
    // This is inconsistent with LSan, which allows common flags in LSAN_FLAGS.
    // This is caused by undefined initialization order between ASan and UBsan,
    // which makes it impossible to make sure that common flags from ASAN_OPTIONS
    // have not been used (in __asan_init) before they are overwritten with flags
    // from UBSAN_OPTIONS.
    CommonFlags cf_ignored;
    RegisterCommonFlags(&parser, &cf_ignored);
  }

  f->SetDefaults();
  // Override from user-specified string.
  parser.ParseString(MaybeCallUbsanDefaultOptions());
  // Override from environment variable.
  parser.ParseString(GetEnv("UBSAN_OPTIONS"));
  SetVerbosity(common_flags()->verbosity);
}

}  // namespace __ubsan

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char *__ubsan_default_options() { return ""; }
}  // extern "C"
#endif
