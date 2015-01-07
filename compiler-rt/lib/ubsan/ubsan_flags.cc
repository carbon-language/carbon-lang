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

namespace __ubsan {

static const char *MaybeCallUbsanDefaultOptions() {
  return (&__ubsan_default_options) ? __ubsan_default_options() : "";
}

void InitializeCommonFlags() {
  SetCommonFlagsDefaults();
  CommonFlags cf;
  cf.CopyFrom(*common_flags());
  cf.print_summary = false;
  OverrideCommonFlags(cf);
  // Override from user-specified string.
  ParseCommonFlagsFromString(MaybeCallUbsanDefaultOptions());
  // Override from environment variable.
  ParseCommonFlagsFromString(GetEnv("UBSAN_OPTIONS"));
}

Flags ubsan_flags;

void Flags::SetDefaults() {
#define UBSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "ubsan_flags.inc"
#undef UBSAN_FLAG
}

void Flags::ParseFromString(const char *str) {
#define UBSAN_FLAG(Type, Name, DefaultValue, Description)                      \
  ParseFlag(str, &Name, #Name, Description);
#include "ubsan_flags.inc"
#undef UBSAN_FLAG
}

void InitializeFlags() {
  Flags *f = flags();
  f->SetDefaults();
  // Override from user-specified string.
  f->ParseFromString(MaybeCallUbsanDefaultOptions());
  // Override from environment variable.
  f->ParseFromString(GetEnv("UBSAN_OPTIONS"));
}

}  // namespace __ubsan

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char *__ubsan_default_options() { return ""; }
}  // extern "C"
#endif
