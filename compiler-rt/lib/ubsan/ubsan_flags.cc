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
  CommonFlags *cf = common_flags();
  SetCommonFlagsDefaults(cf);
  cf->print_summary = false;
  // Override from user-specified string.
  ParseCommonFlagsFromString(cf, MaybeCallUbsanDefaultOptions());
  // Override from environment variable.
  ParseCommonFlagsFromString(cf, GetEnv("UBSAN_OPTIONS"));
}

Flags ubsan_flags;

static void ParseFlagsFromString(Flags *f, const char *str) {
  if (!str)
    return;
  ParseFlag(str, &f->halt_on_error, "halt_on_error",
            "Crash the program after printing the first error report");
  ParseFlag(str, &f->print_stacktrace, "print_stacktrace",
            "Include full stacktrace into an error report");
}

void InitializeFlags() {
  Flags *f = flags();
  // Default values.
  f->halt_on_error = false;
  f->print_stacktrace = false;
  // Override from user-specified string.
  ParseFlagsFromString(f, MaybeCallUbsanDefaultOptions());
  // Override from environment variable.
  ParseFlagsFromString(f, GetEnv("UBSAN_OPTIONS"));
}

}  // namespace __ubsan

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char *__ubsan_default_options() { return ""; }
}  // extern "C"
#endif
