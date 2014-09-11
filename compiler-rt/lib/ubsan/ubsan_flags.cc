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

static const char *GetRuntimeFlagsFromCompileDefinition() {
#ifdef UBSAN_DEFAULT_OPTIONS
// Stringize the macro value
# define UBSAN_STRINGIZE(x) #x
# define UBSAN_STRINGIZE_OPTIONS(options) UBSAN_STRINGIZE(options)
  return UBSAN_STRINGIZE_OPTIONS(UBSAN_DEFAULT_OPTIONS);
#else
  return "";
#endif
}

static void InitializeCommonFlags() {
  CommonFlags *cf = common_flags();
  SetCommonFlagsDefaults(cf);
  cf->print_summary = false;
  // Override from compile definition.
  ParseCommonFlagsFromString(cf, GetRuntimeFlagsFromCompileDefinition());
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
  InitializeCommonFlags();
  Flags *f = flags();
  // Default values.
  f->halt_on_error = false;
  f->print_stacktrace = false;
  // Override from compile definition.
  ParseFlagsFromString(f, GetRuntimeFlagsFromCompileDefinition());
  // Override from environment variable.
  ParseFlagsFromString(f, GetEnv("UBSAN_OPTIONS"));
}

}  // namespace __ubsan
