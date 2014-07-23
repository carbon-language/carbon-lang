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

Flags ubsan_flags;

void InitializeFlags() {
  Flags *f = flags();
  // Default values.
  f->print_stacktrace = false;

  const char *options = GetEnv("UBSAN_OPTIONS");
  if (options) {
    ParseFlag(options, &f->print_stacktrace, "print_stacktrace",
              "Include full stacktrace into an error report");
  }
}

}  // namespace __ubsan
