//===-- sanitizer_common_libcdep.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"

namespace __sanitizer {

bool PrintsToTty() {
  MaybeOpenReportFile();
  return internal_isatty(report_fd) != 0;
}

bool PrintsToTtyCached() {
  // FIXME: Add proper Windows support to AnsiColorDecorator and re-enable color
  // printing on Windows.
  if (SANITIZER_WINDOWS)
    return 0;

  static int cached = 0;
  static bool prints_to_tty;
  if (!cached) {  // Not thread-safe.
    prints_to_tty = PrintsToTty();
    cached = 1;
  }
  return prints_to_tty;
}
}  // namespace __sanitizer
