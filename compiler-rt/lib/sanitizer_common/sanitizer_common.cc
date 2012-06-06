//===-- sanitizer_common.cc -----------------------------------------------===//
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
#include "sanitizer_libc.h"

namespace __sanitizer {

void RawWrite(const char *buffer) {
  static const char *kRawWriteError = "RawWrite can't output requested buffer!";
  uptr length = (uptr)internal_strlen(buffer);
  if (length != internal_write(2, buffer, length)) {
    internal_write(2, kRawWriteError, internal_strlen(kRawWriteError));
    Die();
  }
}

}  // namespace __sanitizer
