//===-- sanitizer_symbolizer_win.cc ---------------------------------------===//
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
// Windows-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_WINDOWS
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

Symbolizer *Symbolizer::PlatformInit(const char *path_to_external) { return 0; }

}  // namespace __sanitizer

#endif  // _WIN32
