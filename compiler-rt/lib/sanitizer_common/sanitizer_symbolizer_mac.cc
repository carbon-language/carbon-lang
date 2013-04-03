//===-- sanitizer_symbolizer_mac.cc ---------------------------------------===//
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
// Mac-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_MAC
#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

bool StartSymbolizerSubprocess(const char *path_to_symbolizer,
                               int *input_fd, int *output_fd) {
  UNIMPLEMENTED();
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules) {
  // FIXME: Actually implement this on Mac. Just using MemoryMappingLayout
  // may be enough for this on Mac.
  return 0;
}

}  // namespace __sanitizer

#endif  // SANITIZER_MAC
