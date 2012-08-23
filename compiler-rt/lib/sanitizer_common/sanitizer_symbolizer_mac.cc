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
#ifdef __APPLE__
#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

bool StartSymbolizerSubprocess(const char *path_to_symbolizer,
                               int *input_fd, int *output_fd) {
  UNIMPLEMENTED();
  return false;
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules) {
  UNIMPLEMENTED();
  return 0;
}

}  // namespace __sanitizer

#endif  // __APPLE__
