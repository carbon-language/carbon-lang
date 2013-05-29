//===-- sanitizer_symbolizer_linux.cc -------------------------------------===//
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
// Linux-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX
#include "sanitizer_common.h"
#include "sanitizer_linux.h"

namespace __sanitizer {

#if SANITIZER_ANDROID
void SymbolizerPrepareForSandboxing() {
  // Do nothing on Android.
}
#else
static char proc_self_exe_cache_str[kMaxPathLength];
static uptr proc_self_exe_cache_len = 0;

uptr ReadBinaryName(/*out*/char *buf, uptr buf_len) {
  uptr module_name_len = internal_readlink(
      "/proc/self/exe", buf, buf_len);
  int readlink_error;
  if (internal_iserror(buf_len, &readlink_error)) {
    if (proc_self_exe_cache_len) {
      // If available, use the cached module name.
      CHECK_LE(proc_self_exe_cache_len, buf_len);
      internal_strncpy(buf, proc_self_exe_cache_str, buf_len);
      module_name_len = internal_strlen(proc_self_exe_cache_str);
    } else {
      // We can't read /proc/self/exe for some reason, assume the name of the
      // binary is unknown.
      Report("WARNING: readlink(\"/proc/self/exe\") failed with errno %d, "
             "some stack frames may not be symbolized\n", readlink_error);
      module_name_len = internal_snprintf(buf, buf_len, "/proc/self/exe");
    }
    CHECK_LT(module_name_len, buf_len);
    buf[module_name_len] = '\0';
  }
  return module_name_len;
}

void SymbolizerPrepareForSandboxing() {
  if (!proc_self_exe_cache_len) {
    proc_self_exe_cache_len =
        ReadBinaryName(proc_self_exe_cache_str, kMaxPathLength);
  }
}
#endif  // SANITIZER_ANDROID

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
