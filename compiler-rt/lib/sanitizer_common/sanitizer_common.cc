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

// Should not add dependency on libstdc++,
// since most of the stuff here is inlinable.
#include <algorithm>

namespace __sanitizer {

void RawWrite(const char *buffer) {
  static const char *kRawWriteError = "RawWrite can't output requested buffer!";
  uptr length = (uptr)internal_strlen(buffer);
  if (length != internal_write(2, buffer, length)) {
    internal_write(2, kRawWriteError, internal_strlen(kRawWriteError));
    Die();
  }
}

uptr ReadFileToBuffer(const char *file_name, char **buff,
                      uptr *buff_size, uptr max_len) {
  const uptr kMinFileLen = kPageSize;
  uptr read_len = 0;
  *buff = 0;
  *buff_size = 0;
  // The files we usually open are not seekable, so try different buffer sizes.
  for (uptr size = kMinFileLen; size <= max_len; size *= 2) {
    fd_t fd = internal_open(file_name, /*write*/ false);
    if (fd == kInvalidFd) return 0;
    UnmapOrDie(*buff, *buff_size);
    *buff = (char*)MmapOrDie(size, __FUNCTION__);
    *buff_size = size;
    // Read up to one page at a time.
    read_len = 0;
    bool reached_eof = false;
    while (read_len + kPageSize <= size) {
      uptr just_read = internal_read(fd, *buff + read_len, kPageSize);
      if (just_read == 0) {
        reached_eof = true;
        break;
      }
      read_len += just_read;
    }
    internal_close(fd);
    if (reached_eof)  // We've read the whole file.
      break;
  }
  return read_len;
}

void SortArray(uptr *array, uptr size) {
  std::sort(array, array + size);
}

}  // namespace __sanitizer
