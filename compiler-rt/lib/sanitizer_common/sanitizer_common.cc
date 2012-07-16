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

// We don't want to use std::sort to avoid including <algorithm>, as
// we may end up with two implementation of std::sort - one in instrumented
// code, and the other in runtime.
// qsort() from stdlib won't work as it calls malloc(), which results
// in deadlock in ASan allocator.
// We re-implement in-place sorting w/o recursion as straightforward heapsort.
void SortArray(uptr *array, uptr size) {
  if (size < 2)
    return;
  // Stage 1: insert elements to the heap.
  for (uptr i = 1; i < size; i++) {
    uptr j, p;
    for (j = i; j > 0; j = p) {
      p = (j - 1) / 2;
      if (array[j] > array[p])
        Swap(array[j], array[p]);
      else
        break;
    }
  }
  // Stage 2: swap largest element with the last one,
  // and sink the new top.
  for (uptr i = size - 1; i > 0; i--) {
    Swap(array[0], array[i]);
    uptr j, max_ind;
    for (j = 0; j < i; j = max_ind) {
      uptr left = 2 * j + 1;
      uptr right = 2 * j + 2;
      max_ind = j;
      if (left < i && array[left] > array[max_ind])
        max_ind = left;
      if (right < i && array[right] > array[max_ind])
        max_ind = right;
      if (max_ind != j)
        Swap(array[j], array[max_ind]);
      else
        break;
    }
  }
}

}  // namespace __sanitizer
