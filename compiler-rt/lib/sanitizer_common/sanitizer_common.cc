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

uptr GetPageSizeCached() {
  static uptr PageSize;
  if (!PageSize)
    PageSize = GetPageSize();
  return PageSize;
}

// By default, dump to stderr. If report_fd is kInvalidFd, try to obtain file
// descriptor by opening file in report_path.
static fd_t report_fd = kStderrFd;
static char report_path[4096];  // Set via __sanitizer_set_report_path.

static void (*DieCallback)(void);
void SetDieCallback(void (*callback)(void)) {
  DieCallback = callback;
}

void NORETURN Die() {
  if (DieCallback) {
    DieCallback();
  }
  Exit(1);
}

static CheckFailedCallbackType CheckFailedCallback;
void SetCheckFailedCallback(CheckFailedCallbackType callback) {
  CheckFailedCallback = callback;
}

void NORETURN CheckFailed(const char *file, int line, const char *cond,
                          u64 v1, u64 v2) {
  if (CheckFailedCallback) {
    CheckFailedCallback(file, line, cond, v1, v2);
  }
  Report("Sanitizer CHECK failed: %s:%d %s (%zd, %zd)\n", file, line, cond,
                                                          v1, v2);
  Die();
}

static void MaybeOpenReportFile() {
  if (report_fd != kInvalidFd)
    return;
  fd_t fd = internal_open(report_path, true);
  if (fd == kInvalidFd) {
    report_fd = kStderrFd;
    Report("ERROR: Can't open file: %s\n", report_path);
    Die();
  }
  report_fd = fd;
}

bool PrintsToTty() {
  MaybeOpenReportFile();
  return internal_isatty(report_fd);
}

void RawWrite(const char *buffer) {
  static const char *kRawWriteError = "RawWrite can't output requested buffer!";
  uptr length = (uptr)internal_strlen(buffer);
  MaybeOpenReportFile();
  if (length != internal_write(report_fd, buffer, length)) {
    internal_write(report_fd, kRawWriteError, internal_strlen(kRawWriteError));
    Die();
  }
}

uptr ReadFileToBuffer(const char *file_name, char **buff,
                      uptr *buff_size, uptr max_len) {
  uptr PageSize = GetPageSizeCached();
  uptr kMinFileLen = PageSize;
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
    while (read_len + PageSize <= size) {
      uptr just_read = internal_read(fd, *buff + read_len, PageSize);
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

// We want to map a chunk of address space aligned to 'alignment'.
// We do it by maping a bit more and then unmaping redundant pieces.
// We probably can do it with fewer syscalls in some OS-dependent way.
void *MmapAlignedOrDie(uptr size, uptr alignment, const char *mem_type) {
// uptr PageSize = GetPageSizeCached();
  CHECK(IsPowerOfTwo(size));
  CHECK(IsPowerOfTwo(alignment));
  uptr map_size = size + alignment;
  uptr map_res = (uptr)MmapOrDie(map_size, mem_type);
  uptr map_end = map_res + map_size;
  uptr res = map_res;
  if (res & (alignment - 1))  // Not aligned.
    res = (map_res + alignment) & ~ (alignment - 1);
  uptr end = res + size;
  if (res != map_res)
    UnmapOrDie((void*)map_res, res - map_res);
  if (end != map_end)
    UnmapOrDie((void*)end, map_end - end);
  return (void*)res;
}

}  // namespace __sanitizer

using namespace __sanitizer;  // NOLINT

extern "C" {
void __sanitizer_set_report_path(const char *path) {
  if (!path) return;
  uptr len = internal_strlen(path);
  if (len > sizeof(report_path) - 100) {
    Report("ERROR: Path is too long: %c%c%c%c%c%c%c%c...\n",
           path[0], path[1], path[2], path[3],
           path[4], path[5], path[6], path[7]);
    Die();
  }
  internal_snprintf(report_path, sizeof(report_path), "%s.%d", path, GetPid());
  report_fd = kInvalidFd;
}

void __sanitizer_set_report_fd(int fd) {
  if (report_fd != kStdoutFd &&
      report_fd != kStderrFd &&
      report_fd != kInvalidFd)
    internal_close(report_fd);
  report_fd = fd;
}
}  // extern "C"
