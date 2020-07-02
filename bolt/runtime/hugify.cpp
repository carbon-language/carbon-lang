//===-- hugify.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// This file contains code that is linked to the final binary with a function
// that is called at program entry to put hot code into a huge page.
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include <sys/mman.h>

// Enables a very verbose logging to stderr useful when debugging
//#define ENABLE_DEBUG

// Function pointers to init routines in the binary, so we can resume
// regular execution of the function that we hooked.
extern void (*__bolt_hugify_init_ptr)();

// The __hot_start and __hot_end symbols set by Bolt. We use them to figure
// out the rage for marking huge pages.
extern uint64_t __hot_start;
extern uint64_t __hot_end;

#ifdef MADV_HUGEPAGE
/// Starting from character at \p buf, find the longest consecutive sequence
/// of digits (0-9) and convert it to uint32_t. The converted value
/// is put into \p ret. \p end marks the end of the buffer to avoid buffer
/// overflow. The function \returns whether a valid uint32_t value is found.
/// \p buf will be updated to the next character right after the digits.
static bool scanUInt32(const char *&buf, const char *end, uint32_t &ret) {
  uint64_t result = 0;
  const char *oldBuf = buf;
  while (buf < end && ((*buf) >= '0' && (*buf) <= '9')) {
    result = result * 10 + (*buf) - '0';
    ++buf;
  }
  if (oldBuf != buf && result <= 0xFFFFFFFFu) {
    ret = static_cast<uint32_t>(result);
    return true;
  }
  return false;
}

/// Check whether the kernel supports THP by checking the kernel version.
/// Only fb kernel 5.2 and latter supports it.
static bool has_pagecache_thp_support() {
  struct utsname u;
  int ret = __uname(&u);
  if (ret) {
    return false;
  }

  const char *buf = u.release;
#ifdef ENABLE_DEBUG
  report("[hugify] uname release: ");
  report(buf);
  report("\n");
#endif
  const char *end = buf + strLen(buf);
  uint32_t nums[5];
  char delims[4][5] = {".", ".", "-", "_fbk"};
  // release should be in the format: %d.%d.%d-%d_fbk%d
  // they represent: major, minor, release, build, fbk.
  for (int i = 0; i < 5; ++i) {
    if (!scanUInt32(buf, end, nums[i])) {
      return false;
    }
    if (i < 4) {
      const char *ptr = delims[i];
      while (*ptr != '\0') {
        if (*ptr != *buf) {
          return false;
        }
        ++ptr;
        ++buf;
      }
    }
  }
  if (nums[0] > 5) {
    // Major is > 5.
    return true;
  }
  if (nums[0] < 5) {
    // Major is < 5.
    return false;
  }
  // minor > 2 || fbk >= 5.
  return nums[1] > 2 || nums[4] >= 5;
}

static void hugify_for_old_kernel(uint8_t *from, uint8_t *to) {
  size_t size = to - from;

  uint8_t *mem = reinterpret_cast<uint8_t *>(
      __mmap(0, size, 0x3 /* PROT_READ | PROT_WRITE*/,
             0x22 /* MAP_PRIVATE | MAP_ANONYMOUS*/, -1, 0));

  if (mem == (void *)MAP_FAILED) {
    char msg[] = "Could not allocate memory for text move\n";
    reportError(msg, sizeof(msg));
  }
#ifdef ENABLE_DEBUG
  reportNumber("Allocated temporary space: ", (uint64_t)mem, 16);
#endif

  // Copy the hot code to a temproary location.
  memCpy(mem, from, size);

  // Maps out the existing hot code.
  if (__mmap(reinterpret_cast<uint64_t>(from), size,
             PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1,
             0) == (void *)MAP_FAILED) {
    char msg[] = "failed to mmap memory for large page move terminating\n";
    reportError(msg, sizeof(msg));
  }

  // Mark the hot code page to be huge page.
  if (__madvise(from, size, MADV_HUGEPAGE) == -1) {
    char msg[] = "failed to allocate large page\n";
    reportError(msg, sizeof(msg));
  }

  // Copy the hot code back.
  memCpy(from, mem, size);

  // Change permission back to read-only, ignore failure
  __mprotect(from, size, PROT_READ | PROT_EXEC);

  __munmap(mem, size);
}
#endif

extern "C" void __bolt_hugify_self_impl() {
#ifdef MADV_HUGEPAGE
  uint8_t *hotStart = (uint8_t *)&__hot_start;
  uint8_t *hotEnd = (uint8_t *)&__hot_end;
  // Make sure the start and end are aligned with huge page address
  const size_t hugePageBytes = 2L * 1024 * 1024;
  uint8_t *from = hotStart - ((intptr_t)hotStart & (hugePageBytes - 1));
  uint8_t *to = hotEnd + (hugePageBytes - 1);
  to -= (intptr_t)to & (hugePageBytes - 1);

#ifdef ENABLE_DEBUG
  reportNumber("[hugify] hot start: ", (uint64_t)hotStart, 16);
  reportNumber("[hugify] hot end: ", (uint64_t)hotEnd, 16);
  reportNumber("[hugify] aligned huge page from: ", (uint64_t)from, 16);
  reportNumber("[hugify] aligned huge page to: ", (uint64_t)to, 16);
#endif

  if (!has_pagecache_thp_support()) {
    hugify_for_old_kernel(from, to);
    return;
  }

  if (__madvise(from, (to - from), MADV_HUGEPAGE) == -1) {
    char msg[] = "failed to allocate large page\n";
    // TODO: allow user to control the failure behavior.
    reportError(msg, sizeof(msg));
  }
#endif
}

/// This is hooking ELF's entry, it needs to save all machine state.
extern "C" __attribute((naked)) void __bolt_hugify_self() {
  __asm__ __volatile__(SAVE_ALL "call __bolt_hugify_self_impl\n" RESTORE_ALL
                                "jmp *__bolt_hugify_init_ptr(%%rip)\n" ::
                                    :);
}
