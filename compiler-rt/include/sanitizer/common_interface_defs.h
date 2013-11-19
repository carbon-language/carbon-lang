//===-- sanitizer/common_interface_defs.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common part of the public sanitizer interface.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_COMMON_INTERFACE_DEFS_H
#define SANITIZER_COMMON_INTERFACE_DEFS_H

#include <stddef.h>
#include <stdint.h>

// GCC does not understand __has_feature.
#if !defined(__has_feature)
# define __has_feature(x) 0
#endif

#ifdef __cplusplus
extern "C" {
#endif
  // Tell the tools to write their reports to "path.<pid>" instead of stderr.
  void __sanitizer_set_report_path(const char *path);

  // Notify the tools that the sandbox is going to be turned on. The reserved
  // parameter will be used in the future to hold a structure with functions
  // that the tools may call to bypass the sandbox.
  void __sanitizer_sandbox_on_notify(void *reserved);

  // This function is called by the tool when it has just finished reporting
  // an error. 'error_summary' is a one-line string that summarizes
  // the error message. This function can be overridden by the client.
  void __sanitizer_report_error_summary(const char *error_summary);

  // Some of the sanitizers (e.g. asan/tsan) may miss bugs that happen
  // in unaligned loads/stores. In order to find such bugs reliably one needs
  // to replace plain unaligned loads/stores with these calls.
  uint16_t __sanitizer_unaligned_load16(const void *p);
  uint32_t __sanitizer_unaligned_load32(const void *p);
  uint64_t __sanitizer_unaligned_load64(const void *p);
  void __sanitizer_unaligned_store16(void *p, uint16_t x);
  void __sanitizer_unaligned_store32(void *p, uint32_t x);
  void __sanitizer_unaligned_store64(void *p, uint64_t x);

  // Record and dump coverage info.
  void __sanitizer_cov_dump();

  // Annotate the current state of a contiguous container, such as
  // std::vector, std::string or similar.
  // A contiguous container is a container that keeps all of its elements
  // in a contiguous region of memory. The container owns the region of memory
  // [beg, end); the memory [beg, mid) is used to store the current elements
  // and the memory [mid, end) is reserved for future elements;
  // end <= mid <= end. For example, in "std::vector<> v"
  //   beg = &v[0];
  //   end = beg + v.capacity() * sizeof(v[0]);
  //   mid = beg + v.size()     * sizeof(v[0]);
  //
  // This annotation tells the Sanitizer tool about the current state of the
  // container so that the tool can report errors when memory from [mid, end)
  // is accessed. Insert this annotation into methods like push_back/pop_back.
  // Supply the old and the new values of mid (old_mid/new_mid).
  // In the initial state mid == end and so should be the final
  // state when the container is destroyed or when it reallocates the storage.
  //
  // Use with caution and don't use for anything other than vector-like classes.
  //
  // For AddressSanitizer, 'beg' should be 8-aligned.
  void __sanitizer_annotate_contiguous_container(const void *beg,
                                                 const void *end,
                                                 const void *old_mid,
                                                 const void *new_mid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_COMMON_INTERFACE_DEFS_H
