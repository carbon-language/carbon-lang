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

  // Tell the tools to write their reports to given file descriptor instead of
  // stderr.
  void __sanitizer_set_report_fd(int fd);

  // Notify the tools that the sandbox is going to be turned on. The reserved
  // parameter will be used in the future to hold a structure with functions
  // that the tools may call to bypass the sandbox.
  void __sanitizer_sandbox_on_notify(void *reserved);

  // This function is called by the tool when it has just finished reporting
  // an error. 'error_summary' is a one-line string that summarizes
  // the error message. This function can be overridden by the client.
  void __sanitizer_report_error_summary(const char *error_summary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_COMMON_INTERFACE_DEFS_H
