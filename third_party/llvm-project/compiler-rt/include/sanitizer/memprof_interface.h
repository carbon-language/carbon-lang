//===-- sanitizer/memprof_interface.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler (MemProf).
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_MEMPROF_INTERFACE_H
#define SANITIZER_MEMPROF_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif
/// Records access to a memory region (<c>[addr, addr+size)</c>).
///
/// This memory must be previously allocated by your program.
///
/// \param addr Start of memory region.
/// \param size Size of memory region.
void __memprof_record_access_range(void const volatile *addr, size_t size);

/// Records access to a memory address <c><i>addr</i></c>.
///
/// This memory must be previously allocated by your program.
///
/// \param addr Accessed memory address
void __memprof_record_access(void const volatile *addr);

/// User-provided callback on MemProf errors.
///
/// You can provide a function that would be called immediately when MemProf
/// detects an error. This is useful in cases when MemProf detects an error but
/// your program crashes before the MemProf report is printed.
void __memprof_on_error(void);

/// Prints accumulated statistics to <c>stderr</c> (useful for calling from the
/// debugger).
void __memprof_print_accumulated_stats(void);

/// User-provided default option settings.
///
/// You can provide your own implementation of this function to return a string
/// containing MemProf runtime options (for example,
/// <c>verbosity=1:print_stats=1</c>).
///
/// \returns Default options string.
const char *__memprof_default_options(void);

/// Prints the memory profile to the current profile file.
///
/// \returns 0 on success.
int __memprof_profile_dump(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_MEMPROF_INTERFACE_H
