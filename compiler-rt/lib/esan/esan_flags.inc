//===-- esan_flags.inc ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Esan runtime flags.
//
//===----------------------------------------------------------------------===//

#ifndef ESAN_FLAG
# error "Define ESAN_FLAG prior to including this file!"
#endif

// ESAN_FLAG(Type, Name, DefaultValue, Description)
// See COMMON_FLAG in sanitizer_flags.inc for more details.

//===----------------------------------------------------------------------===//
// Cross-tool options
//===----------------------------------------------------------------------===//

ESAN_FLAG(int, cache_line_size, 64,
          "The number of bytes in a cache line.  For the working-set tool, this "
          "cannot be changed without also changing the compiler "
          "instrumentation.")

//===----------------------------------------------------------------------===//
// Working set tool options
//===----------------------------------------------------------------------===//

ESAN_FLAG(bool, record_snapshots, true,
          "Working set tool: whether to sample snapshots during a run.")

// Typical profiling uses a 10ms timer.  Our snapshots take some work
// to scan memory so we reduce to 20ms.
// To disable samples, turn off record_snapshots.
ESAN_FLAG(int, sample_freq, 20,
          "Working set tool: sampling frequency in milliseconds.")
