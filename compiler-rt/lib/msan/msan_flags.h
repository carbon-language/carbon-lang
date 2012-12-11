//===-- msan_allocator.cc --------------------------- ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// MemorySanitizer allocator.
//===----------------------------------------------------------------------===//
#ifndef MSAN_FLAGS_H
#define MSAN_FLAGS_H

namespace __msan {

// Flags.
struct Flags {
  int exit_code;
  int num_callers;
  int verbosity;
  bool poison_heap_with_zeroes;  // default: false
  bool poison_stack_with_zeroes;  // default: false
  bool poison_in_malloc;  // default: true
  bool report_umrs;
};

Flags *flags();

}  // namespace __msan

#endif  // MSAN_FLAGS_H
