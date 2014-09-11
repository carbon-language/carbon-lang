//===-- ubsan_flags.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Runtime flags for UndefinedBehaviorSanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef UBSAN_FLAGS_H
#define UBSAN_FLAGS_H

namespace __ubsan {

struct Flags {
  bool halt_on_error;
  bool print_stacktrace;
};

extern Flags ubsan_flags;
inline Flags *flags() { return &ubsan_flags; }

void InitializeFlags();

}  // namespace __ubsan

#endif  // UBSAN_FLAGS_H
