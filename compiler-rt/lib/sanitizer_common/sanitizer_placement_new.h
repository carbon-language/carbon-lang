//===-- sanitizer_placement_new.h -------------------------------*- C++ -*-===//
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
//
// The file provides 'placement new'.
// Do not include it into header files, only into source files.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_PLACEMENT_NEW_H
#define SANITIZER_PLACEMENT_NEW_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {
#if (SANITIZER_WORDSIZE == 64) || SANITIZER_MAC
typedef uptr operator_new_ptr_type;
#else
typedef u32 operator_new_ptr_type;
#endif
}  // namespace __sanitizer

inline void *operator new(__sanitizer::operator_new_ptr_type sz, void *p) {
  return p;
}

#endif  // SANITIZER_PLACEMENT_NEW_H
