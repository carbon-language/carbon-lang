//===-- tsan_placement_new.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// The file provides 'placement new'
// Do not include it into header files, only into source files.
//===----------------------------------------------------------------------===//
#ifndef TSAN_PLACEMENT_NEW_H
#define TSAN_PLACEMENT_NEW_H

#include "tsan_defs.h"

inline void *operator new(__sanitizer::uptr sz, void *p) {
  return p;
}

#endif  // TSAN_PLACEMENT_NEW_H
