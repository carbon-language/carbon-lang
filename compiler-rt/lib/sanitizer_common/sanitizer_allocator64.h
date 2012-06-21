//===-- sanitizer_allocator64.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Specialized allocator which works only in 64-bit address space.
// To be used by ThreadSanitizer, MemorySanitizer and possibly other tools.
// The main feature of this allocator is that the header is located far away
// from the user memory region, so that the tool does not use extra shadow
// for the header.
//
// Status: not yet ready.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_H
#define SANITIZER_ALLOCATOR_H

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// Maps size class to size and back.
class DefaultSizeClassMap {
 private:
  // Here we use a spline composed of 5 polynomials of oder 1.
  // The first size class is l0, then the classes go with step s0
  // untill they reach l1, after which they go with step s1 and so on.
  // Steps should be powers of two for cheap division.
  // The size of the last size class should be a power of two.
  // There should be at most 256 size classes.
  static const uptr l0 = 1 << 4;
  static const uptr l1 = 1 << 9;
  static const uptr l2 = 1 << 12;
  static const uptr l3 = 1 << 15;
  static const uptr l4 = 1 << 18;
  static const uptr l5 = 1 << 21;

  static const uptr s0 = 1 << 4;
  static const uptr s1 = 1 << 6;
  static const uptr s2 = 1 << 9;
  static const uptr s3 = 1 << 12;
  static const uptr s4 = 1 << 15;

  static const uptr u0 = 0  + (l1 - l0) / s0;
  static const uptr u1 = u0 + (l2 - l1) / s1;
  static const uptr u2 = u1 + (l3 - l2) / s2;
  static const uptr u3 = u2 + (l4 - l3) / s3;
  static const uptr u4 = u3 + (l5 - l4) / s4;

 public:
  static const uptr kNumClasses = u4 + 1;
  static const uptr kMaxSize = l5;

  COMPILER_CHECK(kNumClasses <= 256);
  COMPILER_CHECK((kMaxSize & (kMaxSize - 1)) == 0);

  static uptr Size(uptr size_class) {
    if (size_class <= u0) return l0 + s0 * (size_class - 0);
    if (size_class <= u1) return l1 + s1 * (size_class - u0);
    if (size_class <= u2) return l2 + s2 * (size_class - u1);
    if (size_class <= u3) return l3 + s3 * (size_class - u2);
    if (size_class <= u4) return l4 + s4 * (size_class - u3);
    return 0;
  }
  static uptr Class(uptr size) {
    if (size <= l1) return 0  + (size - l0 + s0 - 1) / s0;
    if (size <= l2) return u0 + (size - l1 + s1 - 1) / s1;
    if (size <= l3) return u1 + (size - l2 + s2 - 1) / s2;
    if (size <= l4) return u2 + (size - l3 + s3 - 1) / s3;
    if (size <= l5) return u3 + (size - l4 + s4 - 1) / s4;
    return 0;
  }
};

}  // namespace __sanitizer

#endif  // SANITIZER_ALLOCATOR_H
