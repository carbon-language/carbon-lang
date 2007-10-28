//===----------- Alignment.h - Alignment computation ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duncan Sands and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for computing alignments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALIGNMENT_H
#define LLVM_SUPPORT_ALIGNMENT_H

namespace llvm {

/// MinAlign - A and B are either alignments or offsets.  Return the minimum
/// alignment that may be assumed after adding the two together.

static inline unsigned MinAlign(unsigned A, unsigned B) {
  // The largest power of 2 that divides both A and B.
  return (A | B) & -(A | B);
}

} // end namespace llvm
#endif
