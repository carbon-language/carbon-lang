//===--- UnicodeCharRanges.h - Types and functions for character ranges ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SUPPORT_UNICODECHARRANGES_H
#define LLVM_SUPPORT_UNICODECHARRANGES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/raw_ostream.h"

namespace {

struct UnicodeCharRange {
  uint32_t Lower;
  uint32_t Upper;
};
typedef llvm::ArrayRef<UnicodeCharRange> UnicodeCharSet;

/// Returns true if each of the ranges in \p CharSet is a proper closed range
/// [min, max], and if the ranges themselves are ordered and non-overlapping.
static inline bool isValidCharSet(UnicodeCharSet CharSet) {
#ifndef NDEBUG
  static llvm::SmallPtrSet<const UnicodeCharRange *, 16> Validated;
  static llvm::sys::Mutex ValidationMutex;

  // Check the validation cache.
  {
    llvm::MutexGuard Guard(ValidationMutex);
    if (Validated.count(CharSet.data()))
      return true;
  }

  // Walk through the ranges.
  uint32_t Prev = 0;
  for (UnicodeCharSet::iterator I = CharSet.begin(), E = CharSet.end();
       I != E; ++I) {
    if (I != CharSet.begin() && Prev >= I->Lower) {
      DEBUG(llvm::dbgs() << "Upper bound 0x");
      DEBUG(llvm::dbgs().write_hex(Prev));
      DEBUG(llvm::dbgs() << " should be less than succeeding lower bound 0x");
      DEBUG(llvm::dbgs().write_hex(I->Lower) << "\n");
      return false;
    }
    if (I->Upper < I->Lower) {
      DEBUG(llvm::dbgs() << "Upper bound 0x");
      DEBUG(llvm::dbgs().write_hex(I->Lower));
      DEBUG(llvm::dbgs() << " should not be less than lower bound 0x");
      DEBUG(llvm::dbgs().write_hex(I->Upper) << "\n");
      return false;
    }
    Prev = I->Upper;
  }

  // Update the validation cache.
  {
    llvm::MutexGuard Guard(ValidationMutex);
    Validated.insert(CharSet.data());
  }
#endif
  return true;
}

} // namespace


/// Returns true if the Unicode code point \p C is within the set of
/// characters specified by \p CharSet.
LLVM_READONLY static inline bool isCharInSet(uint32_t C,
                                             UnicodeCharSet CharSet) {
  assert(isValidCharSet(CharSet));

  size_t LowPoint = 0;
  size_t HighPoint = CharSet.size();

  // Binary search the set of char ranges.
  while (HighPoint != LowPoint) {
    size_t MidPoint = (HighPoint + LowPoint) / 2;
    if (C < CharSet[MidPoint].Lower)
      HighPoint = MidPoint;
    else if (C > CharSet[MidPoint].Upper)
      LowPoint = MidPoint + 1;
    else
      return true;
  }

  return false;
}

#endif // LLVM_SUPPORT_UNICODECHARRANGES_H
