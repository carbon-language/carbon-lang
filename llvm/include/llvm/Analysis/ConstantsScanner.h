//==- llvm/Analysis/ConstantsScanner.h - Iterate over constants -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements an iterator to walk through the constants referenced by
// a method.  This is used by the Bitcode & Assembly writers to build constant
// pools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTANTSSCANNER_H
#define LLVM_ANALYSIS_CONSTANTSSCANNER_H

#include "llvm/IR/InstIterator.h"

namespace llvm {

class Constant;

class constant_iterator : public std::iterator<std::forward_iterator_tag,
                                               const Constant, ptrdiff_t> {
  const_inst_iterator InstI;                // Method instruction iterator
  unsigned OpIdx;                           // Operand index

  bool isAtConstant() const {
    assert(!InstI.atEnd() && OpIdx < InstI->getNumOperands() &&
           "isAtConstant called with invalid arguments!");
    return isa<Constant>(InstI->getOperand(OpIdx));
  }

public:
  constant_iterator(const Function *F) : InstI(inst_begin(F)), OpIdx(0) {
    // Advance to first constant... if we are not already at constant or end
    if (InstI != inst_end(F) &&                            // InstI is valid?
        (InstI->getNumOperands() == 0 || !isAtConstant())) // Not at constant?
      operator++();
  }

  constant_iterator(const Function *F, bool) // end ctor
      : InstI(inst_end(F)),
        OpIdx(0) {}

  bool operator==(const constant_iterator &x) const {
    return OpIdx == x.OpIdx && InstI == x.InstI;
  }
  bool operator!=(const constant_iterator &x) const { return !(*this == x); }

  pointer operator*() const {
    assert(isAtConstant() && "Dereferenced an iterator at the end!");
    return cast<Constant>(InstI->getOperand(OpIdx));
  }
  pointer operator->() const { return **this; }

  constant_iterator &operator++() { // Preincrement implementation
    ++OpIdx;
    do {
      unsigned NumOperands = InstI->getNumOperands();
      while (OpIdx < NumOperands && !isAtConstant()) {
        ++OpIdx;
      }

      if (OpIdx < NumOperands) return *this;  // Found a constant!
      ++InstI;
      OpIdx = 0;
    } while (!InstI.atEnd());

    return *this;  // At the end of the method
  }

  onstant_iterator operator++(int) { // Postincrement
    constant_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool atEnd() const { return InstI.atEnd(); }
};

inline constant_iterator constant_begin(const Function *F) {
  return constant_iterator(F);
}

inline constant_iterator constant_end(const Function *F) {
  return constant_iterator(F, true);
}

} // End llvm namespace

#endif
