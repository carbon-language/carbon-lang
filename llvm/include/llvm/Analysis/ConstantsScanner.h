//==-- llvm/Analysis/ConstantsScanner.h - Iterate over constants -*- C++ -*-==//
//
// This class implements an iterator to walk through the constants referenced by
// a method.  This is used by the Bytecode & Assembly writers to build constant
// pools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTANTSSCANNER_H
#define LLVM_ANALYSIS_CONSTANTSSCANNER_H

#include "llvm/Method.h"
#include "llvm/Instruction.h"
#include <iterator>
class ConstPoolVal;

class constant_iterator
  : public std::forward_iterator<const ConstPoolVal, ptrdiff_t> {
  Method::inst_const_iterator InstI;        // Method instruction iterator
  unsigned OpIdx;                           // Operand index

  typedef constant_iterator _Self;

  inline bool isAtConstant() const {
    assert(!InstI.atEnd() && OpIdx < InstI->getNumOperands() &&
	   "isAtConstant called with invalid arguments!");
    return isa<ConstPoolVal>(InstI->getOperand(OpIdx));
  }

public:
  inline constant_iterator(const Method *M) : InstI(M->inst_begin()), OpIdx(0) {
    // Advance to first constant... if we are not already at constant or end
    if (InstI != M->inst_end() &&                          // InstI is valid?
	(InstI->getNumOperands() == 0 || !isAtConstant())) // Not at constant?
      operator++();
  }

  inline constant_iterator(const Method *M, bool)   // end ctor
    : InstI(M->inst_end()), OpIdx(0) {
  }

  inline bool operator==(const _Self& x) const { return OpIdx == x.OpIdx && 
						        InstI == x.InstI; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }

  inline pointer operator*() const {
    assert(isAtConstant() && "Dereferenced an iterator at the end!");
    return cast<ConstPoolVal>(InstI->getOperand(OpIdx));
  }
  inline pointer operator->() const { return operator*(); }

  inline _Self& operator++() {   // Preincrement implementation
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

  inline _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }

  inline bool atEnd() const { return InstI.atEnd(); }
};

inline constant_iterator constant_begin(const Method *M) {
  return constant_iterator(M);
}

inline constant_iterator constant_end(const Method *M) {
  return constant_iterator(M, true);
}

#endif
