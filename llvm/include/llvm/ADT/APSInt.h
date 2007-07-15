//===-- llvm/Support/APSInt.h - Arbitrary Precision Signed Int -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the APSInt class, which is a simple class that
// represents an arbitrary sized integer that knows its signedness.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_APSINT_H
#define LLVM_APSINT_H

#include "llvm/ADT/APInt.h"

namespace llvm {
  
  
class APSInt : public APInt {
  bool IsUnsigned;
public:
  /// APSInt ctor - Create an APSInt with the specified width, default to
  /// unsigned.
  explicit APSInt(unsigned BitWidth) : APInt(BitWidth, 0), IsUnsigned(true) {}
  APSInt(const APInt &I) : APInt(I), IsUnsigned(true) {}

  APSInt &operator=(const APSInt &RHS) {
    APInt::operator=(RHS); 
    IsUnsigned = RHS.IsUnsigned;
    return *this;
  }

  APSInt &operator=(const APInt &RHS) {
    // Retain our current sign.
    APInt::operator=(RHS); 
    return *this;
  }

  APSInt &operator=(uint64_t RHS) {
    // Retain our current sign.
    APInt::operator=(RHS); 
    return *this;
  }

  // Query sign information.
  bool isSigned() const { return !IsUnsigned; }
  bool isUnsigned() const { return IsUnsigned; }
  void setIsUnsigned(bool Val) { IsUnsigned = Val; }
  void setIsSigned(bool Val) { IsUnsigned = !Val; }
  
  
  const APSInt &operator%=(const APSInt &RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    if (IsUnsigned)
      *this = urem(RHS);
    else
      *this = srem(RHS);
    return *this;
  }
  const APSInt &operator/=(const APSInt &RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    if (IsUnsigned)
      *this = udiv(RHS);
    else
      *this = sdiv(RHS);
    return *this;
  }
  APSInt operator%(const APSInt &RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? urem(RHS) : srem(RHS);
  }
  APSInt operator/(const APSInt &RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? udiv(RHS) : sdiv(RHS);
  }
  
  const APSInt &operator>>=(unsigned Amt) {
    *this = *this >> Amt;
    return *this;
  }
  
  APSInt operator>>(unsigned Amt) const {
    return IsUnsigned ? lshr(Amt) : ashr(Amt);
  }
  
  inline bool operator<(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? ult(RHS) : slt(RHS);
  }
  inline bool operator>(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? ugt(RHS) : sgt(RHS);
  }
  inline bool operator<=(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? ule(RHS) : sle(RHS);
  }
  inline bool operator>=(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? uge(RHS) : sge(RHS);
  }
};
  
} // end namespace llvm

#endif
