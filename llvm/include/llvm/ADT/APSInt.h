//===-- llvm/ADT/APSInt.h - Arbitrary Precision Signed Int -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  explicit APSInt(uint32_t BitWidth) : APInt(BitWidth, 0), IsUnsigned(true) {}
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
  
  /// This is used internally to convert an APInt to a string.
  /// @brief Converts an APInt to a std::string
  std::string toString(uint8_t Radix = 10) const {
    return APInt::toString(Radix, isSigned());
  }
  
  
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
  
  APSInt& extend(uint32_t width) {
    if (IsUnsigned)
      zext(width);
    else
      sext(width);
    return *this;
  }
  
  APSInt& extOrTrunc(uint32_t width) {
      if (IsUnsigned)
        zextOrTrunc(width);
      else
        sextOrTrunc(width);
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
