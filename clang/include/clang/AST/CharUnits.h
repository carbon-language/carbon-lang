//===--- CharUnits.h - Character units for sizes and offsets ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CharUnits class
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_CHARUNITS_H
#define LLVM_CLANG_AST_CHARUNITS_H

#include "llvm/ADT/StringExtras.h"

#include <stdint.h>
#include <string>

namespace clang {
  // An opaque type for sizes expressed in character units 
  class CharUnits {
    public:
      typedef int64_t RawType;

    private:
      RawType Quantity;

      explicit CharUnits(RawType C) : Quantity(C) {}

    public:

      /// A default constructor
      CharUnits() : Quantity(0) {}

      /// Zero - Construct a CharUnits quantity of zero
      static CharUnits Zero() {
        return CharUnits(0);
      }

      /// One - Construct a CharUnits quantity of one
      static CharUnits One() {
        return CharUnits(1);
      }

      /// fromRaw - Construct a CharUnits quantity from a raw integer type.
      static CharUnits fromRaw(RawType Quantity) {
        return CharUnits(Quantity); 
      }

      // compound assignment
      CharUnits& operator+= (const CharUnits &Other) {
        Quantity += Other.Quantity;
        return *this;
      }
      CharUnits& operator-= (const CharUnits &Other) {
        Quantity -= Other.Quantity;
        return *this;
      }
       
      // comparison operators
      bool operator== (const CharUnits &Other) const {
        return Quantity == Other.Quantity;
      }
      bool operator!= (const CharUnits &Other) const {
        return Quantity != Other.Quantity;
      }

      // relational operators
      bool operator<  (const CharUnits &Other) const { 
        return Quantity <  Other.Quantity; 
      }
      bool operator<= (const CharUnits &Other) const { 
        return Quantity <= Other.Quantity;
      }
      bool operator>  (const CharUnits &Other) const { 
        return Quantity >  Other.Quantity; 
      }
      bool operator>= (const CharUnits &Other) const { 
        return Quantity >= Other.Quantity; 
      }

      // other predicates
      
      /// isZero - Test whether the quantity equals zero.
      bool isZero() const     { return Quantity == 0; }

      /// isOne - Test whether the quantity equals one.
      bool isOne() const      { return Quantity == 1; }

      /// isPositive - Test whether the quanity is greater than zero.
      bool isPositive() const { return Quantity  > 0; }

      /// isNegative - Test whether the quantity is less than zero.
      bool isNegative() const { return Quantity  < 0; }

      // arithmetic operators
      CharUnits operator* (RawType N) const {
        return CharUnits(Quantity * N);
      }
      CharUnits operator/ (RawType N) const {
        return CharUnits(Quantity / N);
      }
      RawType operator/ (const CharUnits &Other) const {
        return Quantity / Other.Quantity;
      }
      CharUnits operator% (RawType N) const {
        return CharUnits(Quantity % N);
      }
      RawType operator% (const CharUnits &Other) const {
        return Quantity % Other.Quantity;
      }
      CharUnits operator+ (const CharUnits &Other) const {
        return CharUnits(Quantity + Other.Quantity);
      }
      CharUnits operator- (const CharUnits &Other) const {
        return CharUnits(Quantity - Other.Quantity);
      }
      
      // conversions

      /// toString - Convert to a string.
      std::string toString() const {
        return llvm::itostr(Quantity);
      }

      /// getRaw - Get the raw integer representation of this quantity.
      RawType getRaw() const { return Quantity; }


  }; // class CharUnit
} // namespace clang

inline clang::CharUnits operator* (clang::CharUnits::RawType Scale, 
                                   const clang::CharUnits &CU) {
  return CU * Scale;
}

#endif // LLVM_CLANG_AST_CHARUNITS_H
