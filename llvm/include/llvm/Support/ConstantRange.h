//===-- llvm/Support/ConstantRange.h - Represent a range --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Represent a range of possible values that may occur when the program is run
// for an integral value.  This keeps track of a lower and upper bound for the
// constant, which MAY wrap around the end of the numeric range.  To do this, it
// keeps track of a [lower, upper) bound, which specifies an interval just like
// STL iterators.  When used with boolean values, the following are important
// ranges (other integral ranges use min/max values for special range values):
//
//  [F, F) = {}     = Empty set
//  [T, F) = {T}
//  [F, T) = {F}
//  [T, T) = {F, T} = Full set
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CONSTANT_RANGE_H
#define LLVM_SUPPORT_CONSTANT_RANGE_H

#include "Support/DataTypes.h"
#include <iosfwd>

namespace llvm {
class Constant;
class ConstantIntegral;
class Type;

class ConstantRange {
  ConstantIntegral *Lower, *Upper;
 public:
  /// Initialize a full (the default) or empty set for the specified type.
  ///
  ConstantRange(const Type *Ty, bool isFullSet = true);
  
  /// Initialize a range of values explicitly... this will assert out if
  /// Lower==Upper and Lower != Min or Max for its type, if the two constants
  /// have different types, or if the constant are not integral values.
  ///
  ConstantRange(Constant *Lower, Constant *Upper);
  
  /// Initialize a set of values that all satisfy the condition with C.
  ///
  ConstantRange(unsigned SetCCOpcode, ConstantIntegral *C);

  /// getLower - Return the lower value for this range...
  ///
  ConstantIntegral *getLower() const { return Lower; }

  /// getUpper - Return the upper value for this range...
  ///
  ConstantIntegral *getUpper() const { return Upper; }

  /// getType - Return the LLVM data type of this range.
  ///
  const Type *getType() const;
  
  /// isFullSet - Return true if this set contains all of the elements possible
  /// for this data-type
  ///
  bool isFullSet() const;
  
  /// isEmptySet - Return true if this set contains no members.
  ///
  bool isEmptySet() const;

  /// isWrappedSet - Return true if this set wraps around the top of the range,
  /// for example: [100, 8)
  ///
  bool isWrappedSet() const;
  
  /// getSingleElement - If this set contains a single element, return it,
  /// otherwise return null.
  ///
  ConstantIntegral *getSingleElement() const;
  
  /// isSingleElement - Return true if this set contains exactly one member.
  ///
  bool isSingleElement() const { return getSingleElement() != 0; }

  /// getSetSize - Return the number of elements in this set.
  ///
  uint64_t getSetSize() const;

  /// operator== - Return true if this range is equal to another range.
  ///
  bool operator==(const ConstantRange &CR) const {
    return Lower == CR.Lower && Upper == CR.Upper;
  }
  bool operator!=(const ConstantRange &CR) const {
    return !operator==(CR);
  }

  /// intersect - Return the range that results from the intersection of this
  /// range with another range.  The resultant range is pruned as much as
  /// possible, but there may be cases where elements are included that are in
  /// one of the sets but not the other.  For example: [100, 8) intersect [3,
  /// 120) yields [3, 120)
  ///
  ConstantRange intersectWith(const ConstantRange &CR) const;

  /// union - Return the range that results from the union of this range with
  /// another range.  The resultant range is guaranteed to include the elements
  /// of both sets, but may contain more.  For example, [3, 9) union [12,15) is
  /// [3, 15), which includes 9, 10, and 11, which were not included in either
  /// set before.
  ///
  ConstantRange unionWith(const ConstantRange &CR) const;

  /// print - Print out the bounds to a stream...
  ///
  void print(std::ostream &OS) const;

  /// dump - Allow printing from a debugger easily...
  ///
  void dump() const;
};

inline std::ostream &operator<<(std::ostream &OS, const ConstantRange &CR) {
  CR.print(OS);
  return OS;
}

} // End llvm namespace

#endif
