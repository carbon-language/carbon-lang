//===-- llvm/ConstantRangesSet.h - The constant set of ranges ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains class that implements constant set of ranges:
/// [<Low0,High0>,...,<LowN,HighN>]. Mainly, this set is used by SwitchInst and
/// represents case value that may contain multiple ranges for a single
/// successor.
///
//
//===----------------------------------------------------------------------===//

#ifndef CONSTANTRANGESSET_H_
#define CONSTANTRANGESSET_H_

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"

namespace llvm {
  
class ConstantRangesSet;  
  
template <bool IsReadonly> struct CRSConstantTypes {
  typedef ConstantInt ConstantIntTy;
  typedef ConstantRangesSet ConstantRangesSetTy;  
};

template <>
struct CRSConstantTypes<true> {
  typedef const ConstantInt ConstantIntTy;
  typedef const ConstantRangesSet ConstantRangesSetTy;
};  
  
//===----------------------------------------------------------------------===//
/// ConstantRangesSet - class that implements constant set of ranges.
/// It is a wrapper for some real "holder" class (currently ConstantArray).
/// It contains functions, that allows to parse "holder" like a set of ranges.
/// Note: It is assumed that "holder" is inherited from Constant object.
///       ConstantRangesSet may be converted to and from Constant* pointer.
///
class ConstantRangesSet {
  Constant *Array;
public:
  
  // implicit
  ConstantRangesSet(Constant *V) : Array(V) {}
  
  operator Constant*() { return Array; }
  operator const Constant*() const { return Array; }
  Constant *operator->() { return Array; }
  const Constant *operator->() const { return Array; }
   
  template <bool IsReadonly>
  struct RangeT {
    
    typedef typename CRSConstantTypes<IsReadonly>::ConstantIntTy ConstantIntTy;
    typedef std::pair<RangeT, RangeT> SubRes;
    
    ConstantIntTy *Low;
    ConstantIntTy *High;
   
    RangeT() : Low(0), High(0) {}
    RangeT(const RangeT<false> &RHS) : Low(RHS.Low), High(RHS.High) {}
    RangeT(ConstantIntTy *C) : Low(C), High(C) {}
    RangeT(ConstantIntTy *L, ConstantIntTy *H) : Low(L), High(H) {}
   
    bool operator<(const RangeT &RHS) const {
      assert(Low && High && "Case range is not initialized.");
      assert(RHS.Low && RHS.High && "Right case range is not initialized.");
      const APInt &LowInt = Low->getValue();
      const APInt &HighInt = High->getValue();
      const APInt &RHSLowInt = RHS.Low->getValue();
      const APInt &RHSHighInt = RHS.High->getValue();
      if (LowInt.getBitWidth() == RHSLowInt.getBitWidth()) {
        if (LowInt.eq(RHSLowInt)) {
          if (HighInt.ult(RHSHighInt))
            return true;
          return false;
        }
        if (LowInt.ult(RHSLowInt))
          return true;
        return false;
      } else
        return LowInt.getBitWidth() < RHSLowInt.getBitWidth();      
    }

    bool operator==(const RangeT &RHS) const {
      assert(Low && High && "Case range is not initialized.");
      assert(RHS.Low && RHS.High && "Right case range is not initialized.");
      if (Low->getValue().getBitWidth() != RHS.Low->getValue().getBitWidth())
        return false;
      return Low->getValue() == RHS.Low->getValue() &&
             High->getValue() == RHS.High->getValue();      
    }
 
    bool operator!=(const RangeT &RHS) const {
      return !operator ==(RHS);      
    }
 
    static bool LessBySize(const RangeT &LHS, const RangeT &RHS) {
      assert(LHS.Low->getBitWidth() == RHS.Low->getBitWidth() && 
          "This type of comparison requires equal bit width for LHS and RHS");
      APInt LSize = LHS.High->getValue() - LHS.Low->getValue();
      APInt RSize = RHS.High->getValue() - RHS.Low->getValue();;
      return LSize.ult(RSize);      
    }
 
    bool isInRange(const APInt &IntVal) const {
      assert(Low && High && "Case range is not initialized.");
      if (IntVal.getBitWidth() != Low->getValue().getBitWidth())
        return false;
      return IntVal.uge(Low->getValue()) && IntVal.ule(High->getValue());      
    }    
  
    bool isInRange(const ConstantIntTy *CI) const {
      const APInt& IntVal = CI->getValue();
      return isInRange(IntVal);
    }
  
    SubRes sub(const RangeT &RHS) const {
      SubRes Res;
      
      // RHS is either more global and includes this range or
      // if it doesn't intersected with this range.
      if (!isInRange(RHS.Low) && !isInRange(RHS.High)) {
        
        // If RHS more global (it is enough to check
        // only one border in this case.
        if (RHS.isInRange(Low))
          return std::make_pair(RangeT(Low, High), RangeT()); 
        
        return Res;
      }
      
      const APInt& LoInt = Low->getValue();
      const APInt& HiInt = High->getValue();
      APInt RHSLoInt = RHS.Low->getValue();
      APInt RHSHiInt = RHS.High->getValue();
      if (LoInt.ult(RHSLoInt)) {
        Res.first.Low = Low;
        Res.first.High = ConstantIntTy::get(RHS.Low->getContext(), --RHSLoInt);
      }
      if (HiInt.ugt(RHSHiInt)) {
        Res.second.Low = ConstantIntTy::get(RHS.High->getContext(), ++RHSHiInt);
        Res.second.High = High;
      }
      return Res;      
    }
  };      

  typedef RangeT<false> Range;
 
  /// Checks is the given constant satisfies this case. Returns
  /// true if it equals to one of contained values or belongs to the one of
  /// contained ranges.
  bool isSatisfies(const ConstantInt *C) const {
    const APInt &CheckingVal = C->getValue();
    for (unsigned i = 0, e = getNumItems(); i < e; ++i) {
      const Constant *CV = Array->getAggregateElement(i);
      unsigned VecSize = cast<VectorType>(CV->getType())->getNumElements();
      switch (VecSize) {
      case 1:
        if (cast<const ConstantInt>(CV->getAggregateElement(0U))->getValue() ==
            CheckingVal)
          return true;
        break;
      case 2: {
        const APInt &Lo =
            cast<const ConstantInt>(CV->getAggregateElement(0U))->getValue();
        const APInt &Hi =
            cast<const ConstantInt>(CV->getAggregateElement(1))->getValue();
        if (Lo.uge(CheckingVal) && Hi.ule(CheckingVal))
          return true;
      }
        break;
      default:
        assert(0 && "Only pairs and single numbers are allowed here.");
        break;
      }
    }
    return false;    
  }
  
  /// Returns set's item with given index.
  Range getItem(unsigned idx) {
    Constant *CV = Array->getAggregateElement(idx);
    unsigned NumEls = cast<VectorType>(CV->getType())->getNumElements();
    switch (NumEls) {
    case 1:
      return Range(cast<ConstantInt>(CV->getAggregateElement(0U)),
                   cast<ConstantInt>(CV->getAggregateElement(0U)));
    case 2:
      return Range(cast<ConstantInt>(CV->getAggregateElement(0U)),
                   cast<ConstantInt>(CV->getAggregateElement(1)));
    default:
      assert(0 && "Only pairs and single numbers are allowed here.");
      return Range();
    }    
  }
  
  const Range getItem(unsigned idx) const {
    const Constant *CV = Array->getAggregateElement(idx);
    
    unsigned NumEls = cast<VectorType>(CV->getType())->getNumElements();
    switch (NumEls) {
    case 1:
      return Range(cast<ConstantInt>(
                     const_cast<Constant*>(CV->getAggregateElement(0U))),
                   cast<ConstantInt>(
                     const_cast<Constant*>(CV->getAggregateElement(0U))));
    case 2:
      return Range(cast<ConstantInt>(
                     const_cast<Constant*>(CV->getAggregateElement(0U))),
                   cast<ConstantInt>(
                     const_cast<Constant*>(CV->getAggregateElement(1))));
    default:
      assert(0 && "Only pairs and single numbers are allowed here.");
      return Range();
    }    
  }
  
  /// Return number of items (ranges) stored in set.
  unsigned getNumItems() const {
    return cast<ArrayType>(Array->getType())->getNumElements();
  }
  
  /// Returns set the size, that equals number of all values + sizes of all
  /// ranges.
  /// Ranges set is considered as flat numbers collection.
  /// E.g.: for range [<0>, <1>, <4,8>] the size will 7;
  ///       for range [<0>, <1>, <5>] the size will 3
  unsigned getSize() const {
    APInt sz(getItem(0).Low->getBitWidth(), 0);
    for (unsigned i = 0, e = getNumItems(); i != e; ++i) {
      const APInt &S = getItem(i).High->getValue() - getItem(i).Low->getValue();
      sz += S;
    }
    return sz.getZExtValue();    
  }
  
  /// Allows to access single value even if it belongs to some range.
  /// Ranges set is considered as flat numbers collection.
  /// [<1>, <4,8>] is considered as [1,4,5,6,7,8] 
  /// For range [<1>, <4,8>] getSingleValue(3) returns 6.
  APInt getSingleValue(unsigned idx) const {
    APInt sz(getItem(0).Low->getBitWidth(), 0);
    for (unsigned i = 0, e = getNumItems(); i != e; ++i) {
      const APInt& S = getItem(i).High->getValue() - getItem(i).Low->getValue();
      APInt oldSz = sz;
      sz += S;
      if (oldSz.uge(i) && sz.ult(i)) {
        APInt Res = getItem(i).Low->getValue();
        APInt Offset(oldSz.getBitWidth(), i);
        Offset -= oldSz;
        Res += Offset;
        return Res;
      }
    }
    assert(0 && "Index exceeds high border.");
    return sz;    
  }
};  

}

#endif /* CONSTANTRANGESSET_H_ */
