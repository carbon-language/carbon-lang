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
#include "llvm/LLVMContext.h"

namespace llvm {
  
template <class ImplTy>
class IntItemBase {
protected:
  ImplTy Implementation;
  typedef IntItemBase<ImplTy> self;
public:
  
  IntItemBase() {}
  
  IntItemBase(const ImplTy &impl) : Implementation(impl) {}
  
  // implicit
  IntItemBase(const APInt& src) : Implementation(src) {}
  
  operator const APInt&() const {
    return (const APInt&)Implementation;
  }
  bool operator<(const self& RHS) const {
    return ((const APInt&)*this).ult(RHS);
  }
  bool operator==(const self& RHS) const {
    return (const APInt&)*this == (const APInt&)RHS;
  }
  bool operator!=(const self& RHS) const {
    return (const APInt&)*this != (const APInt&)RHS;
  }  
  self& operator=(const ImplTy& RHS) {
    Implementation = RHS;
    return *this;
  }
  const APInt* operator->() const {
    return &((const APInt&)Implementation);
  }
  const APInt& operator*() const {
    return ((const APInt&)Implementation);
  }
  // FIXME: Hack. Will removed.
  ImplTy& getImplementation() {
    return Implementation;
  }
};
 
class IntItemConstantIntImpl {
  const ConstantInt *ConstantIntVal;
public:
  IntItemConstantIntImpl() : ConstantIntVal(0) {}
  IntItemConstantIntImpl(const ConstantInt *Val) : ConstantIntVal(Val) {}
  IntItemConstantIntImpl(LLVMContext &Ctx, const APInt& src) {
    ConstantIntVal = cast<ConstantInt>(ConstantInt::get(Ctx, src));
  }
  explicit IntItemConstantIntImpl(const APInt& src) {
    ConstantIntVal =
        cast<ConstantInt>(ConstantInt::get(llvm::getGlobalContext(), src));
  }
  operator const APInt&() const {
    return ConstantIntVal->getValue();
  }  
  operator const ConstantInt*() {
    return ConstantIntVal;
  }
};

class IntItem : public IntItemBase<IntItemConstantIntImpl> {
  typedef IntItemBase<IntItemConstantIntImpl> ParentTy;
  IntItem(const IntItemConstantIntImpl& Impl) : ParentTy(Impl) {}
public:
  
  IntItem() {}
  
  // implicit
  IntItem(const APInt& src) : ParentTy(src) {}  
  
  static IntItem fromConstantInt(const ConstantInt *V) {
    IntItemConstantIntImpl Impl(V);
    return IntItem(Impl);
  }
  static IntItem fromType(Type* Ty, const APInt& V) {
    ConstantInt *C = cast<ConstantInt>(ConstantInt::get(Ty, V));
    return fromConstantInt(C);
  }
  ConstantInt *toConstantInt() {
    return const_cast<ConstantInt*>((const ConstantInt*)Implementation);
  }
};

// TODO: it should be a class in next commit.
struct IntRange {

    IntItem Low;
    IntItem High;
    bool IsEmpty : 1;
    bool IsSingleNumber : 1;
// TODO: 
// public:
    
    typedef std::pair<IntRange, IntRange> SubRes;
    
    IntRange() : IsEmpty(true) {}
    IntRange(const IntRange &RHS) :
      Low(RHS.Low), High(RHS.High), IsEmpty(false), IsSingleNumber(false) {}
    IntRange(const IntItem &C) :
      Low(C), High(C), IsEmpty(false), IsSingleNumber(true) {}
    IntRange(const IntItem &L, const IntItem &H) : Low(L), High(H),
        IsEmpty(false), IsSingleNumber(false) {}
    
    bool isEmpty() const { return IsEmpty; }
    bool isSingleNumber() const { return IsSingleNumber; }
    
    const IntItem& getLow() {
      assert(!IsEmpty && "Range is empty.");
      return Low;
    }
    const IntItem& getHigh() {
      assert(!IsEmpty && "Range is empty.");
      return High;
    }
   
    bool operator<(const IntRange &RHS) const {
      assert(!IsEmpty && "Left range is empty.");
      assert(!RHS.IsEmpty && "Right range is empty.");
      if (Low->getBitWidth() == RHS.Low->getBitWidth()) {
        if (Low->eq(RHS.Low)) {
          if (High->ult(RHS.High))
            return true;
          return false;
        }
        if (Low->ult(RHS.Low))
          return true;
        return false;
      } else
        return Low->getBitWidth() < RHS.Low->getBitWidth();      
    }

    bool operator==(const IntRange &RHS) const {
      assert(!IsEmpty && "Left range is empty.");
      assert(!RHS.IsEmpty && "Right range is empty.");
      if (Low->getBitWidth() != RHS.Low->getBitWidth())
        return false;
      return Low == RHS.Low && High == RHS.High;      
    }
 
    bool operator!=(const IntRange &RHS) const {
      return !operator ==(RHS);      
    }
 
    static bool LessBySize(const IntRange &LHS, const IntRange &RHS) {
      assert(LHS.Low->getBitWidth() == RHS.Low->getBitWidth() && 
          "This type of comparison requires equal bit width for LHS and RHS");
      APInt LSize = *LHS.High - *LHS.Low;
      APInt RSize = *RHS.High - *RHS.Low;
      return LSize.ult(RSize);      
    }
 
    bool isInRange(const APInt &IntVal) const {
      assert(!IsEmpty && "Range is empty.");
      if (IntVal.getBitWidth() != Low->getBitWidth())
        return false;
      return IntVal.uge(Low) && IntVal.ule(High);      
    }    
  
    SubRes sub(const IntRange &RHS) const {
      SubRes Res;
      
      // RHS is either more global and includes this range or
      // if it doesn't intersected with this range.
      if (!isInRange(RHS.Low) && !isInRange(RHS.High)) {
        
        // If RHS more global (it is enough to check
        // only one border in this case.
        if (RHS.isInRange(Low))
          return std::make_pair(IntRange(Low, High), IntRange()); 
        
        return Res;
      }
      
      if (Low->ult(RHS.Low)) {
        Res.first.Low = Low;
        APInt NewHigh = RHS.Low;
        --NewHigh;
        Res.first.High = NewHigh;
      }
      if (High->ugt(RHS.High)) {
        APInt NewLow = RHS.High;
        ++NewLow;
        Res.second.Low = NewLow;
        Res.second.High = High;
      }
      return Res;      
    }
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
  
  bool IsWide;
  
  // implicit
  ConstantRangesSet(Constant *V) : Array(V) {
    ArrayType *ArrTy = cast<ArrayType>(Array->getType());
    VectorType *VecTy = cast<VectorType>(ArrTy->getElementType());
    IntegerType *IntTy = cast<IntegerType>(VecTy->getElementType());
    IsWide = IntTy->getBitWidth() > 64;    
  }
  
  operator Constant*() { return Array; }
  operator const Constant*() const { return Array; }
  Constant *operator->() { return Array; }
  const Constant *operator->() const { return Array; }
  
  typedef IntRange Range;
 
  /// Checks is the given constant satisfies this case. Returns
  /// true if it equals to one of contained values or belongs to the one of
  /// contained ranges.
  bool isSatisfies(const IntItem &CheckingVal) const {
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
      return Range(IntItem::fromConstantInt(
                    cast<ConstantInt>(CV->getAggregateElement(0U))));
    case 2:
      return Range(IntItem::fromConstantInt(
                     cast<ConstantInt>(CV->getAggregateElement(0U))),
                   IntItem::fromConstantInt(
                     cast<ConstantInt>(CV->getAggregateElement(1U))));
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
      return Range(IntItem::fromConstantInt(
                     cast<ConstantInt>(CV->getAggregateElement(0U))),
                   IntItem::fromConstantInt(cast<ConstantInt>(
                     cast<ConstantInt>(CV->getAggregateElement(0U)))));
    case 2:
      return Range(IntItem::fromConstantInt(
                     cast<ConstantInt>(CV->getAggregateElement(0U))),
                   IntItem::fromConstantInt(
                   cast<ConstantInt>(CV->getAggregateElement(1))));
    default:
      assert(0 && "Only pairs and single numbers are allowed here.");
      return Range();
    }    
  }
  
  /// Return number of items (ranges) stored in set.
  unsigned getNumItems() const {
    return cast<ArrayType>(Array->getType())->getNumElements();
  }
  
  bool isWideNumberFormat() const { return IsWide; }
  
  bool isSingleNumber(unsigned idx) const {
    Constant *CV = Array->getAggregateElement(idx);
    return cast<VectorType>(CV->getType())->getNumElements() == 1;
  }
  
  /// Returns set the size, that equals number of all values + sizes of all
  /// ranges.
  /// Ranges set is considered as flat numbers collection.
  /// E.g.: for range [<0>, <1>, <4,8>] the size will 7;
  ///       for range [<0>, <1>, <5>] the size will 3
  unsigned getSize() const {
    APInt sz(getItem(0).Low->getBitWidth(), 0);
    for (unsigned i = 0, e = getNumItems(); i != e; ++i) {
      const APInt &Low = getItem(i).Low;
      const APInt &High = getItem(i).High;
      const APInt &S = High - Low;
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
      const APInt &Low = getItem(i).Low;
      const APInt &High = getItem(i).High;      
      const APInt& S = High - Low;
      APInt oldSz = sz;
      sz += S;
      if (oldSz.uge(i) && sz.ult(i)) {
        APInt Res = Low;
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
