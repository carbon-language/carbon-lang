//===- llvm/Support/ValueHandle.h - Value Smart Pointer classes -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ValueHandle class and its sub-classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VALUEHANDLE_H
#define LLVM_SUPPORT_VALUEHANDLE_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Value.h"

namespace llvm {
class ValueHandleBase;

// ValueHandleBase** is only 4-byte aligned.
template<>
class PointerLikeTypeTraits<ValueHandleBase**> {
public:
  static inline void *getAsVoidPointer(ValueHandleBase** P) { return P; }
  static inline ValueHandleBase **getFromVoidPointer(void *P) {
    return static_cast<ValueHandleBase**>(P);
  }
  enum { NumLowBitsAvailable = 2 };
};

/// ValueHandleBase - This is the common base class of value handles.
/// ValueHandle's are smart pointers to Value's that have special behavior when
/// the value is deleted or ReplaceAllUsesWith'd.  See the specific handles
/// below for details.
///
class ValueHandleBase {
  friend class Value;
protected:
  /// HandleBaseKind - This indicates what base class the handle actually is.
  /// This is to avoid having a vtable for the light-weight handle pointers. The
  /// fully generally Callback version does have a vtable.
  enum HandleBaseKind {
    Assert,
    Weak,
    Callback
  };
private:
  
  PointerIntPair<ValueHandleBase**, 2, HandleBaseKind> PrevPair;
  ValueHandleBase *Next;
  Value *VP;
public:
  ValueHandleBase(HandleBaseKind Kind) : PrevPair(0, Kind), Next(0), VP(0) {}
  ValueHandleBase(HandleBaseKind Kind, Value *V)
    : PrevPair(0, Kind), Next(0), VP(V) {
    if (V)
      AddToUseList();
  }
  ValueHandleBase(HandleBaseKind Kind, const ValueHandleBase &RHS)
    : PrevPair(0, Kind), Next(0), VP(RHS.VP) {
    if (VP)
      AddToExistingUseList(RHS.getPrevPtr());
  }
  ~ValueHandleBase() {
    if (VP)
      RemoveFromUseList();   
  }
  
  Value *operator=(Value *RHS) {
    if (VP == RHS) return RHS;
    if (VP) RemoveFromUseList();
    VP = RHS;
    if (VP) AddToUseList();
    return RHS;
  }

  Value *operator=(const ValueHandleBase &RHS) {
    if (VP == RHS.VP) return RHS.VP;
    if (VP) RemoveFromUseList();
    VP = RHS.VP;
    if (VP) AddToExistingUseList(RHS.getPrevPtr());
    return VP;
  }
  
  Value *operator->() const { return getValPtr(); }
  Value &operator*() const { return *getValPtr(); }
  
  bool operator==(const Value *RHS) const { return VP == RHS; }
  bool operator==(const ValueHandleBase &RHS) const { return VP == RHS.VP; }
  bool operator!=(const Value *RHS) const { return VP != RHS; }
  bool operator!=(const ValueHandleBase &RHS) const { return VP != RHS.VP; }
  bool operator<(const Value *RHS) const { return VP < RHS; }
  bool operator<(const ValueHandleBase &RHS) const { return VP < RHS.VP; }
  bool operator>(const Value *RHS) const { return VP > RHS; }
  bool operator>(const ValueHandleBase &RHS) const { return VP > RHS.VP; }
  bool operator<=(const Value *RHS) const { return VP <= RHS; }
  bool operator<=(const ValueHandleBase &RHS) const { return VP <= RHS.VP; }
  bool operator>=(const Value *RHS) const { return VP >= RHS; }
  bool operator>=(const ValueHandleBase &RHS) const { return VP >= RHS.VP; }
  
protected:
  Value *getValPtr() const { return VP; }
private:
  // Callbacks made from Value.
  static void ValueIsDeleted(Value *V);
  static void ValueIsRAUWd(Value *Old, Value *New);
  
  // Internal implementation details.
  ValueHandleBase **getPrevPtr() const { return PrevPair.getPointer(); }
  HandleBaseKind getKind() const { return PrevPair.getInt(); }
  void setPrevPtr(ValueHandleBase **Ptr) { PrevPair.setPointer(Ptr); }
  
  /// AddToUseList - Add this ValueHandle to the use list for VP, where List is
  /// known to point into the existing use list.
  void AddToExistingUseList(ValueHandleBase **List);
  
  /// AddToUseList - Add this ValueHandle to the use list for VP.
  void AddToUseList();
  /// RemoveFromUseList - Remove this ValueHandle from its current use list.
  void RemoveFromUseList();
};
  
/// WeakVH - This is a value handle that tries hard to point to a Value, even
/// across RAUW operations, but will null itself out if the value is destroyed.
/// this is useful for advisory sorts of information, but should not be used as
/// the key of a map (since the map would have to rearrange itself when the
/// pointer changes).
class WeakVH : public ValueHandleBase {
public:
  WeakVH() : ValueHandleBase(Weak) {}
  WeakVH(Value *P) : ValueHandleBase(Weak, P) {}
  WeakVH(const WeakVH &RHS)
    : ValueHandleBase(Weak, RHS) {}

  operator Value*() const {
    return getValPtr();
  }
};  
  
/// AssertingVH - This is a Value Handle that points to a value and asserts out
/// if the value is destroyed while the handle is still live.  This is very
/// useful for catching dangling pointer bugs and other things which can be
/// non-obvious.  One particularly useful place to use this is as the Key of a
/// map.  Dangling pointer bugs often lead to really subtle bugs that only occur
/// if another object happens to get allocated to the same address as the old
/// one.  Using an AssertingVH ensures that an assert is triggered as soon as
/// the bad delete occurs.
///
/// Note that an AssertingVH handle does *not* follow values across RAUW
/// operations.  This means that RAUW's need to explicitly update the
/// AssertingVH's as it moves.  This is required because in non-assert mode this
/// class turns into a trivial wrapper around a pointer.
template <typename ValueTy>
class AssertingVH 
#ifndef NDEBUG
  : public ValueHandleBase
#endif
  {

#ifndef NDEBUG
  ValueTy *getValPtr() const {
    return static_cast<ValueTy*>(ValueHandleBase::getValPtr());
  }
  void setValPtr(ValueTy *P) {
    ValueHandleBase::operator=(P);
  }
#else
  ValueTy *ThePtr;
  ValueTy *getValPtr() const { return ThePtr; }
  void setValPtr(ValueTy *P) { ThePtr = P; }
#endif

public:
#ifndef NDEBUG
  AssertingVH() : ValueHandleBase(Assert) {}
  AssertingVH(ValueTy *P) : ValueHandleBase(Assert, P) {}
  AssertingVH(const AssertingVH &RHS) : ValueHandleBase(Assert, RHS) {}
#else
  AssertingVH() : ThePtr(0) {}
  AssertingVH(ValueTy *P) : ThePtr(P) {}
#endif

  operator ValueTy*() const {
    return getValPtr();
  }

  ValueTy *operator=(ValueTy *RHS) {
    setValPtr(RHS);
    return getValPtr();
  }
  ValueTy *operator=(AssertingVH<ValueTy> &RHS) {
    setValPtr(RHS.getValPtr());
    return getValPtr();
  }

  ValueTy *operator->() const { return getValPtr(); }
  ValueTy &operator*() const { return *getValPtr(); }

  // Duplicate these from the base class so that they work when assertions are
  // off.
  bool operator==(const Value *RHS) const { return getValPtr() == RHS; }
  bool operator!=(const Value *RHS) const { return getValPtr() != RHS; }
  bool operator<(const Value *RHS) const { return getValPtr() < RHS; }
  bool operator>(const Value *RHS) const { return getValPtr() > RHS; }
  bool operator<=(const Value *RHS) const { return getValPtr() <= RHS; }
  bool operator>=(const Value *RHS) const { return getValPtr() >= RHS; }
  bool operator==(const AssertingVH &RHS) const {
    return getValPtr() == RHS.getValPtr();
  }
  bool operator!=(const AssertingVH &RHS) const {
    return getValPtr() != RHS.getValPtr();
  }
  bool operator<(const AssertingVH &RHS) const {
    return getValPtr() < RHS.getValPtr();
  }
  bool operator>(const AssertingVH &RHS) const {
    return getValPtr() > RHS.getValPtr();
  }
  bool operator<=(const AssertingVH &RHS) const {
    return getValPtr() <= RHS.getValPtr();
  }
  bool operator>=(const AssertingVH &RHS) const {
    return getValPtr() >= RHS.getValPtr();
  }
};

} // End llvm namespace

#endif
