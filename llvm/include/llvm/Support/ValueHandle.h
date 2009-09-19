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

#include "llvm/ADT/DenseMapInfo.h"
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
  /// HandleBaseKind - This indicates what sub class the handle actually is.
  /// This is to avoid having a vtable for the light-weight handle pointers. The
  /// fully general Callback version does have a vtable.
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
  explicit ValueHandleBase(HandleBaseKind Kind)
    : PrevPair(0, Kind), Next(0), VP(0) {}
  ValueHandleBase(HandleBaseKind Kind, Value *V)
    : PrevPair(0, Kind), Next(0), VP(V) {
    if (isValid(VP))
      AddToUseList();
  }
  ValueHandleBase(HandleBaseKind Kind, const ValueHandleBase &RHS)
    : PrevPair(0, Kind), Next(0), VP(RHS.VP) {
    if (isValid(VP))
      AddToExistingUseList(RHS.getPrevPtr());
  }
  ~ValueHandleBase() {
    if (isValid(VP))
      RemoveFromUseList();
  }

  Value *operator=(Value *RHS) {
    if (VP == RHS) return RHS;
    if (isValid(VP)) RemoveFromUseList();
    VP = RHS;
    if (isValid(VP)) AddToUseList();
    return RHS;
  }

  Value *operator=(const ValueHandleBase &RHS) {
    if (VP == RHS.VP) return RHS.VP;
    if (isValid(VP)) RemoveFromUseList();
    VP = RHS.VP;
    if (isValid(VP)) AddToExistingUseList(RHS.getPrevPtr());
    return VP;
  }

  Value *operator->() const { return getValPtr(); }
  Value &operator*() const { return *getValPtr(); }

protected:
  Value *getValPtr() const { return VP; }
private:
  static bool isValid(Value *V) {
    return V &&
           V != DenseMapInfo<Value *>::getEmptyKey() &&
           V != DenseMapInfo<Value *>::getTombstoneKey();
  }

  // Callbacks made from Value.
  static void ValueIsDeleted(Value *V);
  static void ValueIsRAUWd(Value *Old, Value *New);

  // Internal implementation details.
  ValueHandleBase **getPrevPtr() const { return PrevPair.getPointer(); }
  HandleBaseKind getKind() const { return PrevPair.getInt(); }
  void setPrevPtr(ValueHandleBase **Ptr) { PrevPair.setPointer(Ptr); }

  /// AddToExistingUseList - Add this ValueHandle to the use list for VP,
  /// where List is known to point into the existing use list.
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

// Specialize simplify_type to allow WeakVH to participate in
// dyn_cast, isa, etc.
template<typename From> struct simplify_type;
template<> struct simplify_type<const WeakVH> {
  typedef Value* SimpleType;
  static SimpleType getSimplifiedValue(const WeakVH &WVH) {
    return static_cast<Value *>(WVH);
  }
};
template<> struct simplify_type<WeakVH> : public simplify_type<const WeakVH> {};

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
    ValueHandleBase::operator=(GetAsValue(P));
  }
#else
  ValueTy *ThePtr;
  ValueTy *getValPtr() const { return ThePtr; }
  void setValPtr(ValueTy *P) { ThePtr = P; }
#endif

  // Convert a ValueTy*, which may be const, to the type the base
  // class expects.
  static Value *GetAsValue(Value *V) { return V; }
  static Value *GetAsValue(const Value *V) { return const_cast<Value*>(V); }

public:
#ifndef NDEBUG
  AssertingVH() : ValueHandleBase(Assert) {}
  AssertingVH(ValueTy *P) : ValueHandleBase(Assert, GetAsValue(P)) {}
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
  ValueTy *operator=(const AssertingVH<ValueTy> &RHS) {
    setValPtr(RHS.getValPtr());
    return getValPtr();
  }

  ValueTy *operator->() const { return getValPtr(); }
  ValueTy &operator*() const { return *getValPtr(); }
};

// Specialize simplify_type to allow AssertingVH to participate in
// dyn_cast, isa, etc.
template<typename From> struct simplify_type;
template<> struct simplify_type<const AssertingVH<Value> > {
  typedef Value* SimpleType;
  static SimpleType getSimplifiedValue(const AssertingVH<Value> &AVH) {
    return static_cast<Value *>(AVH);
  }
};
template<> struct simplify_type<AssertingVH<Value> >
  : public simplify_type<const AssertingVH<Value> > {};

/// CallbackVH - This is a value handle that allows subclasses to define
/// callbacks that run when the underlying Value has RAUW called on it or is
/// destroyed.  This class can be used as the key of a map, as long as the user
/// takes it out of the map before calling setValPtr() (since the map has to
/// rearrange itself when the pointer changes).  Unlike ValueHandleBase, this
/// class has a vtable and a virtual destructor.
class CallbackVH : public ValueHandleBase {
protected:
  CallbackVH(const CallbackVH &RHS)
    : ValueHandleBase(Callback, RHS) {}

  virtual ~CallbackVH();

  void setValPtr(Value *P) {
    ValueHandleBase::operator=(P);
  }

public:
  CallbackVH() : ValueHandleBase(Callback) {}
  CallbackVH(Value *P) : ValueHandleBase(Callback, P) {}

  operator Value*() const {
    return getValPtr();
  }

  /// Called when this->getValPtr() is destroyed, inside ~Value(), so you may
  /// call any non-virtual Value method on getValPtr(), but no subclass methods.
  /// If WeakVH were implemented as a CallbackVH, it would use this method to
  /// call setValPtr(NULL).  AssertingVH would use this method to cause an
  /// assertion failure.
  ///
  /// All implementations must remove the reference from this object to the
  /// Value that's being destroyed.
  virtual void deleted() {
    setValPtr(NULL);
  }

  /// Called when this->getValPtr()->replaceAllUsesWith(new_value) is called,
  /// _before_ any of the uses have actually been replaced.  If WeakVH were
  /// implemented as a CallbackVH, it would use this method to call
  /// setValPtr(new_value).  AssertingVH would do nothing in this method.
  virtual void allUsesReplacedWith(Value *new_value) {}
};

// Specialize simplify_type to allow CallbackVH to participate in
// dyn_cast, isa, etc.
template<typename From> struct simplify_type;
template<> struct simplify_type<const CallbackVH> {
  typedef Value* SimpleType;
  static SimpleType getSimplifiedValue(const CallbackVH &CVH) {
    return static_cast<Value *>(CVH);
  }
};
template<> struct simplify_type<CallbackVH>
  : public simplify_type<const CallbackVH> {};

} // End llvm namespace

#endif
