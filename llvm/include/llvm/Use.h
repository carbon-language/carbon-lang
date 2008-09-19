//===-- llvm/Use.h - Definition of the Use class ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines the Use class.  The Use class represents the operand of an
// instruction or some other User instance which refers to a Value.  The Use
// class keeps the "use list" of the referenced value up to date.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_USE_H
#define LLVM_USE_H

#include "llvm/Support/Casting.h"
#include "llvm/ADT/iterator.h"

namespace llvm {

class Value;
class User;


//===----------------------------------------------------------------------===//
//                          Generic Tagging Functions
//===----------------------------------------------------------------------===//

/// Tag - generic tag type for (at least 32 bit) pointers
enum Tag { noTag, tagOne, tagTwo, tagThree };

/// addTag - insert tag bits into an (untagged) pointer
template <typename T, typename TAG>
inline T *addTag(const T *P, TAG Tag) {
    return reinterpret_cast<T*>(ptrdiff_t(P) | Tag);
}

/// stripTag - remove tag bits from a pointer,
/// making it dereferencable
template <ptrdiff_t MASK, typename T>
inline T *stripTag(const T *P) {
  return reinterpret_cast<T*>(ptrdiff_t(P) & ~MASK);
}

/// extractTag - extract tag bits from a pointer
template <typename TAG, TAG MASK, typename T>
inline TAG extractTag(const T *P) {
  return TAG(ptrdiff_t(P) & MASK);
}

/// transferTag - transfer tag bits from a pointer,
/// to an untagged pointer
template <ptrdiff_t MASK, typename T>
inline T *transferTag(const T *From, const T *To) {
  return reinterpret_cast<T*>((ptrdiff_t(From) & MASK) | ptrdiff_t(To));
}


//===----------------------------------------------------------------------===//
//                                  Use Class
//===----------------------------------------------------------------------===//

// Use is here to make keeping the "use" list of a Value up-to-date really easy.
//
class Use {
  class UseWaymark;
  friend class UseWaymark;
  Value *getValue() const;
  /// nilUse - returns a 'token' that marks the end of the def/use chain
  static Use *nilUse(const Value *V) {
    return addTag((Use*)V, fullStopTagN);
  }
  static bool isNil(Use *U) { return extractTag<NextPtrTag, tagMaskN>(U) == fullStopTagN; }
  void showWaymarks() const;
  static bool isStop(Use *U) {
    return isStopTag(extractTag<NextPtrTag, tagMaskN>(U));
  }
public:
  /// init - specify Value and User
  /// @deprecated in 2.4, will be removed soon
  inline void init(Value *V, User *U);
  /// swap - provide a fast substitute to std::swap<Use>
  /// that also works with less standard-compliant compilers
  void swap(Use &RHS);

private:
  /// Copy ctor - do not implement
  Use(const Use &U);

  /// Destructor - Only for zap()
  inline ~Use() {
    if (Val1) removeFromList();
  }

  /// Default ctor - This leaves the Use completely uninitialized.  The only thing
  /// that is valid to do with this use is to call the "init" method.

  inline Use() {}
  enum PrevPtrTag { zeroDigitTag = noTag
                  , oneDigitTag = tagOne
                  , stopTag = tagTwo
                  , fullStopTag = tagThree
                  , tagMask = tagThree };

  enum NextPtrTag { zeroDigitTagN = tagTwo
                  , oneDigitTagN = tagOne
                  , stopTagN = noTag
                  , fullStopTagN = tagThree
                  , tagMaskN = tagThree };

  static bool isStopTag(NextPtrTag T) {
    bool P[] = { true, false, false, true };
    return P[T];
  }
public:
  operator Value*() const { return get(); }
  inline Value *get() const;
  User *getUser() const;
  const Use* getImpliedUser() const;
  static Use *initTags(Use *Start, Use *Stop, ptrdiff_t Done = 0);
  static void zap(Use *Start, const Use *Stop, bool del = false);

  inline void set(Value *Val);

  Value *operator=(Value *RHS) {
    set(RHS);
    return RHS;
  }
  const Use &operator=(const Use &RHS) {
    set(RHS.Val1);
    return *this;
  }

        Value *operator->()       { return get(); }
  const Value *operator->() const { return get(); }

  Use *getNext() const { return extractTag<NextPtrTag, tagMaskN>(Next) == fullStopTagN
			   ? 0
			   : stripTag<tagMaskN>(Next); }
private:
  Value *Val1;
  Use *Next, **Prev;

  void setPrev(Use **NewPrev) {
    Prev = transferTag<tagMask>(Prev, NewPrev);
  }
  void addToList(Use **List) {
    Next = *List;
    Use *StrippedNext(getNext());
    if (StrippedNext) StrippedNext->setPrev(&Next);
    setPrev(List);
    *List = this;
  }
  void removeFromList() {
    // __builtin_prefetch(Next);
    Use **StrippedPrev = stripTag<tagMask>(Prev);
    Use *StrippedNext(getNext());
    if (isStop(Next))
      assert((isStop(*StrippedPrev) || (StrippedNext ? isStop(StrippedNext->Next) : true)) && "joining digits?");
    *StrippedPrev = Next;
    if (StrippedNext) StrippedNext->setPrev(StrippedPrev);
  }

  friend class Value;
};

// simplify_type - Allow clients to treat uses just like values when using
// casting operators.
template<> struct simplify_type<Use> {
  typedef Value* SimpleType;
  static SimpleType getSimplifiedValue(const Use &Val) {
    return static_cast<SimpleType>(Val.get());
  }
};
template<> struct simplify_type<const Use> {
  typedef Value* SimpleType;
  static SimpleType getSimplifiedValue(const Use &Val) {
    return static_cast<SimpleType>(Val.get());
  }
};



template<typename UserTy>  // UserTy == 'User' or 'const User'
class value_use_iterator : public forward_iterator<UserTy*, ptrdiff_t> {
  typedef forward_iterator<UserTy*, ptrdiff_t> super;
  typedef value_use_iterator<UserTy> _Self;

  Use *U;
  explicit value_use_iterator(Use *u) : U(extractTag<Use::NextPtrTag, Use::tagMaskN>(u)
					  == Use::fullStopTagN
					  ? 0
					  : stripTag<Use::tagMaskN>(u)) {}
  friend class Value;
public:
  typedef typename super::reference reference;
  typedef typename super::pointer pointer;

  value_use_iterator(const _Self &I) : U(I.U) {}
  value_use_iterator() {}

  bool operator==(const _Self &x) const {
    return U == x.U;
  }
  bool operator!=(const _Self &x) const {
    return !operator==(x);
  }

  /// atEnd - return true if this iterator is equal to use_end() on the value.
  bool atEnd() const { return !U; }

  // Iterator traversal: forward iteration only
  _Self &operator++() {          // Preincrement
    assert(!atEnd() && "Cannot increment end iterator!");
    U = U->getNext();
    return *this;
  }
  _Self operator++(int) {        // Postincrement
    _Self tmp = *this; ++*this; return tmp;
  }

  // Retrieve a reference to the current User
  UserTy *operator*() const {
    assert(!atEnd() && "Cannot dereference end iterator!");
    return U->getUser();
  }

  UserTy *operator->() const { return operator*(); }

  Use &getUse() const { return *U; }
  
  /// getOperandNo - Return the operand # of this use in its User.  Defined in
  /// User.h
  ///
  unsigned getOperandNo() const;
};

Value *Use::get() const {
  return fullStopTagN == extractTag<NextPtrTag, tagMaskN>(Next)
    ? reinterpret_cast<Value*>(stripTag<tagMaskN>(Next))
    : (Val1 == getValue() ? Val1 : 0); // should crash if not equal!
}

template<> struct simplify_type<value_use_iterator<User> > {
  typedef User* SimpleType;
  
  static SimpleType getSimplifiedValue(const value_use_iterator<User> &Val) {
    return *Val;
  }
};

template<> struct simplify_type<const value_use_iterator<User> >
 : public simplify_type<value_use_iterator<User> > {};

template<> struct simplify_type<value_use_iterator<const User> > {
  typedef const User* SimpleType;
  
  static SimpleType getSimplifiedValue(const 
                                       value_use_iterator<const User> &Val) {
    return *Val;
  }
};

template<> struct simplify_type<const value_use_iterator<const User> >
  : public simplify_type<value_use_iterator<const User> > {};

} // End llvm namespace

#endif
