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

// We adhere to the following convention: The type of a tagged pointer
// to T is T volatile*. This means that functions that superpose a tag
// on a pointer will be supplied a T* (or T const*) and will return a
// tagged one: T volatile*. Untagging functions do it the other way
// 'round. While this scheme does not prevent dereferencing of tagged
// pointers, proper type annotations do catch most inappropriate uses.

/// Tag - generic tag type for (at least 32 bit) pointers
enum Tag { noTag, tagOne, tagTwo, tagThree };

/// addTag - insert tag bits into an (untagged) pointer
template <typename T, typename TAG>
inline volatile T *addTag(const T *P, TAG Tag) {
  return reinterpret_cast<T*>(ptrdiff_t(P) | Tag);
}

/// stripTag - remove tag bits from a pointer,
/// making it dereferencable
template <ptrdiff_t MASK, typename T>
inline T *stripTag(const volatile T *P) {
  return reinterpret_cast<T*>(ptrdiff_t(P) & ~MASK);
}

/// extractTag - extract tag bits from a pointer
template <typename TAG, TAG MASK, typename T>
inline TAG extractTag(const volatile T *P) {
  return TAG(ptrdiff_t(P) & MASK);
}

/// transferTag - transfer tag bits from a pointer,
/// to an untagged pointer
template <ptrdiff_t MASK, typename T>
inline volatile T *transferTag(const volatile T *From, const T *To) {
  return reinterpret_cast<T*>((ptrdiff_t(From) & MASK) | ptrdiff_t(To));
}


//===----------------------------------------------------------------------===//
//                                  Use Class
//===----------------------------------------------------------------------===//

/// Use is here to make keeping the "use" list of a Value up-to-date really
/// easy.
class Use {
public:
  /// swap - provide a fast substitute to std::swap<Use>
  /// that also works with less standard-compliant compilers
  void swap(Use &RHS);

private:
  /// Copy ctor - do not implement
  Use(const Use &U);

  /// Destructor - Only for zap()
  inline ~Use() {
    if (Val) removeFromList();
  }

  /// Default ctor - This leaves the Use completely uninitialized.  The only
  /// thing that is valid to do with this use is to call the "init" method.
  inline Use() {}
  enum PrevPtrTag { zeroDigitTag = noTag
                  , oneDigitTag = tagOne
                  , stopTag = tagTwo
                  , fullStopTag = tagThree };

public:
  /// Normally Use will just implicitly convert to a Value* that it holds.
  operator Value*() const { return Val; }
  
  /// If implicit conversion to Value* doesn't work, the get() method returns
  /// the Value*.
  Value *get() const { return Val; }
  
  /// getUser - This returns the User that contains this Use.  For an
  /// instruction operand, for example, this will return the instruction.
  User *getUser() const;

  inline void set(Value *Val);

  Value *operator=(Value *RHS) {
    set(RHS);
    return RHS;
  }
  const Use &operator=(const Use &RHS) {
    set(RHS.Val);
    return *this;
  }

        Value *operator->()       { return Val; }
  const Value *operator->() const { return Val; }

  Use *getNext() const { return Next; }

  
  /// zap - This is used to destroy Use operands when the number of operands of
  /// a User changes.
  static void zap(Use *Start, const Use *Stop, bool del = false);

private:
  const Use* getImpliedUser() const;
  static Use *initTags(Use *Start, Use *Stop, ptrdiff_t Done = 0);
  
  Value *Val;
  Use *Next, *volatile*Prev;

  void setPrev(Use **NewPrev) {
    Prev = transferTag<fullStopTag>(Prev, NewPrev);
  }
  void addToList(Use **List) {
    Next = *List;
    if (Next) Next->setPrev(&Next);
    setPrev(List);
    *List = this;
  }
  void removeFromList() {
    Use **StrippedPrev = stripTag<fullStopTag>(Prev);
    *StrippedPrev = Next;
    if (Next) Next->setPrev(StrippedPrev);
  }

  friend class Value;
  friend class User;
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
  explicit value_use_iterator(Use *u) : U(u) {}
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
  bool atEnd() const { return U == 0; }

  // Iterator traversal: forward iteration only
  _Self &operator++() {          // Preincrement
    assert(U && "Cannot increment end iterator!");
    U = U->getNext();
    return *this;
  }
  _Self operator++(int) {        // Postincrement
    _Self tmp = *this; ++*this; return tmp;
  }

  // Retrieve a pointer to the current User.
  UserTy *operator*() const {
    assert(U && "Cannot dereference end iterator!");
    return U->getUser();
  }

  UserTy *operator->() const { return operator*(); }

  Use &getUse() const { return *U; }
  
  /// getOperandNo - Return the operand # of this use in its User.  Defined in
  /// User.h
  ///
  unsigned getOperandNo() const;
};


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
