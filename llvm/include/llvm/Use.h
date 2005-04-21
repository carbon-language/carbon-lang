//===-- llvm/Use.h - Definition of the Use class ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "llvm/ADT/iterator"

namespace llvm {

class Value;
class User;


//===----------------------------------------------------------------------===//
//                                  Use Class
//===----------------------------------------------------------------------===//

// Use is here to make keeping the "use" list of a Value up-to-date really easy.
//
class Use {
public:
  inline void init(Value *V, User *U);

  Use(Value *V, User *U) { init(V, U); }
  Use(const Use &U) { init(U.Val, U.U); }
  inline ~Use();

  /// Default ctor - This leaves the Use completely unitialized.  The only thing
  /// that is valid to do with this use is to call the "init" method.
  inline Use() : Val(0) {}


  operator Value*() const { return Val; }
  Value *get() const { return Val; }
  User *getUser() const { return U; }

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
private:
  Use *Next, **Prev;
  Value *Val;
  User *U;

  void addToList(Use **List) {
    Next = *List;
    if (Next) Next->Prev = &Next;
    Prev = List;
    *List = this;
  }
  void removeFromList() {
    *Prev = Next;
    if (Next) Next->Prev = Prev;
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
  value_use_iterator(Use *u) : U(u) {}
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

  // Iterator traversal: forward iteration only
  _Self &operator++() {          // Preincrement
    assert(U && "Cannot increment end iterator!");
    U = U->getNext();
    return *this;
  }
  _Self operator++(int) {        // Postincrement
    _Self tmp = *this; ++*this; return tmp;
  }

  // Retrieve a reference to the current SCC
   UserTy *operator*() const {
    assert(U && "Cannot increment end iterator!");
    return U->getUser();
  }

  UserTy *operator->() const { return operator*(); }

  Use &getUse() const { return *U; }
};

} // End llvm namespace

#endif
