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

#include "llvm/ADT/ilist"

namespace llvm {

template<typename NodeTy> struct ilist_traits;
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

private:
  // NOTE!! The Next/Prev fields MUST stay at the start of this structure.  The
  // end-token for the ilist is allocated as JUST the next/prev pair to reduce
  // memory usage instead of allocating an entire Use.
  struct NextPrevPtrs {
    Use *Next, *Prev;
  } UseLinks;

  Value *Val;
  User *U;
  friend struct ilist_traits<Use>;
};

template<>
struct ilist_traits<Use> {
  static Use *getPrev(Use *N) { return N->UseLinks.Prev; }
  static Use *getNext(Use *N) { return N->UseLinks.Next; }
  static const Use *getPrev(const Use *N) { return N->UseLinks.Prev; }
  static const Use *getNext(const Use *N) { return N->UseLinks.Next; }
  static void setPrev(Use *N, Use *Prev) { N->UseLinks.Prev = Prev; }
  static void setNext(Use *N, Use *Next) { N->UseLinks.Next = Next; }

  /// createSentinel - this is used to create the end marker for the use list.
  /// Note that we only allocate a UseLinks structure, which is just enough to
  /// hold the next/prev pointers.  This saves us 8 bytes of memory for every
  /// Value allocated.
  static Use *createSentinel() { return (Use*)new Use::NextPrevPtrs(); }
  static void destroySentinel(Use *S) { delete (Use::NextPrevPtrs*)S; }

  void addNodeToList(Use *NTy) {}
  void removeNodeFromList(Use *NTy) {}
  void transferNodesFromList(iplist<Use, ilist_traits> &L2,
                             ilist_iterator<Use> first,
                             ilist_iterator<Use> last) {}
};


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

struct UseListIteratorWrapper : public iplist<Use>::iterator {
  typedef iplist<Use>::iterator Super;
  UseListIteratorWrapper() {}
  UseListIteratorWrapper(const Super &RHS) : Super(RHS) {}

  UseListIteratorWrapper &operator=(const Super &RHS) {
    Super::operator=(RHS);
    return *this;
  }

  inline User *operator*() const;
  User *operator->() const { return operator*(); }

  UseListIteratorWrapper operator--() { return Super::operator--(); }
  UseListIteratorWrapper operator++() { return Super::operator++(); }

  UseListIteratorWrapper operator--(int) {    // postdecrement operators...
    UseListIteratorWrapper tmp = *this;
    --*this;
    return tmp;
  }
  UseListIteratorWrapper operator++(int) {    // postincrement operators...
    UseListIteratorWrapper tmp = *this;
    ++*this;
    return tmp;
  }
};

struct UseListConstIteratorWrapper : public iplist<Use>::const_iterator {
  typedef iplist<Use>::const_iterator Super;
  UseListConstIteratorWrapper() {}
  UseListConstIteratorWrapper(const Super &RHS) : Super(RHS) {}

  // Allow conversion from non-const to const iterators
  UseListConstIteratorWrapper(const UseListIteratorWrapper &RHS) : Super(RHS) {}
  UseListConstIteratorWrapper(const iplist<Use>::iterator &RHS) : Super(RHS) {}

  UseListConstIteratorWrapper &operator=(const Super &RHS) {
    Super::operator=(RHS);
    return *this;
  }

  inline const User *operator*() const;
  const User *operator->() const { return operator*(); }

  UseListConstIteratorWrapper operator--() { return Super::operator--(); }
  UseListConstIteratorWrapper operator++() { return Super::operator++(); }

  UseListConstIteratorWrapper operator--(int) {    // postdecrement operators...
    UseListConstIteratorWrapper tmp = *this;
    --*this;
    return tmp;
  }
  UseListConstIteratorWrapper operator++(int) {    // postincrement operators...
    UseListConstIteratorWrapper tmp = *this;
    ++*this;
    return tmp;
  }
};

} // End llvm namespace

#endif
