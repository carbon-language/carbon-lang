//===-- llvm/User.h - User class definition ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class defines the interface that one who uses a Value must implement.
// Each instance of the Value class keeps track of what User's have handles
// to it.
//
//  * Instructions are the largest class of Users.
//  * Constants may be users of other constants (think arrays and stuff)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_USER_H
#define LLVM_IR_USER_H

#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// OperandTraits - Compile-time customization of
/// operand-related allocators and accessors
/// for use of the User class
template <class>
struct OperandTraits;

class User : public Value {
  User(const User &) LLVM_DELETED_FUNCTION;
  void *operator new(size_t) LLVM_DELETED_FUNCTION;
  template <unsigned>
  friend struct HungoffOperandTraits;
  virtual void anchor();
protected:
  /// NumOperands - The number of values used by this User.
  ///
  unsigned NumOperands;

  /// OperandList - This is a pointer to the array of Uses for this User.
  /// For nodes of fixed arity (e.g. a binary operator) this array will live
  /// prefixed to some derived class instance.  For nodes of resizable variable
  /// arity (e.g. PHINodes, SwitchInst etc.), this memory will be dynamically
  /// allocated and should be destroyed by the classes' virtual dtor.
  Use *OperandList;

  void *operator new(size_t s, unsigned Us);
  User(Type *ty, unsigned vty, Use *OpList, unsigned NumOps)
    : Value(ty, vty), NumOperands(NumOps), OperandList(OpList) {}
  Use *allocHungoffUses(unsigned) const;
  void dropHungoffUses() {
    Use::zap(OperandList, OperandList + NumOperands, true);
    OperandList = nullptr;
    // Reset NumOperands so User::operator delete() does the right thing.
    NumOperands = 0;
  }
public:
  ~User() {
    Use::zap(OperandList, OperandList + NumOperands);
  }
  /// operator delete - free memory allocated for User and Use objects
  void operator delete(void *Usr);
  /// placement delete - required by std, but never called.
  void operator delete(void*, unsigned) {
    llvm_unreachable("Constructor throws?");
  }
  /// placement delete - required by std, but never called.
  void operator delete(void*, unsigned, bool) {
    llvm_unreachable("Constructor throws?");
  }
protected:
  template <int Idx, typename U> static Use &OpFrom(const U *that) {
    return Idx < 0
      ? OperandTraits<U>::op_end(const_cast<U*>(that))[Idx]
      : OperandTraits<U>::op_begin(const_cast<U*>(that))[Idx];
  }
  template <int Idx> Use &Op() {
    return OpFrom<Idx>(this);
  }
  template <int Idx> const Use &Op() const {
    return OpFrom<Idx>(this);
  }
public:
  Value *getOperand(unsigned i) const {
    assert(i < NumOperands && "getOperand() out of range!");
    return OperandList[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < NumOperands && "setOperand() out of range!");
    assert((!isa<Constant>((const Value*)this) ||
            isa<GlobalValue>((const Value*)this)) &&
           "Cannot mutate a constant with setOperand!");
    OperandList[i] = Val;
  }
  const Use &getOperandUse(unsigned i) const {
    assert(i < NumOperands && "getOperandUse() out of range!");
    return OperandList[i];
  }
  Use &getOperandUse(unsigned i) {
    assert(i < NumOperands && "getOperandUse() out of range!");
    return OperandList[i];
  }

  unsigned getNumOperands() const { return NumOperands; }

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  typedef Use*       op_iterator;
  typedef const Use* const_op_iterator;
  typedef iterator_range<op_iterator> op_range;
  typedef iterator_range<const_op_iterator> const_op_range;

  inline op_iterator       op_begin()       { return OperandList; }
  inline const_op_iterator op_begin() const { return OperandList; }
  inline op_iterator       op_end()         { return OperandList+NumOperands; }
  inline const_op_iterator op_end()   const { return OperandList+NumOperands; }
  inline op_range operands() {
    return op_range(op_begin(), op_end());
  }
  inline const_op_range operands() const {
    return const_op_range(op_begin(), op_end());
  }

  /// Convenience iterator for directly iterating over the Values in the
  /// OperandList
  struct value_op_iterator
      : iterator_adaptor_base<value_op_iterator, op_iterator,
                              std::random_access_iterator_tag, Value *,
                              ptrdiff_t, Value *, Value *> {
    explicit value_op_iterator(Use *U = nullptr) : iterator_adaptor_base(U) {}

    Value *operator*() const { return *I; }
    Value *operator->() const { return operator*(); }
  };

  inline value_op_iterator value_op_begin() {
    return value_op_iterator(op_begin());
  }
  inline value_op_iterator value_op_end() {
    return value_op_iterator(op_end());
  }
  inline iterator_range<value_op_iterator> operand_values() {
    return iterator_range<value_op_iterator>(value_op_begin(), value_op_end());
  }

  // dropAllReferences() - This function is in charge of "letting go" of all
  // objects that this User refers to.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references...  First all references are dropped, and all use counts go to
  // zero.  Then everything is deleted for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator
  // delete.
  //
  void dropAllReferences() {
    for (Use &U : operands())
      U.set(nullptr);
  }

  /// replaceUsesOfWith - Replaces all references to the "From" definition with
  /// references to the "To" definition.
  ///
  void replaceUsesOfWith(Value *From, Value *To);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) || isa<Constant>(V);
  }
};

template<> struct simplify_type<User::op_iterator> {
  typedef Value* SimpleType;
  static SimpleType getSimplifiedValue(User::op_iterator &Val) {
    return Val->get();
  }
};
template<> struct simplify_type<User::const_op_iterator> {
  typedef /*const*/ Value* SimpleType;
  static SimpleType getSimplifiedValue(User::const_op_iterator &Val) {
    return Val->get();
  }
};

} // End llvm namespace

#endif
