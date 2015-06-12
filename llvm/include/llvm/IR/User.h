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

/// \brief Compile-time customization of User operands.
///
/// Customizes operand-related allocators and accessors.
template <class>
struct OperandTraits;

class User : public Value {
  User(const User &) = delete;
  void *operator new(size_t) = delete;
  template <unsigned>
  friend struct HungoffOperandTraits;
  virtual void anchor();
protected:
  /// \brief This is a pointer to the array of Uses for this User.
  ///
  /// For nodes of fixed arity (e.g. a binary operator) this array will live
  /// prefixed to some derived class instance.  For nodes of resizable variable
  /// arity (e.g. PHINodes, SwitchInst etc.), this memory will be dynamically
  /// allocated and should be destroyed by the classes' virtual dtor.
  Use *LegacyOperandList;

protected:
  void *operator new(size_t s, unsigned Us);

  User(Type *ty, unsigned vty, Use *OpList, unsigned NumOps)
      : Value(ty, vty) {
    setOperandList(OpList);
    assert(NumOps < (1u << NumUserOperandsBits) && "Too many operands");
    NumUserOperands = NumOps;
  }

  /// \brief Allocate the array of Uses, followed by a pointer
  /// (with bottom bit set) to the User.
  /// \param IsPhi identifies callers which are phi nodes and which need
  /// N BasicBlock* allocated along with N
  void allocHungoffUses(unsigned N, bool IsPhi = false);

  /// \brief Grow the number of hung off uses.  Note that allocHungoffUses
  /// should be called if there are no uses.
  void growHungoffUses(unsigned N, bool IsPhi = false);

public:
  ~User() override {
    // drop the hung off uses.
    Use::zap(getOperandList(), getOperandList() + NumUserOperands,
             HasHungOffUses);
    if (HasHungOffUses) {
      setOperandList(nullptr);
      // Reset NumOperands so User::operator delete() does the right thing.
      NumUserOperands = 0;
    }
  }
  /// \brief Free memory allocated for User and Use objects.
  void operator delete(void *Usr);
  /// \brief Placement delete - required by std, but never called.
  void operator delete(void*, unsigned) {
    llvm_unreachable("Constructor throws?");
  }
  /// \brief Placement delete - required by std, but never called.
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
private:
  void setOperandList(Use *NewList) {
    LegacyOperandList = NewList;
  }
public:
  Use *getOperandList() const {
    return LegacyOperandList;
  }
  Value *getOperand(unsigned i) const {
    assert(i < NumUserOperands && "getOperand() out of range!");
    return getOperandList()[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < NumUserOperands && "setOperand() out of range!");
    assert((!isa<Constant>((const Value*)this) ||
            isa<GlobalValue>((const Value*)this)) &&
           "Cannot mutate a constant with setOperand!");
    getOperandList()[i] = Val;
  }
  const Use &getOperandUse(unsigned i) const {
    assert(i < NumUserOperands && "getOperandUse() out of range!");
    return getOperandList()[i];
  }
  Use &getOperandUse(unsigned i) {
    assert(i < NumUserOperands && "getOperandUse() out of range!");
    return getOperandList()[i];
  }

  unsigned getNumOperands() const { return NumUserOperands; }

  /// Set the number of operands on a GlobalVariable.
  ///
  /// GlobalVariable always allocates space for a single operands, but
  /// doesn't always use it.
  ///
  /// FIXME: As that the number of operands is used to find the start of
  /// the allocated memory in operator delete, we need to always think we have
  /// 1 operand before delete.
  void setGlobalVariableNumOperands(unsigned NumOps) {
    assert(NumOps <= 1 && "GlobalVariable can only have 0 or 1 operands");
    NumUserOperands = NumOps;
  }

  /// \brief Subclasses with hung off uses need to manage the operand count
  /// themselves.  In these instances, the operand count isn't used to find the
  /// OperandList, so there's no issue in having the operand count change.
  void setNumHungOffUseOperands(unsigned NumOps) {
    assert(HasHungOffUses && "Must have hung off uses to use this method");
    assert(NumOps < (1u << NumUserOperandsBits) && "Too many operands");
    NumUserOperands = NumOps;
  }

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  typedef Use*       op_iterator;
  typedef const Use* const_op_iterator;
  typedef iterator_range<op_iterator> op_range;
  typedef iterator_range<const_op_iterator> const_op_range;

  inline op_iterator       op_begin()       { return getOperandList(); }
  inline const_op_iterator op_begin() const { return getOperandList(); }
  inline op_iterator       op_end()         {
    return getOperandList() + NumUserOperands;
  }
  inline const_op_iterator op_end()   const {
    return getOperandList() + NumUserOperands;
  }
  inline op_range operands() {
    return op_range(op_begin(), op_end());
  }
  inline const_op_range operands() const {
    return const_op_range(op_begin(), op_end());
  }

  /// \brief Iterator for directly iterating over the operand Values.
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

  /// \brief Drop all references to operands.
  ///
  /// This function is in charge of "letting go" of all objects that this User
  /// refers to.  This allows one to 'delete' a whole class at a time, even
  /// though there may be circular references...  First all references are
  /// dropped, and all use counts go to zero.  Then everything is deleted for
  /// real.  Note that no operations are valid on an object that has "dropped
  /// all references", except operator delete.
  void dropAllReferences() {
    for (Use &U : operands())
      U.set(nullptr);
  }

  /// \brief Replace uses of one Value with another.
  ///
  /// Replaces all references to the "From" definition with references to the
  /// "To" definition.
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
