//===-- llvm/User.h - User class definition ----------------------*- C++ -*--=//
//
// This class defines the interface that one who 'use's a Value must implement.
// Each instance of the Value class keeps track of what User's have handles
// to it.
//
//  * Instructions are the largest class of User's.
//  * Constants may be users of other constants (think arrays and stuff)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_USER_H
#define LLVM_USER_H

#include "llvm/Value.h"

class User : public Value {
  User(const User &);             // Do not implement
protected:
  std::vector<Use> Operands;
public:
  User(const Type *Ty, ValueTy vty, const std::string &name = "");
  virtual ~User() { dropAllReferences(); }

  inline Value *getOperand(unsigned i) { 
    assert(i < Operands.size() && "getOperand() out of range!");
    return Operands[i];
  }
  inline const Value *getOperand(unsigned i) const {
    assert(i < Operands.size() && "getOperand() const out of range!");
    return Operands[i];
  }
  inline void setOperand(unsigned i, Value *Val) {
    assert(i < Operands.size() && "setOperand() out of range!");
    Operands[i] = Val;
  }
  inline unsigned getNumOperands() const { return Operands.size(); }

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  typedef std::vector<Use>::iterator       op_iterator;
  typedef std::vector<Use>::const_iterator const_op_iterator;

  inline op_iterator       op_begin()       { return Operands.begin(); }
  inline const_op_iterator op_begin() const { return Operands.begin(); }
  inline op_iterator       op_end()         { return Operands.end(); }
  inline const_op_iterator op_end()   const { return Operands.end(); }

  // dropAllReferences() - This function is in charge of "letting go" of all
  // objects that this User refers to.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references... first all references are dropped, and all use counts go to
  // zero.  Then everything is delete'd for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator 
  // delete.
  //
  inline void dropAllReferences() {
    Operands.clear();
  }

  /// replaceUsesOfWith - Replaces all references to the "From" definition with
  /// references to the "To" definition.
  ///
  void replaceUsesOfWith(Value *From, Value *To);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const User *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::GlobalVariableVal ||
           V->getValueType() == Value::ConstantVal ||
           V->getValueType() == Value::InstructionVal;
  }
};

template<> struct simplify_type<User::op_iterator> {
  typedef Value* SimpleType;
  
  static SimpleType getSimplifiedValue(const User::op_iterator &Val) {
    return (SimpleType)Val->get();
  }
};
template<> struct simplify_type<const User::op_iterator>
  : public simplify_type<User::op_iterator> {};

template<> struct simplify_type<User::const_op_iterator> {
  typedef Value* SimpleType;
  
  static SimpleType getSimplifiedValue(const User::const_op_iterator &Val) {
    return (SimpleType)Val->get();
  }
};
template<> struct simplify_type<const User::const_op_iterator>
  : public simplify_type<User::const_op_iterator> {};

#endif
