//===-- llvm/Constant.h - Constant class definition -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Constant class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANT_H
#define LLVM_CONSTANT_H

#include "llvm/User.h"

namespace llvm {

class Constant : public User {
protected:
  Constant(const Type *Ty, ValueTy vty, Use *Ops, unsigned NumOps,
           const std::string& Name = "")
    : User(Ty, vty, Ops, NumOps, Name) {}

  void destroyConstantImpl();
public:
  /// Static constructor to get a '0' constant of arbitrary type...
  ///
  static Constant *getNullValue(const Type *Ty);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const = 0;

  virtual void print(std::ostream &O) const;

  // Specialize get/setOperand for Constant's as their operands are always
  // constants as well.
  Constant *getOperand(unsigned i) {
    return static_cast<Constant*>(User::getOperand(i));
  }
  const Constant *getOperand(unsigned i) const {
    return static_cast<const Constant*>(User::getOperand(i));
  }
  void setOperand(unsigned i, Constant *C) {
    User::setOperand(i, C);
  }

  /// destroyConstant - Called if some element of this constant is no longer
  /// valid.  At this point only other constants may be on the use_list for this
  /// constant.  Any constants on our Use list must also be destroy'd.  The
  /// implementation must be sure to remove the constant from the list of
  /// available cached constants.  Implementations should call
  /// destroyConstantImpl as the last thing they do, to destroy all users and
  /// delete this.
  virtual void destroyConstant() { assert(0 && "Not reached!"); }

  //// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Constant *) { return true; }
  static inline bool classof(const GlobalValue *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::SimpleConstantVal ||
           V->getValueType() == Value::ConstantExprVal ||
           V->getValueType() == Value::ConstantAggregateZeroVal ||
           V->getValueType() == Value::FunctionVal ||
           V->getValueType() == Value::GlobalVariableVal ||
           V->getValueType() == Value::UndefValueVal;
  }

  /// replaceUsesOfWithOnConstant - This method is a special form of
  /// User::replaceUsesOfWith (which does not work on constants) that does work
  /// on constants.  Basically this method goes through the trouble of building
  /// a new constant that is equivalent to the current one, with all uses of
  /// From replaced with uses of To.  After this construction is completed, all
  /// of the users of 'this' are replaced to use the new constant, and then
  /// 'this' is deleted.  In general, you should not call this method, instead,
  /// use Value::replaceAllUsesWith, which automatically dispatches to this
  /// method as needed.
  ///
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To,
                                           bool DisableChecking = false) {
    // Provide a default implementation for constants (like integers) that
    // cannot use any other values.  This cannot be called at runtime, but needs
    // to be here to avoid link errors.
    assert(getNumOperands() == 0 && "replaceUsesOfWithOnConstant must be "
           "implemented for all constants that have operands!");
    assert(0 && "Constants that do not have operands cannot be using 'From'!");
  }

  /// clearAllValueMaps - This method frees all internal memory used by the
  /// constant subsystem, which can be used in environments where this memory
  /// is otherwise reported as a leak.
  static void clearAllValueMaps();
};

} // End llvm namespace

#endif
