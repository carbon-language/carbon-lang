//===-- llvm/GlobalValue.h - Class to represent a global value ---*- C++ -*--=//
//
// This file is a common base class of all globally definable objects.  As such,
// it is subclassed by GlobalVariable and by Function.  This is used because you
// can do certain things with these global objects that you can't do to anything
// else.  For example, use the address of one as a constant.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBALVALUE_H
#define LLVM_GLOBALVALUE_H

#include "llvm/User.h"
class PointerType;
class Module;

class GlobalValue : public User {
  GlobalValue(const GlobalValue &);             // do not implement
protected:
  GlobalValue(const Type *Ty, ValueTy vty, bool hasInternalLinkage,
	      const std::string &name = "")
    : User(Ty, vty, name), HasInternalLinkage(hasInternalLinkage), Parent(0) {}

  bool HasInternalLinkage;    // Is this value accessable externally?
  Module *Parent;
public:
  ~GlobalValue() {}

  /// getType - Global values are always pointers.
  inline const PointerType *getType() const {
    return (const PointerType*)User::getType();
  }

  /// Internal Linkage - True if the global value is inaccessible to 
  bool hasInternalLinkage() const { return HasInternalLinkage; }
  bool hasExternalLinkage() const { return !HasInternalLinkage; }
  void setInternalLinkage(bool HIL) { HasInternalLinkage = HIL; }

  /// isExternal - Return true if the primary definition of this global value is
  /// outside of the current translation unit...
  virtual bool isExternal() const = 0;

  /// getParent - Get the module that this global value is contained inside
  /// of...
  inline Module *getParent() { return Parent; }
  inline const Module *getParent() const { return Parent; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalValue *T) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::FunctionVal || 
           V->getValueType() == Value::GlobalVariableVal;
  }
};

#endif
