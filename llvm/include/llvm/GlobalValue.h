//===-- llvm/GlobalValue.h - Class to represent a global value ---*- C++ -*--=//
//
// This file is a common base class of all globally definable objects.  As such,
// it is subclassed by GlobalVariable and by Method.  This is used because you
// can do certain things with these global objects that you can't do to anything
// else.  For example, use the address of one as a constant.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBALVALUE_H
#define LLVM_GLOBALVALUE_H

#include "llvm/User.h"
class PointerType;

class GlobalValue : public User {
  GlobalValue(const GlobalValue &);             // do not implement
protected:
  GlobalValue(const Type *Ty, ValueTy vty, const string &name = "")
    : User(Ty, vty, name) {}
public:

  // getType - Global values are always pointers (FIXME, methods should be ptrs too!)
  inline const PointerType *getType() const {
    return (const PointerType*)User::getType();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalValue *T) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::MethodVal || 
           V->getValueType() == Value::GlobalVariableVal;
  }
};

#endif
