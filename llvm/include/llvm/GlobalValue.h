//===-- llvm/GlobalValue.h - Class to represent a global value --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
public:
  enum LinkageTypes {
    ExternalLinkage,   // Externally visible function
    LinkOnceLinkage,   // Keep one copy of named function when linking (inline)
    WeakLinkage,       // Keep one copy of named function when linking (weak)
    AppendingLinkage,  // Special purpose, only applies to global arrays
    InternalLinkage    // Rename collisions when linking (static functions)
  };
protected:
  GlobalValue(const Type *Ty, ValueTy vty, LinkageTypes linkage,
	      const std::string &name = "")
    : User(Ty, vty, name), Linkage(linkage), Parent(0) {}

  LinkageTypes Linkage;   // The linkage of this global
  Module *Parent;
public:
  ~GlobalValue() {}

  /// getType - Global values are always pointers.
  inline const PointerType *getType() const {
    return (const PointerType*)User::getType();
  }

  bool hasExternalLinkage()  const { return Linkage == ExternalLinkage; }
  bool hasLinkOnceLinkage()  const { return Linkage == LinkOnceLinkage; }
  bool hasWeakLinkage()      const { return Linkage == WeakLinkage; }
  bool hasAppendingLinkage() const { return Linkage == AppendingLinkage; }
  bool hasInternalLinkage()  const { return Linkage == InternalLinkage; }
  void setLinkage(LinkageTypes LT) { Linkage = LT; }
  LinkageTypes getLinkage() const { return Linkage; }

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
