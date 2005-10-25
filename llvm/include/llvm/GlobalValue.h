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

#include "llvm/Constant.h"

namespace llvm {

class PointerType;
class Module;

class GlobalValue : public Constant {
  GlobalValue(const GlobalValue &);             // do not implement
public:
  enum LinkageTypes {
    ExternalLinkage,   /// Externally visible function
    LinkOnceLinkage,   /// Keep one copy of named function when linking (inline)
    WeakLinkage,       /// Keep one copy of named function when linking (weak)
    AppendingLinkage,  /// Special purpose, only applies to global arrays
    InternalLinkage,   /// Rename collisions when linking (static functions)
    GhostLinkage       /// Stand-in functions for streaming fns from BC files
  };
protected:
  GlobalValue(const Type *Ty, ValueTy vty, Use *Ops, unsigned NumOps,
              LinkageTypes linkage, const std::string &name = "")
    : Constant(Ty, vty, Ops, NumOps, name), Linkage(linkage), Parent(0) { }

  LinkageTypes Linkage;   // The linkage of this global
  Module *Parent;
public:
  ~GlobalValue() {
    removeDeadConstantUsers();   // remove any dead constants using this.
  }

  /// If the usage is empty (except transitively dead constants), then this
  /// global value can can be safely deleted since the destructor will
  /// delete the dead constants as well.
  /// @brief Determine if the usage of this global value is empty except
  /// for transitively dead constants.
  bool use_empty_except_constants();

  /// getType - Global values are always pointers.
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(User::getType());
  }

  bool hasExternalLinkage()  const { return Linkage == ExternalLinkage; }
  bool hasLinkOnceLinkage()  const { return Linkage == LinkOnceLinkage; }
  bool hasWeakLinkage()      const { return Linkage == WeakLinkage; }
  bool hasAppendingLinkage() const { return Linkage == AppendingLinkage; }
  bool hasInternalLinkage()  const { return Linkage == InternalLinkage; }
  void setLinkage(LinkageTypes LT) { Linkage = LT; }
  LinkageTypes getLinkage() const { return Linkage; }

  /// hasNotBeenReadFromBytecode - If a module provider is being used to lazily
  /// stream in functions from disk, this method can be used to check to see if
  /// the function has been read in yet or not.  Unless you are working on the
  /// JIT or something else that streams stuff in lazily, you don't need to
  /// worry about this.
  bool hasNotBeenReadFromBytecode() const { return Linkage == GhostLinkage; }

  /// Override from Constant class. No GlobalValue's are null values so this
  /// always returns false.
  virtual bool isNullValue() const { return false; }

  /// Override from Constant class.
  virtual void destroyConstant();

  /// isExternal - Return true if the primary definition of this global value is
  /// outside of the current translation unit...
  virtual bool isExternal() const = 0;

  /// getParent - Get the module that this global value is contained inside
  /// of...
  inline Module *getParent() { return Parent; }
  inline const Module *getParent() const { return Parent; }

  /// removeDeadConstantUsers - If there are any dead constant users dangling
  /// off of this global value, remove them.  This method is useful for clients
  /// that want to check to see if a global is unused, but don't want to deal
  /// with potentially dead constants hanging off of the globals.
  ///
  /// This method tries to make the global dead.  If it detects a user that
  /// would prevent it from becoming completely dead, it gives up early,
  /// potentially leaving some dead constant users around.
  void removeDeadConstantUsers();

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalValue *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::FunctionVal ||
           V->getValueType() == Value::GlobalVariableVal;
  }
};

} // End llvm namespace

#endif
