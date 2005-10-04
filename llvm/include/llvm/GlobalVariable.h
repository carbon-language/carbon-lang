//===-- llvm/GlobalVariable.h - GlobalVariable class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the GlobalVariable class, which
// represents a single global variable (or constant) in the VM.
//
// Global variables are constant pointers that refer to hunks of space that are
// allocated by either the VM, or by the linker in a static compiler.  A global
// variable may have an intial value, which is copied into the executables .data
// area.  Global Constants are required to have initializers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBAL_VARIABLE_H
#define LLVM_GLOBAL_VARIABLE_H

#include "llvm/GlobalValue.h"

namespace llvm {

class Module;
class Constant;
class PointerType;
template<typename SC> struct ilist_traits;
template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass> class SymbolTableListTraits;

class GlobalVariable : public GlobalValue {
  friend class SymbolTableListTraits<GlobalVariable, Module, Module,
                                     ilist_traits<GlobalVariable> >;
  void setParent(Module *parent);

  GlobalVariable *Prev, *Next;
  void setNext(GlobalVariable *N) { Next = N; }
  void setPrev(GlobalVariable *N) { Prev = N; }

  bool isConstantGlobal;               // Is this a global constant?
  Use Initializer;

public:
  /// GlobalVariable ctor - If a parent module is specified, the global is
  /// automatically inserted into the end of the specified modules global list.
  ///
  GlobalVariable(const Type *Ty, bool isConstant, LinkageTypes Linkage,
                 Constant *Initializer = 0, const std::string &Name = "",
                 Module *Parent = 0);

  /// isExternal - Is this global variable lacking an initializer?  If so, the
  /// global variable is defined in some other translation unit, and is thus
  /// externally defined here.
  ///
  virtual bool isExternal() const { return getNumOperands() == 0; }

  /// hasInitializer - Unless a global variable isExternal(), it has an
  /// initializer.  The initializer for the global variable/constant is held by
  /// Initializer if an initializer is specified.
  ///
  inline bool hasInitializer() const { return !isExternal(); }

  /// getInitializer - Return the initializer for this global variable.  It is
  /// illegal to call this method if the global is external, because we cannot
  /// tell what the value is initialized to!
  ///
  inline Constant *getInitializer() const {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return reinterpret_cast<Constant*>(Initializer.get());
  }
  inline Constant *getInitializer() {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return reinterpret_cast<Constant*>(Initializer.get());
  }
  inline void setInitializer(Constant *CPV) {
    if (CPV == 0) {
      if (hasInitializer()) {
        Initializer.set(0);
        NumOperands = 0;
      }
    } else {
      if (!hasInitializer())
        NumOperands = 1;
      Initializer.set(CPV);
    }
  }

  // getNext/Prev - Return the next or previous global variable in the list.
        GlobalVariable *getNext()       { return Next; }
  const GlobalVariable *getNext() const { return Next; }
        GlobalVariable *getPrev()       { return Prev; }
  const GlobalVariable *getPrev() const { return Prev; }

  /// If the value is a global constant, its value is immutable throughout the
  /// runtime execution of the program.  Assigning a value into the constant
  /// leads to undefined behavior.
  ///
  bool isConstant() const { return isConstantGlobal; }
  void setConstant(bool Value) { isConstantGlobal = Value; }

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  void eraseFromParent();

  /// Override Constant's implementation of this method so we can
  /// replace constant initializers.
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  virtual void print(std::ostream &OS) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalVariable *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::GlobalVariableVal;
  }
};

} // End llvm namespace

#endif
