//===-- llvm/Global.h - Class to represent a global variable -----*- C++ -*--=//
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

#include "llvm/User.h"
class Module;
class ConstPoolVal;
class PointerType;

class GlobalVariable : public User {
  Module *Parent;                  // The module that contains this method

  friend class ValueHolder<GlobalVariable, Module, Module>;
  void setParent(Module *parent) { Parent = parent; }

  bool Constant;                   // Is this a global constant?
public:
  GlobalVariable(const Type *Ty, bool isConstant, ConstPoolVal *Initializer = 0,
		 const string &Name = "");
  ~GlobalVariable() {}

  // getType - Global variables are always pointers
  inline const PointerType *getType() const {
    return (const PointerType*)User::getType();
  }

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name, SymbolTable *ST = 0);

  inline       Module *getParent()       { return Parent; }
  inline const Module *getParent() const { return Parent; }

  // The initializer for the global variable/constant is held by Operands[0] if
  // an initializer is specified.
  //
  inline bool hasInitializer() const { return !Operands.empty(); }
  inline const ConstPoolVal *getInitializer() const {
    return Operands[0]->castConstantAsserting();
  }
  inline ConstPoolVal *getInitializer() {
    return Operands[0]->castConstantAsserting();
  }
  inline void setInitializer(ConstPoolVal *CPV) { Operands[0] = (Value*)CPV; }


  // If the value is a global constant, its value is immutable throughout the
  // runtime execution of the program.  Assigning a value into the constant
  // leads to undefined behavior.
  //
  inline bool isConstant() const { return Constant; }
};

#endif
