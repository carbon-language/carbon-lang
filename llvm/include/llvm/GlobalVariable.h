//===-- llvm/Global.h - Class to represent a global variable -----*- C++ -*--=//
//
// This file contains the declaration of the GlobalVariable class, which
// represents a single global variable in the VM.
//
// Global variables are constant pointers that refer to hunks of space that are
// allocated by either the VM, or by the linker in a static compiler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBAL_VARIABLE_H
#define LLVM_GLOBAL_VARIABLE_H

#include "llvm/Value.h"
class Module;

class GlobalVariable : public Value {
  Module *Parent;                  // The module that contains this method

  friend class ValueHolder<GlobalVariable, Module, Module>;
  void setParent(Module *parent) { Parent = parent; }

public:
  GlobalVariable(const Type *Ty, const string &Name = "");
  ~GlobalVariable() {}

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name, SymbolTable *ST = 0);

  inline       Module *getParent()       { return Parent; }
  inline const Module *getParent() const { return Parent; }
};

#endif
