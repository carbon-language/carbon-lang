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

#include "llvm/GlobalValue.h"
class Module;
class Constant;
class PointerType;

class GlobalVariable : public GlobalValue {
  friend class ValueHolder<GlobalVariable, Module, Module>;
  void setParent(Module *parent) { Parent = parent; }

  bool isConstantGlobal;               // Is this a global constant?
public:
  GlobalVariable(const Type *Ty, bool isConstant, bool isInternal,
		 Constant *Initializer = 0, const string &Name = "");
  ~GlobalVariable() {}

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name, SymbolTable *ST = 0);

  // The initializer for the global variable/constant is held by Operands[0] if
  // an initializer is specified.
  //
  inline bool hasInitializer() const { return !Operands.empty(); }
  inline Constant *getInitializer() const {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return (Constant*)Operands[0].get();
  }
  inline Constant *getInitializer() {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return (Constant*)Operands[0].get();
  }
  inline void setInitializer(Constant *CPV) {
    if (CPV == 0) {
      if (hasInitializer()) Operands.pop_back();
    } else {
      if (!hasInitializer()) Operands.push_back(Use(0, this));
      Operands[0] = (Value*)CPV;
    }
  }


  // If the value is a global constant, its value is immutable throughout the
  // runtime execution of the program.  Assigning a value into the constant
  // leads to undefined behavior.
  //
  inline bool isConstant() const { return isConstantGlobal; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalVariable *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::GlobalVariableVal;
  }
};

#endif
