//===-- llvm/Argument.h - Definition of the Argument class -------*- C++ -*--=//
//
// This file defines the Argument class, which represents and incoming formal
// argument to a function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARGUMENT_H
#define LLVM_ARGUMENT_H

#include "llvm/Value.h"

class Argument : public Value {  // Defined in the InstrType.cpp file
  Function *Parent;

  friend class ValueHolder<Argument, Function, Function>;
  inline void setParent(Function *parent) { Parent = parent; }

public:
  Argument(const Type *Ty, const std::string &Name = "") 
    : Value(Ty, Value::ArgumentVal, Name) {
    Parent = 0;
  }

  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  inline const Function *getParent() const { return Parent; }
  inline       Function *getParent()       { return Parent; }

  virtual void print(std::ostream &OS) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Argument *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == ArgumentVal;
  }
};

#endif
