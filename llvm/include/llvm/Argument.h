//===-- llvm/Argument.h - Definition of the Argument class ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the Argument class, which represents an incoming formal
// argument to a Function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARGUMENT_H
#define LLVM_ARGUMENT_H

#include "llvm/Value.h"

class Argument : public Value {  // Defined in the Function.cpp file
  Function *Parent;

  Argument *Prev, *Next; // Next and Prev links for our intrusive linked list
  void setNext(Argument *N) { Next = N; }
  void setPrev(Argument *N) { Prev = N; }
  friend class SymbolTableListTraits<Argument, Function, Function,
                                     ilist_traits<Argument> >;
  void setParent(Function *parent);

public:
  /// Argument ctor - If Function argument is specified, this argument is
  /// inserted at the end of the argument list for the function.
  ///
  Argument(const Type *Ty, const std::string &Name = "", Function *F = 0);

  /// setName - Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  inline const Function *getParent() const { return Parent; }
  inline       Function *getParent()       { return Parent; }
 
  // getNext/Prev - Return the next or previous argument in the list.
        Argument *getNext()       { return Next; }
  const Argument *getNext() const { return Next; }
        Argument *getPrev()       { return Prev; }
  const Argument *getPrev() const { return Prev; }

  virtual void print(std::ostream &OS) const;

  /// classof - Methods for support type inquiry through isa, cast, and
  /// dyn_cast:
  ///
  static inline bool classof(const Argument *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == ArgumentVal;
  }
};

#endif
