//===-- llvm/Argument.h - Definition of the Argument class ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Argument class. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARGUMENT_H
#define LLVM_ARGUMENT_H

#include "llvm/Value.h"

namespace llvm {

template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

/// A class to represent an incoming formal argument to a Function. An argument
/// is a very simple Value. It is essentially a named (optional) type. When used
/// in the body of a function, it represents the value of the actual argument
/// the function was called with.
/// @brief LLVM Argument representation  
class Argument : public Value {  // Defined in the Function.cpp file
  Function *Parent;

  Argument *Prev, *Next; // Next and Prev links for our intrusive linked list
  void setNext(Argument *N) { Next = N; }
  void setPrev(Argument *N) { Prev = N; }
  friend class SymbolTableListTraits<Argument, Function>;
  void setParent(Function *parent);

public:
  /// Argument ctor - If Function argument is specified, this argument is
  /// inserted at the end of the argument list for the function.
  ///
  explicit Argument(const Type *Ty,
                    const std::string &Name = "",
                    Function *F = 0);

  inline const Function *getParent() const { return Parent; }
  inline       Function *getParent()       { return Parent; }

  virtual void print(std::ostream &OS) const;
  void print(std::ostream *OS) const {
    if (OS) print(*OS);
  }

  /// classof - Methods for support type inquiry through isa, cast, and
  /// dyn_cast:
  ///
  static inline bool classof(const Argument *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueID() == ArgumentVal;
  }
  
private:
  // getNext/Prev - Return the next or previous argument in the list.
  Argument *getNext()       { return Next; }
  const Argument *getNext() const { return Next; }
  Argument *getPrev()       { return Prev; }
  const Argument *getPrev() const { return Prev; }
};

} // End llvm namespace

#endif
