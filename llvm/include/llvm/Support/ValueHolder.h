//===-- llvm/Support/ValueHolder.h - Wrapper for Value's --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class defines a simple subclass of User, which keeps a pointer to a
// Value, which automatically updates when Value::replaceAllUsesWith is called.
// This is useful when you have pointers to Value's in your pass, but the
// pointers get invalidated when some other portion of the algorithm is
// replacing Values with other Values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VALUEHOLDER_H
#define LLVM_SUPPORT_VALUEHOLDER_H

#include "llvm/User.h"

struct ValueHolder : public User {
  ValueHolder(Value *V = 0);
  ValueHolder(const ValueHolder &VH) : User(VH.getType(), Value::TypeVal) {
    Operands.push_back(Use(VH.get(), this));
  }

  // Getters...
  Value *get() const { return (Value*)getOperand(0); }
  operator Value*() const { return (Value*)getOperand(0); }

  // Setters...
  const ValueHolder &operator=(Value *V) {
    setOperand(0, V);
    return *this;
  }

  const ValueHolder &operator=(ValueHolder &VH) {
    setOperand(0, VH);
    return *this;
  }

  virtual void print(std::ostream& OS) const {
    OS << "ValueHolder";
  }
};

#endif
