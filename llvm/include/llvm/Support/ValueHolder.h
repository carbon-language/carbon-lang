//===-- llvm/Support/ValueHolder.h - Wrapper for Value's --------*- C++ -*-===//
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

  // Getters...
  const Value *get() const { return getOperand(0); }
  operator const Value*() const { return getOperand(0); }
  Value *get() { return getOperand(0); }
  operator Value*() { return getOperand(0); }

  // Setters...
  const ValueHolder &operator=(Value *V) {
    setOperand(0, V);
    return *this;
  }

  virtual void print(std::ostream& OS) const {
    OS << "ValueHolder";
  }
};

#endif
