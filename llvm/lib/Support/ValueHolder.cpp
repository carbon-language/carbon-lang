//===-- ValueHolder.cpp - Wrapper for Value implementation ----------------===//
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

#include "llvm/Support/ValueHolder.h"
#include "llvm/Type.h"

ValueHolder::ValueHolder(Value *V) : User(Type::TypeTy, Value::TypeVal) {
  Operands.push_back(Use(V, this));
}
