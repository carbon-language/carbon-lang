//===-- llvm/User.h - User class definition ----------------------*- C++ -*--=//
//
// This class defines the interface that one who 'use's a Value must implement.
// Each instance of the Value class keeps track of what User's have handles
// to it.
//
//  * Instructions are the largest class of User's.
//  * Constants may be users of other constants (think arrays and stuff)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_USER_H
#define LLVM_USER_H

#include "llvm/Value.h"

class User : public Value {
  User(const User &);             // Do not implement
public:
  User(const Type *Ty, ValueTy vty, const string &name = "");
  virtual ~User() {}

  // if i > the number of operands, then getOperand() returns 0, and setOperand
  // returns false.  setOperand() may also return false if the operand is of
  // the wrong type.
  //
  virtual Value *getOperand(unsigned i) = 0;
  virtual const Value *getOperand(unsigned i) const = 0;
  virtual bool setOperand(unsigned i, Value *Val) = 0;

  // dropAllReferences() - This virtual function should be overridden to "let
  // go" of all references that this user is maintaining.  This allows one to 
  // 'delete' a whole class at a time, even though there may be circular
  // references... first all references are dropped, and all use counts go to
  // zero.  Then everything is delete'd for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator 
  // delete.
  //
  virtual void dropAllReferences() = 0;

  // replaceUsesOfWith - Replaces all references to the "From" definition with
  // references to the "To" definition.  (defined in Value.cpp)
  //
  void replaceUsesOfWith(Value *From, Value *To);
};

#endif
