//===-- llvm/iBinary.h - Binary Operator node definitions --------*- C++ -*--=//
//
// This file contains the declarations of all of the Binary Operator classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IBINARY_H
#define LLVM_IBINARY_H

#include "llvm/InstrTypes.h"

//===----------------------------------------------------------------------===//
//                 Classes to represent Binary operators
//===----------------------------------------------------------------------===//
//
// All of these classes are subclasses of the BinaryOperator class...
//

class AddInst : public BinaryOperator {
public:
  AddInst(Value *S1, Value *S2, const string &Name = "")
      : BinaryOperator(Instruction::Add, S1, S2, Name) {
  }

  virtual string getOpcode() const { return "add"; }
};


class SubInst : public BinaryOperator {
public:
  SubInst(Value *S1, Value *S2, const string &Name = "") 
    : BinaryOperator(Instruction::Sub, S1, S2, Name) {
  }

  virtual string getOpcode() const { return "sub"; }
};


class SetCondInst : public BinaryOperator {
  BinaryOps OpType;
public:
  SetCondInst(BinaryOps opType, Value *S1, Value *S2, 
	      const string &Name = "");

  virtual string getOpcode() const;
};

#endif
