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

class GenericBinaryInst : public BinaryOperator {
public:
  GenericBinaryInst(BinaryOps Opcode, Value *S1, Value *S2, 
		    const std::string &Name = "")
    : BinaryOperator(Opcode, S1, S2, Name) {
  }
};

class SetCondInst : public BinaryOperator {
  BinaryOps OpType;
public:
  SetCondInst(BinaryOps opType, Value *S1, Value *S2, 
	      const std::string &Name = "");

  // getInverseCondition - Return the inverse of the current condition opcode.
  // For example seteq -> setne, setgt -> setle, setlt -> setge, etc...
  //
  BinaryOps getInverseCondition() const;
};

#endif
