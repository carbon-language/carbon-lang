//===-- llvm/iBinary.h - Binary Operator node definitions --------*- C++ -*--=//
//
// This file contains the declarations of all of the Binary Operator classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IBINARY_H
#define LLVM_IBINARY_H

#include "llvm/InstrTypes.h"

//===----------------------------------------------------------------------===//
//                   Class to represent Unary operators
//===----------------------------------------------------------------------===//
//
class GenericUnaryInst : public UnaryOperator {
public:
  GenericUnaryInst(UnaryOps Opcode, Value *S1, const std::string &Name = "")
    : UnaryOperator(S1, Opcode, Name) {
  }
};

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
};

#endif
