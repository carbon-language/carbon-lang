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
  const char *OpcodeString;
public:
  GenericBinaryInst(unsigned Opcode, Value *S1, Value *S2, 
		    const char *OpcodeStr, const string &Name = "")
    : BinaryOperator(Opcode, S1, S2, Name) {
    OpcodeString = OpcodeStr;
  }

  virtual string getOpcode() const { return OpcodeString; }
};

class SetCondInst : public BinaryOperator {
  BinaryOps OpType;
public:
  SetCondInst(BinaryOps opType, Value *S1, Value *S2, 
	      const string &Name = "");

  virtual string getOpcode() const;
};

#endif
