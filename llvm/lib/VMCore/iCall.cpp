//===-- iCall.cpp - Implement the call & icall instructions ------*- C++ -*--=//
//
// This file implements the call and icall instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Method.h"

CallInst::CallInst(Method *M, const vector<Value*> &params, 
                   const string &Name) 
  : Instruction(M->getReturnType(), Instruction::Call, Name) {

  Operands.reserve(1+params.size());
  Operands.push_back(Use(M, this));

  const MethodType::ParamTypes &PL = M->getMethodType()->getParamTypes();
  assert((params.size() == PL.size()) || 
	 (M->getMethodType()->isVarArg() && params.size() > PL.size()) &&
	 "Calling a function with bad signature");
#ifndef NDEBUG
  MethodType::ParamTypes::const_iterator It = PL.begin();
#endif
  for (unsigned i = 0; i < params.size(); i++)
    Operands.push_back(Use(params[i], this));
}

CallInst::CallInst(const CallInst &CI) 
  : Instruction(CI.getType(), Instruction::Call) {
  Operands.reserve(CI.Operands.size());
  for (unsigned i = 0; i < CI.Operands.size(); ++i)
    Operands.push_back(Use(CI.Operands[i], this));
}

