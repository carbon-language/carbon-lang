//===-- iCall.cpp - Implement the Call & Invoke instructions -----*- C++ -*--=//
//
// This file implements the call and invoke instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Method.h"

CallInst::CallInst(Method *m, vector<Value*> &params, 
                   const string &Name) 
  : Instruction(m->getReturnType(), Instruction::Call, Name), M(m, this) {

  const MethodType* MT = M->getMethodType();
  const MethodType::ParamTypes &PL = MT->getParamTypes();
  assert(params.size() == PL.size() && "Calling a function with bad signature");
#ifndef NDEBUG
  MethodType::ParamTypes::const_iterator It = PL.begin();
#endif
  for (unsigned i = 0; i < params.size(); i++) {
    assert(*It++ == params[i]->getType());
    Params.push_back(Use(params[i], this));
  }
}

CallInst::CallInst(const CallInst &CI) 
  : Instruction(CI.getType(), Instruction::Call), M(CI.M, this) {
  for (unsigned i = 0; i < CI.Params.size(); i++)
    Params.push_back(Use(CI.Params[i], this));
}

void CallInst::dropAllReferences() { 
  M = 0;
  Params.clear(); 
}

bool CallInst::setOperand(unsigned i, Value *Val) {
  if (i > Params.size()) return false;
  if (i == 0) {
    M = Val->castMethodAsserting();
  } else {
    // TODO: assert = method arg type
    Params[i-1] = Val;
  }
  return true;
}
