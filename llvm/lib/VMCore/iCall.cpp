//===-- iCall.cpp - Implement the call & invoke instructions --------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the call and invoke instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Support/CallSite.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                        CallInst Implementation
//===----------------------------------------------------------------------===//

CallInst::CallInst(Value *Func, const std::vector<Value*> &params, 
                   const std::string &Name, Instruction *InsertBefore) 
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
				 ->getElementType())->getReturnType(),
		Instruction::Call, Name, InsertBefore) {
  Operands.reserve(1+params.size());
  Operands.push_back(Use(Func, this));

  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  const FunctionType::ParamTypes &PL = MTy->getParamTypes();
  assert(params.size() == PL.size() || 
	 (MTy->isVarArg() && params.size() > PL.size()) &&
	 "Calling a function with bad signature");
  for (unsigned i = 0; i < params.size(); i++)
    Operands.push_back(Use(params[i], this));
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  Operands.reserve(1);
  Operands.push_back(Use(Func, this));
  
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  const FunctionType::ParamTypes &PL = MTy->getParamTypes();
  assert(PL.empty() && "Calling a function with bad signature");
}

CallInst::CallInst(Value *Func, Value* A, const std::string &Name,
                   Instruction  *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  Operands.reserve(2);
  Operands.push_back(Use(Func, this));
  
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  const FunctionType::ParamTypes &PL = MTy->getParamTypes();
  assert(PL.size() == 1 || (MTy->isVarArg() && PL.empty()) &&
	 "Calling a function with bad signature");
  Operands.push_back(Use(A, this));
}

CallInst::CallInst(const CallInst &CI) 
  : Instruction(CI.getType(), Instruction::Call) {
  Operands.reserve(CI.Operands.size());
  for (unsigned i = 0; i < CI.Operands.size(); ++i)
    Operands.push_back(Use(CI.Operands[i], this));
}

const Function *CallInst::getCalledFunction() const {
  if (const Function *F = dyn_cast<Function>(Operands[0]))
    return F;
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Operands[0]))
    return cast<Function>(CPR->getValue());
  return 0;
}
Function *CallInst::getCalledFunction() {
  if (Function *F = dyn_cast<Function>(Operands[0]))
    return F;
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Operands[0]))
    return cast<Function>(CPR->getValue());
  return 0;
}


//===----------------------------------------------------------------------===//
//                        InvokeInst Implementation
//===----------------------------------------------------------------------===//

InvokeInst::InvokeInst(Value *Func, BasicBlock *IfNormal,
		       BasicBlock *IfException,
                       const std::vector<Value*> &params,
		       const std::string &Name, Instruction *InsertBefore)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Func->getType())
				    ->getElementType())->getReturnType(),
		   Instruction::Invoke, Name, InsertBefore) {
  Operands.reserve(3+params.size());
  Operands.push_back(Use(Func, this));
  Operands.push_back(Use((Value*)IfNormal, this));
  Operands.push_back(Use((Value*)IfException, this));
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  
  const FunctionType::ParamTypes &PL = MTy->getParamTypes();
  assert((params.size() == PL.size()) || 
	 (MTy->isVarArg() && params.size() > PL.size()) &&
	 "Calling a function with bad signature");
  
  for (unsigned i = 0; i < params.size(); i++)
    Operands.push_back(Use(params[i], this));
}

InvokeInst::InvokeInst(Value *Func, BasicBlock *IfNormal,
		       BasicBlock *IfException,
                       const std::vector<Value*> &params,
		       const std::string &Name, BasicBlock *InsertAtEnd)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Func->getType())
				    ->getElementType())->getReturnType(),
		   Instruction::Invoke, Name) {
  Operands.reserve(3+params.size());
  Operands.push_back(Use(Func, this));
  Operands.push_back(Use((Value*)IfNormal, this));
  Operands.push_back(Use((Value*)IfException, this));
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  
  const FunctionType::ParamTypes &PL = MTy->getParamTypes();
  assert((params.size() == PL.size()) || 
	 (MTy->isVarArg() && params.size() > PL.size()) &&
	 "Calling a function with bad signature");
  
  for (unsigned i = 0; i < params.size(); i++)
    Operands.push_back(Use(params[i], this));

  if (InsertAtEnd)
    InsertAtEnd->getInstList().push_back(this);
}

InvokeInst::InvokeInst(const InvokeInst &CI) 
  : TerminatorInst(CI.getType(), Instruction::Invoke) {
  Operands.reserve(CI.Operands.size());
  for (unsigned i = 0; i < CI.Operands.size(); ++i)
    Operands.push_back(Use(CI.Operands[i], this));
}

const Function *InvokeInst::getCalledFunction() const {
  if (const Function *F = dyn_cast<Function>(Operands[0]))
    return F;
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Operands[0]))
    return cast<Function>(CPR->getValue());
  return 0;
}
Function *InvokeInst::getCalledFunction() {
  if (Function *F = dyn_cast<Function>(Operands[0]))
    return F;
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Operands[0]))
    return cast<Function>(CPR->getValue());
  return 0;
}

Function *CallSite::getCalledFunction() const {
  Value *Callee = getCalledValue();
  if (Function *F = dyn_cast<Function>(Callee))
    return F;
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Callee))
    return cast<Function>(CPR->getValue());
  return 0;
}
