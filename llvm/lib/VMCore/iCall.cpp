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

void CallInst::init(Value *Func, const std::vector<Value*> &Params)
{
  Operands.reserve(1+Params.size());
  Operands.push_back(Use(Func, this));

  const FunctionType *FTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((Params.size() == FTy->getNumParams() || 
          (FTy->isVarArg() && Params.size() > FTy->getNumParams())) &&
	 "Calling a function with bad signature");
  for (unsigned i = 0; i != Params.size(); i++)
    Operands.push_back(Use(Params[i], this));
}

void CallInst::init(Value *Func, Value *Actual)
{
  Operands.reserve(2);
  Operands.push_back(Use(Func, this));
  
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((MTy->getNumParams() == 1 ||
          (MTy->isVarArg() && MTy->getNumParams() == 0)) &&
	 "Calling a function with bad signature");
  Operands.push_back(Use(Actual, this));
}

void CallInst::init(Value *Func)
{
  Operands.reserve(1);
  Operands.push_back(Use(Func, this));
  
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert(MTy->getNumParams() == 0 && "Calling a function with bad signature");
}

CallInst::CallInst(Value *Func, const std::vector<Value*> &Params, 
                   const std::string &Name, Instruction *InsertBefore) 
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
				 ->getElementType())->getReturnType(),
		Instruction::Call, Name, InsertBefore) {
  init(Func, Params);
}

CallInst::CallInst(Value *Func, const std::vector<Value*> &Params, 
                   const std::string &Name, BasicBlock *InsertAtEnd) 
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
				 ->getElementType())->getReturnType(),
		Instruction::Call, Name, InsertAtEnd) {
  init(Func, Params);
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   Instruction  *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  init(Func, Actual);
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   BasicBlock  *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertAtEnd) {
  init(Func, Actual);
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  init(Func);
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   BasicBlock *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertAtEnd) {
  init(Func);
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

void InvokeInst::init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
                      const std::vector<Value*> &Params)
{
  Operands.reserve(3+Params.size());
  Operands.push_back(Use(Fn, this));
  Operands.push_back(Use((Value*)IfNormal, this));
  Operands.push_back(Use((Value*)IfException, this));
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Fn->getType())->getElementType());
  
  assert((Params.size() == MTy->getNumParams()) || 
	 (MTy->isVarArg() && Params.size() > MTy->getNumParams()) &&
	 "Calling a function with bad signature");
  
  for (unsigned i = 0; i < Params.size(); i++)
    Operands.push_back(Use(Params[i], this));
}

InvokeInst::InvokeInst(Value *Fn, BasicBlock *IfNormal,
		       BasicBlock *IfException,
                       const std::vector<Value*> &Params,
		       const std::string &Name, Instruction *InsertBefore)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Fn->getType())
				    ->getElementType())->getReturnType(),
		   Instruction::Invoke, Name, InsertBefore) {
  init(Fn, IfNormal, IfException, Params);
}

InvokeInst::InvokeInst(Value *Fn, BasicBlock *IfNormal,
		       BasicBlock *IfException,
                       const std::vector<Value*> &Params,
		       const std::string &Name, BasicBlock *InsertAtEnd)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Fn->getType())
				    ->getElementType())->getReturnType(),
		   Instruction::Invoke, Name, InsertAtEnd) {
  init(Fn, IfNormal, IfException, Params);
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
