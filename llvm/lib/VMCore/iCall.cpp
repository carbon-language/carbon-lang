//===-- iCall.cpp - Implement the call & invoke instructions -----*- C++ -*--=//
//
// This file implements the call and invoke instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"

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

InvokeInst::InvokeInst(const InvokeInst &CI) 
  : TerminatorInst(CI.getType(), Instruction::Invoke) {
  Operands.reserve(CI.Operands.size());
  for (unsigned i = 0; i < CI.Operands.size(); ++i)
    Operands.push_back(Use(CI.Operands[i], this));
}

