//===-- InstrinsicInst.cpp - Intrinsic Instruction Wrappers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IntrinsicInst.h"

#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
/// DbgInfoIntrinsic - This is the common base class for debug info intrinsics
///

static Value *CastOperand(Value *C) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (CE->getOpcode() == Instruction::Cast)
      return CE->getOperand(0);
  return NULL;
}

Value *DbgInfoIntrinsic::StripCast(Value *C) {
  if (Value *CO = CastOperand(C)) {
      return StripCast(CO);
  } else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    if (GV->hasInitializer())
      if (Value *CO = CastOperand(GV->getInitializer()))
        return StripCast(CO);
  }
  return C;
}

//===----------------------------------------------------------------------===//
/// DbgStopPointInst - This represents the llvm.dbg.stoppoint instruction.
///

std::string DbgStopPointInst::getFileName() const {
  GlobalVariable *GV = cast<GlobalVariable>(getContext());
  ConstantStruct *CS = cast<ConstantStruct>(GV->getInitializer());
  return CS->getOperand(4)->getStringValue();
}

std::string DbgStopPointInst::getDirectory() const {
  GlobalVariable *GV = cast<GlobalVariable>(getContext());
  ConstantStruct *CS = cast<ConstantStruct>(GV->getInitializer());
  return CS->getOperand(5)->getStringValue();
}

//===----------------------------------------------------------------------===//
