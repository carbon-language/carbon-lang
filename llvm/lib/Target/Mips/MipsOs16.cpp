//===---- MipsOs16.cpp for Mips Option -Os16                         --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an optimization phase for the MIPS target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-os16"
#include "MipsOs16.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace {

  // Figure out if we need float point based on the function signature.
  // We need to move variables in and/or out of floating point
  // registers because of the ABI
  //
  bool needsFPFromSig(Function &F) {
    Type* RetType = F.getReturnType();
    switch (RetType->getTypeID()) {
    case Type::FloatTyID:
    case Type::DoubleTyID:
      return true;
    default:
      ;
    }
    if (F.arg_size() >=1) {
      Argument &Arg = F.getArgumentList().front();
      switch (Arg.getType()->getTypeID()) {
        case Type::FloatTyID:
        case Type::DoubleTyID:
          return true;
        default:
          ;
      }
    }
    return false;
  }

  // Figure out if the function will need floating point operations
  //
  bool needsFP(Function &F) {
    if (needsFPFromSig(F))
      return true;
    for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
        const Instruction &Inst = *I;
        switch (Inst.getOpcode()) {
        case Instruction::FAdd:
        case Instruction::FSub:
        case Instruction::FMul:
        case Instruction::FDiv:
        case Instruction::FRem:
        case Instruction::FPToUI:
        case Instruction::FPToSI:
        case Instruction::UIToFP:
        case Instruction::SIToFP:
        case Instruction::FPTrunc:
        case Instruction::FPExt:
        case Instruction::FCmp:
          return true;
        default:
          ;
        }
        if (const CallInst *CI = dyn_cast<CallInst>(I)) {
          DEBUG(dbgs() << "Working on call" << "\n");
          Function &F_ =  *CI->getCalledFunction();
          if (needsFPFromSig(F_))
            return true;
        }
      }
    return false;
  }
}
namespace llvm {


bool MipsOs16::runOnModule(Module &M) {
  DEBUG(errs() << "Run on Module MipsOs16\n");
  bool modified = false;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    DEBUG(dbgs() << "Working on " << F->getName() << "\n");
    if (needsFP(*F)) {
      DEBUG(dbgs() << " need to compile as nomips16 \n");
      F->addFnAttr("nomips16");
    }
    else {
      F->addFnAttr("mips16");
      DEBUG(dbgs() << " no need to compile as nomips16 \n");
    }
  }
  return modified;
}

char MipsOs16::ID = 0;

}

ModulePass *llvm::createMipsOs16(MipsTargetMachine &TM) {
  return new MipsOs16;
}


