//===-- llvm/CodeGen/GlobalISel/IRTranslator.cpp - IRTranslator --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the IRTranslator class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/IRTranslator.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#define DEBUG_TYPE "irtranslator"

using namespace llvm;

char IRTranslator::ID = 0;

IRTranslator::IRTranslator() : MachineFunctionPass(ID), MRI(nullptr) {
}

const VRegsSequence &IRTranslator::getOrCreateVRegs(const Value *Val) {
  VRegsSequence &ValRegSequence = ValToVRegs[Val];
  // Check if this is the first time we see Val.
  if (ValRegSequence.empty()) {
    // Fill ValRegsSequence with the sequence of registers
    // we need to concat together to produce the value.
    assert(Val->getType()->isSized() &&
           "Don't know how to create an empty vreg");
    assert(!Val->getType()->isAggregateType() && "Not yet implemented");
    unsigned Size = Val->getType()->getPrimitiveSizeInBits();
    unsigned VReg = MRI->createGenericVirtualRegister(Size);
    ValRegSequence.push_back(VReg);
    assert(!isa<Constant>(Val) && "Not yet implemented");
  }
  assert(ValRegSequence.size() == 1 &&
         "We support only one vreg per value at the moment");
  return ValRegSequence;
}

MachineBasicBlock &IRTranslator::getOrCreateBB(const BasicBlock *BB) {
  MachineBasicBlock *&MBB = BBToMBB[BB];
  if (!MBB) {
    MachineFunction &MF = MIRBuilder.getMF();
    MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);
  }
  return *MBB;
}

bool IRTranslator::translateADD(const Instruction &Inst) {
  // Get or create a virtual register for each value.
  // Unless the value is a Constant => loadimm cst?
  // or inline constant each time?
  // Creation of a virtual register needs to have a size.
  unsigned Op0 = *getOrCreateVRegs(Inst.getOperand(0)).begin();
  unsigned Op1 = *getOrCreateVRegs(Inst.getOperand(1)).begin();
  unsigned Res = *getOrCreateVRegs(&Inst).begin();
  MIRBuilder.buildInstr(TargetOpcode::G_ADD, Res, Op0, Op1);
  return true;
}

bool IRTranslator::translate(const Instruction &Inst) {
  MIRBuilder.setDebugLoc(Inst.getDebugLoc());
  switch(Inst.getOpcode()) {
    case Instruction::Add: {
      return translateADD(Inst);
    default:
      llvm_unreachable("Opcode not supported");
    }
  }
}


void IRTranslator::finalize() {
  // Release the memory used by the different maps we
  // needed during the translation.
  ValToVRegs.clear();
  Constants.clear();
}

bool IRTranslator::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = *MF.getFunction();
  MIRBuilder.setFunction(MF);
  MRI = &MF.getRegInfo();
  for (const BasicBlock &BB: F) {
    MachineBasicBlock &MBB = getOrCreateBB(&BB);
    MIRBuilder.setBasicBlock(MBB);
    for (const Instruction &Inst: BB) {
      bool Succeeded = translate(Inst);
      if (!Succeeded) {
        DEBUG(dbgs() << "Cannot translate: " << Inst << '\n');
        report_fatal_error("Unable to translate instruction");
      }
    }
  }
  return false;
}
