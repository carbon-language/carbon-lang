//===-- R600LowerConstCopy.cpp - Propagate ConstCopy / lower them to MOV---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass is intended to handle remaining ConstCopy pseudo MachineInstr.
/// ISel will fold each Const Buffer read inside scalar ALU. However it cannot
/// fold them inside vector instruction, like DOT4 or Cube ; ISel emits
/// ConstCopy instead. This pass (executed after ExpandingSpecialInstr) will try
/// to fold them if possible or replace them by MOV otherwise.
/// TODO : Implement the folding part, using Copy Propagation algorithm.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "R600InstrInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/GlobalValue.h"

namespace llvm {

class R600LowerConstCopy : public MachineFunctionPass {
private:
  static char ID;
  const R600InstrInfo *TII;
public:
  R600LowerConstCopy(TargetMachine &tm);
  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const { return "R600 Eliminate Symbolic Operand"; }
};

char R600LowerConstCopy::ID = 0;


R600LowerConstCopy::R600LowerConstCopy(TargetMachine &tm) :
    MachineFunctionPass(ID),
    TII (static_cast<const R600InstrInfo *>(tm.getInstrInfo()))
{
}

bool R600LowerConstCopy::runOnMachineFunction(MachineFunction &MF) {
  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                      I != E;) {
      MachineInstr &MI = *I;
      I = llvm::next(I);
      if (MI.getOpcode() != AMDGPU::CONST_COPY)
        continue;
      MachineInstr *NewMI = TII->buildDefaultInstruction(MBB, I, AMDGPU::MOV,
          MI.getOperand(0).getReg(), AMDGPU::ALU_CONST);
      NewMI->getOperand(9).setImm(MI.getOperand(1).getImm());
      MI.eraseFromParent();
    }
  }
  return false;
}

FunctionPass *createR600LowerConstCopy(TargetMachine &tm) {
  return new R600LowerConstCopy(tm);
}

}


