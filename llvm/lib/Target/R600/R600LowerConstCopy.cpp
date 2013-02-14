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

  struct ConstPairs {
    unsigned XYPair;
    unsigned ZWPair;
  };

  bool canFoldInBundle(ConstPairs &UsedConst, unsigned ReadConst) const;
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

bool R600LowerConstCopy::canFoldInBundle(ConstPairs &UsedConst,
    unsigned ReadConst) const {
  unsigned ReadConstChan = ReadConst & 3;
  unsigned ReadConstIndex = ReadConst & (~3);
  if (ReadConstChan < 2) {
    if (!UsedConst.XYPair) {
      UsedConst.XYPair = ReadConstIndex;
    }
    return UsedConst.XYPair == ReadConstIndex;
  } else {
    if (!UsedConst.ZWPair) {
      UsedConst.ZWPair = ReadConstIndex;
    }
    return UsedConst.ZWPair == ReadConstIndex;
  }
}

static bool isControlFlow(const MachineInstr &MI) {
  return (MI.getOpcode() == AMDGPU::IF_PREDICATE_SET) ||
  (MI.getOpcode() == AMDGPU::ENDIF) ||
  (MI.getOpcode() == AMDGPU::ELSE) ||
  (MI.getOpcode() == AMDGPU::WHILELOOP) ||
  (MI.getOpcode() == AMDGPU::BREAK);
}

bool R600LowerConstCopy::runOnMachineFunction(MachineFunction &MF) {

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    DenseMap<unsigned, MachineInstr *> RegToConstIndex;
    for (MachineBasicBlock::instr_iterator I = MBB.instr_begin(),
        E = MBB.instr_end(); I != E;) {

      if (I->getOpcode() == AMDGPU::CONST_COPY) {
        MachineInstr &MI = *I;
        I = llvm::next(I);
        unsigned DstReg = MI.getOperand(0).getReg();
        DenseMap<unsigned, MachineInstr *>::iterator SrcMI =
            RegToConstIndex.find(DstReg);
        if (SrcMI != RegToConstIndex.end()) {
          SrcMI->second->eraseFromParent();
          RegToConstIndex.erase(SrcMI);
        }
        MachineInstr *NewMI = 
            TII->buildDefaultInstruction(MBB, &MI, AMDGPU::MOV,
            MI.getOperand(0).getReg(), AMDGPU::ALU_CONST);
        TII->setImmOperand(NewMI, R600Operands::SRC0_SEL,
            MI.getOperand(1).getImm());
        RegToConstIndex[DstReg] = NewMI;
        MI.eraseFromParent();
        continue;
      }

      std::vector<unsigned> Defs;
      // We consider all Instructions as bundled because algorithm that  handle
      // const read port limitations inside an IG is still valid with single
      // instructions.
      std::vector<MachineInstr *> Bundle;

      if (I->isBundle()) {
        unsigned BundleSize = I->getBundleSize();
        for (unsigned i = 0; i < BundleSize; i++) {
          I = llvm::next(I);
          Bundle.push_back(I);
        }
      } else if (TII->isALUInstr(I->getOpcode())){
        Bundle.push_back(I);
      } else if (isControlFlow(*I)) {
          RegToConstIndex.clear();
          I = llvm::next(I);
          continue;
      } else {
        MachineInstr &MI = *I;
        for (MachineInstr::mop_iterator MOp = MI.operands_begin(),
            MOpE = MI.operands_end(); MOp != MOpE; ++MOp) {
          MachineOperand &MO = *MOp;
          if (!MO.isReg())
            continue;
          if (MO.isDef()) {
            Defs.push_back(MO.getReg());
          } else {
            // Either a TEX or an Export inst, prevent from erasing def of used
            // operand
            RegToConstIndex.erase(MO.getReg());
            for (MCSubRegIterator SR(MO.getReg(), &TII->getRegisterInfo());
                SR.isValid(); ++SR) {
              RegToConstIndex.erase(*SR);
            }
          }
        }
      }


      R600Operands::Ops OpTable[3][2] = {
        {R600Operands::SRC0, R600Operands::SRC0_SEL},
        {R600Operands::SRC1, R600Operands::SRC1_SEL},
        {R600Operands::SRC2, R600Operands::SRC2_SEL},
      };

      for(std::vector<MachineInstr *>::iterator It = Bundle.begin(),
          ItE = Bundle.end(); It != ItE; ++It) {
        MachineInstr *MI = *It;
        if (TII->isPredicated(MI)) {
          // We don't want to erase previous assignment
          RegToConstIndex.erase(MI->getOperand(0).getReg());
        } else {
          int WriteIDX = TII->getOperandIdx(MI->getOpcode(), R600Operands::WRITE);
          if (WriteIDX < 0 || MI->getOperand(WriteIDX).getImm())
            Defs.push_back(MI->getOperand(0).getReg());
        }
      }

      ConstPairs CP = {0,0};
      for (unsigned SrcOp = 0; SrcOp < 3; SrcOp++) {
        for(std::vector<MachineInstr *>::iterator It = Bundle.begin(),
            ItE = Bundle.end(); It != ItE; ++It) {
          MachineInstr *MI = *It;
          int SrcIdx = TII->getOperandIdx(MI->getOpcode(), OpTable[SrcOp][0]);
          if (SrcIdx < 0)
            continue;
          MachineOperand &MO = MI->getOperand(SrcIdx);
          DenseMap<unsigned, MachineInstr *>::iterator SrcMI =
              RegToConstIndex.find(MO.getReg());
          if (SrcMI != RegToConstIndex.end()) {
            MachineInstr *CstMov = SrcMI->second;
            int ConstMovSel =
                TII->getOperandIdx(CstMov->getOpcode(), R600Operands::SRC0_SEL);
            unsigned ConstIndex = CstMov->getOperand(ConstMovSel).getImm();
            if (MI->isInsideBundle() && canFoldInBundle(CP, ConstIndex)) {
              TII->setImmOperand(MI, OpTable[SrcOp][1], ConstIndex);
              MI->getOperand(SrcIdx).setReg(AMDGPU::ALU_CONST);
            } else {
              RegToConstIndex.erase(SrcMI);
            }
          }
        }
      }

      for (std::vector<unsigned>::iterator It = Defs.begin(), ItE = Defs.end();
          It != ItE; ++It) {
        DenseMap<unsigned, MachineInstr *>::iterator SrcMI =
            RegToConstIndex.find(*It);
        if (SrcMI != RegToConstIndex.end()) {
          SrcMI->second->eraseFromParent();
          RegToConstIndex.erase(SrcMI);
        }
      }
      I = llvm::next(I);
    }

    if (MBB.succ_empty()) {
      for (DenseMap<unsigned, MachineInstr *>::iterator
          DI = RegToConstIndex.begin(), DE = RegToConstIndex.end();
          DI != DE; ++DI) {
        DI->second->eraseFromParent();
      }
    }
  }
  return false;
}

FunctionPass *createR600LowerConstCopy(TargetMachine &tm) {
  return new R600LowerConstCopy(tm);
}

}


