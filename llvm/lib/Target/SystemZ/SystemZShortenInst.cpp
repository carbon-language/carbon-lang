//===-- SystemZShortenInst.cpp - Instruction-shortening pass --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass tries to replace instructions with shorter forms.  For example,
// IILF can be replaced with LLILL or LLILH if the constant fits and if the
// other 32 bits of the GR64 destination are not live.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-shorten-inst"

#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

namespace {
  class SystemZShortenInst : public MachineFunctionPass {
  public:
    static char ID;
    SystemZShortenInst(const SystemZTargetMachine &tm);

    virtual const char *getPassName() const {
      return "SystemZ Instruction Shortening";
    }

    bool processBlock(MachineBasicBlock *MBB);
    bool runOnMachineFunction(MachineFunction &F);

  private:
    bool shortenIIF(MachineInstr &MI, unsigned *GPRMap, unsigned LiveOther,
                    unsigned LLIxL, unsigned LLIxH);

    const SystemZInstrInfo *TII;

    // LowGPRs[I] has bit N set if LLVM register I includes the low
    // word of GPR N.  HighGPRs is the same for the high word.
    unsigned LowGPRs[SystemZ::NUM_TARGET_REGS];
    unsigned HighGPRs[SystemZ::NUM_TARGET_REGS];
  };

  char SystemZShortenInst::ID = 0;
} // end of anonymous namespace

FunctionPass *llvm::createSystemZShortenInstPass(SystemZTargetMachine &TM) {
  return new SystemZShortenInst(TM);
}

SystemZShortenInst::SystemZShortenInst(const SystemZTargetMachine &tm)
  : MachineFunctionPass(ID), TII(0), LowGPRs(), HighGPRs() {
  // Set up LowGPRs and HighGPRs.
  for (unsigned I = 0; I < 16; ++I) {
    LowGPRs[SystemZMC::GR32Regs[I]] |= 1 << I;
    LowGPRs[SystemZMC::GR64Regs[I]] |= 1 << I;
    HighGPRs[SystemZMC::GRH32Regs[I]] |= 1 << I;
    HighGPRs[SystemZMC::GR64Regs[I]] |= 1 << I;
    if (unsigned GR128 = SystemZMC::GR128Regs[I]) {
      LowGPRs[GR128] |= 3 << I;
      HighGPRs[GR128] |= 3 << I;
    }
  }
}

// MI loads one word of a GPR using an IIxF instruction and LLIxL and LLIxH
// are the halfword immediate loads for the same word.  Try to use one of them
// instead of IIxF.  If MI loads the high word, GPRMap[X] is the set of high
// words referenced by LLVM register X while LiveOther is the mask of low
// words that are currently live, and vice versa.
bool SystemZShortenInst::shortenIIF(MachineInstr &MI, unsigned *GPRMap,
                                    unsigned LiveOther, unsigned LLIxL,
                                    unsigned LLIxH) {
  unsigned Reg = MI.getOperand(0).getReg();
  assert(Reg < SystemZ::NUM_TARGET_REGS && "Invalid register number");
  unsigned GPRs = GPRMap[Reg];
  assert(GPRs != 0 && "Register must be a GPR");
  if (GPRs & LiveOther)
    return false;

  uint64_t Imm = MI.getOperand(1).getImm();
  if (SystemZ::isImmLL(Imm)) {
    MI.setDesc(TII->get(LLIxL));
    MI.getOperand(0).setReg(SystemZMC::getRegAsGR64(Reg));
    return true;
  }
  if (SystemZ::isImmLH(Imm)) {
    MI.setDesc(TII->get(LLIxH));
    MI.getOperand(0).setReg(SystemZMC::getRegAsGR64(Reg));
    MI.getOperand(1).setImm(Imm >> 16);
    return true;
  }
  return false;
}

// Process all instructions in MBB.  Return true if something changed.
bool SystemZShortenInst::processBlock(MachineBasicBlock *MBB) {
  bool Changed = false;

  // Work out which words are live on exit from the block.
  unsigned LiveLow = 0;
  unsigned LiveHigh = 0;
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    for (MachineBasicBlock::livein_iterator LI = (*SI)->livein_begin(),
           LE = (*SI)->livein_end(); LI != LE; ++LI) {
      unsigned Reg = *LI;
      assert(Reg < SystemZ::NUM_TARGET_REGS && "Invalid register number");
      LiveLow |= LowGPRs[Reg];
      LiveHigh |= HighGPRs[Reg];
    }
  }

  // Iterate backwards through the block looking for instructions to change.
  for (MachineBasicBlock::reverse_iterator MBBI = MBB->rbegin(),
         MBBE = MBB->rend(); MBBI != MBBE; ++MBBI) {
    MachineInstr &MI = *MBBI;
    unsigned Opcode = MI.getOpcode();
    if (Opcode == SystemZ::IILF)
      Changed |= shortenIIF(MI, LowGPRs, LiveHigh, SystemZ::LLILL,
                            SystemZ::LLILH);
    else if (Opcode == SystemZ::IIHF)
      Changed |= shortenIIF(MI, HighGPRs, LiveLow, SystemZ::LLIHL,
                            SystemZ::LLIHH);
    unsigned UsedLow = 0;
    unsigned UsedHigh = 0;
    for (MachineInstr::mop_iterator MOI = MI.operands_begin(),
           MOE = MI.operands_end(); MOI != MOE; ++MOI) {
      MachineOperand &MO = *MOI;
      if (MO.isReg()) {
        if (unsigned Reg = MO.getReg()) {
          assert(Reg < SystemZ::NUM_TARGET_REGS && "Invalid register number");
          if (MO.isDef()) {
            LiveLow &= ~LowGPRs[Reg];
            LiveHigh &= ~HighGPRs[Reg];
          } else if (!MO.isUndef()) {
            UsedLow |= LowGPRs[Reg];
            UsedHigh |= HighGPRs[Reg];
          }
        }
      }
    }
    LiveLow |= UsedLow;
    LiveHigh |= UsedHigh;
  }

  return Changed;
}

bool SystemZShortenInst::runOnMachineFunction(MachineFunction &F) {
  TII = static_cast<const SystemZInstrInfo *>(F.getTarget().getInstrInfo());

  bool Changed = false;
  for (MachineFunction::iterator MFI = F.begin(), MFE = F.end();
       MFI != MFE; ++MFI)
    Changed |= processBlock(MFI);

  return Changed;
}
