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

#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "systemz-shorten-inst"

namespace {
class SystemZShortenInst : public MachineFunctionPass {
public:
  static char ID;
  SystemZShortenInst(const SystemZTargetMachine &tm);

  const char *getPassName() const override {
    return "SystemZ Instruction Shortening";
  }

  bool processBlock(MachineBasicBlock &MBB);
  bool runOnMachineFunction(MachineFunction &F) override;

private:
  bool shortenIIF(MachineInstr &MI, unsigned *GPRMap, unsigned LiveOther,
                  unsigned LLIxL, unsigned LLIxH);
  bool shortenOn0(MachineInstr &MI, unsigned Opcode);
  bool shortenOn01(MachineInstr &MI, unsigned Opcode);
  bool shortenOn001(MachineInstr &MI, unsigned Opcode);
  bool shortenFPConv(MachineInstr &MI, unsigned Opcode);

  const SystemZInstrInfo *TII;

  // LowGPRs[I] has bit N set if LLVM register I includes the low
  // word of GPR N.  HighGPRs is the same for the high word.
  unsigned LowGPRs[SystemZ::NUM_TARGET_REGS];
  unsigned HighGPRs[SystemZ::NUM_TARGET_REGS];
};

char SystemZShortenInst::ID = 0;
} // end anonymous namespace

FunctionPass *llvm::createSystemZShortenInstPass(SystemZTargetMachine &TM) {
  return new SystemZShortenInst(TM);
}

SystemZShortenInst::SystemZShortenInst(const SystemZTargetMachine &tm)
  : MachineFunctionPass(ID), TII(nullptr), LowGPRs(), HighGPRs() {
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

// Change MI's opcode to Opcode if register operand 0 has a 4-bit encoding.
bool SystemZShortenInst::shortenOn0(MachineInstr &MI, unsigned Opcode) {
  if (SystemZMC::getFirstReg(MI.getOperand(0).getReg()) < 16) {
    MI.setDesc(TII->get(Opcode));
    return true;
  }
  return false;
}

// Change MI's opcode to Opcode if register operands 0 and 1 have a
// 4-bit encoding.
bool SystemZShortenInst::shortenOn01(MachineInstr &MI, unsigned Opcode) {
  if (SystemZMC::getFirstReg(MI.getOperand(0).getReg()) < 16 &&
      SystemZMC::getFirstReg(MI.getOperand(1).getReg()) < 16) {
    MI.setDesc(TII->get(Opcode));
    return true;
  }
  return false;
}

// Change MI's opcode to Opcode if register operands 0, 1 and 2 have a
// 4-bit encoding and if operands 0 and 1 are tied.
bool SystemZShortenInst::shortenOn001(MachineInstr &MI, unsigned Opcode) {
  if (SystemZMC::getFirstReg(MI.getOperand(0).getReg()) < 16 &&
      MI.getOperand(1).getReg() == MI.getOperand(0).getReg() &&
      SystemZMC::getFirstReg(MI.getOperand(2).getReg()) < 16) {
    MI.setDesc(TII->get(Opcode));
    return true;
  }
  return false;
}

// MI is a vector-style conversion instruction with the operand order:
// destination, source, exact-suppress, rounding-mode.  If both registers
// have a 4-bit encoding then change it to Opcode, which has operand order:
// destination, rouding-mode, source, exact-suppress.
bool SystemZShortenInst::shortenFPConv(MachineInstr &MI, unsigned Opcode) {
  if (SystemZMC::getFirstReg(MI.getOperand(0).getReg()) < 16 &&
      SystemZMC::getFirstReg(MI.getOperand(1).getReg()) < 16) {
    MachineOperand Dest(MI.getOperand(0));
    MachineOperand Src(MI.getOperand(1));
    MachineOperand Suppress(MI.getOperand(2));
    MachineOperand Mode(MI.getOperand(3));
    MI.RemoveOperand(3);
    MI.RemoveOperand(2);
    MI.RemoveOperand(1);
    MI.RemoveOperand(0);
    MI.setDesc(TII->get(Opcode));
    MachineInstrBuilder(*MI.getParent()->getParent(), &MI)
      .addOperand(Dest)
      .addOperand(Mode)
      .addOperand(Src)
      .addOperand(Suppress);
    return true;
  }
  return false;
}

// Process all instructions in MBB.  Return true if something changed.
bool SystemZShortenInst::processBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  // Work out which words are live on exit from the block.
  unsigned LiveLow = 0;
  unsigned LiveHigh = 0;
  for (auto SI = MBB.succ_begin(), SE = MBB.succ_end(); SI != SE; ++SI) {
    for (const auto &LI : (*SI)->liveins()) {
      unsigned Reg = LI.PhysReg;
      assert(Reg < SystemZ::NUM_TARGET_REGS && "Invalid register number");
      LiveLow |= LowGPRs[Reg];
      LiveHigh |= HighGPRs[Reg];
    }
  }

  // Iterate backwards through the block looking for instructions to change.
  for (auto MBBI = MBB.rbegin(), MBBE = MBB.rend(); MBBI != MBBE; ++MBBI) {
    MachineInstr &MI = *MBBI;
    switch (MI.getOpcode()) {
    case SystemZ::IILF:
      Changed |= shortenIIF(MI, LowGPRs, LiveHigh, SystemZ::LLILL,
                            SystemZ::LLILH);
      break;

    case SystemZ::IIHF:
      Changed |= shortenIIF(MI, HighGPRs, LiveLow, SystemZ::LLIHL,
                            SystemZ::LLIHH);
      break;

    case SystemZ::WFADB:
      Changed |= shortenOn001(MI, SystemZ::ADBR);
      break;

    case SystemZ::WFDDB:
      Changed |= shortenOn001(MI, SystemZ::DDBR);
      break;

    case SystemZ::WFIDB:
      Changed |= shortenFPConv(MI, SystemZ::FIDBRA);
      break;

    case SystemZ::WLDEB:
      Changed |= shortenOn01(MI, SystemZ::LDEBR);
      break;

    case SystemZ::WLEDB:
      Changed |= shortenFPConv(MI, SystemZ::LEDBRA);
      break;

    case SystemZ::WFMDB:
      Changed |= shortenOn001(MI, SystemZ::MDBR);
      break;

    case SystemZ::WFLCDB:
      Changed |= shortenOn01(MI, SystemZ::LCDFR);
      break;

    case SystemZ::WFLNDB:
      Changed |= shortenOn01(MI, SystemZ::LNDFR);
      break;

    case SystemZ::WFLPDB:
      Changed |= shortenOn01(MI, SystemZ::LPDFR);
      break;

    case SystemZ::WFSQDB:
      Changed |= shortenOn01(MI, SystemZ::SQDBR);
      break;

    case SystemZ::WFSDB:
      Changed |= shortenOn001(MI, SystemZ::SDBR);
      break;

    case SystemZ::WFCDB:
      Changed |= shortenOn01(MI, SystemZ::CDBR);
      break;

    case SystemZ::VL32:
      // For z13 we prefer LDE over LE to avoid partial register dependencies.
      Changed |= shortenOn0(MI, SystemZ::LDE32);
      break;

    case SystemZ::VST32:
      Changed |= shortenOn0(MI, SystemZ::STE);
      break;

    case SystemZ::VL64:
      Changed |= shortenOn0(MI, SystemZ::LD);
      break;

    case SystemZ::VST64:
      Changed |= shortenOn0(MI, SystemZ::STD);
      break;
    }

    unsigned UsedLow = 0;
    unsigned UsedHigh = 0;
    for (auto MOI = MI.operands_begin(), MOE = MI.operands_end();
         MOI != MOE; ++MOI) {
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
  TII = static_cast<const SystemZInstrInfo *>(F.getSubtarget().getInstrInfo());

  bool Changed = false;
  for (auto &MBB : F)
    Changed |= processBlock(MBB);

  return Changed;
}
