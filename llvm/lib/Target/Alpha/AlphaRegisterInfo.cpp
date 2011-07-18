//===- AlphaRegisterInfo.cpp - Alpha Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reginfo"
#include "Alpha.h"
#include "AlphaRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>

#define GET_REGINFO_TARGET_DESC
#include "AlphaGenRegisterInfo.inc"

using namespace llvm;

AlphaRegisterInfo::AlphaRegisterInfo(const TargetInstrInfo &tii)
  : AlphaGenRegisterInfo(Alpha::R26), TII(tii) {
}

static long getUpper16(long l) {
  long y = l / Alpha::IMM_MULT;
  if (l % Alpha::IMM_MULT > Alpha::IMM_HIGH)
    ++y;
  return y;
}

static long getLower16(long l) {
  long h = getUpper16(l);
  return l - h * Alpha::IMM_MULT;
}

const unsigned* AlphaRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF)
                                                                         const {
  static const unsigned CalleeSavedRegs[] = {
    Alpha::R9, Alpha::R10,
    Alpha::R11, Alpha::R12,
    Alpha::R13, Alpha::R14,
    Alpha::F2, Alpha::F3,
    Alpha::F4, Alpha::F5,
    Alpha::F6, Alpha::F7,
    Alpha::F8, Alpha::F9,  0
  };
  return CalleeSavedRegs;
}

BitVector AlphaRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(Alpha::R15);
  Reserved.set(Alpha::R29);
  Reserved.set(Alpha::R30);
  Reserved.set(Alpha::R31);
  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

void AlphaRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (TFI->hasFP(MF)) {
    // If we have a frame pointer, turn the adjcallstackup instruction into a
    // 'sub ESP, <amt>' and the adjcallstackdown instruction into 'add ESP,
    // <amt>'
    MachineInstr *Old = I;
    uint64_t Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = TFI->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      MachineInstr *New;
      if (Old->getOpcode() == Alpha::ADJUSTSTACKDOWN) {
        New=BuildMI(MF, Old->getDebugLoc(), TII.get(Alpha::LDA), Alpha::R30)
          .addImm(-Amount).addReg(Alpha::R30);
      } else {
         assert(Old->getOpcode() == Alpha::ADJUSTSTACKUP);
         New=BuildMI(MF, Old->getDebugLoc(), TII.get(Alpha::LDA), Alpha::R30)
          .addImm(Amount).addReg(Alpha::R30);
      }

      // Replace the pseudo instruction with a new instruction...
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

//Alpha has a slightly funny stack:
//Args
//<- incoming SP
//fixed locals (and spills, callee saved, etc)
//<- FP
//variable locals
//<- SP

void
AlphaRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                       int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  bool FP = TFI->hasFP(MF);

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  // Add the base register of R30 (SP) or R15 (FP).
  MI.getOperand(i + 1).ChangeToRegister(FP ? Alpha::R15 : Alpha::R30, false);

  // Now add the frame object offset to the offset from the virtual frame index.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  DEBUG(errs() << "FI: " << FrameIndex << " Offset: " << Offset << "\n");

  Offset += MF.getFrameInfo()->getStackSize();

  DEBUG(errs() << "Corrected Offset " << Offset
       << " for stack size: " << MF.getFrameInfo()->getStackSize() << "\n");

  if (Offset > Alpha::IMM_HIGH || Offset < Alpha::IMM_LOW) {
    DEBUG(errs() << "Unconditionally using R28 for evil purposes Offset: "
          << Offset << "\n");
    //so in this case, we need to use a temporary register, and move the
    //original inst off the SP/FP
    //fix up the old:
    MI.getOperand(i + 1).ChangeToRegister(Alpha::R28, false);
    MI.getOperand(i).ChangeToImmediate(getLower16(Offset));
    //insert the new
    MachineInstr* nMI=BuildMI(MF, MI.getDebugLoc(),
                              TII.get(Alpha::LDAH), Alpha::R28)
      .addImm(getUpper16(Offset)).addReg(FP ? Alpha::R15 : Alpha::R30);
    MBB.insert(II, nMI);
  } else {
    MI.getOperand(i).ChangeToImmediate(Offset);
  }
}

unsigned AlphaRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  return TFI->hasFP(MF) ? Alpha::R15 : Alpha::R30;
}

unsigned AlphaRegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
  return 0;
}

unsigned AlphaRegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
  return 0;
}

std::string AlphaRegisterInfo::getPrettyName(unsigned reg)
{
  std::string s(AlphaRegDesc[reg].Name);
  return s;
}
