//=======- MipsFrameLowering.cpp - Mips Frame Information ------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MipsFrameLowering.h"
#include "MipsInstrInfo.h"
#include "MipsMachineFunction.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;


//===----------------------------------------------------------------------===//
//
// Stack Frame Processing methods
// +----------------------------+
//
// The stack is allocated decrementing the stack pointer on
// the first instruction of a function prologue. Once decremented,
// all stack references are done thought a positive offset
// from the stack/frame pointer, so the stack is considering
// to grow up! Otherwise terrible hacks would have to be made
// to get this stack ABI compliant :)
//
//  The stack frame required by the ABI (after call):
//  Offset
//
//  0                 ----------
//  4                 Args to pass
//  .                 saved $GP  (used in PIC)
//  .                 Alloca allocations
//  .                 Local Area
//  .                 CPU "Callee Saved" Registers
//  .                 saved FP
//  .                 saved RA
//  .                 FPU "Callee Saved" Registers
//  StackSize         -----------
//
// Offset - offset from sp after stack allocation on function prologue
//
// The sp is the stack pointer subtracted/added from the stack size
// at the Prologue/Epilogue
//
// References to the previous stack (to obtain arguments) are done
// with offsets that exceeds the stack size: (stacksize+(4*(num_arg-1))
//
// Examples:
// - reference to the actual stack frame
//   for any local area var there is smt like : FI >= 0, StackOffset: 4
//     sw REGX, 4(SP)
//
// - reference to previous stack frame
//   suppose there's a load to the 5th arguments : FI < 0, StackOffset: 16.
//   The emitted instruction will be something like:
//     lw REGX, 16+StackSize(SP)
//
// Since the total stack size is unknown on LowerFormalArguments, all
// stack references (ObjectOffset) created to reference the function
// arguments, are negative numbers. This way, on eliminateFrameIndex it's
// possible to detect those references and the offsets are adjusted to
// their real location.
//
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool MipsFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return DisableFramePointerElim(MF) || MFI->hasVarSizedObjects();
}

void MipsFrameLowering::adjustMipsStackFrame(MachineFunction &MF) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  unsigned StackAlign = getStackAlignment();
  unsigned RegSize = STI.isGP32bit() ? 4 : 8;
  bool HasGP = MipsFI->needGPSaveRestore();

  // Min and Max CSI FrameIndex.
  int MinCSFI = -1, MaxCSFI = -1;

  // See the description at MipsMachineFunction.h
  int TopCPUSavedRegOff = -1, TopFPUSavedRegOff = -1;

  // Replace the dummy '0' SPOffset by the negative offsets, as explained on
  // LowerFormalArguments. Leaving '0' for while is necessary to avoid the
  // approach done by calculateFrameObjectOffsets to the stack frame.
  MipsFI->adjustLoadArgsFI(MFI);
  MipsFI->adjustStoreVarArgsFI(MFI);

  // It happens that the default stack frame allocation order does not directly
  // map to the convention used for mips. So we must fix it. We move the callee
  // save register slots after the local variables area, as described in the
  // stack frame above.
  unsigned CalleeSavedAreaSize = 0;
  if (!CSI.empty()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size()-1].getFrameIdx();
  }
  for (unsigned i = 0, e = CSI.size(); i != e; ++i)
    CalleeSavedAreaSize += MFI->getObjectAlignment(CSI[i].getFrameIdx());

  unsigned StackOffset = HasGP ? (MipsFI->getGPStackOffset()+RegSize)
                : (STI.isABI_O32() ? 16 : 0);

  // Adjust local variables. They should come on the stack right
  // after the arguments.
  int LastOffsetFI = -1;
  for (int i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (i >= MinCSFI && i <= MaxCSFI)
      continue;
    if (MFI->isDeadObjectIndex(i))
      continue;
    unsigned Offset =
      StackOffset + MFI->getObjectOffset(i) - CalleeSavedAreaSize;
    if (LastOffsetFI == -1)
      LastOffsetFI = i;
    if (Offset > MFI->getObjectOffset(LastOffsetFI))
      LastOffsetFI = i;
    MFI->setObjectOffset(i, Offset);
  }

  // Adjust CPU Callee Saved Registers Area. Registers RA and FP must
  // be saved in this CPU Area. This whole area must be aligned to the
  // default Stack Alignment requirements.
  if (LastOffsetFI >= 0)
    StackOffset = MFI->getObjectOffset(LastOffsetFI)+
                  MFI->getObjectSize(LastOffsetFI);
  StackOffset = ((StackOffset+StackAlign-1)/StackAlign*StackAlign);

  for (unsigned i = 0, e = CSI.size(); i != e ; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (!Mips::CPURegsRegisterClass->contains(Reg))
      break;
    MFI->setObjectOffset(CSI[i].getFrameIdx(), StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += MFI->getObjectAlignment(CSI[i].getFrameIdx());
  }

  // Stack locations for FP and RA. If only one of them is used,
  // the space must be allocated for both, otherwise no space at all.
  if (hasFP(MF) || MFI->adjustsStack()) {
    // FP stack location
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize, true),
                         StackOffset);
    MipsFI->setFPStackOffset(StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += RegSize;

    // SP stack location
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize, true),
                         StackOffset);
    MipsFI->setRAStackOffset(StackOffset);
    StackOffset += RegSize;

    if (MFI->adjustsStack())
      TopCPUSavedRegOff += RegSize;
  }

  StackOffset = ((StackOffset+StackAlign-1)/StackAlign*StackAlign);

  // Adjust FPU Callee Saved Registers Area. This Area must be
  // aligned to the default Stack Alignment requirements.
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (Mips::CPURegsRegisterClass->contains(Reg))
      continue;
    MFI->setObjectOffset(CSI[i].getFrameIdx(), StackOffset);
    TopFPUSavedRegOff = StackOffset;
    StackOffset += MFI->getObjectAlignment(CSI[i].getFrameIdx());
  }
  StackOffset = ((StackOffset+StackAlign-1)/StackAlign*StackAlign);

  // Update frame info
  MFI->setStackSize(StackOffset);

  // Recalculate the final tops offset. The final values must be '0'
  // if there isn't a callee saved register for CPU or FPU, otherwise
  // a negative offset is needed.
  if (TopCPUSavedRegOff >= 0)
    MipsFI->setCPUTopSavedRegOff(TopCPUSavedRegOff-StackOffset);

  if (TopFPUSavedRegOff >= 0)
    MipsFI->setFPUTopSavedRegOff(TopFPUSavedRegOff-StackOffset);
}


// expand pair of register and immediate if the immediate doesn't fit in the 16-bit
// offset field.
// e.g.
//  if OrigImm = 0x10000, OrigReg = $sp:
//    generate the following sequence of instrs:
//      lui  $at, hi(0x10000)
//      addu $at, $sp, $at
//
//    (NewReg, NewImm) = ($at, lo(Ox10000))
//    return true
static bool expandRegLargeImmPair(unsigned OrigReg, int OrigImm,
                                  unsigned& NewReg, int& NewImm,
                                  MachineBasicBlock& MBB, MachineBasicBlock::iterator I) {
  // OrigImm fits in the 16-bit field
  if (OrigImm < 0x8000 && OrigImm >= -0x8000) {
    NewReg = OrigReg;
    NewImm = OrigImm;
    return false;
  }

  MachineFunction* MF = MBB.getParent();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();
  DebugLoc DL = I->getDebugLoc();
  int ImmLo = OrigImm & 0xffff;
  int ImmHi = (((unsigned)OrigImm & 0xffff0000) >> 16) + ((OrigImm & 0x8000) != 0);

  // FIXME: change this when mips goes MC".
  BuildMI(MBB, I, DL, TII->get(Mips::NOAT));
  BuildMI(MBB, I, DL, TII->get(Mips::LUi), Mips::AT).addImm(ImmHi);
  BuildMI(MBB, I, DL, TII->get(Mips::ADDu), Mips::AT).addReg(OrigReg).addReg(Mips::AT);
  NewReg = Mips::AT;
  NewImm = ImmLo;

  return true;
}

void MipsFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
  const MipsRegisterInfo *RegInfo =
    static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const MipsInstrInfo &TII =
    *static_cast<const MipsInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  bool isPIC = (MF.getTarget().getRelocationModel() == Reloc::PIC_);
  unsigned NewReg = 0;
  int NewImm = 0;
  bool ATUsed;

  // Get the right frame order for Mips.
  adjustMipsStackFrame(MF);

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->adjustsStack()) return;

  int FPOffset = MipsFI->getFPStackOffset();
  int RAOffset = MipsFI->getRAStackOffset();

  BuildMI(MBB, MBBI, dl, TII.get(Mips::NOREORDER));

  // TODO: check need from GP here.
  if (isPIC && STI.isABI_O32())
    BuildMI(MBB, MBBI, dl, TII.get(Mips::CPLOAD))
      .addReg(RegInfo->getPICCallReg());
  BuildMI(MBB, MBBI, dl, TII.get(Mips::NOMACRO));

  // Adjust stack : addi sp, sp, (-imm)
  ATUsed = expandRegLargeImmPair(Mips::SP, -StackSize, NewReg, NewImm, MBB,
                                 MBBI);
  BuildMI(MBB, MBBI, dl, TII.get(Mips::ADDiu), Mips::SP)
    .addReg(NewReg).addImm(NewImm);

  // FIXME: change this when mips goes MC".
  if (ATUsed)
    BuildMI(MBB, MBBI, dl, TII.get(Mips::ATMACRO));

  // Save the return address only if the function isn't a leaf one.
  // sw  $ra, stack_loc($sp)
  if (MFI->adjustsStack()) {
    ATUsed = expandRegLargeImmPair(Mips::SP, RAOffset, NewReg, NewImm, MBB,
                                   MBBI);
    BuildMI(MBB, MBBI, dl, TII.get(Mips::SW))
      .addReg(Mips::RA).addImm(NewImm).addReg(NewReg);

    // FIXME: change this when mips goes MC".
    if (ATUsed)
      BuildMI(MBB, MBBI, dl, TII.get(Mips::ATMACRO));
  }

  // if framepointer enabled, save it and set it
  // to point to the stack pointer
  if (hasFP(MF)) {
    // sw  $fp,stack_loc($sp)
    ATUsed = expandRegLargeImmPair(Mips::SP, FPOffset, NewReg, NewImm, MBB,
                                   MBBI);
    BuildMI(MBB, MBBI, dl, TII.get(Mips::SW))
      .addReg(Mips::FP).addImm(NewImm).addReg(NewReg);

    // FIXME: change this when mips goes MC".
    if (ATUsed)
      BuildMI(MBB, MBBI, dl, TII.get(Mips::ATMACRO));

    // move $fp, $sp
    BuildMI(MBB, MBBI, dl, TII.get(Mips::ADDu), Mips::FP)
      .addReg(Mips::SP).addReg(Mips::ZERO);
  }

  // Restore GP from the saved stack location
  if (MipsFI->needGPSaveRestore())
    BuildMI(MBB, MBBI, dl, TII.get(Mips::CPRESTORE))
      .addImm(MipsFI->getGPStackOffset());
}

void MipsFrameLowering::emitEpilogue(MachineFunction &MF,
                                 MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI         = MF.getInfo<MipsFunctionInfo>();
  const MipsInstrInfo &TII =
    *static_cast<const MipsInstrInfo*>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();

  // Get the number of bytes from FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Get the FI's where RA and FP are saved.
  int FPOffset = MipsFI->getFPStackOffset();
  int RAOffset = MipsFI->getRAStackOffset();

  unsigned NewReg = 0;
  int NewImm = 0;
  bool ATUsed = false;

  // if framepointer enabled, restore it and restore the
  // stack pointer
  if (hasFP(MF)) {
    // move $sp, $fp
    BuildMI(MBB, MBBI, dl, TII.get(Mips::ADDu), Mips::SP)
      .addReg(Mips::FP).addReg(Mips::ZERO);

    // lw  $fp,stack_loc($sp)
    ATUsed = expandRegLargeImmPair(Mips::SP, FPOffset, NewReg, NewImm, MBB,
                                   MBBI);
    BuildMI(MBB, MBBI, dl, TII.get(Mips::LW), Mips::FP)
      .addImm(NewImm).addReg(NewReg);

    // FIXME: change this when mips goes MC".
    if (ATUsed)
      BuildMI(MBB, MBBI, dl, TII.get(Mips::ATMACRO));
  }

  // Restore the return address only if the function isn't a leaf one.
  // lw  $ra, stack_loc($sp)
  if (MFI->adjustsStack()) {
    ATUsed = expandRegLargeImmPair(Mips::SP, RAOffset, NewReg, NewImm, MBB,
                                   MBBI);
    BuildMI(MBB, MBBI, dl, TII.get(Mips::LW), Mips::RA)
      .addImm(NewImm).addReg(NewReg);

    // FIXME: change this when mips goes MC".
    if (ATUsed)
      BuildMI(MBB, MBBI, dl, TII.get(Mips::ATMACRO));
  }

  // adjust stack  : insert addi sp, sp, (imm)
  if (NumBytes) {
    ATUsed = expandRegLargeImmPair(Mips::SP, NumBytes, NewReg, NewImm, MBB,
                                   MBBI);
    BuildMI(MBB, MBBI, dl, TII.get(Mips::ADDiu), Mips::SP)
      .addReg(NewReg).addImm(NewImm);

    // FIXME: change this when mips goes MC".
    if (ATUsed)
      BuildMI(MBB, MBBI, dl, TII.get(Mips::ATMACRO));
  }
}

void MipsFrameLowering::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
  const MipsRegisterInfo *RegInfo =
    static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());
  RegInfo->processFunctionBeforeFrameFinalized(MF);
}
