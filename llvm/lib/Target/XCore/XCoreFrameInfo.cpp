//===-- XCoreFrameInfo.cpp - Frame info for XCore Target ---------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains XCore frame information that doesn't fit anywhere else
// cleanly...
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "XCoreFrameInfo.h"
#include "XCoreInstrInfo.h"
#include "XCoreMachineFunctionInfo.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// helper functions. FIXME: Eliminate.
static inline bool isImmUs(unsigned val) {
  return val <= 11;
}

static inline bool isImmU6(unsigned val) {
  return val < (1 << 6);
}

static inline bool isImmU16(unsigned val) {
  return val < (1 << 16);
}

static void loadFromStack(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator I,
                          unsigned DstReg, int Offset, DebugLoc dl,
                          const TargetInstrInfo &TII) {
  assert(Offset%4 == 0 && "Misaligned stack offset");
  Offset/=4;
  bool isU6 = isImmU6(Offset);
  if (!isU6 && !isImmU16(Offset))
    report_fatal_error("loadFromStack offset too big " + Twine(Offset));
  int Opcode = isU6 ? XCore::LDWSP_ru6 : XCore::LDWSP_lru6;
  BuildMI(MBB, I, dl, TII.get(Opcode), DstReg)
    .addImm(Offset);
}


static void storeToStack(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator I,
                         unsigned SrcReg, int Offset, DebugLoc dl,
                         const TargetInstrInfo &TII) {
  assert(Offset%4 == 0 && "Misaligned stack offset");
  Offset/=4;
  bool isU6 = isImmU6(Offset);
  if (!isU6 && !isImmU16(Offset))
    report_fatal_error("storeToStack offset too big " + Twine(Offset));
  int Opcode = isU6 ? XCore::STWSP_ru6 : XCore::STWSP_lru6;
  BuildMI(MBB, I, dl, TII.get(Opcode))
    .addReg(SrcReg)
    .addImm(Offset);
}


//===----------------------------------------------------------------------===//
// XCoreFrameInfo:
//===----------------------------------------------------------------------===//

XCoreFrameInfo::XCoreFrameInfo(const XCoreSubtarget &sti)
  : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 4, 0),
    STI(sti) {
  // Do nothing
}

void XCoreFrameInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo *MMI = &MF.getMMI();
  const XCoreRegisterInfo *RegInfo =
    static_cast<const XCoreRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const XCoreInstrInfo &TII =
    *static_cast<const XCoreInstrInfo*>(MF.getTarget().getInstrInfo());
  XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  bool FP = RegInfo->hasFP(MF);

  // Work out frame sizes.
  int FrameSize = MFI->getStackSize();
  assert(FrameSize%4 == 0 && "Misaligned frame size");
  FrameSize/=4;

  bool isU6 = isImmU6(FrameSize);

  if (!isU6 && !isImmU16(FrameSize)) {
    // FIXME could emit multiple instructions.
    report_fatal_error("emitPrologue Frame size too big: " + Twine(FrameSize));
  }
  bool emitFrameMoves = RegInfo->needsFrameMoves(MF);

  // Do we need to allocate space on the stack?
  if (FrameSize) {
    bool saveLR = XFI->getUsesLR();
    bool LRSavedOnEntry = false;
    int Opcode;
    if (saveLR && (MFI->getObjectOffset(XFI->getLRSpillSlot()) == 0)) {
      Opcode = (isU6) ? XCore::ENTSP_u6 : XCore::ENTSP_lu6;
      MBB.addLiveIn(XCore::LR);
      saveLR = false;
      LRSavedOnEntry = true;
    } else {
      Opcode = (isU6) ? XCore::EXTSP_u6 : XCore::EXTSP_lu6;
    }
    BuildMI(MBB, MBBI, dl, TII.get(Opcode)).addImm(FrameSize);

    if (emitFrameMoves) {
      std::vector<MachineMove> &Moves = MMI->getFrameMoves();

      // Show update of SP.
      MCSymbol *FrameLabel = MMI->getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, dl, TII.get(XCore::PROLOG_LABEL)).addSym(FrameLabel);

      MachineLocation SPDst(MachineLocation::VirtualFP);
      MachineLocation SPSrc(MachineLocation::VirtualFP, -FrameSize * 4);
      Moves.push_back(MachineMove(FrameLabel, SPDst, SPSrc));

      if (LRSavedOnEntry) {
        MachineLocation CSDst(MachineLocation::VirtualFP, 0);
        MachineLocation CSSrc(XCore::LR);
        Moves.push_back(MachineMove(FrameLabel, CSDst, CSSrc));
      }
    }
    if (saveLR) {
      int LRSpillOffset = MFI->getObjectOffset(XFI->getLRSpillSlot());
      storeToStack(MBB, MBBI, XCore::LR, LRSpillOffset + FrameSize*4, dl, TII);
      MBB.addLiveIn(XCore::LR);

      if (emitFrameMoves) {
        MCSymbol *SaveLRLabel = MMI->getContext().CreateTempSymbol();
        BuildMI(MBB, MBBI, dl, TII.get(XCore::PROLOG_LABEL)).addSym(SaveLRLabel);
        MachineLocation CSDst(MachineLocation::VirtualFP, LRSpillOffset);
        MachineLocation CSSrc(XCore::LR);
        MMI->getFrameMoves().push_back(MachineMove(SaveLRLabel, CSDst, CSSrc));
      }
    }
  }

  if (FP) {
    // Save R10 to the stack.
    int FPSpillOffset = MFI->getObjectOffset(XFI->getFPSpillSlot());
    storeToStack(MBB, MBBI, XCore::R10, FPSpillOffset + FrameSize*4, dl, TII);
    // R10 is live-in. It is killed at the spill.
    MBB.addLiveIn(XCore::R10);
    if (emitFrameMoves) {
      MCSymbol *SaveR10Label = MMI->getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, dl, TII.get(XCore::PROLOG_LABEL)).addSym(SaveR10Label);
      MachineLocation CSDst(MachineLocation::VirtualFP, FPSpillOffset);
      MachineLocation CSSrc(XCore::R10);
      MMI->getFrameMoves().push_back(MachineMove(SaveR10Label, CSDst, CSSrc));
    }
    // Set the FP from the SP.
    unsigned FramePtr = XCore::R10;
    BuildMI(MBB, MBBI, dl, TII.get(XCore::LDAWSP_ru6), FramePtr)
      .addImm(0);
    if (emitFrameMoves) {
      // Show FP is now valid.
      MCSymbol *FrameLabel = MMI->getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, dl, TII.get(XCore::PROLOG_LABEL)).addSym(FrameLabel);
      MachineLocation SPDst(FramePtr);
      MachineLocation SPSrc(MachineLocation::VirtualFP);
      MMI->getFrameMoves().push_back(MachineMove(FrameLabel, SPDst, SPSrc));
    }
  }

  if (emitFrameMoves) {
    // Frame moves for callee saved.
    std::vector<MachineMove> &Moves = MMI->getFrameMoves();
    std::vector<std::pair<MCSymbol*, CalleeSavedInfo> >&SpillLabels =
        XFI->getSpillLabels();
    for (unsigned I = 0, E = SpillLabels.size(); I != E; ++I) {
      MCSymbol *SpillLabel = SpillLabels[I].first;
      CalleeSavedInfo &CSI = SpillLabels[I].second;
      int Offset = MFI->getObjectOffset(CSI.getFrameIdx());
      unsigned Reg = CSI.getReg();
      MachineLocation CSDst(MachineLocation::VirtualFP, Offset);
      MachineLocation CSSrc(Reg);
      Moves.push_back(MachineMove(SpillLabel, CSDst, CSSrc));
    }
  }
}

void XCoreFrameInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  const XCoreRegisterInfo *RegInfo =
    static_cast<const XCoreRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const XCoreInstrInfo &TII =
    *static_cast<const XCoreInstrInfo*>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();

  bool FP = RegInfo->hasFP(MF);

  if (FP) {
    // Restore the stack pointer.
    unsigned FramePtr = XCore::R10;
    BuildMI(MBB, MBBI, dl, TII.get(XCore::SETSP_1r))
      .addReg(FramePtr);
  }

  // Work out frame sizes.
  int FrameSize = MFI->getStackSize();

  assert(FrameSize%4 == 0 && "Misaligned frame size");

  FrameSize/=4;

  bool isU6 = isImmU6(FrameSize);

  if (!isU6 && !isImmU16(FrameSize)) {
    // FIXME could emit multiple instructions.
    report_fatal_error("emitEpilogue Frame size too big: " + Twine(FrameSize));
  }

  if (FrameSize) {
    XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();

    if (FP) {
      // Restore R10
      int FPSpillOffset = MFI->getObjectOffset(XFI->getFPSpillSlot());
      FPSpillOffset += FrameSize*4;
      loadFromStack(MBB, MBBI, XCore::R10, FPSpillOffset, dl, TII);
    }
    bool restoreLR = XFI->getUsesLR();
    if (restoreLR && MFI->getObjectOffset(XFI->getLRSpillSlot()) != 0) {
      int LRSpillOffset = MFI->getObjectOffset(XFI->getLRSpillSlot());
      LRSpillOffset += FrameSize*4;
      loadFromStack(MBB, MBBI, XCore::LR, LRSpillOffset, dl, TII);
      restoreLR = false;
    }
    if (restoreLR) {
      // Fold prologue into return instruction
      assert(MBBI->getOpcode() == XCore::RETSP_u6
        || MBBI->getOpcode() == XCore::RETSP_lu6);
      int Opcode = (isU6) ? XCore::RETSP_u6 : XCore::RETSP_lu6;
      BuildMI(MBB, MBBI, dl, TII.get(Opcode)).addImm(FrameSize);
      MBB.erase(MBBI);
    } else {
      int Opcode = (isU6) ? XCore::LDAWSP_ru6_RRegs : XCore::LDAWSP_lru6_RRegs;
      BuildMI(MBB, MBBI, dl, TII.get(Opcode), XCore::SP).addImm(FrameSize);
    }
  }
}
