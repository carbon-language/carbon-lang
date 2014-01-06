//===-- XCoreFrameLowering.cpp - Frame info for XCore Target --------------===//
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

#include "XCoreFrameLowering.h"
#include "XCore.h"
#include "XCoreInstrInfo.h"
#include "XCoreMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

static const unsigned FramePtr = XCore::R10;
static const int MaxImmU16 = (1<<16) - 1;

// helper functions. FIXME: Eliminate.
static inline bool isImmU6(unsigned val) {
  return val < (1 << 6);
}

static inline bool isImmU16(unsigned val) {
  return val < (1 << 16);
}

static void EmitDefCfaRegister(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI, DebugLoc dl,
                               const TargetInstrInfo &TII,
                               MachineModuleInfo *MMI, unsigned DRegNum) {
  MCSymbol *Label = MMI->getContext().CreateTempSymbol();
  BuildMI(MBB, MBBI, dl, TII.get(XCore::PROLOG_LABEL)).addSym(Label);
  MMI->addFrameInst(MCCFIInstruction::createDefCfaRegister(Label, DRegNum));
}

static void EmitDefCfaOffset(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MBBI, DebugLoc dl,
                             const TargetInstrInfo &TII,
                             MachineModuleInfo *MMI, int Offset) {
  MCSymbol *Label = MMI->getContext().CreateTempSymbol();
  BuildMI(MBB, MBBI, dl, TII.get(XCore::PROLOG_LABEL)).addSym(Label);
  MMI->addFrameInst(MCCFIInstruction::createDefCfaOffset(Label, -Offset));
}

static void EmitCfiOffset(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MBBI, DebugLoc dl,
                          const TargetInstrInfo &TII, MachineModuleInfo *MMI,
                          unsigned DRegNum, int Offset, MCSymbol *Label) {
  if (!Label) {
    Label = MMI->getContext().CreateTempSymbol();
    BuildMI(MBB, MBBI, dl, TII.get(XCore::PROLOG_LABEL)).addSym(Label);
  }
  MMI->addFrameInst(MCCFIInstruction::createOffset(Label, DRegNum, Offset));
}

/// The SP register is moved in steps of 'MaxImmU16' towards the bottom of the
/// frame. During these steps, it may be necessary to spill registers.
/// IfNeededExtSP emits the necessary EXTSP instructions to move the SP only
/// as far as to make 'OffsetFromBottom' reachable using an STWSP_lru6.
/// \param OffsetFromTop the spill offset from the top of the frame.
/// \param [in,out] Adjusted the current SP offset from the top of the frame.
static void IfNeededExtSP(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MBBI, DebugLoc dl,
                          const TargetInstrInfo &TII, MachineModuleInfo *MMI,
                          int OffsetFromTop, int &Adjusted, int FrameSize,
                          bool emitFrameMoves) {
  while (OffsetFromTop > Adjusted) {
    assert(Adjusted < FrameSize && "OffsetFromTop is beyond FrameSize");
    int remaining = FrameSize - Adjusted;
    int OpImm = (remaining > MaxImmU16) ? MaxImmU16 : remaining;
    int Opcode = isImmU6(OpImm) ? XCore::EXTSP_u6 : XCore::EXTSP_lu6;
    BuildMI(MBB, MBBI, dl, TII.get(Opcode)).addImm(OpImm);
    Adjusted += OpImm;
    if (emitFrameMoves)
      EmitDefCfaOffset(MBB, MBBI, dl, TII, MMI, Adjusted*4);
  }
}

/// The SP register is moved in steps of 'MaxImmU16' towards the top of the
/// frame. During these steps, it may be necessary to re-load registers.
/// IfNeededLDAWSP emits the necessary LDAWSP instructions to move the SP only
/// as far as to make 'OffsetFromTop' reachable using an LDAWSP_lru6.
/// \param OffsetFromTop the spill offset from the top of the frame.
/// \param [in,out] RemainingAdj the current SP offset from the top of the frame.
static void IfNeededLDAWSP(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, DebugLoc dl,
                           const TargetInstrInfo &TII, int OffsetFromTop,
                           int &RemainingAdj) {
  while (OffsetFromTop < RemainingAdj - MaxImmU16) {
    assert(RemainingAdj && "OffsetFromTop is beyond FrameSize");
    int OpImm = (RemainingAdj > MaxImmU16) ? MaxImmU16 : RemainingAdj;
    int Opcode = isImmU6(OpImm) ? XCore::LDAWSP_ru6 : XCore::LDAWSP_lru6;
    BuildMI(MBB, MBBI, dl, TII.get(Opcode), XCore::SP).addImm(OpImm);
    RemainingAdj -= OpImm;
  }
}

/// Creates an ordered list of registers that are spilled
/// during the emitPrologue/emitEpilogue.
/// Registers are ordered according to their frame offset.
static void GetSpillList(SmallVectorImpl<std::pair<unsigned,int> > &SpillList,
                         MachineFrameInfo *MFI, XCoreFunctionInfo *XFI,
                         bool fetchLR, bool fetchFP) {
  int LRSpillOffset = fetchLR? MFI->getObjectOffset(XFI->getLRSpillSlot()) : 0;
  int FPSpillOffset = fetchFP? MFI->getObjectOffset(XFI->getFPSpillSlot()) : 0;
  if (fetchLR && fetchFP && LRSpillOffset > FPSpillOffset) {
    SpillList.push_back(std::pair<unsigned, int>(XCore::LR, LRSpillOffset));
    fetchLR = false;
  }
  if (fetchFP)
    SpillList.push_back(std::pair<unsigned, int>(FramePtr, FPSpillOffset));
  if (fetchLR)
    SpillList.push_back(std::pair<unsigned, int>(XCore::LR, LRSpillOffset));
}


//===----------------------------------------------------------------------===//
// XCoreFrameLowering:
//===----------------------------------------------------------------------===//

XCoreFrameLowering::XCoreFrameLowering(const XCoreSubtarget &sti)
  : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 4, 0) {
  // Do nothing
}

bool XCoreFrameLowering::hasFP(const MachineFunction &MF) const {
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         MF.getFrameInfo()->hasVarSizedObjects();
}

void XCoreFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo *MMI = &MF.getMMI();
  const MCRegisterInfo *MRI = MMI->getContext().getRegisterInfo();
  const XCoreInstrInfo &TII =
    *static_cast<const XCoreInstrInfo*>(MF.getTarget().getInstrInfo());
  XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  if (MFI->getMaxAlignment() > getStackAlignment())
    report_fatal_error("emitPrologue unsupported alignment: "
                       + Twine(MFI->getMaxAlignment()));

  const AttributeSet &PAL = MF.getFunction()->getAttributes();
  if (PAL.hasAttrSomewhere(Attribute::Nest))
    BuildMI(MBB, MBBI, dl, TII.get(XCore::LDWSP_ru6), XCore::R11).addImm(0);

  // Work out frame sizes.
  // We will adjust the SP in stages towards the final FrameSize.
  assert(MFI->getStackSize()%4 == 0 && "Misaligned frame size");
  const int FrameSize = MFI->getStackSize() / 4;
  int Adjusted = 0;

  bool saveLR = XFI->hasLRSpillSlot();
  bool UseENTSP = saveLR && FrameSize
                  && (MFI->getObjectOffset(XFI->getLRSpillSlot()) == 0);
  if (UseENTSP)
    saveLR = false;
  bool FP = hasFP(MF);
  bool emitFrameMoves = XCoreRegisterInfo::needsFrameMoves(MF);

  if (UseENTSP) {
    // Allocate space on the stack at the same time as saving LR.
    Adjusted = (FrameSize > MaxImmU16) ? MaxImmU16 : FrameSize;
    int Opcode = isImmU6(Adjusted) ? XCore::ENTSP_u6 : XCore::ENTSP_lu6;
    MBB.addLiveIn(XCore::LR);
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(Opcode));
    MIB.addImm(Adjusted);
    MIB->addRegisterKilled(XCore::LR, MF.getTarget().getRegisterInfo(), true);
    if (emitFrameMoves) {
      EmitDefCfaOffset(MBB, MBBI, dl, TII, MMI, Adjusted*4);
      unsigned DRegNum = MRI->getDwarfRegNum(XCore::LR, true);
      EmitCfiOffset(MBB, MBBI, dl, TII, MMI, DRegNum, 0, NULL);
    }
  }

  // If necessary, save LR and FP to the stack, as we EXTSP.
  SmallVector<std::pair<unsigned,int>,2> SpillList;
  GetSpillList(SpillList, MFI, XFI, saveLR, FP);
  for (unsigned i = 0, e = SpillList.size(); i != e; ++i) {
    unsigned SpillReg = SpillList[i].first;
    int SpillOffset = SpillList[i].second;
    assert(SpillOffset % 4 == 0 && "Misaligned stack offset");
    assert(SpillOffset <= 0 && "Unexpected positive stack offset");
    int OffsetFromTop = - SpillOffset/4;
    IfNeededExtSP(MBB, MBBI, dl, TII, MMI, OffsetFromTop, Adjusted, FrameSize,
                  emitFrameMoves);
    int Offset = Adjusted - OffsetFromTop;
    int Opcode = isImmU6(Offset) ? XCore::STWSP_ru6 : XCore::STWSP_lru6;
    MBB.addLiveIn(SpillReg);
    BuildMI(MBB, MBBI, dl, TII.get(Opcode)).addReg(SpillReg, RegState::Kill)
                                           .addImm(Offset);
    if (emitFrameMoves) {
      unsigned DRegNum = MRI->getDwarfRegNum(SpillReg, true);
      EmitCfiOffset(MBB, MBBI, dl, TII, MMI, DRegNum, SpillOffset, NULL);
    }
  }

  // Complete any remaining Stack adjustment.
  IfNeededExtSP(MBB, MBBI, dl, TII, MMI, FrameSize, Adjusted, FrameSize,
                emitFrameMoves);
  assert(Adjusted==FrameSize && "IfNeededExtSP has not completed adjustment");

  if (FP) {
    // Set the FP from the SP.
    BuildMI(MBB, MBBI, dl, TII.get(XCore::LDAWSP_ru6), FramePtr).addImm(0);
    if (emitFrameMoves)
      EmitDefCfaRegister(MBB, MBBI, dl, TII, MMI,
                         MRI->getDwarfRegNum(FramePtr, true));
  }

  if (emitFrameMoves) {
    // Frame moves for callee saved.
    std::vector<std::pair<MCSymbol*, CalleeSavedInfo> >&SpillLabels =
        XFI->getSpillLabels();
    for (unsigned I = 0, E = SpillLabels.size(); I != E; ++I) {
      MCSymbol *SpillLabel = SpillLabels[I].first;
      CalleeSavedInfo &CSI = SpillLabels[I].second;
      int Offset = MFI->getObjectOffset(CSI.getFrameIdx());
      unsigned DRegNum = MRI->getDwarfRegNum(CSI.getReg(), true);
      EmitCfiOffset(MBB, MBBI, dl, TII, MMI, DRegNum, Offset, SpillLabel);
    }
  }
}

void XCoreFrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const XCoreInstrInfo &TII =
    *static_cast<const XCoreInstrInfo*>(MF.getTarget().getInstrInfo());
  XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();
  DebugLoc dl = MBBI->getDebugLoc();
  unsigned RetOpcode = MBBI->getOpcode();

  if (RetOpcode == XCore::EH_RETURN) {
    unsigned EhStackReg = MBBI->getOperand(0).getReg();
    unsigned EhHandlerReg = MBBI->getOperand(1).getReg();
    BuildMI(MBB, MBBI, dl, TII.get(XCore::SETSP_1r)).addReg(EhStackReg);
    BuildMI(MBB, MBBI, dl, TII.get(XCore::BAU_1r)).addReg(EhHandlerReg);
    MBB.erase(MBBI);  // Erase the previous return instruction.
    return;
  }

  // Work out frame sizes.
  // We will adjust the SP in stages towards the final FrameSize.
  int RemainingAdj = MFI->getStackSize();
  assert(RemainingAdj%4 == 0 && "Misaligned frame size");
  RemainingAdj /= 4;

  bool restoreLR = XFI->hasLRSpillSlot();
  bool UseRETSP = restoreLR && RemainingAdj
                  && (MFI->getObjectOffset(XFI->getLRSpillSlot()) == 0);
  if (UseRETSP)
    restoreLR = false;
  bool FP = hasFP(MF);

  if (FP) // Restore the stack pointer.
    BuildMI(MBB, MBBI, dl, TII.get(XCore::SETSP_1r)).addReg(FramePtr);

  // If necessary, restore LR and FP from the stack, as we EXTSP.
  SmallVector<std::pair<unsigned,int>,2> SpillList;
  GetSpillList(SpillList, MFI, XFI, restoreLR, FP);
  unsigned i = SpillList.size();
  while (i--) {
    unsigned SpilledReg = SpillList[i].first;
    int SpillOffset = SpillList[i].second;
    assert(SpillOffset % 4 == 0 && "Misaligned stack offset");
    assert(SpillOffset <= 0 && "Unexpected positive stack offset");
    int OffsetFromTop = - SpillOffset/4;
    IfNeededLDAWSP(MBB, MBBI, dl, TII, OffsetFromTop, RemainingAdj);
    int Offset = RemainingAdj - OffsetFromTop;
    int Opcode = isImmU6(Offset) ? XCore::LDWSP_ru6 : XCore::LDWSP_lru6;
    BuildMI(MBB, MBBI, dl, TII.get(Opcode), SpilledReg).addImm(Offset);
  }

  if (RemainingAdj) {
    // Complete all but one of the remaining Stack adjustments.
    IfNeededLDAWSP(MBB, MBBI, dl, TII, 0, RemainingAdj);
    if (UseRETSP) {
      // Fold prologue into return instruction
      assert(RetOpcode == XCore::RETSP_u6
             || RetOpcode == XCore::RETSP_lu6);
      int Opcode = isImmU6(RemainingAdj) ? XCore::RETSP_u6 : XCore::RETSP_lu6;
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(Opcode))
                                  .addImm(RemainingAdj);
      for (unsigned i = 3, e = MBBI->getNumOperands(); i < e; ++i)
        MIB->addOperand(MBBI->getOperand(i)); // copy any variadic operands
      MBB.erase(MBBI);  // Erase the previous return instruction.
    } else {
      int Opcode = isImmU6(RemainingAdj) ? XCore::LDAWSP_ru6 :
                                           XCore::LDAWSP_lru6;
      BuildMI(MBB, MBBI, dl, TII.get(Opcode), XCore::SP).addImm(RemainingAdj);
      // Don't erase the return instruction.
    }
  } // else Don't erase the return instruction.
}

bool XCoreFrameLowering::
spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          const std::vector<CalleeSavedInfo> &CSI,
                          const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();
  XCoreFunctionInfo *XFI = MF->getInfo<XCoreFunctionInfo>();
  bool emitFrameMoves = XCoreRegisterInfo::needsFrameMoves(*MF);

  DebugLoc DL;
  if (MI != MBB.end())
    DL = MI->getDebugLoc();

  for (std::vector<CalleeSavedInfo>::const_iterator it = CSI.begin();
                                                    it != CSI.end(); ++it) {
    unsigned Reg = it->getReg();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(MBB, MI, Reg, true, it->getFrameIdx(), RC, TRI);
    if (emitFrameMoves) {
      MCSymbol *SaveLabel = MF->getContext().CreateTempSymbol();
      BuildMI(MBB, MI, DL, TII.get(XCore::PROLOG_LABEL)).addSym(SaveLabel);
      XFI->getSpillLabels().push_back(std::make_pair(SaveLabel, *it));
    }
  }
  return true;
}

bool XCoreFrameLowering::
restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const std::vector<CalleeSavedInfo> &CSI,
                            const TargetRegisterInfo *TRI) const{
  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();

  bool AtStart = MI == MBB.begin();
  MachineBasicBlock::iterator BeforeI = MI;
  if (!AtStart)
    --BeforeI;
  for (std::vector<CalleeSavedInfo>::const_iterator it = CSI.begin();
                                                    it != CSI.end(); ++it) {
    unsigned Reg = it->getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.loadRegFromStackSlot(MBB, MI, Reg, it->getFrameIdx(), RC, TRI);
    assert(MI != MBB.begin() &&
           "loadRegFromStackSlot didn't insert any code!");
    // Insert in reverse order.  loadRegFromStackSlot can insert multiple
    // instructions.
    if (AtStart)
      MI = MBB.begin();
    else {
      MI = BeforeI;
      ++MI;
    }
  }
  return true;
}

// This function eliminates ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void XCoreFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  const XCoreInstrInfo &TII =
    *static_cast<const XCoreInstrInfo*>(MF.getTarget().getInstrInfo());
  if (!hasReservedCallFrame(MF)) {
    // Turn the adjcallstackdown instruction into 'extsp <amt>' and the
    // adjcallstackup instruction into 'ldaw sp, sp[<amt>]'
    MachineInstr *Old = I;
    uint64_t Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      assert(Amount%4 == 0);
      Amount /= 4;

      bool isU6 = isImmU6(Amount);
      if (!isU6 && !isImmU16(Amount)) {
        // FIX could emit multiple instructions in this case.
#ifndef NDEBUG
        errs() << "eliminateCallFramePseudoInstr size too big: "
               << Amount << "\n";
#endif
        llvm_unreachable(0);
      }

      MachineInstr *New;
      if (Old->getOpcode() == XCore::ADJCALLSTACKDOWN) {
        int Opcode = isU6 ? XCore::EXTSP_u6 : XCore::EXTSP_lu6;
        New=BuildMI(MF, Old->getDebugLoc(), TII.get(Opcode))
          .addImm(Amount);
      } else {
        assert(Old->getOpcode() == XCore::ADJCALLSTACKUP);
        int Opcode = isU6 ? XCore::LDAWSP_ru6 : XCore::LDAWSP_lru6;
        New=BuildMI(MF, Old->getDebugLoc(), TII.get(Opcode), XCore::SP)
          .addImm(Amount);
      }

      // Replace the pseudo instruction with a new instruction...
      MBB.insert(I, New);
    }
  }
  
  MBB.erase(I);
}

void XCoreFrameLowering::
processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                     RegScavenger *RS) const {
  XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();

  bool LRUsed = MF.getRegInfo().isPhysRegUsed(XCore::LR);
  // If we need to extend the stack it is more efficient to use entsp / retsp.
  // We force the LR to be saved so these instructions are used.
  if (!LRUsed && !MF.getFunction()->isVarArg() &&
      MF.getFrameInfo()->estimateStackSize(MF))
    LRUsed = true;

  // We will handling LR in the prologue/epilogue
  // and space on the stack ourselves.
  if (LRUsed) {
    MF.getRegInfo().setPhysRegUnused(XCore::LR);
    XFI->createLRSpillSlot(MF);
  }

  // A callee save register is used to hold the FP.
  // This needs saving / restoring in the epilogue / prologue.
  if (hasFP(MF))
    XFI->createFPSpillSlot(MF);
}

void XCoreFrameLowering::
processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                    RegScavenger *RS) const {
  assert(RS && "requiresRegisterScavenging failed");
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetRegisterClass *RC = &XCore::GRRegsRegClass;
  XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();
  // Reserve slots close to SP or frame pointer for Scavenging spills.
  // When using SP for small frames, we don't need any scratch registers.
  // When using SP for large frames, we may need 2 scratch registers.
  // When using FP, for large or small frames, we may need 1 scratch register.
  if (XFI->isLargeFrame(MF) || hasFP(MF))
    RS->addScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                       RC->getAlignment(),
                                                       false));
  if (XFI->isLargeFrame(MF) && !hasFP(MF))
    RS->addScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                       RC->getAlignment(),
                                                       false));
}
