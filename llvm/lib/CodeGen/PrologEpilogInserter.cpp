//===-- PrologEpilogInserter.cpp - Insert Prolog/Epilog code in function --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
// This pass provides an optional shrink wrapping variant of prolog/epilog
// insertion, enabled via --shrink-wrap. See ShrinkWrapping.cpp.
//
//===----------------------------------------------------------------------===//

#include "PrologEpilogInserter.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>

using namespace llvm;

char PEI::ID = 0;

static RegisterPass<PEI>
X("prologepilog", "Prologue/Epilogue Insertion");

/// createPrologEpilogCodeInserter - This function returns a pass that inserts
/// prolog and epilog code, and eliminates abstract frame references.
///
FunctionPass *llvm::createPrologEpilogCodeInserter() { return new PEI(); }

/// runOnMachineFunction - Insert prolog/epilog code and replace abstract
/// frame indexes with appropriate references.
///
bool PEI::runOnMachineFunction(MachineFunction &Fn) {
  const Function* F = Fn.getFunction();
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
  RS = TRI->requiresRegisterScavenging(Fn) ? new RegScavenger() : NULL;
  FrameIndexVirtualScavenging = TRI->requiresFrameIndexScavenging(Fn);

  // Get MachineModuleInfo so that we can track the construction of the
  // frame.
  if (MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>())
    Fn.getFrameInfo()->setMachineModuleInfo(MMI);

  // Calculate the MaxCallFrameSize and HasCalls variables for the function's
  // frame information. Also eliminates call frame pseudo instructions.
  calculateCallsInformation(Fn);

  // Allow the target machine to make some adjustments to the function
  // e.g. UsedPhysRegs before calculateCalleeSavedRegisters.
  TRI->processFunctionBeforeCalleeSavedScan(Fn, RS);

  // Scan the function for modified callee saved registers and insert spill code
  // for any callee saved registers that are modified.
  calculateCalleeSavedRegisters(Fn);

  // Determine placement of CSR spill/restore code:
  //  - with shrink wrapping, place spills and restores to tightly
  //    enclose regions in the Machine CFG of the function where
  //    they are used. Without shrink wrapping
  //  - default (no shrink wrapping), place all spills in the
  //    entry block, all restores in return blocks.
  placeCSRSpillsAndRestores(Fn);

  // Add the code to save and restore the callee saved registers
  if (!F->hasFnAttr(Attribute::Naked))
    insertCSRSpillsAndRestores(Fn);

  // Allow the target machine to make final modifications to the function
  // before the frame layout is finalized.
  TRI->processFunctionBeforeFrameFinalized(Fn);

  // Calculate actual frame offsets for all abstract stack objects...
  calculateFrameObjectOffsets(Fn);

  // Add prolog and epilog code to the function.  This function is required
  // to align the stack frame as necessary for any stack variables or
  // called functions.  Because of this, calculateCalleeSavedRegisters
  // must be called before this function in order to set the HasCalls
  // and MaxCallFrameSize variables.
  if (!F->hasFnAttr(Attribute::Naked))
    insertPrologEpilogCode(Fn);

  // Replace all MO_FrameIndex operands with physical register references
  // and actual offsets.
  //
  replaceFrameIndices(Fn);

  // If register scavenging is needed, as we've enabled doing it as a
  // post-pass, scavenge the virtual registers that frame index elimiation
  // inserted.
  if (TRI->requiresRegisterScavenging(Fn) && FrameIndexVirtualScavenging)
    scavengeFrameVirtualRegs(Fn);

  delete RS;
  clearAllSets();
  return true;
}

#if 0
void PEI::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  if (ShrinkWrapping || ShrinkWrapFunc != "") {
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<MachineDominatorTree>();
  }
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreserved<MachineDominatorTree>();
  MachineFunctionPass::getAnalysisUsage(AU);
}
#endif

/// calculateCallsInformation - Calculate the MaxCallFrameSize and HasCalls
/// variables for the function's frame information and eliminate call frame
/// pseudo instructions.
void PEI::calculateCallsInformation(MachineFunction &Fn) {
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();

  unsigned MaxCallFrameSize = 0;
  bool HasCalls = false;

  // Get the function call frame set-up and tear-down instruction opcode
  int FrameSetupOpcode   = RegInfo->getCallFrameSetupOpcode();
  int FrameDestroyOpcode = RegInfo->getCallFrameDestroyOpcode();

  // Early exit for targets which have no call frame setup/destroy pseudo
  // instructions.
  if (FrameSetupOpcode == -1 && FrameDestroyOpcode == -1)
    return;

  std::vector<MachineBasicBlock::iterator> FrameSDOps;
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        assert(I->getNumOperands() >= 1 && "Call Frame Setup/Destroy Pseudo"
               " instructions should have a single immediate argument!");
        unsigned Size = I->getOperand(0).getImm();
        if (Size > MaxCallFrameSize) MaxCallFrameSize = Size;
        HasCalls = true;
        FrameSDOps.push_back(I);
      } else if (I->getOpcode() == TargetInstrInfo::INLINEASM) {
        // An InlineAsm might be a call; assume it is to get the stack frame
        // aligned correctly for calls.
        HasCalls = true;
      }

  MachineFrameInfo *FFI = Fn.getFrameInfo();
  FFI->setHasCalls(HasCalls);
  FFI->setMaxCallFrameSize(MaxCallFrameSize);

  for (std::vector<MachineBasicBlock::iterator>::iterator
         i = FrameSDOps.begin(), e = FrameSDOps.end(); i != e; ++i) {
    MachineBasicBlock::iterator I = *i;

    // If call frames are not being included as part of the stack frame, and
    // there is no dynamic allocation (therefore referencing frame slots off
    // sp), leave the pseudo ops alone. We'll eliminate them later.
    if (RegInfo->hasReservedCallFrame(Fn) || RegInfo->hasFP(Fn))
      RegInfo->eliminateCallFramePseudoInstr(Fn, *I->getParent(), I);
  }
}


/// calculateCalleeSavedRegisters - Scan the function for modified callee saved
/// registers.
void PEI::calculateCalleeSavedRegisters(MachineFunction &Fn) {
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  const TargetFrameInfo *TFI = Fn.getTarget().getFrameInfo();
  MachineFrameInfo *FFI = Fn.getFrameInfo();

  // Get the callee saved register list...
  const unsigned *CSRegs = RegInfo->getCalleeSavedRegs(&Fn);

  // These are used to keep track the callee-save area. Initialize them.
  MinCSFrameIndex = INT_MAX;
  MaxCSFrameIndex = 0;

  // Early exit for targets which have no callee saved registers.
  if (CSRegs == 0 || CSRegs[0] == 0)
    return;

  // Figure out which *callee saved* registers are modified by the current
  // function, thus needing to be saved and restored in the prolog/epilog.
  const TargetRegisterClass * const *CSRegClasses =
    RegInfo->getCalleeSavedRegClasses(&Fn);

  std::vector<CalleeSavedInfo> CSI;
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (Fn.getRegInfo().isPhysRegUsed(Reg)) {
      // If the reg is modified, save it!
      CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
    } else {
      for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
           *AliasSet; ++AliasSet) {  // Check alias registers too.
        if (Fn.getRegInfo().isPhysRegUsed(*AliasSet)) {
          CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
          break;
        }
      }
    }
  }

  if (CSI.empty())
    return;   // Early exit if no callee saved registers are modified!

  unsigned NumFixedSpillSlots;
  const TargetFrameInfo::SpillSlot *FixedSpillSlots =
    TFI->getCalleeSavedSpillSlots(NumFixedSpillSlots);

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  for (std::vector<CalleeSavedInfo>::iterator
         I = CSI.begin(), E = CSI.end(); I != E; ++I) {
    unsigned Reg = I->getReg();
    const TargetRegisterClass *RC = I->getRegClass();

    int FrameIdx;
    if (RegInfo->hasReservedSpillSlot(Fn, Reg, FrameIdx)) {
      I->setFrameIdx(FrameIdx);
      continue;
    }

    // Check to see if this physreg must be spilled to a particular stack slot
    // on this target.
    const TargetFrameInfo::SpillSlot *FixedSlot = FixedSpillSlots;
    while (FixedSlot != FixedSpillSlots+NumFixedSpillSlots &&
           FixedSlot->Reg != Reg)
      ++FixedSlot;

    if (FixedSlot == FixedSpillSlots + NumFixedSpillSlots) {
      // Nope, just spill it anywhere convenient.
      unsigned Align = RC->getAlignment();
      unsigned StackAlign = TFI->getStackAlignment();

      // We may not be able to satisfy the desired alignment specification of
      // the TargetRegisterClass if the stack alignment is smaller. Use the
      // min.
      Align = std::min(Align, StackAlign);
      FrameIdx = FFI->CreateStackObject(RC->getSize(), Align);
      if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
      if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;
    } else {
      // Spill it to the stack where we must.
      FrameIdx = FFI->CreateFixedObject(RC->getSize(), FixedSlot->Offset);
    }

    I->setFrameIdx(FrameIdx);
  }

  FFI->setCalleeSavedInfo(CSI);
}

/// insertCSRSpillsAndRestores - Insert spill and restore code for
/// callee saved registers used in the function, handling shrink wrapping.
///
void PEI::insertCSRSpillsAndRestores(MachineFunction &Fn) {
  // Get callee saved register information.
  MachineFrameInfo *FFI = Fn.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = FFI->getCalleeSavedInfo();

  FFI->setCalleeSavedInfoValid(true);

  // Early exit if no callee saved registers are modified!
  if (CSI.empty())
    return;

  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  MachineBasicBlock::iterator I;

  if (! ShrinkWrapThisFunction) {
    // Spill using target interface.
    I = EntryBlock->begin();
    if (!TII.spillCalleeSavedRegisters(*EntryBlock, I, CSI)) {
      for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
        // Add the callee-saved register as live-in.
        // It's killed at the spill.
        EntryBlock->addLiveIn(CSI[i].getReg());

        // Insert the spill to the stack frame.
        TII.storeRegToStackSlot(*EntryBlock, I, CSI[i].getReg(), true,
                                CSI[i].getFrameIdx(), CSI[i].getRegClass());
      }
    }

    // Restore using target interface.
    for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri) {
      MachineBasicBlock* MBB = ReturnBlocks[ri];
      I = MBB->end(); --I;

      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = I;
      while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
        I = I2;

      bool AtStart = I == MBB->begin();
      MachineBasicBlock::iterator BeforeI = I;
      if (!AtStart)
        --BeforeI;

      // Restore all registers immediately before the return and any
      // terminators that preceed it.
      if (!TII.restoreCalleeSavedRegisters(*MBB, I, CSI)) {
        for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
          TII.loadRegFromStackSlot(*MBB, I, CSI[i].getReg(),
                                   CSI[i].getFrameIdx(),
                                   CSI[i].getRegClass());
          assert(I != MBB->begin() &&
                 "loadRegFromStackSlot didn't insert any code!");
          // Insert in reverse order.  loadRegFromStackSlot can insert
          // multiple instructions.
          if (AtStart)
            I = MBB->begin();
          else {
            I = BeforeI;
            ++I;
          }
        }
      }
    }
    return;
  }

  // Insert spills.
  std::vector<CalleeSavedInfo> blockCSI;
  for (CSRegBlockMap::iterator BI = CSRSave.begin(),
         BE = CSRSave.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet save = BI->second;

    if (save.empty())
      continue;

    blockCSI.clear();
    for (CSRegSet::iterator RI = save.begin(),
           RE = save.end(); RI != RE; ++RI) {
      blockCSI.push_back(CSI[*RI]);
    }
    assert(blockCSI.size() > 0 &&
           "Could not collect callee saved register info");

    I = MBB->begin();

    // When shrink wrapping, use stack slot stores/loads.
    for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
      // Add the callee-saved register as live-in.
      // It's killed at the spill.
      MBB->addLiveIn(blockCSI[i].getReg());

      // Insert the spill to the stack frame.
      TII.storeRegToStackSlot(*MBB, I, blockCSI[i].getReg(),
                              true,
                              blockCSI[i].getFrameIdx(),
                              blockCSI[i].getRegClass());
    }
  }

  for (CSRegBlockMap::iterator BI = CSRRestore.begin(),
         BE = CSRRestore.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet restore = BI->second;

    if (restore.empty())
      continue;

    blockCSI.clear();
    for (CSRegSet::iterator RI = restore.begin(),
           RE = restore.end(); RI != RE; ++RI) {
      blockCSI.push_back(CSI[*RI]);
    }
    assert(blockCSI.size() > 0 &&
           "Could not find callee saved register info");

    // If MBB is empty and needs restores, insert at the _beginning_.
    if (MBB->empty()) {
      I = MBB->begin();
    } else {
      I = MBB->end();
      --I;

      // Skip over all terminator instructions, which are part of the
      // return sequence.
      if (! I->getDesc().isTerminator()) {
        ++I;
      } else {
        MachineBasicBlock::iterator I2 = I;
        while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
          I = I2;
      }
    }

    bool AtStart = I == MBB->begin();
    MachineBasicBlock::iterator BeforeI = I;
    if (!AtStart)
      --BeforeI;

    // Restore all registers immediately before the return and any
    // terminators that preceed it.
    for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
      TII.loadRegFromStackSlot(*MBB, I, blockCSI[i].getReg(),
                               blockCSI[i].getFrameIdx(),
                               blockCSI[i].getRegClass());
      assert(I != MBB->begin() &&
             "loadRegFromStackSlot didn't insert any code!");
      // Insert in reverse order.  loadRegFromStackSlot can insert
      // multiple instructions.
      if (AtStart)
        I = MBB->begin();
      else {
        I = BeforeI;
        ++I;
      }
    }
  }
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
static inline void
AdjustStackOffset(MachineFrameInfo *FFI, int FrameIdx,
                  bool StackGrowsDown, int64_t &Offset,
                  unsigned &MaxAlign) {
  // If the stack grows down, add the object size to find the lowest address.
  if (StackGrowsDown)
    Offset += FFI->getObjectSize(FrameIdx);

  unsigned Align = FFI->getObjectAlignment(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Align);

  // Adjust to alignment boundary.
  Offset = (Offset + Align - 1) / Align * Align;

  if (StackGrowsDown) {
    FFI->setObjectOffset(FrameIdx, -Offset); // Set the computed offset
  } else {
    FFI->setObjectOffset(FrameIdx, Offset);
    Offset += FFI->getObjectSize(FrameIdx);
  }
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void PEI::calculateFrameObjectOffsets(MachineFunction &Fn) {
  const TargetFrameInfo &TFI = *Fn.getTarget().getFrameInfo();

  bool StackGrowsDown =
    TFI.getStackGrowthDirection() == TargetFrameInfo::StackGrowsDown;

  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo *FFI = Fn.getFrameInfo();

  unsigned MaxAlign = 1;

  // Start at the beginning of the local area.
  // The Offset is the distance from the stack top in the direction
  // of stack growth -- so it's always nonnegative.
  int LocalAreaOffset = TFI.getOffsetOfLocalArea();
  if (StackGrowsDown)
    LocalAreaOffset = -LocalAreaOffset;
  assert(LocalAreaOffset >= 0
         && "Local area offset should be in direction of stack growth");
  int64_t Offset = LocalAreaOffset;

  // If there are fixed sized objects that are preallocated in the local area,
  // non-fixed objects can't be allocated right at the start of local area.
  // We currently don't support filling in holes in between fixed sized
  // objects, so we adjust 'Offset' to point to the end of last fixed sized
  // preallocated object.
  for (int i = FFI->getObjectIndexBegin(); i != 0; ++i) {
    int64_t FixedOff;
    if (StackGrowsDown) {
      // The maximum distance from the stack pointer is at lower address of
      // the object -- which is given by offset. For down growing stack
      // the offset is negative, so we negate the offset to get the distance.
      FixedOff = -FFI->getObjectOffset(i);
    } else {
      // The maximum distance from the start pointer is at the upper
      // address of the object.
      FixedOff = FFI->getObjectOffset(i) + FFI->getObjectSize(i);
    }
    if (FixedOff > Offset) Offset = FixedOff;
  }

  // First assign frame offsets to stack objects that are used to spill
  // callee saved registers.
  if (StackGrowsDown) {
    for (unsigned i = MinCSFrameIndex; i <= MaxCSFrameIndex; ++i) {
      // If stack grows down, we need to add size of find the lowest
      // address of the object.
      Offset += FFI->getObjectSize(i);

      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack,
      // then increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, -Offset);        // Set the computed offset
    }
  } else {
    int MaxCSFI = MaxCSFrameIndex, MinCSFI = MinCSFrameIndex;
    for (int i = MaxCSFI; i >= MinCSFI ; --i) {
      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack,
      // then increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, Offset);
      Offset += FFI->getObjectSize(i);
    }
  }

  // Make sure the special register scavenging spill slot is closest to the
  // frame pointer if a frame pointer is required.
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  if (RS && RegInfo->hasFP(Fn)) {
    int SFI = RS->getScavengingFrameIndex();
    if (SFI >= 0)
      AdjustStackOffset(FFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure that the stack protector comes before the local variables on the
  // stack.
  if (FFI->getStackProtectorIndex() >= 0)
    AdjustStackOffset(FFI, FFI->getStackProtectorIndex(), StackGrowsDown,
                      Offset, MaxAlign);

  // Then assign frame offsets to stack objects that are not used to spill
  // callee saved registers.
  for (unsigned i = 0, e = FFI->getObjectIndexEnd(); i != e; ++i) {
    if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
      continue;
    if (RS && (int)i == RS->getScavengingFrameIndex())
      continue;
    if (FFI->isDeadObjectIndex(i))
      continue;
    if (FFI->getStackProtectorIndex() == (int)i)
      continue;

    AdjustStackOffset(FFI, i, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure the special register scavenging spill slot is closest to the
  // stack pointer.
  if (RS && !RegInfo->hasFP(Fn)) {
    int SFI = RS->getScavengingFrameIndex();
    if (SFI >= 0)
      AdjustStackOffset(FFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  if (!RegInfo->targetHandlesStackFrameRounding()) {
    // If we have reserved argument space for call sites in the function
    // immediately on entry to the current function, count it as part of the
    // overall stack size.
    if (FFI->hasCalls() && RegInfo->hasReservedCallFrame(Fn))
      Offset += FFI->getMaxCallFrameSize();

    // Round up the size to a multiple of the alignment.  If the function has
    // any calls or alloca's, align to the target's StackAlignment value to
    // ensure that the callee's frame or the alloca data is suitably aligned;
    // otherwise, for leaf functions, align to the TransientStackAlignment
    // value.
    unsigned StackAlign;
    if (FFI->hasCalls() || FFI->hasVarSizedObjects() ||
        (RegInfo->needsStackRealignment(Fn) && FFI->getObjectIndexEnd() != 0))
      StackAlign = TFI.getStackAlignment();
    else
      StackAlign = TFI.getTransientStackAlignment();
    // If the frame pointer is eliminated, all frame offsets will be relative
    // to SP not FP; align to MaxAlign so this works.
    StackAlign = std::max(StackAlign, MaxAlign);
    unsigned AlignMask = StackAlign - 1;
    Offset = (Offset + AlignMask) & ~uint64_t(AlignMask);
  }

  // Update frame info to pretend that this is part of the stack...
  FFI->setStackSize(Offset - LocalAreaOffset);

  // Remember the required stack alignment in case targets need it to perform
  // dynamic stack alignment.
  if (MaxAlign > FFI->getMaxAlignment())
    FFI->setMaxAlignment(MaxAlign);
}


/// insertPrologEpilogCode - Scan the function for modified callee saved
/// registers, insert spill code for these callee saved registers, then add
/// prolog and epilog code to the function.
///
void PEI::insertPrologEpilogCode(MachineFunction &Fn) {
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();

  // Add prologue to the function...
  TRI->emitPrologue(Fn);

  // Add epilogue to restore the callee-save registers in each exiting block
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    // If last instruction is a return instruction, add an epilogue
    if (!I->empty() && I->back().getDesc().isReturn())
      TRI->emitEpilogue(Fn, *I);
  }
}


/// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
/// register references and actual offsets.
///
void PEI::replaceFrameIndices(MachineFunction &Fn) {
  if (!Fn.getFrameInfo()->hasStackObjects()) return; // Nothing to do?

  const TargetMachine &TM = Fn.getTarget();
  assert(TM.getRegisterInfo() && "TM::getRegisterInfo() must be implemented!");
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();
  const TargetFrameInfo *TFI = TM.getFrameInfo();
  bool StackGrowsDown =
    TFI->getStackGrowthDirection() == TargetFrameInfo::StackGrowsDown;
  int FrameSetupOpcode   = TRI.getCallFrameSetupOpcode();
  int FrameDestroyOpcode = TRI.getCallFrameDestroyOpcode();

  for (MachineFunction::iterator BB = Fn.begin(),
         E = Fn.end(); BB != E; ++BB) {
    int SPAdj = 0;  // SP offset due to call frame setup / destroy.
    if (RS && !FrameIndexVirtualScavenging) RS->enterBasicBlock(BB);

    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {

      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        // Remember how much SP has been adjusted to create the call
        // frame.
        int Size = I->getOperand(0).getImm();

        if ((!StackGrowsDown && I->getOpcode() == FrameSetupOpcode) ||
            (StackGrowsDown && I->getOpcode() == FrameDestroyOpcode))
          Size = -Size;

        SPAdj += Size;

        MachineBasicBlock::iterator PrevI = BB->end();
        if (I != BB->begin()) PrevI = prior(I);
        TRI.eliminateCallFramePseudoInstr(Fn, *BB, I);

        // Visit the instructions created by eliminateCallFramePseudoInstr().
        if (PrevI == BB->end())
          I = BB->begin();     // The replaced instr was the first in the block.
        else
          I = next(PrevI);
        continue;
      }

      MachineInstr *MI = I;
      bool DoIncr = true;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
        if (MI->getOperand(i).isFI()) {
          // Some instructions (e.g. inline asm instructions) can have
          // multiple frame indices and/or cause eliminateFrameIndex
          // to insert more than one instruction. We need the register
          // scavenger to go through all of these instructions so that
          // it can update its register information. We keep the
          // iterator at the point before insertion so that we can
          // revisit them in full.
          bool AtBeginning = (I == BB->begin());
          if (!AtBeginning) --I;

          // If this instruction has a FrameIndex operand, we need to
          // use that target machine register info object to eliminate
          // it.
          int Value;
          unsigned VReg =
            TRI.eliminateFrameIndex(MI, SPAdj, &Value,
                                    FrameIndexVirtualScavenging ?  NULL : RS);
          if (VReg) {
            assert (FrameIndexVirtualScavenging &&
                    "Not scavenging, but virtual returned from "
                    "eliminateFrameIndex()!");
            FrameConstantRegMap[VReg] = FrameConstantEntry(Value, SPAdj);
          }

          // Reset the iterator if we were at the beginning of the BB.
          if (AtBeginning) {
            I = BB->begin();
            DoIncr = false;
          }

          MI = 0;
          break;
        }

      if (DoIncr && I != BB->end()) ++I;

      // Update register states.
      if (RS && !FrameIndexVirtualScavenging && MI) RS->forward(MI);
    }

    assert(SPAdj == 0 && "Unbalanced call frame setup / destroy pairs?");
  }
}

/// findLastUseReg - find the killing use of the specified register within
/// the instruciton range. Return the operand number of the kill in Operand.
static MachineBasicBlock::iterator
findLastUseReg(MachineBasicBlock::iterator I, MachineBasicBlock::iterator ME,
               unsigned Reg, unsigned *Operand) {
  // Scan forward to find the last use of this virtual register
  for (++I; I != ME; ++I) {
    MachineInstr *MI = I;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if (MI->getOperand(i).isReg()) {
        unsigned OpReg = MI->getOperand(i).getReg();
        if (OpReg == 0 || !TargetRegisterInfo::isVirtualRegister(OpReg))
          continue;
        assert (OpReg == Reg
                && "overlapping use of scavenged index register!");
        // If this is the killing use, we're done
        if (MI->getOperand(i).isKill()) {
          if (Operand)
            *Operand = i;
          return I;
        }
      }
  }
  // If we hit the end of the basic block, there was no kill of
  // the virtual register, which is wrong.
  assert (0 && "scavenged index register never killed!");
  return ME;
}

/// scavengeFrameVirtualRegs - Replace all frame index virtual registers
/// with physical registers. Use the register scavenger to find an
/// appropriate register to use.
void PEI::scavengeFrameVirtualRegs(MachineFunction &Fn) {
  // Run through the instructions and find any virtual registers.
  for (MachineFunction::iterator BB = Fn.begin(),
       E = Fn.end(); BB != E; ++BB) {
    RS->enterBasicBlock(BB);

    unsigned CurrentVirtReg = 0;
    unsigned CurrentScratchReg = 0;
    bool havePrevValue = false;
    unsigned PrevScratchReg = 0;
    int PrevValue = 0;
    MachineInstr *PrevLastUseMI = NULL;
    unsigned PrevLastUseOp = 0;
    bool trackingCurrentValue = false;
    int SPAdj = 0;
    int Value = 0;

    // The instruction stream may change in the loop, so check BB->end()
    // directly.
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
      MachineInstr *MI = I;
      // Likewise, call getNumOperands() each iteration, as the MI may change
      // inside the loop (with 'i' updated accordingly).
      for (unsigned i = 0; i != MI->getNumOperands(); ++i)
        if (MI->getOperand(i).isReg()) {
          MachineOperand &MO = MI->getOperand(i);
          unsigned Reg = MO.getReg();
          if (Reg == 0)
            continue;
          if (!TargetRegisterInfo::isVirtualRegister(Reg)) {
            // If we have an active scavenged register, we shouldn't be
            // seeing any references to it.
            assert (Reg != CurrentScratchReg
                    && "overlapping use of scavenged frame index register!");

            // If we have a previous scratch reg, check and see if anything
            // here kills whatever value is in there.
            if (Reg == PrevScratchReg) {
              if (MO.isUse()) {
                // Two-address operands implicitly kill
                if (MO.isKill() || MI->isRegTiedToDefOperand(i)) {
                  havePrevValue = false;
                  PrevScratchReg = 0;
                }
              } else {
                assert (MO.isDef());
                havePrevValue = false;
                PrevScratchReg = 0;
              }
            }
            continue;
          }

          // Have we already allocated a scratch register for this virtual?
          if (Reg != CurrentVirtReg) {
            // When we first encounter a new virtual register, it
            // must be a definition.
            assert(MI->getOperand(i).isDef() &&
                   "frame index virtual missing def!");
            // We can't have nested virtual register live ranges because
            // there's only a guarantee of one scavenged register at a time.
            assert (CurrentVirtReg == 0 &&
                    "overlapping frame index virtual registers!");

            // If the target gave us information about what's in the register,
            // we can use that to re-use scratch regs.
            DenseMap<unsigned, FrameConstantEntry>::iterator Entry =
              FrameConstantRegMap.find(Reg);
            trackingCurrentValue = Entry != FrameConstantRegMap.end();
            if (trackingCurrentValue) {
              SPAdj = (*Entry).second.second;
              Value = (*Entry).second.first;
            } else
              SPAdj = Value = 0;

            // If the scratch register from the last allocation is still
            // available, see if the value matches. If it does, just re-use it.
            if (trackingCurrentValue && havePrevValue && PrevValue == Value) {
              // FIXME: This assumes that the instructions in the live range
              // for the virtual register are exclusively for the purpose
              // of populating the value in the register. That's reasonable
              // for these frame index registers, but it's still a very, very
              // strong assumption. Perhaps this implies that the frame index
              // elimination should be before register allocation, with
              // conservative heuristics since we'll know less then, and
              // the reuse calculations done directly when doing the code-gen?

              // Find the last use of the new virtual register. Remove all
              // instruction between here and there, and update the current
              // instruction to reference the last use insn instead.
              MachineBasicBlock::iterator LastUseMI =
                findLastUseReg(I, BB->end(), Reg, &i);
              // Remove all instructions up 'til the last use, since they're
              // just calculating the value we already have.
              BB->erase(I, LastUseMI);
              MI = I = LastUseMI;

              CurrentScratchReg = PrevScratchReg;
              // Extend the live range of the register
              PrevLastUseMI->getOperand(PrevLastUseOp).setIsKill(false);
              RS->setUsed(CurrentScratchReg);
            } else {
              CurrentVirtReg = Reg;
              const TargetRegisterClass *RC = Fn.getRegInfo().getRegClass(Reg);
              CurrentScratchReg = RS->FindUnusedReg(RC);
              if (CurrentScratchReg == 0)
                // No register is "free". Scavenge a register.
                CurrentScratchReg = RS->scavengeRegister(RC, I, SPAdj);

              PrevValue = Value;
            }
          }
          assert (CurrentScratchReg && "Missing scratch register!");
          MI->getOperand(i).setReg(CurrentScratchReg);

          // If this is the last use of the register, stop tracking it.
          if (MI->getOperand(i).isKill()) {
            PrevScratchReg = CurrentScratchReg;
            PrevLastUseMI = MI;
            PrevLastUseOp = i;
            CurrentScratchReg = CurrentVirtReg = 0;
            havePrevValue = trackingCurrentValue;
          }
        }
      RS->forward(MI);
    }
  }
}
