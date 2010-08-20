//===- LocalStackSlotAllocation.cpp - Pre-allocate locals to stack slots --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass assigns local frame indices to stack slots relative to one another
// and allocates additional base registers to access them when the target
// estimates the are likely to be out of range of stack pointer and frame
// pointer relative addressing.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "localstackalloc"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"

using namespace llvm;

STATISTIC(NumAllocations, "Number of frame indices allocated into local block");
STATISTIC(NumBaseRegisters, "Number of virtual frame base registers allocated");
STATISTIC(NumReplacements, "Number of frame indices references replaced");

namespace {
  class LocalStackSlotPass: public MachineFunctionPass {
    SmallVector<int64_t,16> LocalOffsets;

    void AdjustStackOffset(MachineFrameInfo *MFI, int FrameIdx, int64_t &Offset,
                           unsigned &MaxAlign);
    void calculateFrameObjectOffsets(MachineFunction &Fn);
    bool insertFrameReferenceRegisters(MachineFunction &Fn);
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit LocalStackSlotPass() : MachineFunctionPass(ID) { }
    bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    const char *getPassName() const {
      return "Local Stack Slot Allocation";
    }

  private:
  };
} // end anonymous namespace

char LocalStackSlotPass::ID = 0;

FunctionPass *llvm::createLocalStackSlotAllocationPass() {
  return new LocalStackSlotPass();
}

bool LocalStackSlotPass::runOnMachineFunction(MachineFunction &MF) {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned LocalObjectCount = MFI->getObjectIndexEnd();

  // Early exit if there are no locals to consider
  if (!LocalObjectCount)
    return true;

  // Make sure we have enough space to store the local offsets.
  LocalOffsets.resize(MFI->getObjectIndexEnd());

  // Lay out the local blob.
  calculateFrameObjectOffsets(MF);

  // Insert virtual base registers to resolve frame index references.
  bool UsedBaseRegs = insertFrameReferenceRegisters(MF);

  // Tell MFI whether any base registers were allocated. PEI will only
  // want to use the local block allocations from this pass if there were any.
  // Otherwise, PEI can do a bit better job of getting the alignment right
  // without a hole at the start since it knows the alignment of the stack
  // at the start of local allocation, and this pass doesn't.
  MFI->setUseLocalStackAllocationBlock(UsedBaseRegs);

  return true;
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
void LocalStackSlotPass::AdjustStackOffset(MachineFrameInfo *MFI,
                                           int FrameIdx, int64_t &Offset,
                                           unsigned &MaxAlign) {
  unsigned Align = MFI->getObjectAlignment(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Align);

  // Adjust to alignment boundary.
  Offset = (Offset + Align - 1) / Align * Align;

  DEBUG(dbgs() << "Allocate FI(" << FrameIdx << ") to local offset "
        << Offset << "\n");
  // Keep the offset available for base register allocation
  LocalOffsets[FrameIdx] = Offset;
  // And tell MFI about it for PEI to use later
  MFI->mapLocalFrameObject(FrameIdx, Offset);
  Offset += MFI->getObjectSize(FrameIdx);

  ++NumAllocations;
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void LocalStackSlotPass::calculateFrameObjectOffsets(MachineFunction &Fn) {
  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo *MFI = Fn.getFrameInfo();
  int64_t Offset = 0;
  unsigned MaxAlign = 0;

  // Make sure that the stack protector comes before the local variables on the
  // stack.
  SmallSet<int, 16> LargeStackObjs;
  if (MFI->getStackProtectorIndex() >= 0) {
    AdjustStackOffset(MFI, MFI->getStackProtectorIndex(), Offset, MaxAlign);

    // Assign large stack objects first.
    for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
      if (MFI->isDeadObjectIndex(i))
        continue;
      if (MFI->getStackProtectorIndex() == (int)i)
        continue;
      if (!MFI->MayNeedStackProtector(i))
        continue;

      AdjustStackOffset(MFI, i, Offset, MaxAlign);
      LargeStackObjs.insert(i);
    }
  }

  // Then assign frame offsets to stack objects that are not used to spill
  // callee saved registers.
  for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (MFI->isDeadObjectIndex(i))
      continue;
    if (MFI->getStackProtectorIndex() == (int)i)
      continue;
    if (LargeStackObjs.count(i))
      continue;

    AdjustStackOffset(MFI, i, Offset, MaxAlign);
  }

  // Remember how big this blob of stack space is
  MFI->setLocalFrameSize(Offset);
  MFI->setLocalFrameMaxAlign(MaxAlign);
}

static inline bool
lookupCandidateBaseReg(const SmallVector<std::pair<unsigned, int64_t>, 8> &Regs,
                       std::pair<unsigned, int64_t> &RegOffset,
                       int64_t LocalFrameOffset,
                       const MachineInstr *MI,
                       const TargetRegisterInfo *TRI) {
  unsigned e = Regs.size();
  for (unsigned i = 0; i < e; ++i) {
    RegOffset = Regs[i];
    // Check if the relative offset from the where the base register references
    // to the target address is in range for the instruction.
    int64_t Offset = LocalFrameOffset - RegOffset.second;
    if (TRI->isFrameOffsetLegal(MI, Offset))
      return true;
  }
  return false;
}

bool LocalStackSlotPass::insertFrameReferenceRegisters(MachineFunction &Fn) {
  // Scan the function's instructions looking for frame index references.
  // For each, ask the target if it wants a virtual base register for it
  // based on what we can tell it about where the local will end up in the
  // stack frame. If it wants one, re-use a suitable one we've previously
  // allocated, or if there isn't one that fits the bill, allocate a new one
  // and ask the target to create a defining instruction for it.
  bool UsedBaseReg = false;

  MachineFrameInfo *MFI = Fn.getFrameInfo();
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();

  for (MachineFunction::iterator BB = Fn.begin(),
         E = Fn.end(); BB != E; ++BB) {
    // A base register definition is a register+offset pair.
    SmallVector<std::pair<unsigned, int64_t>, 8> BaseRegisters;

    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
      MachineInstr *MI = I;
      // Debug value instructions can't be out of range, so they don't need
      // any updates.
      // FIXME: When we extend this stuff to handle functions with both
      // VLAs and dynamic realignment, we should update the debug values
      // to reference the new base pointer when possible.
      if (MI->isDebugValue())
        continue;

      // For now, allocate the base register(s) within the basic block
      // where they're used, and don't try to keep them around outside
      // of that. It may be beneficial to try sharing them more broadly
      // than that, but the increased register pressure makes that a
      // tricky thing to balance. Investigate if re-materializing these
      // becomes an issue.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        // Consider replacing all frame index operands that reference
        // an object allocated in the local block.
        if (MI->getOperand(i).isFI()) {
          int FrameIdx = MI->getOperand(i).getIndex();

          // Don't try this with values not in the local block.
          if (!MFI->isObjectPreAllocated(FrameIdx))
            continue;

          DEBUG(dbgs() << "Considering: " << *MI);
          if (TRI->needsFrameBaseReg(MI, i)) {
            unsigned BaseReg = 0;
            int64_t Offset = 0;

            DEBUG(dbgs() << "  Replacing FI in: " << *MI);

            // If we have a suitable base register available, use it; otherwise
            // create a new one. Note that any offset encoded in the
            // instruction itself will be taken into account by the target,
            // so we don't have to adjust for it here when reusing a base
            // register.
            std::pair<unsigned, int64_t> RegOffset;
            if (lookupCandidateBaseReg(BaseRegisters, RegOffset,
                                       LocalOffsets[FrameIdx], MI, TRI)) {
              DEBUG(dbgs() << "  Reusing base register " <<
                    RegOffset.first << "\n");
              // We found a register to reuse.
              BaseReg = RegOffset.first;
              Offset = LocalOffsets[FrameIdx] - RegOffset.second;
            } else {
              // No previously defined register was in range, so create a
              // new one.
              int64_t InstrOffset = TRI->getFrameIndexInstrOffset(MI, i);
              const TargetRegisterClass *RC = TRI->getPointerRegClass();
              BaseReg = Fn.getRegInfo().createVirtualRegister(RC);

              DEBUG(dbgs() << "  Materializing base register " << BaseReg <<
                    " at frame local offset " <<
                    LocalOffsets[FrameIdx] + InstrOffset << "\n");
              // Tell the target to insert the instruction to initialize
              // the base register.
              TRI->materializeFrameBaseRegister(I, BaseReg, FrameIdx,
                                                InstrOffset);

              // The base register already includes any offset specified
              // by the instruction, so account for that so it doesn't get
              // applied twice.
              Offset = -InstrOffset;

              BaseRegisters.push_back(
                std::pair<unsigned, int64_t>(BaseReg,
                                      LocalOffsets[FrameIdx] + InstrOffset));
              ++NumBaseRegisters;
              UsedBaseReg = true;
            }
            assert(BaseReg != 0 && "Unable to allocate virtual base register!");

            // Modify the instruction to use the new base register rather
            // than the frame index operand.
            TRI->resolveFrameIndex(I, BaseReg, Offset);
            DEBUG(dbgs() << "Resolved: " << *MI);

            ++NumReplacements;
          }

        }
      }
    }
  }
  return UsedBaseReg;
}
