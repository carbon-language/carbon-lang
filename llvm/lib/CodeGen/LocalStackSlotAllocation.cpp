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
#include "llvm/ADT/Statistic.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"

using namespace llvm;

STATISTIC(NumAllocations, "Number of frame indices processed");

namespace {
  class LocalStackSlotPass: public MachineFunctionPass {
    void calculateFrameObjectOffsets(MachineFunction &Fn);
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
  calculateFrameObjectOffsets(MF);
  return true;
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
static inline void
AdjustStackOffset(MachineFrameInfo *MFI, int FrameIdx, int64_t &Offset,
                  unsigned &MaxAlign) {
  unsigned Align = MFI->getObjectAlignment(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Align);

  // Adjust to alignment boundary.
  Offset = (Offset + Align - 1) / Align * Align;

  DEBUG(dbgs() << "Allocate FI(" << FrameIdx << ") to local offset "
        << Offset << "\n");
  MFI->mapLocalFrameObject(FrameIdx, Offset);
  Offset += MFI->getObjectSize(FrameIdx);

  ++NumAllocations;
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void LocalStackSlotPass::calculateFrameObjectOffsets(MachineFunction &Fn) {
  const TargetFrameInfo &TFI = *Fn.getTarget().getFrameInfo();

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
