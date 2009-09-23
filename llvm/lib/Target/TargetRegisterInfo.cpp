//===- TargetRegisterInfo.cpp - Target Register Information Implementation ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetRegisterInfo interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/ADT/BitVector.h"

using namespace llvm;

TargetRegisterInfo::TargetRegisterInfo(const TargetRegisterDesc *D, unsigned NR,
                             regclass_iterator RCB, regclass_iterator RCE,
                             int CFSO, int CFDO,
                             const unsigned* subregs, const unsigned subregsize,
                         const unsigned* superregs, const unsigned superregsize,
                         const unsigned* aliases, const unsigned aliasessize)
  : SubregHash(subregs), SubregHashSize(subregsize),
    SuperregHash(superregs), SuperregHashSize(superregsize),
    AliasesHash(aliases), AliasesHashSize(aliasessize),
    Desc(D), NumRegs(NR), RegClassBegin(RCB), RegClassEnd(RCE) {
  assert(NumRegs < FirstVirtualRegister &&
         "Target has too many physical registers!");

  CallFrameSetupOpcode   = CFSO;
  CallFrameDestroyOpcode = CFDO;
}

TargetRegisterInfo::~TargetRegisterInfo() {}

/// getPhysicalRegisterRegClass - Returns the Register Class of a physical
/// register of the given type. If type is EVT::Other, then just return any
/// register class the register belongs to.
const TargetRegisterClass *
TargetRegisterInfo::getPhysicalRegisterRegClass(unsigned reg, EVT VT) const {
  assert(isPhysicalRegister(reg) && "reg must be a physical register");

  // Pick the most super register class of the right type that contains
  // this physreg.
  const TargetRegisterClass* BestRC = 0;
  for (regclass_iterator I = regclass_begin(), E = regclass_end(); I != E; ++I){
    const TargetRegisterClass* RC = *I;
    if ((VT == MVT::Other || RC->hasType(VT)) && RC->contains(reg) &&
        (!BestRC || BestRC->hasSuperClass(RC)))
      BestRC = RC;
  }

  assert(BestRC && "Couldn't find the register class");
  return BestRC;
}

/// getAllocatableSetForRC - Toggle the bits that represent allocatable
/// registers for the specific register class.
static void getAllocatableSetForRC(MachineFunction &MF,
                                   const TargetRegisterClass *RC, BitVector &R){  
  for (TargetRegisterClass::iterator I = RC->allocation_order_begin(MF),
         E = RC->allocation_order_end(MF); I != E; ++I)
    R.set(*I);
}

BitVector TargetRegisterInfo::getAllocatableSet(MachineFunction &MF,
                                          const TargetRegisterClass *RC) const {
  BitVector Allocatable(NumRegs);
  if (RC) {
    getAllocatableSetForRC(MF, RC, Allocatable);
    return Allocatable;
  }

  for (TargetRegisterInfo::regclass_iterator I = regclass_begin(),
         E = regclass_end(); I != E; ++I)
    getAllocatableSetForRC(MF, *I, Allocatable);
  return Allocatable;
}

/// getFrameIndexOffset - Returns the displacement from the frame register to
/// the stack frame of the specified index. This is the default implementation
/// which is overridden for some targets.
int TargetRegisterInfo::getFrameIndexOffset(MachineFunction &MF, int FI) const {
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->getObjectOffset(FI) + MFI->getStackSize() -
    TFI.getOffsetOfLocalArea() + MFI->getOffsetAdjustment();
}

/// getInitialFrameState - Returns a list of machine moves that are assumed
/// on entry to a function.
void
TargetRegisterInfo::getInitialFrameState(std::vector<MachineMove> &Moves) const {
  // Default is to do nothing.
}

const TargetRegisterClass *
llvm::getCommonSubClass(const TargetRegisterClass *A,
                        const TargetRegisterClass *B) {
  // First take care of the trivial cases
  if (A == B)
    return A;
  if (!A || !B)
    return 0;

  // If B is a subclass of A, it will be handled in the loop below
  if (B->hasSubClass(A))
    return A;

  const TargetRegisterClass *Best = 0;
  for (TargetRegisterClass::sc_iterator I = A->subclasses_begin();
       const TargetRegisterClass *X = *I; ++I) {
    if (X == B)
      return B;                 // B is a subclass of A

    // X must be a common subclass of A and B
    if (!B->hasSubClass(X))
      continue;

    // A superclass is definitely better.
    if (!Best || Best->hasSuperClass(X)) {
      Best = X;
      continue;
    }

    // A subclass is definitely worse
    if (Best->hasSubClass(X))
      continue;

    // Best and *I have no super/sub class relation - pick the larger class, or
    // the smaller spill size.
    int nb = std::distance(Best->begin(), Best->end());
    int ni = std::distance(X->begin(), X->end());
    if (ni>nb || (ni==nb && X->getSize() < Best->getSize()))
      Best = X;
  }
  return Best;
}
