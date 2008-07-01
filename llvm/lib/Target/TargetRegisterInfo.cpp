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
                             const unsigned* subregs, const unsigned subregsize)
  : Desc(D), NumRegs(NR), RegClassBegin(RCB), RegClassEnd(RCE),
    SubregHash(subregs), SubregHashSize(subregsize) {
  assert(NumRegs < FirstVirtualRegister &&
         "Target has too many physical registers!");

  CallFrameSetupOpcode   = CFSO;
  CallFrameDestroyOpcode = CFDO;
}

TargetRegisterInfo::~TargetRegisterInfo() {}

namespace {
  // Sort according to super- / sub- class relations.
  // i.e. super- register class < sub- register class.
  struct RCCompare {
    bool operator()(const TargetRegisterClass* const &LHS,
                    const TargetRegisterClass* const &RHS) {
      return RHS->hasSuperClass(LHS);
    }
  };
}

/// getPhysicalRegisterRegClass - Returns the Register Class of a physical
/// register of the given type. If type is MVT::Other, then just return any
/// register class the register belongs to.
const TargetRegisterClass *
TargetRegisterInfo::getPhysicalRegisterRegClass(unsigned reg, MVT VT) const {
  assert(isPhysicalRegister(reg) && "reg must be a physical register");

  // Pick the register class of the right type that contains this physreg.
  SmallVector<const TargetRegisterClass*, 4> RCs;
  for (regclass_iterator I = regclass_begin(), E = regclass_end(); I != E; ++I){
    if ((VT == MVT::Other || (*I)->hasType(VT)) && (*I)->contains(reg))
      RCs.push_back(*I);
  }

  if (RCs.size() == 1)
    return RCs[0];

  if (RCs.size()) {
    // Multiple compatible register classes. Get the super- class.
    std::stable_sort(RCs.begin(), RCs.end(), RCCompare());
    return RCs[0];
  }

  assert(false && "Couldn't find the register class");
  return 0;
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
/// which is likely incorrect for the target.
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

