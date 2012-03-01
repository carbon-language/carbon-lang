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
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

TargetRegisterInfo::TargetRegisterInfo(const TargetRegisterInfoDesc *ID,
                             regclass_iterator RCB, regclass_iterator RCE,
                             const char *const *subregindexnames)
  : InfoDesc(ID), SubRegIndexNames(subregindexnames),
    RegClassBegin(RCB), RegClassEnd(RCE) {
}

TargetRegisterInfo::~TargetRegisterInfo() {}

void PrintReg::print(raw_ostream &OS) const {
  if (!Reg)
    OS << "%noreg";
  else if (TargetRegisterInfo::isStackSlot(Reg))
    OS << "SS#" << TargetRegisterInfo::stackSlot2Index(Reg);
  else if (TargetRegisterInfo::isVirtualRegister(Reg))
    OS << "%vreg" << TargetRegisterInfo::virtReg2Index(Reg);
  else if (TRI && Reg < TRI->getNumRegs())
    OS << '%' << TRI->getName(Reg);
  else
    OS << "%physreg" << Reg;
  if (SubIdx) {
    if (TRI)
      OS << ':' << TRI->getSubRegIndexName(SubIdx);
    else
      OS << ":sub(" << SubIdx << ')';
  }
}

/// getMinimalPhysRegClass - Returns the Register Class of a physical
/// register of the given type, picking the most sub register class of
/// the right type that contains this physreg.
const TargetRegisterClass *
TargetRegisterInfo::getMinimalPhysRegClass(unsigned reg, EVT VT) const {
  assert(isPhysicalRegister(reg) && "reg must be a physical register");

  // Pick the most sub register class of the right type that contains
  // this physreg.
  const TargetRegisterClass* BestRC = 0;
  for (regclass_iterator I = regclass_begin(), E = regclass_end(); I != E; ++I){
    const TargetRegisterClass* RC = *I;
    if ((VT == MVT::Other || RC->hasType(VT)) && RC->contains(reg) &&
        (!BestRC || BestRC->hasSubClass(RC)))
      BestRC = RC;
  }

  assert(BestRC && "Couldn't find the register class");
  return BestRC;
}

/// getAllocatableSetForRC - Toggle the bits that represent allocatable
/// registers for the specific register class.
static void getAllocatableSetForRC(const MachineFunction &MF,
                                   const TargetRegisterClass *RC, BitVector &R){
  ArrayRef<unsigned> Order = RC->getRawAllocationOrder(MF);
  for (unsigned i = 0; i != Order.size(); ++i)
    R.set(Order[i]);
}

BitVector TargetRegisterInfo::getAllocatableSet(const MachineFunction &MF,
                                          const TargetRegisterClass *RC) const {
  BitVector Allocatable(getNumRegs());
  if (RC) {
    getAllocatableSetForRC(MF, RC, Allocatable);
  } else {
    for (TargetRegisterInfo::regclass_iterator I = regclass_begin(),
         E = regclass_end(); I != E; ++I)
      if ((*I)->isAllocatable())
        getAllocatableSetForRC(MF, *I, Allocatable);
  }

  // Mask out the reserved registers
  BitVector Reserved = getReservedRegs(MF);
  Allocatable &= Reserved.flip();

  return Allocatable;
}

const TargetRegisterClass *
TargetRegisterInfo::getCommonSubClass(const TargetRegisterClass *A,
                                      const TargetRegisterClass *B) const {
  // First take care of the trivial cases.
  if (A == B)
    return A;
  if (!A || !B)
    return 0;

  // Register classes are ordered topologically, so the largest common
  // sub-class it the common sub-class with the smallest ID.
  const unsigned *SubA = A->getSubClassMask();
  const unsigned *SubB = B->getSubClassMask();

  // We could start the search from max(A.ID, B.ID), but we are only going to
  // execute 2-3 iterations anyway.
  for (unsigned Base = 0, BaseE = getNumRegClasses(); Base < BaseE; Base += 32)
    if (unsigned Common = *SubA++ & *SubB++)
      return getRegClass(Base + CountTrailingZeros_32(Common));

  // No common sub-class exists.
  return NULL;
}
