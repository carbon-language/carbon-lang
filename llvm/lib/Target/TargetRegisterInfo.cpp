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

/// getAllocatableClass - Return the maximal subclass of the given register
/// class that is alloctable, or NULL.
const TargetRegisterClass *
TargetRegisterInfo::getAllocatableClass(const TargetRegisterClass *RC) const {
  if (!RC || RC->isAllocatable())
    return RC;

  const unsigned *SubClass = RC->getSubClassMask();
  for (unsigned Base = 0, BaseE = getNumRegClasses();
       Base < BaseE; Base += 32) {
    unsigned Idx = Base;
    for (unsigned Mask = *SubClass++; Mask; Mask >>= 1) {
      unsigned Offset = CountTrailingZeros_32(Mask);
      const TargetRegisterClass *SubRC = getRegClass(Idx + Offset);
      if (SubRC->isAllocatable())
        return SubRC;
      Mask >>= Offset;
      Idx += Offset + 1;
    }
  }
  return NULL;
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
  assert(RC->isAllocatable() && "invalid for nonallocatable sets");
  ArrayRef<uint16_t> Order = RC->getRawAllocationOrder(MF);
  for (unsigned i = 0; i != Order.size(); ++i)
    R.set(Order[i]);
}

BitVector TargetRegisterInfo::getAllocatableSet(const MachineFunction &MF,
                                          const TargetRegisterClass *RC) const {
  BitVector Allocatable(getNumRegs());
  if (RC) {
    // A register class with no allocatable subclass returns an empty set.
    const TargetRegisterClass *SubClass = getAllocatableClass(RC);
    if (SubClass)
      getAllocatableSetForRC(MF, SubClass, Allocatable);
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

const TargetRegisterClass *
TargetRegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                             const TargetRegisterClass *B,
                                             unsigned Idx) const {
  assert(A && B && "Missing register class");
  assert(Idx && "Bad sub-register index");

  // Find Idx in the list of super-register indices.
  const uint16_t *SRI = B->getSuperRegIndices();
  unsigned Offset = 0;
  while (SRI[Offset] != Idx) {
    if (!SRI[Offset])
      return 0;
    ++Offset;
  }

  // The register class bit mask corresponding to SRI[Offset]. The bit mask
  // contains all register classes that are projected into B by Idx. Find a
  // class that is also a sub-class of A.
  const unsigned RCMaskWords = (getNumRegClasses()+31)/32;
  const uint32_t *TV = B->getSubClassMask() + (Offset + 1) * RCMaskWords;
  const uint32_t *SC = A->getSubClassMask();

  // Find the first common register class in TV and SC.
  for (unsigned i = 0; i != RCMaskWords ; ++i)
    if (unsigned Common = TV[i] & SC[i])
      return getRegClass(32*i + CountTrailingZeros_32(Common));
  return 0;
}
