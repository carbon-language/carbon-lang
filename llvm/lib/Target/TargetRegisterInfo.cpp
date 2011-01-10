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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

TargetRegisterInfo::TargetRegisterInfo(const TargetRegisterDesc *D, unsigned NR,
                             regclass_iterator RCB, regclass_iterator RCE,
                             const char *const *subregindexnames,
                             int CFSO, int CFDO,
                             const unsigned* subregs, const unsigned subregsize,
                         const unsigned* aliases, const unsigned aliasessize)
  : SubregHash(subregs), SubregHashSize(subregsize),
    AliasesHash(aliases), AliasesHashSize(aliasessize),
    Desc(D), SubRegIndexNames(subregindexnames), NumRegs(NR),
    RegClassBegin(RCB), RegClassEnd(RCE) {
  assert(isPhysicalRegister(NumRegs) &&
         "Target has too many physical registers!");

  CallFrameSetupOpcode   = CFSO;
  CallFrameDestroyOpcode = CFDO;
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
  for (TargetRegisterClass::iterator I = RC->allocation_order_begin(MF),
         E = RC->allocation_order_end(MF); I != E; ++I)
    R.set(*I);
}

BitVector TargetRegisterInfo::getAllocatableSet(const MachineFunction &MF,
                                          const TargetRegisterClass *RC) const {
  BitVector Allocatable(NumRegs);
  if (RC) {
    getAllocatableSetForRC(MF, RC, Allocatable);
  } else {
    for (TargetRegisterInfo::regclass_iterator I = regclass_begin(),
         E = regclass_end(); I != E; ++I)
      getAllocatableSetForRC(MF, *I, Allocatable);
  }

  // Mask out the reserved registers
  BitVector Reserved = getReservedRegs(MF);
  Allocatable &= Reserved.flip();

  return Allocatable;
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
