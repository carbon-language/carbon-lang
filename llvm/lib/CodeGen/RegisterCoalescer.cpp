//===- RegisterCoalescer.cpp - Generic Register Coalescing Interface -------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic RegisterCoalescer interface which
// is used as the common interface used by all clients and
// implementations of register coalescing.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Pass.h"

using namespace llvm;

// Register the RegisterCoalescer interface, providing a nice name to refer to.
static RegisterAnalysisGroup<RegisterCoalescer> Z("Register Coalescer");
char RegisterCoalescer::ID = 0;

// RegisterCoalescer destructor: DO NOT move this to the header file
// for RegisterCoalescer or else clients of the RegisterCoalescer
// class may not depend on the RegisterCoalescer.o file in the current
// .a file, causing alias analysis support to not be included in the
// tool correctly!
//
RegisterCoalescer::~RegisterCoalescer() {}

unsigned CoalescerPair::compose(unsigned a, unsigned b) const {
  if (!a) return b;
  if (!b) return a;
  return tri_.composeSubRegIndices(a, b);
}

bool CoalescerPair::isMoveInstr(const MachineInstr *MI,
                                unsigned &Src, unsigned &Dst,
                                unsigned &SrcSub, unsigned &DstSub) const {
  if (MI->isCopy()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = MI->getOperand(0).getSubReg();
    Src = MI->getOperand(1).getReg();
    SrcSub = MI->getOperand(1).getSubReg();
  } else if (MI->isSubregToReg()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = compose(MI->getOperand(0).getSubReg(), MI->getOperand(3).getImm());
    Src = MI->getOperand(2).getReg();
    SrcSub = MI->getOperand(2).getSubReg();
  } else
    return false;
  return true;
}

bool CoalescerPair::setRegisters(const MachineInstr *MI) {
  srcReg_ = dstReg_ = subIdx_ = 0;
  newRC_ = 0;
  flipped_ = crossClass_ = false;

  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(MI, Src, Dst, SrcSub, DstSub))
    return false;
  partial_ = SrcSub || DstSub;

  // If one register is a physreg, it must be Dst.
  if (TargetRegisterInfo::isPhysicalRegister(Src)) {
    if (TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
    flipped_ = true;
  }

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();

  if (TargetRegisterInfo::isPhysicalRegister(Dst)) {
    // Eliminate DstSub on a physreg.
    if (DstSub) {
      Dst = tri_.getSubReg(Dst, DstSub);
      if (!Dst) return false;
      DstSub = 0;
    }

    // Eliminate SrcSub by picking a corresponding Dst superregister.
    if (SrcSub) {
      Dst = tri_.getMatchingSuperReg(Dst, SrcSub, MRI.getRegClass(Src));
      if (!Dst) return false;
      SrcSub = 0;
    } else if (!MRI.getRegClass(Src)->contains(Dst)) {
      return false;
    }
  } else {
    // Both registers are virtual.

    // Both registers have subreg indices.
    if (SrcSub && DstSub) {
      // For now we only handle the case of identical indices in commensurate
      // registers: Dreg:ssub_1 + Dreg:ssub_1 -> Dreg
      // FIXME: Handle Qreg:ssub_3 + Dreg:ssub_1 as QReg:dsub_1 + Dreg.
      if (SrcSub != DstSub)
        return false;
      const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
      const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);
      if (!getCommonSubClass(DstRC, SrcRC))
        return false;
      SrcSub = DstSub = 0;
    }

    // There can be no SrcSub.
    if (SrcSub) {
      std::swap(Src, Dst);
      DstSub = SrcSub;
      SrcSub = 0;
      assert(!flipped_ && "Unexpected flip");
      flipped_ = true;
    }

    // Find the new register class.
    const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);
    if (DstSub)
      newRC_ = tri_.getMatchingSuperRegClass(DstRC, SrcRC, DstSub);
    else
      newRC_ = getCommonSubClass(DstRC, SrcRC);
    if (!newRC_)
      return false;
    crossClass_ = newRC_ != DstRC || newRC_ != SrcRC;
  }
  // Check our invariants
  assert(TargetRegisterInfo::isVirtualRegister(Src) && "Src must be virtual");
  assert(!(TargetRegisterInfo::isPhysicalRegister(Dst) && DstSub) &&
         "Cannot have a physical SubIdx");
  srcReg_ = Src;
  dstReg_ = Dst;
  subIdx_ = DstSub;
  return true;
}

bool CoalescerPair::flip() {
  if (subIdx_ || TargetRegisterInfo::isPhysicalRegister(dstReg_))
    return false;
  std::swap(srcReg_, dstReg_);
  flipped_ = !flipped_;
  return true;
}

bool CoalescerPair::isCoalescable(const MachineInstr *MI) const {
  if (!MI)
    return false;
  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(MI, Src, Dst, SrcSub, DstSub))
    return false;

  // Find the virtual register that is srcReg_.
  if (Dst == srcReg_) {
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
  } else if (Src != srcReg_) {
    return false;
  }

  // Now check that Dst matches dstReg_.
  if (TargetRegisterInfo::isPhysicalRegister(dstReg_)) {
    if (!TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    assert(!subIdx_ && "Inconsistent CoalescerPair state.");
    // DstSub could be set for a physreg from INSERT_SUBREG.
    if (DstSub)
      Dst = tri_.getSubReg(Dst, DstSub);
    // Full copy of Src.
    if (!SrcSub)
      return dstReg_ == Dst;
    // This is a partial register copy. Check that the parts match.
    return tri_.getSubReg(dstReg_, SrcSub) == Dst;
  } else {
    // dstReg_ is virtual.
    if (dstReg_ != Dst)
      return false;
    // Registers match, do the subregisters line up?
    return compose(subIdx_, SrcSub) == DstSub;
  }
}

// Because of the way .a files work, we must force the SimpleRC
// implementation to be pulled in if the RegisterCoalescer classes are
// pulled in.  Otherwise we run the risk of RegisterCoalescer being
// used, but the default implementation not being linked into the tool
// that uses it.
DEFINING_FILE_FOR(RegisterCoalescer)
