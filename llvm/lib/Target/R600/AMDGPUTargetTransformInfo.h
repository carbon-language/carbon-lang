//===-- AMDGPUTargetTransformInfo.h - AMDGPU specific TTI -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// AMDGPU target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_AMDGPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_R600_AMDGPUTARGETTRANSFORMINFO_H

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

class AMDGPUTTIImpl : public BasicTTIImplBase<AMDGPUTTIImpl> {
  typedef BasicTTIImplBase<AMDGPUTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;

  const AMDGPUSubtarget *ST;

public:
  explicit AMDGPUTTIImpl(const AMDGPUTargetMachine *TM = nullptr)
      : BaseT(TM), ST(TM->getSubtargetImpl()) {}

  // Provide value semantics. MSVC requires that we spell all of these out.
  AMDGPUTTIImpl(const AMDGPUTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)), ST(Arg.ST) {}
  AMDGPUTTIImpl(AMDGPUTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))), ST(std::move(Arg.ST)) {}
  AMDGPUTTIImpl &operator=(const AMDGPUTTIImpl &RHS) {
    BaseT::operator=(static_cast<const BaseT &>(RHS));
    ST = RHS.ST;
    return *this;
  }
  AMDGPUTTIImpl &operator=(AMDGPUTTIImpl &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    ST = std::move(RHS.ST);
    return *this;
  }

  bool hasBranchDivergence() { return true; }

  void getUnrollingPreferences(const Function *F, Loop *L,
                               TTI::UnrollingPreferences &UP);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) {
    assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
    return ST->hasBCNT(TyWidth) ? TTI::PSK_FastHardware : TTI::PSK_Software;
  }

  unsigned getNumberOfRegisters(bool Vector);
  unsigned getRegisterBitWidth(bool Vector);
  unsigned getMaxInterleaveFactor();
};

} // end namespace llvm

#endif
