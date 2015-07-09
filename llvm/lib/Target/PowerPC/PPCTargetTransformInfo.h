//===-- PPCTargetTransformInfo.h - PPC specific TTI -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// PPC target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_POWERPC_PPCTARGETTRANSFORMINFO_H

#include "PPC.h"
#include "PPCTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

class PPCTTIImpl : public BasicTTIImplBase<PPCTTIImpl> {
  typedef BasicTTIImplBase<PPCTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const PPCSubtarget *ST;
  const PPCTargetLowering *TLI;

  const PPCSubtarget *getST() const { return ST; }
  const PPCTargetLowering *getTLI() const { return TLI; }

public:
  explicit PPCTTIImpl(const PPCTargetMachine *TM, Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  // Provide value semantics. MSVC requires that we spell all of these out.
  PPCTTIImpl(const PPCTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)), ST(Arg.ST), TLI(Arg.TLI) {}
  PPCTTIImpl(PPCTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))), ST(std::move(Arg.ST)),
        TLI(std::move(Arg.TLI)) {}

  /// \name Scalar TTI Implementations
  /// @{

  using BaseT::getIntImmCost;
  unsigned getIntImmCost(const APInt &Imm, Type *Ty);

  unsigned getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                         Type *Ty);
  unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                         Type *Ty);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth);
  void getUnrollingPreferences(Loop *L, TTI::UnrollingPreferences &UP);

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  bool enableAggressiveInterleaving(bool LoopHasReductions);
  unsigned getNumberOfRegisters(bool Vector);
  unsigned getRegisterBitWidth(bool Vector);
  unsigned getMaxInterleaveFactor(unsigned VF);
  unsigned getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None);
  unsigned getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                          Type *SubTp);
  unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src);
  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy);
  unsigned getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                           unsigned AddressSpace);

  /// @}
};

} // end namespace llvm

#endif
