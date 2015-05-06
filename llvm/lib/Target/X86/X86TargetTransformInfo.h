//===-- X86TargetTransformInfo.h - X86 specific TTI -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// X86 target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86TARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_X86_X86TARGETTRANSFORMINFO_H

#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

class X86TTIImpl : public BasicTTIImplBase<X86TTIImpl> {
  typedef BasicTTIImplBase<X86TTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const X86Subtarget *ST;
  const X86TargetLowering *TLI;

  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract);

  const X86Subtarget *getST() const { return ST; }
  const X86TargetLowering *getTLI() const { return TLI; }

public:
  explicit X86TTIImpl(const X86TargetMachine *TM, Function &F)
      : BaseT(TM), ST(TM->getSubtargetImpl(F)), TLI(ST->getTargetLowering()) {}

  // Provide value semantics. MSVC requires that we spell all of these out.
  X86TTIImpl(const X86TTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)), ST(Arg.ST), TLI(Arg.TLI) {}
  X86TTIImpl(X86TTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))), ST(std::move(Arg.ST)),
        TLI(std::move(Arg.TLI)) {}
  X86TTIImpl &operator=(const X86TTIImpl &RHS) {
    BaseT::operator=(static_cast<const BaseT &>(RHS));
    ST = RHS.ST;
    TLI = RHS.TLI;
    return *this;
  }
  X86TTIImpl &operator=(X86TTIImpl &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    ST = std::move(RHS.ST);
    TLI = std::move(RHS.TLI);
    return *this;
  }

  /// \name Scalar TTI Implementations
  /// @{
  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth);

  /// @}

  /// \name Vector TTI Implementations
  /// @{

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
  unsigned getMaskedMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace);

  unsigned getAddressComputationCost(Type *PtrTy, bool IsComplex);

  unsigned getReductionCost(unsigned Opcode, Type *Ty, bool IsPairwiseForm);

  unsigned getIntImmCost(int64_t);

  unsigned getIntImmCost(const APInt &Imm, Type *Ty);

  unsigned getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                         Type *Ty);
  unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                         Type *Ty);
  bool isLegalMaskedLoad(Type *DataType, int Consecutive);
  bool isLegalMaskedStore(Type *DataType, int Consecutive);

  /// @}
};

} // end namespace llvm

#endif
