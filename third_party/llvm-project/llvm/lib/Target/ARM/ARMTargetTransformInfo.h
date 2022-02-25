//===- ARMTargetTransformInfo.h - ARM specific TTI --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// ARM target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H

#include "ARM.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/SubtargetFeature.h"

namespace llvm {

class APInt;
class ARMTargetLowering;
class Instruction;
class Loop;
class SCEV;
class ScalarEvolution;
class Type;
class Value;

namespace TailPredication {
  enum Mode {
    Disabled = 0,
    EnabledNoReductions,
    Enabled,
    ForceEnabledNoReductions,
    ForceEnabled
  };
}

// For controlling conversion of memcpy into Tail Predicated loop.
namespace TPLoop {
enum MemTransfer { ForceDisabled = 0, ForceEnabled, Allow };
}

class ARMTTIImpl : public BasicTTIImplBase<ARMTTIImpl> {
  using BaseT = BasicTTIImplBase<ARMTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const ARMSubtarget *ST;
  const ARMTargetLowering *TLI;

  // Currently the following features are excluded from InlineFeaturesAllowed.
  // ModeThumb, FeatureNoARM, ModeSoftFloat, FeatureFP64, FeatureD32
  // Depending on whether they are set or unset, different
  // instructions/registers are available. For example, inlining a callee with
  // -thumb-mode in a caller with +thumb-mode, may cause the assembler to
  // fail if the callee uses ARM only instructions, e.g. in inline asm.
  const FeatureBitset InlineFeaturesAllowed = {
      ARM::FeatureVFP2, ARM::FeatureVFP3, ARM::FeatureNEON, ARM::FeatureThumb2,
      ARM::FeatureFP16, ARM::FeatureVFP4, ARM::FeatureFPARMv8,
      ARM::FeatureFullFP16, ARM::FeatureFP16FML, ARM::FeatureHWDivThumb,
      ARM::FeatureHWDivARM, ARM::FeatureDB, ARM::FeatureV7Clrex,
      ARM::FeatureAcquireRelease, ARM::FeatureSlowFPBrcc,
      ARM::FeaturePerfMon, ARM::FeatureTrustZone, ARM::Feature8MSecExt,
      ARM::FeatureCrypto, ARM::FeatureCRC, ARM::FeatureRAS,
      ARM::FeatureFPAO, ARM::FeatureFuseAES, ARM::FeatureZCZeroing,
      ARM::FeatureProfUnpredicate, ARM::FeatureSlowVGETLNi32,
      ARM::FeatureSlowVDUP32, ARM::FeaturePreferVMOVSR,
      ARM::FeaturePrefISHSTBarrier, ARM::FeatureMuxedUnits,
      ARM::FeatureSlowOddRegister, ARM::FeatureSlowLoadDSubreg,
      ARM::FeatureDontWidenVMOVS, ARM::FeatureExpandMLx,
      ARM::FeatureHasVMLxHazards, ARM::FeatureNEONForFPMovs,
      ARM::FeatureNEONForFP, ARM::FeatureCheckVLDnAlign,
      ARM::FeatureHasSlowFPVMLx, ARM::FeatureHasSlowFPVFMx,
      ARM::FeatureVMLxForwarding, ARM::FeaturePref32BitThumb,
      ARM::FeatureAvoidPartialCPSR, ARM::FeatureCheapPredicableCPSR,
      ARM::FeatureAvoidMOVsShOp, ARM::FeatureHasRetAddrStack,
      ARM::FeatureHasNoBranchPredictor, ARM::FeatureDSP, ARM::FeatureMP,
      ARM::FeatureVirtualization, ARM::FeatureMClass, ARM::FeatureRClass,
      ARM::FeatureAClass, ARM::FeatureNaClTrap, ARM::FeatureStrictAlign,
      ARM::FeatureLongCalls, ARM::FeatureExecuteOnly, ARM::FeatureReserveR9,
      ARM::FeatureNoMovt, ARM::FeatureNoNegativeImmediates
  };

  const ARMSubtarget *getST() const { return ST; }
  const ARMTargetLowering *getTLI() const { return TLI; }

public:
  explicit ARMTTIImpl(const ARMBaseTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;

  bool enableInterleavedAccessVectorization() { return true; }

  TTI::AddressingModeKind
    getPreferredAddressingMode(const Loop *L, ScalarEvolution *SE) const;

  /// Floating-point computation using ARMv8 AArch32 Advanced
  /// SIMD instructions remains unchanged from ARMv7. Only AArch64 SIMD
  /// and Arm MVE are IEEE-754 compliant.
  bool isFPVectorizationPotentiallyUnsafe() {
    return !ST->isTargetDarwin() && !ST->hasMVEFloatOps();
  }

  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) const;

  /// \name Scalar TTI Implementations
  /// @{

  InstructionCost getIntImmCodeSizeCost(unsigned Opcode, unsigned Idx,
                                        const APInt &Imm, Type *Ty);

  using BaseT::getIntImmCost;
  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind);

  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr);

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const {
    bool Vector = (ClassID == 1);
    if (Vector) {
      if (ST->hasNEON())
        return 16;
      if (ST->hasMVEIntegerOps())
        return 8;
      return 0;
    }

    if (ST->isThumb1Only())
      return 8;
    return 13;
  }

  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
    switch (K) {
    case TargetTransformInfo::RGK_Scalar:
      return TypeSize::getFixed(32);
    case TargetTransformInfo::RGK_FixedWidthVector:
      if (ST->hasNEON())
        return TypeSize::getFixed(128);
      if (ST->hasMVEIntegerOps())
        return TypeSize::getFixed(128);
      return TypeSize::getFixed(0);
    case TargetTransformInfo::RGK_ScalableVector:
      return TypeSize::getScalable(0);
    }
    llvm_unreachable("Unsupported register kind");
  }

  unsigned getMaxInterleaveFactor(unsigned VF) {
    return ST->getMaxInterleaveFactor();
  }

  bool isProfitableLSRChainElement(Instruction *I);

  bool isLegalMaskedLoad(Type *DataTy, Align Alignment);

  bool isLegalMaskedStore(Type *DataTy, Align Alignment) {
    return isLegalMaskedLoad(DataTy, Alignment);
  }

  bool isLegalMaskedGather(Type *Ty, Align Alignment);

  bool isLegalMaskedScatter(Type *Ty, Align Alignment) {
    return isLegalMaskedGather(Ty, Alignment);
  }

  InstructionCost getMemcpyCost(const Instruction *I);

  int getNumMemOps(const IntrinsicInst *I) const;

  InstructionCost getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask, int Index,
                                 VectorType *SubTp);

  bool preferInLoopReduction(unsigned Opcode, Type *Ty,
                             TTI::ReductionFlags Flags) const;

  bool preferPredicatedReductionSelect(unsigned Opcode, Type *Ty,
                                       TTI::ReductionFlags Flags) const;

  bool shouldExpandReduction(const IntrinsicInst *II) const { return false; }

  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr);

  InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                   TTI::CastContextHint CCH,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I = nullptr);

  InstructionCost getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                                     CmpInst::Predicate VecPred,
                                     TTI::TargetCostKind CostKind,
                                     const Instruction *I = nullptr);

  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                     unsigned Index);

  InstructionCost getAddressComputationCost(Type *Val, ScalarEvolution *SE,
                                            const SCEV *Ptr);

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Op1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Op2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);

  InstructionCost getMemoryOpCost(unsigned Opcode, Type *Src,
                                  MaybeAlign Alignment, unsigned AddressSpace,
                                  TTI::TargetCostKind CostKind,
                                  const Instruction *I = nullptr);

  InstructionCost getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                        Align Alignment, unsigned AddressSpace,
                                        TTI::TargetCostKind CostKind);

  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);

  InstructionCost getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                         const Value *Ptr, bool VariableMask,
                                         Align Alignment,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I = nullptr);

  InstructionCost getArithmeticReductionCost(unsigned Opcode, VectorType *ValTy,
                                             Optional<FastMathFlags> FMF,
                                             TTI::TargetCostKind CostKind);
  InstructionCost getExtendedAddReductionCost(bool IsMLA, bool IsUnsigned,
                                              Type *ResTy, VectorType *ValTy,
                                              TTI::TargetCostKind CostKind);

  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind);

  bool maybeLoweredToCall(Instruction &I);
  bool isLoweredToCall(const Function *F);
  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC,
                                TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo);
  bool preferPredicateOverEpilogue(Loop *L, LoopInfo *LI,
                                   ScalarEvolution &SE,
                                   AssumptionCache &AC,
                                   TargetLibraryInfo *TLI,
                                   DominatorTree *DT,
                                   const LoopAccessInfo *LAI);
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE);

  bool emitGetActiveLaneMask() const;

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);
  bool shouldBuildLookupTablesForConstant(Constant *C) const {
    // In the ROPI and RWPI relocation models we can't have pointers to global
    // variables or functions in constant data, so don't convert switches to
    // lookup tables if any of the values would need relocation.
    if (ST->isROPI() || ST->isRWPI())
      return !C->needsDynamicRelocation();

    return true;
  }
  /// @}
};

/// isVREVMask - Check if a vector shuffle corresponds to a VREV
/// instruction with the specified blocksize.  (The order of the elements
/// within each block of the vector is reversed.)
inline bool isVREVMask(ArrayRef<int> M, EVT VT, unsigned BlockSize) {
  assert((BlockSize == 16 || BlockSize == 32 || BlockSize == 64) &&
         "Only possible block sizes for VREV are: 16, 32, 64");

  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz != 8 && EltSz != 16 && EltSz != 32)
    return false;

  unsigned BlockElts = M[0] + 1;
  // If the first shuffle index is UNDEF, be optimistic.
  if (M[0] < 0)
    BlockElts = BlockSize / EltSz;

  if (BlockSize <= EltSz || BlockSize != BlockElts * EltSz)
    return false;

  for (unsigned i = 0, e = M.size(); i < e; ++i) {
    if (M[i] < 0)
      continue; // ignore UNDEF indices
    if ((unsigned)M[i] != (i - i % BlockElts) + (BlockElts - 1 - i % BlockElts))
      return false;
  }

  return true;
}

} // end namespace llvm

#endif // LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H
