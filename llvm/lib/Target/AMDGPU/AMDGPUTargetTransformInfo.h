//===- AMDGPUTargetTransformInfo.h - AMDGPU specific TTI --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// AMDGPU target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H

#include "AMDGPU.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class AMDGPUTargetMachine;
class GCNSubtarget;
class InstCombiner;
class Loop;
class ScalarEvolution;
class SITargetLowering;
class Type;
class Value;

class AMDGPUTTIImpl final : public BasicTTIImplBase<AMDGPUTTIImpl> {
  using BaseT = BasicTTIImplBase<AMDGPUTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  Triple TargetTriple;

  const TargetSubtargetInfo *ST;
  const TargetLoweringBase *TLI;

  const TargetSubtargetInfo *getST() const { return ST; }
  const TargetLoweringBase *getTLI() const { return TLI; }

public:
  explicit AMDGPUTTIImpl(const AMDGPUTargetMachine *TM, const Function &F);

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE);

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);
};

class GCNTTIImpl final : public BasicTTIImplBase<GCNTTIImpl> {
  using BaseT = BasicTTIImplBase<GCNTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const GCNSubtarget *ST;
  const SITargetLowering *TLI;
  AMDGPUTTIImpl CommonTTI;
  bool IsGraphics;
  bool HasFP32Denormals;
  bool HasFP64FP16Denormals;
  unsigned MaxVGPRs;

  static const FeatureBitset InlineFeatureIgnoreList;

  const GCNSubtarget *getST() const { return ST; }
  const SITargetLowering *getTLI() const { return TLI; }

  static inline int getFullRateInstrCost() {
    return TargetTransformInfo::TCC_Basic;
  }

  static inline int getHalfRateInstrCost(
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) {
    return CostKind == TTI::TCK_CodeSize ? 2
                                         : 2 * TargetTransformInfo::TCC_Basic;
  }

  // TODO: The size is usually 8 bytes, but takes 4x as many cycles. Maybe
  // should be 2 or 4.
  static inline int getQuarterRateInstrCost(
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) {
    return CostKind == TTI::TCK_CodeSize ? 2
                                         : 4 * TargetTransformInfo::TCC_Basic;
  }

  // On some parts, normal fp64 operations are half rate, and others
  // quarter. This also applies to some integer operations.
  int get64BitInstrCost(
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) const;

public:
  explicit GCNTTIImpl(const AMDGPUTargetMachine *TM, const Function &F);

  bool hasBranchDivergence() { return true; }
  bool useGPUDivergenceAnalysis() const;

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE);

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) {
    assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
    return TTI::PSK_FastHardware;
  }

  unsigned getHardwareNumberOfRegisters(bool Vector) const;
  unsigned getNumberOfRegisters(bool Vector) const;
  unsigned getNumberOfRegisters(unsigned RCID) const;
  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind Vector) const;
  unsigned getMinVectorRegisterBitWidth() const;
  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const;
  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const;
  unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                unsigned ChainSizeInBytes,
                                VectorType *VecTy) const;
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const;

  bool isLegalToVectorizeMemChain(unsigned ChainSizeInBytes, Align Alignment,
                                  unsigned AddrSpace) const;
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const;
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const;
  Type *getMemcpyLoopLoweringType(LLVMContext &Context, Value *Length,
                                  unsigned SrcAddrSpace, unsigned DestAddrSpace,
                                  unsigned SrcAlign, unsigned DestAlign) const;

  void getMemcpyLoopResidualLoweringType(SmallVectorImpl<Type *> &OpsOut,
                                         LLVMContext &Context,
                                         unsigned RemainingBytes,
                                         unsigned SrcAddrSpace,
                                         unsigned DestAddrSpace,
                                         unsigned SrcAlign,
                                         unsigned DestAlign) const;
  unsigned getMaxInterleaveFactor(unsigned VF);

  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info) const;

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);

  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr);

  bool isInlineAsmSourceOfDivergence(const CallInst *CI,
                                     ArrayRef<unsigned> Indices = {}) const;

  InstructionCost getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                     unsigned Index);
  bool isSourceOfDivergence(const Value *V) const;
  bool isAlwaysUniform(const Value *V) const;

  unsigned getFlatAddressSpace() const {
    // Don't bother running InferAddressSpaces pass on graphics shaders which
    // don't use flat addressing.
    if (IsGraphics)
      return -1;
    return AMDGPUAS::FLAT_ADDRESS;
  }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const;
  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const;

  bool canSimplifyLegacyMulToMul(const Value *Op0, const Value *Op1,
                                 InstCombiner &IC) const;
  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) const;
  Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const;

  InstructionCost getVectorSplitCost() { return 0; }

  InstructionCost getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask, int Index,
                                 VectorType *SubTp);

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;

  unsigned getInliningThresholdMultiplier() { return 11; }
  unsigned adjustInliningThreshold(const CallBase *CB) const;

  int getInlinerVectorBonusPercent() { return 0; }

  InstructionCost getArithmeticReductionCost(
      unsigned Opcode, VectorType *Ty, Optional<FastMathFlags> FMF,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput);

  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind);
  InstructionCost getMinMaxReductionCost(
      VectorType *Ty, VectorType *CondTy, bool IsUnsigned,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H
