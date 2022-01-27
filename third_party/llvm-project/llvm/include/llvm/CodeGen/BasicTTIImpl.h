//===- BasicTTIImpl.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides a helper that implements much of the TTI interface in
/// terms of the target-independent code generator and TargetLowering
/// interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BASICTTIIMPL_H
#define LLVM_CODEGEN_BASICTTIIMPL_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>

namespace llvm {

class Function;
class GlobalValue;
class LLVMContext;
class ScalarEvolution;
class SCEV;
class TargetMachine;

extern cl::opt<unsigned> PartialUnrollingThreshold;

/// Base class which can be used to help build a TTI implementation.
///
/// This class provides as much implementation of the TTI interface as is
/// possible using the target independent parts of the code generator.
///
/// In order to subclass it, your class must implement a getST() method to
/// return the subtarget, and a getTLI() method to return the target lowering.
/// We need these methods implemented in the derived class so that this class
/// doesn't have to duplicate storage for them.
template <typename T>
class BasicTTIImplBase : public TargetTransformInfoImplCRTPBase<T> {
private:
  using BaseT = TargetTransformInfoImplCRTPBase<T>;
  using TTI = TargetTransformInfo;

  /// Helper function to access this as a T.
  T *thisT() { return static_cast<T *>(this); }

  /// Estimate a cost of Broadcast as an extract and sequence of insert
  /// operations.
  InstructionCost getBroadcastShuffleOverhead(FixedVectorType *VTy) {
    InstructionCost Cost = 0;
    // Broadcast cost is equal to the cost of extracting the zero'th element
    // plus the cost of inserting it into every element of the result vector.
    Cost += thisT()->getVectorInstrCost(Instruction::ExtractElement, VTy, 0);

    for (int i = 0, e = VTy->getNumElements(); i < e; ++i) {
      Cost += thisT()->getVectorInstrCost(Instruction::InsertElement, VTy, i);
    }
    return Cost;
  }

  /// Estimate a cost of shuffle as a sequence of extract and insert
  /// operations.
  InstructionCost getPermuteShuffleOverhead(FixedVectorType *VTy) {
    InstructionCost Cost = 0;
    // Shuffle cost is equal to the cost of extracting element from its argument
    // plus the cost of inserting them onto the result vector.

    // e.g. <4 x float> has a mask of <0,5,2,7> i.e we need to extract from
    // index 0 of first vector, index 1 of second vector,index 2 of first
    // vector and finally index 3 of second vector and insert them at index
    // <0,1,2,3> of result vector.
    for (int i = 0, e = VTy->getNumElements(); i < e; ++i) {
      Cost += thisT()->getVectorInstrCost(Instruction::InsertElement, VTy, i);
      Cost += thisT()->getVectorInstrCost(Instruction::ExtractElement, VTy, i);
    }
    return Cost;
  }

  /// Estimate a cost of subvector extraction as a sequence of extract and
  /// insert operations.
  InstructionCost getExtractSubvectorOverhead(VectorType *VTy, int Index,
                                       FixedVectorType *SubVTy) {
    assert(VTy && SubVTy &&
           "Can only extract subvectors from vectors");
    int NumSubElts = SubVTy->getNumElements();
    assert((!isa<FixedVectorType>(VTy) ||
            (Index + NumSubElts) <=
                (int)cast<FixedVectorType>(VTy)->getNumElements()) &&
           "SK_ExtractSubvector index out of range");

    InstructionCost Cost = 0;
    // Subvector extraction cost is equal to the cost of extracting element from
    // the source type plus the cost of inserting them into the result vector
    // type.
    for (int i = 0; i != NumSubElts; ++i) {
      Cost += thisT()->getVectorInstrCost(Instruction::ExtractElement, VTy,
                                          i + Index);
      Cost +=
          thisT()->getVectorInstrCost(Instruction::InsertElement, SubVTy, i);
    }
    return Cost;
  }

  /// Estimate a cost of subvector insertion as a sequence of extract and
  /// insert operations.
  InstructionCost getInsertSubvectorOverhead(VectorType *VTy, int Index,
                                      FixedVectorType *SubVTy) {
    assert(VTy && SubVTy &&
           "Can only insert subvectors into vectors");
    int NumSubElts = SubVTy->getNumElements();
    assert((!isa<FixedVectorType>(VTy) ||
            (Index + NumSubElts) <=
                (int)cast<FixedVectorType>(VTy)->getNumElements()) &&
           "SK_InsertSubvector index out of range");

    InstructionCost Cost = 0;
    // Subvector insertion cost is equal to the cost of extracting element from
    // the source type plus the cost of inserting them into the result vector
    // type.
    for (int i = 0; i != NumSubElts; ++i) {
      Cost +=
          thisT()->getVectorInstrCost(Instruction::ExtractElement, SubVTy, i);
      Cost += thisT()->getVectorInstrCost(Instruction::InsertElement, VTy,
                                          i + Index);
    }
    return Cost;
  }

  /// Local query method delegates up to T which *must* implement this!
  const TargetSubtargetInfo *getST() const {
    return static_cast<const T *>(this)->getST();
  }

  /// Local query method delegates up to T which *must* implement this!
  const TargetLoweringBase *getTLI() const {
    return static_cast<const T *>(this)->getTLI();
  }

  static ISD::MemIndexedMode getISDIndexedMode(TTI::MemIndexedMode M) {
    switch (M) {
      case TTI::MIM_Unindexed:
        return ISD::UNINDEXED;
      case TTI::MIM_PreInc:
        return ISD::PRE_INC;
      case TTI::MIM_PreDec:
        return ISD::PRE_DEC;
      case TTI::MIM_PostInc:
        return ISD::POST_INC;
      case TTI::MIM_PostDec:
        return ISD::POST_DEC;
    }
    llvm_unreachable("Unexpected MemIndexedMode");
  }

  InstructionCost getCommonMaskedMemoryOpCost(unsigned Opcode, Type *DataTy,
                                              Align Alignment,
                                              bool VariableMask,
                                              bool IsGatherScatter,
                                              TTI::TargetCostKind CostKind) {
    auto *VT = cast<FixedVectorType>(DataTy);
    // Assume the target does not have support for gather/scatter operations
    // and provide a rough estimate.
    //
    // First, compute the cost of the individual memory operations.
    InstructionCost AddrExtractCost =
        IsGatherScatter
            ? getVectorInstrCost(Instruction::ExtractElement,
                                 FixedVectorType::get(
                                     PointerType::get(VT->getElementType(), 0),
                                     VT->getNumElements()),
                                 -1)
            : 0;
    InstructionCost LoadCost =
        VT->getNumElements() *
        (AddrExtractCost +
         getMemoryOpCost(Opcode, VT->getElementType(), Alignment, 0, CostKind));

    // Next, compute the cost of packing the result in a vector.
    InstructionCost PackingCost = getScalarizationOverhead(
        VT, Opcode != Instruction::Store, Opcode == Instruction::Store);

    InstructionCost ConditionalCost = 0;
    if (VariableMask) {
      // Compute the cost of conditionally executing the memory operations with
      // variable masks. This includes extracting the individual conditions, a
      // branches and PHIs to combine the results.
      // NOTE: Estimating the cost of conditionally executing the memory
      // operations accurately is quite difficult and the current solution
      // provides a very rough estimate only.
      ConditionalCost =
          VT->getNumElements() *
          (getVectorInstrCost(
               Instruction::ExtractElement,
               FixedVectorType::get(Type::getInt1Ty(DataTy->getContext()),
                                    VT->getNumElements()),
               -1) +
           getCFInstrCost(Instruction::Br, CostKind) +
           getCFInstrCost(Instruction::PHI, CostKind));
    }

    return LoadCost + PackingCost + ConditionalCost;
  }

protected:
  explicit BasicTTIImplBase(const TargetMachine *TM, const DataLayout &DL)
      : BaseT(DL) {}
  virtual ~BasicTTIImplBase() = default;

  using TargetTransformInfoImplBase::DL;

public:
  /// \name Scalar TTI Implementations
  /// @{
  bool allowsMisalignedMemoryAccesses(LLVMContext &Context, unsigned BitWidth,
                                      unsigned AddressSpace, Align Alignment,
                                      bool *Fast) const {
    EVT E = EVT::getIntegerVT(Context, BitWidth);
    return getTLI()->allowsMisalignedMemoryAccesses(
        E, AddressSpace, Alignment, MachineMemOperand::MONone, Fast);
  }

  bool hasBranchDivergence() { return false; }

  bool useGPUDivergenceAnalysis() { return false; }

  bool isSourceOfDivergence(const Value *V) { return false; }

  bool isAlwaysUniform(const Value *V) { return false; }

  unsigned getFlatAddressSpace() {
    // Return an invalid address space.
    return -1;
  }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const {
    return false;
  }

  bool isNoopAddrSpaceCast(unsigned FromAS, unsigned ToAS) const {
    return getTLI()->getTargetMachine().isNoopAddrSpaceCast(FromAS, ToAS);
  }

  unsigned getAssumedAddrSpace(const Value *V) const {
    return getTLI()->getTargetMachine().getAssumedAddrSpace(V);
  }

  std::pair<const Value *, unsigned>
  getPredicatedAddrSpace(const Value *V) const {
    return getTLI()->getTargetMachine().getPredicatedAddrSpace(V);
  }

  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const {
    return nullptr;
  }

  bool isLegalAddImmediate(int64_t imm) {
    return getTLI()->isLegalAddImmediate(imm);
  }

  bool isLegalICmpImmediate(int64_t imm) {
    return getTLI()->isLegalICmpImmediate(imm);
  }

  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale,
                             unsigned AddrSpace, Instruction *I = nullptr) {
    TargetLoweringBase::AddrMode AM;
    AM.BaseGV = BaseGV;
    AM.BaseOffs = BaseOffset;
    AM.HasBaseReg = HasBaseReg;
    AM.Scale = Scale;
    return getTLI()->isLegalAddressingMode(DL, AM, Ty, AddrSpace, I);
  }

  bool isIndexedLoadLegal(TTI::MemIndexedMode M, Type *Ty,
                          const DataLayout &DL) const {
    EVT VT = getTLI()->getValueType(DL, Ty);
    return getTLI()->isIndexedLoadLegal(getISDIndexedMode(M), VT);
  }

  bool isIndexedStoreLegal(TTI::MemIndexedMode M, Type *Ty,
                           const DataLayout &DL) const {
    EVT VT = getTLI()->getValueType(DL, Ty);
    return getTLI()->isIndexedStoreLegal(getISDIndexedMode(M), VT);
  }

  bool isLSRCostLess(TTI::LSRCost C1, TTI::LSRCost C2) {
    return TargetTransformInfoImplBase::isLSRCostLess(C1, C2);
  }

  bool isNumRegsMajorCostOfLSR() {
    return TargetTransformInfoImplBase::isNumRegsMajorCostOfLSR();
  }

  bool isProfitableLSRChainElement(Instruction *I) {
    return TargetTransformInfoImplBase::isProfitableLSRChainElement(I);
  }

  InstructionCost getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                       int64_t BaseOffset, bool HasBaseReg,
                                       int64_t Scale, unsigned AddrSpace) {
    TargetLoweringBase::AddrMode AM;
    AM.BaseGV = BaseGV;
    AM.BaseOffs = BaseOffset;
    AM.HasBaseReg = HasBaseReg;
    AM.Scale = Scale;
    return getTLI()->getScalingFactorCost(DL, AM, Ty, AddrSpace);
  }

  bool isTruncateFree(Type *Ty1, Type *Ty2) {
    return getTLI()->isTruncateFree(Ty1, Ty2);
  }

  bool isProfitableToHoist(Instruction *I) {
    return getTLI()->isProfitableToHoist(I);
  }

  bool useAA() const { return getST()->useAA(); }

  bool isTypeLegal(Type *Ty) {
    EVT VT = getTLI()->getValueType(DL, Ty);
    return getTLI()->isTypeLegal(VT);
  }

  InstructionCost getRegUsageForType(Type *Ty) {
    InstructionCost Val = getTLI()->getTypeLegalizationCost(DL, Ty).first;
    assert(Val >= 0 && "Negative cost!");
    return Val;
  }

  InstructionCost getGEPCost(Type *PointeeType, const Value *Ptr,
                             ArrayRef<const Value *> Operands,
                             TTI::TargetCostKind CostKind) {
    return BaseT::getGEPCost(PointeeType, Ptr, Operands, CostKind);
  }

  unsigned getEstimatedNumberOfCaseClusters(const SwitchInst &SI,
                                            unsigned &JumpTableSize,
                                            ProfileSummaryInfo *PSI,
                                            BlockFrequencyInfo *BFI) {
    /// Try to find the estimated number of clusters. Note that the number of
    /// clusters identified in this function could be different from the actual
    /// numbers found in lowering. This function ignore switches that are
    /// lowered with a mix of jump table / bit test / BTree. This function was
    /// initially intended to be used when estimating the cost of switch in
    /// inline cost heuristic, but it's a generic cost model to be used in other
    /// places (e.g., in loop unrolling).
    unsigned N = SI.getNumCases();
    const TargetLoweringBase *TLI = getTLI();
    const DataLayout &DL = this->getDataLayout();

    JumpTableSize = 0;
    bool IsJTAllowed = TLI->areJTsAllowed(SI.getParent()->getParent());

    // Early exit if both a jump table and bit test are not allowed.
    if (N < 1 || (!IsJTAllowed && DL.getIndexSizeInBits(0u) < N))
      return N;

    APInt MaxCaseVal = SI.case_begin()->getCaseValue()->getValue();
    APInt MinCaseVal = MaxCaseVal;
    for (auto CI : SI.cases()) {
      const APInt &CaseVal = CI.getCaseValue()->getValue();
      if (CaseVal.sgt(MaxCaseVal))
        MaxCaseVal = CaseVal;
      if (CaseVal.slt(MinCaseVal))
        MinCaseVal = CaseVal;
    }

    // Check if suitable for a bit test
    if (N <= DL.getIndexSizeInBits(0u)) {
      SmallPtrSet<const BasicBlock *, 4> Dests;
      for (auto I : SI.cases())
        Dests.insert(I.getCaseSuccessor());

      if (TLI->isSuitableForBitTests(Dests.size(), N, MinCaseVal, MaxCaseVal,
                                     DL))
        return 1;
    }

    // Check if suitable for a jump table.
    if (IsJTAllowed) {
      if (N < 2 || N < TLI->getMinimumJumpTableEntries())
        return N;
      uint64_t Range =
          (MaxCaseVal - MinCaseVal)
              .getLimitedValue(std::numeric_limits<uint64_t>::max() - 1) + 1;
      // Check whether a range of clusters is dense enough for a jump table
      if (TLI->isSuitableForJumpTable(&SI, N, Range, PSI, BFI)) {
        JumpTableSize = Range;
        return 1;
      }
    }
    return N;
  }

  bool shouldBuildLookupTables() {
    const TargetLoweringBase *TLI = getTLI();
    return TLI->isOperationLegalOrCustom(ISD::BR_JT, MVT::Other) ||
           TLI->isOperationLegalOrCustom(ISD::BRIND, MVT::Other);
  }

  bool shouldBuildRelLookupTables() const {
    const TargetMachine &TM = getTLI()->getTargetMachine();
    // If non-PIC mode, do not generate a relative lookup table.
    if (!TM.isPositionIndependent())
      return false;

    /// Relative lookup table entries consist of 32-bit offsets.
    /// Do not generate relative lookup tables for large code models
    /// in 64-bit achitectures where 32-bit offsets might not be enough.
    if (TM.getCodeModel() == CodeModel::Medium ||
        TM.getCodeModel() == CodeModel::Large)
      return false;

    Triple TargetTriple = TM.getTargetTriple();
    if (!TargetTriple.isArch64Bit())
      return false;

    // TODO: Triggers issues on aarch64 on darwin, so temporarily disable it
    // there.
    if (TargetTriple.getArch() == Triple::aarch64 && TargetTriple.isOSDarwin())
      return false;

    return true;
  }

  bool haveFastSqrt(Type *Ty) {
    const TargetLoweringBase *TLI = getTLI();
    EVT VT = TLI->getValueType(DL, Ty);
    return TLI->isTypeLegal(VT) &&
           TLI->isOperationLegalOrCustom(ISD::FSQRT, VT);
  }

  bool isFCmpOrdCheaperThanFCmpZero(Type *Ty) {
    return true;
  }

  InstructionCost getFPOpCost(Type *Ty) {
    // Check whether FADD is available, as a proxy for floating-point in
    // general.
    const TargetLoweringBase *TLI = getTLI();
    EVT VT = TLI->getValueType(DL, Ty);
    if (TLI->isOperationLegalOrCustomOrPromote(ISD::FADD, VT))
      return TargetTransformInfo::TCC_Basic;
    return TargetTransformInfo::TCC_Expensive;
  }

  unsigned getInliningThresholdMultiplier() { return 1; }
  unsigned adjustInliningThreshold(const CallBase *CB) { return 0; }

  int getInlinerVectorBonusPercent() { return 150; }

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) {
    // This unrolling functionality is target independent, but to provide some
    // motivation for its intended use, for x86:

    // According to the Intel 64 and IA-32 Architectures Optimization Reference
    // Manual, Intel Core models and later have a loop stream detector (and
    // associated uop queue) that can benefit from partial unrolling.
    // The relevant requirements are:
    //  - The loop must have no more than 4 (8 for Nehalem and later) branches
    //    taken, and none of them may be calls.
    //  - The loop can have no more than 18 (28 for Nehalem and later) uops.

    // According to the Software Optimization Guide for AMD Family 15h
    // Processors, models 30h-4fh (Steamroller and later) have a loop predictor
    // and loop buffer which can benefit from partial unrolling.
    // The relevant requirements are:
    //  - The loop must have fewer than 16 branches
    //  - The loop must have less than 40 uops in all executed loop branches

    // The number of taken branches in a loop is hard to estimate here, and
    // benchmarking has revealed that it is better not to be conservative when
    // estimating the branch count. As a result, we'll ignore the branch limits
    // until someone finds a case where it matters in practice.

    unsigned MaxOps;
    const TargetSubtargetInfo *ST = getST();
    if (PartialUnrollingThreshold.getNumOccurrences() > 0)
      MaxOps = PartialUnrollingThreshold;
    else if (ST->getSchedModel().LoopMicroOpBufferSize > 0)
      MaxOps = ST->getSchedModel().LoopMicroOpBufferSize;
    else
      return;

    // Scan the loop: don't unroll loops with calls.
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
          if (const Function *F = cast<CallBase>(I).getCalledFunction()) {
            if (!thisT()->isLoweredToCall(F))
              continue;
          }

          if (ORE) {
            ORE->emit([&]() {
              return OptimizationRemark("TTI", "DontUnroll", L->getStartLoc(),
                                        L->getHeader())
                     << "advising against unrolling the loop because it "
                        "contains a "
                     << ore::NV("Call", &I);
            });
          }
          return;
        }
      }
    }

    // Enable runtime and partial unrolling up to the specified size.
    // Enable using trip count upper bound to unroll loops.
    UP.Partial = UP.Runtime = UP.UpperBound = true;
    UP.PartialThreshold = MaxOps;

    // Avoid unrolling when optimizing for size.
    UP.OptSizeThreshold = 0;
    UP.PartialOptSizeThreshold = 0;

    // Set number of instructions optimized when "back edge"
    // becomes "fall through" to default value of 2.
    UP.BEInsns = 2;
  }

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) {
    PP.PeelCount = 0;
    PP.AllowPeeling = true;
    PP.AllowLoopNestsPeeling = false;
    PP.PeelProfiledIterations = true;
  }

  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC,
                                TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo) {
    return BaseT::isHardwareLoopProfitable(L, SE, AC, LibInfo, HWLoopInfo);
  }

  bool preferPredicateOverEpilogue(Loop *L, LoopInfo *LI, ScalarEvolution &SE,
                                   AssumptionCache &AC, TargetLibraryInfo *TLI,
                                   DominatorTree *DT,
                                   const LoopAccessInfo *LAI) {
    return BaseT::preferPredicateOverEpilogue(L, LI, SE, AC, TLI, DT, LAI);
  }

  bool emitGetActiveLaneMask() {
    return BaseT::emitGetActiveLaneMask();
  }

  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) {
    return BaseT::instCombineIntrinsic(IC, II);
  }

  Optional<Value *> simplifyDemandedUseBitsIntrinsic(InstCombiner &IC,
                                                     IntrinsicInst &II,
                                                     APInt DemandedMask,
                                                     KnownBits &Known,
                                                     bool &KnownBitsComputed) {
    return BaseT::simplifyDemandedUseBitsIntrinsic(IC, II, DemandedMask, Known,
                                                   KnownBitsComputed);
  }

  Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) {
    return BaseT::simplifyDemandedVectorEltsIntrinsic(
        IC, II, DemandedElts, UndefElts, UndefElts2, UndefElts3,
        SimplifyAndSetOp);
  }

  InstructionCost getInstructionLatency(const Instruction *I) {
    if (isa<LoadInst>(I))
      return getST()->getSchedModel().DefaultLoadLatency;

    return BaseT::getInstructionLatency(I);
  }

  virtual Optional<unsigned>
  getCacheSize(TargetTransformInfo::CacheLevel Level) const {
    return Optional<unsigned>(
      getST()->getCacheSize(static_cast<unsigned>(Level)));
  }

  virtual Optional<unsigned>
  getCacheAssociativity(TargetTransformInfo::CacheLevel Level) const {
    Optional<unsigned> TargetResult =
        getST()->getCacheAssociativity(static_cast<unsigned>(Level));

    if (TargetResult)
      return TargetResult;

    return BaseT::getCacheAssociativity(Level);
  }

  virtual unsigned getCacheLineSize() const {
    return getST()->getCacheLineSize();
  }

  virtual unsigned getPrefetchDistance() const {
    return getST()->getPrefetchDistance();
  }

  virtual unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                        unsigned NumStridedMemAccesses,
                                        unsigned NumPrefetches,
                                        bool HasCall) const {
    return getST()->getMinPrefetchStride(NumMemAccesses, NumStridedMemAccesses,
                                         NumPrefetches, HasCall);
  }

  virtual unsigned getMaxPrefetchIterationsAhead() const {
    return getST()->getMaxPrefetchIterationsAhead();
  }

  virtual bool enableWritePrefetching() const {
    return getST()->enableWritePrefetching();
  }

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
    return TypeSize::getFixed(32);
  }

  Optional<unsigned> getMaxVScale() const { return None; }
  Optional<unsigned> getVScaleForTuning() const { return None; }

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the demanded result elements need to be inserted and/or
  /// extracted from vectors.
  InstructionCost getScalarizationOverhead(VectorType *InTy,
                                           const APInt &DemandedElts,
                                           bool Insert, bool Extract) {
    /// FIXME: a bitfield is not a reasonable abstraction for talking about
    /// which elements are needed from a scalable vector
    auto *Ty = cast<FixedVectorType>(InTy);

    assert(DemandedElts.getBitWidth() == Ty->getNumElements() &&
           "Vector size mismatch");

    InstructionCost Cost = 0;

    for (int i = 0, e = Ty->getNumElements(); i < e; ++i) {
      if (!DemandedElts[i])
        continue;
      if (Insert)
        Cost += thisT()->getVectorInstrCost(Instruction::InsertElement, Ty, i);
      if (Extract)
        Cost += thisT()->getVectorInstrCost(Instruction::ExtractElement, Ty, i);
    }

    return Cost;
  }

  /// Helper wrapper for the DemandedElts variant of getScalarizationOverhead.
  InstructionCost getScalarizationOverhead(VectorType *InTy, bool Insert,
                                           bool Extract) {
    auto *Ty = cast<FixedVectorType>(InTy);

    APInt DemandedElts = APInt::getAllOnes(Ty->getNumElements());
    return thisT()->getScalarizationOverhead(Ty, DemandedElts, Insert, Extract);
  }

  /// Estimate the overhead of scalarizing an instructions unique
  /// non-constant operands. The (potentially vector) types to use for each of
  /// argument are passes via Tys.
  InstructionCost getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                                   ArrayRef<Type *> Tys) {
    assert(Args.size() == Tys.size() && "Expected matching Args and Tys");

    InstructionCost Cost = 0;
    SmallPtrSet<const Value*, 4> UniqueOperands;
    for (int I = 0, E = Args.size(); I != E; I++) {
      // Disregard things like metadata arguments.
      const Value *A = Args[I];
      Type *Ty = Tys[I];
      if (!Ty->isIntOrIntVectorTy() && !Ty->isFPOrFPVectorTy() &&
          !Ty->isPtrOrPtrVectorTy())
        continue;

      if (!isa<Constant>(A) && UniqueOperands.insert(A).second) {
        if (auto *VecTy = dyn_cast<VectorType>(Ty))
          Cost += getScalarizationOverhead(VecTy, false, true);
      }
    }

    return Cost;
  }

  /// Estimate the overhead of scalarizing the inputs and outputs of an
  /// instruction, with return type RetTy and arguments Args of type Tys. If
  /// Args are unknown (empty), then the cost associated with one argument is
  /// added as a heuristic.
  InstructionCost getScalarizationOverhead(VectorType *RetTy,
                                           ArrayRef<const Value *> Args,
                                           ArrayRef<Type *> Tys) {
    InstructionCost Cost = getScalarizationOverhead(RetTy, true, false);
    if (!Args.empty())
      Cost += getOperandsScalarizationOverhead(Args, Tys);
    else
      // When no information on arguments is provided, we add the cost
      // associated with one argument as a heuristic.
      Cost += getScalarizationOverhead(RetTy, false, true);

    return Cost;
  }

  unsigned getMaxInterleaveFactor(unsigned VF) { return 1; }

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr) {
    // Check if any of the operands are vector operands.
    const TargetLoweringBase *TLI = getTLI();
    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    assert(ISD && "Invalid opcode");

    // TODO: Handle more cost kinds.
    if (CostKind != TTI::TCK_RecipThroughput)
      return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind,
                                           Opd1Info, Opd2Info,
                                           Opd1PropInfo, Opd2PropInfo,
                                           Args, CxtI);

    std::pair<InstructionCost, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);

    bool IsFloat = Ty->isFPOrFPVectorTy();
    // Assume that floating point arithmetic operations cost twice as much as
    // integer operations.
    InstructionCost OpCost = (IsFloat ? 2 : 1);

    if (TLI->isOperationLegalOrPromote(ISD, LT.second)) {
      // The operation is legal. Assume it costs 1.
      // TODO: Once we have extract/insert subvector cost we need to use them.
      return LT.first * OpCost;
    }

    if (!TLI->isOperationExpand(ISD, LT.second)) {
      // If the operation is custom lowered, then assume that the code is twice
      // as expensive.
      return LT.first * 2 * OpCost;
    }

    // An 'Expand' of URem and SRem is special because it may default
    // to expanding the operation into a sequence of sub-operations
    // i.e. X % Y -> X-(X/Y)*Y.
    if (ISD == ISD::UREM || ISD == ISD::SREM) {
      bool IsSigned = ISD == ISD::SREM;
      if (TLI->isOperationLegalOrCustom(IsSigned ? ISD::SDIVREM : ISD::UDIVREM,
                                        LT.second) ||
          TLI->isOperationLegalOrCustom(IsSigned ? ISD::SDIV : ISD::UDIV,
                                        LT.second)) {
        unsigned DivOpc = IsSigned ? Instruction::SDiv : Instruction::UDiv;
        InstructionCost DivCost = thisT()->getArithmeticInstrCost(
            DivOpc, Ty, CostKind, Opd1Info, Opd2Info, Opd1PropInfo,
            Opd2PropInfo);
        InstructionCost MulCost =
            thisT()->getArithmeticInstrCost(Instruction::Mul, Ty, CostKind);
        InstructionCost SubCost =
            thisT()->getArithmeticInstrCost(Instruction::Sub, Ty, CostKind);
        return DivCost + MulCost + SubCost;
      }
    }

    // We cannot scalarize scalable vectors, so return Invalid.
    if (isa<ScalableVectorType>(Ty))
      return InstructionCost::getInvalid();

    // Else, assume that we need to scalarize this op.
    // TODO: If one of the types get legalized by splitting, handle this
    // similarly to what getCastInstrCost() does.
    if (auto *VTy = dyn_cast<FixedVectorType>(Ty)) {
      InstructionCost Cost = thisT()->getArithmeticInstrCost(
          Opcode, VTy->getScalarType(), CostKind, Opd1Info, Opd2Info,
          Opd1PropInfo, Opd2PropInfo, Args, CxtI);
      // Return the cost of multiple scalar invocation plus the cost of
      // inserting and extracting the values.
      SmallVector<Type *> Tys(Args.size(), Ty);
      return getScalarizationOverhead(VTy, Args, Tys) +
             VTy->getNumElements() * Cost;
    }

    // We don't know anything about this scalar instruction.
    return OpCost;
  }

  TTI::ShuffleKind improveShuffleKindFromMask(TTI::ShuffleKind Kind,
                                              ArrayRef<int> Mask) const {
    int Limit = Mask.size() * 2;
    if (Mask.empty() ||
        // Extra check required by isSingleSourceMaskImpl function (called by
        // ShuffleVectorInst::isSingleSourceMask).
        any_of(Mask, [Limit](int I) { return I >= Limit; }))
      return Kind;
    switch (Kind) {
    case TTI::SK_PermuteSingleSrc:
      if (ShuffleVectorInst::isReverseMask(Mask))
        return TTI::SK_Reverse;
      if (ShuffleVectorInst::isZeroEltSplatMask(Mask))
        return TTI::SK_Broadcast;
      break;
    case TTI::SK_PermuteTwoSrc:
      if (ShuffleVectorInst::isSelectMask(Mask))
        return TTI::SK_Select;
      if (ShuffleVectorInst::isTransposeMask(Mask))
        return TTI::SK_Transpose;
      break;
    case TTI::SK_Select:
    case TTI::SK_Reverse:
    case TTI::SK_Broadcast:
    case TTI::SK_Transpose:
    case TTI::SK_InsertSubvector:
    case TTI::SK_ExtractSubvector:
    case TTI::SK_Splice:
      break;
    }
    return Kind;
  }

  InstructionCost getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask, int Index,
                                 VectorType *SubTp) {

    switch (improveShuffleKindFromMask(Kind, Mask)) {
    case TTI::SK_Broadcast:
      if (auto *FVT = dyn_cast<FixedVectorType>(Tp))
        return getBroadcastShuffleOverhead(FVT);
      return InstructionCost::getInvalid();
    case TTI::SK_Select:
    case TTI::SK_Splice:
    case TTI::SK_Reverse:
    case TTI::SK_Transpose:
    case TTI::SK_PermuteSingleSrc:
    case TTI::SK_PermuteTwoSrc:
      if (auto *FVT = dyn_cast<FixedVectorType>(Tp))
        return getPermuteShuffleOverhead(FVT);
      return InstructionCost::getInvalid();
    case TTI::SK_ExtractSubvector:
      return getExtractSubvectorOverhead(Tp, Index,
                                         cast<FixedVectorType>(SubTp));
    case TTI::SK_InsertSubvector:
      return getInsertSubvectorOverhead(Tp, Index,
                                        cast<FixedVectorType>(SubTp));
    }
    llvm_unreachable("Unknown TTI::ShuffleKind");
  }

  InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                   TTI::CastContextHint CCH,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I = nullptr) {
    if (BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I) == 0)
      return 0;

    const TargetLoweringBase *TLI = getTLI();
    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    assert(ISD && "Invalid opcode");
    std::pair<InstructionCost, MVT> SrcLT =
        TLI->getTypeLegalizationCost(DL, Src);
    std::pair<InstructionCost, MVT> DstLT =
        TLI->getTypeLegalizationCost(DL, Dst);

    TypeSize SrcSize = SrcLT.second.getSizeInBits();
    TypeSize DstSize = DstLT.second.getSizeInBits();
    bool IntOrPtrSrc = Src->isIntegerTy() || Src->isPointerTy();
    bool IntOrPtrDst = Dst->isIntegerTy() || Dst->isPointerTy();

    switch (Opcode) {
    default:
      break;
    case Instruction::Trunc:
      // Check for NOOP conversions.
      if (TLI->isTruncateFree(SrcLT.second, DstLT.second))
        return 0;
      LLVM_FALLTHROUGH;
    case Instruction::BitCast:
      // Bitcast between types that are legalized to the same type are free and
      // assume int to/from ptr of the same size is also free.
      if (SrcLT.first == DstLT.first && IntOrPtrSrc == IntOrPtrDst &&
          SrcSize == DstSize)
        return 0;
      break;
    case Instruction::FPExt:
      if (I && getTLI()->isExtFree(I))
        return 0;
      break;
    case Instruction::ZExt:
      if (TLI->isZExtFree(SrcLT.second, DstLT.second))
        return 0;
      LLVM_FALLTHROUGH;
    case Instruction::SExt:
      if (I && getTLI()->isExtFree(I))
        return 0;

      // If this is a zext/sext of a load, return 0 if the corresponding
      // extending load exists on target and the result type is legal.
      if (CCH == TTI::CastContextHint::Normal) {
        EVT ExtVT = EVT::getEVT(Dst);
        EVT LoadVT = EVT::getEVT(Src);
        unsigned LType =
          ((Opcode == Instruction::ZExt) ? ISD::ZEXTLOAD : ISD::SEXTLOAD);
        if (DstLT.first == SrcLT.first &&
            TLI->isLoadExtLegal(LType, ExtVT, LoadVT))
          return 0;
      }
      break;
    case Instruction::AddrSpaceCast:
      if (TLI->isFreeAddrSpaceCast(Src->getPointerAddressSpace(),
                                   Dst->getPointerAddressSpace()))
        return 0;
      break;
    }

    auto *SrcVTy = dyn_cast<VectorType>(Src);
    auto *DstVTy = dyn_cast<VectorType>(Dst);

    // If the cast is marked as legal (or promote) then assume low cost.
    if (SrcLT.first == DstLT.first &&
        TLI->isOperationLegalOrPromote(ISD, DstLT.second))
      return SrcLT.first;

    // Handle scalar conversions.
    if (!SrcVTy && !DstVTy) {
      // Just check the op cost. If the operation is legal then assume it costs
      // 1.
      if (!TLI->isOperationExpand(ISD, DstLT.second))
        return 1;

      // Assume that illegal scalar instruction are expensive.
      return 4;
    }

    // Check vector-to-vector casts.
    if (DstVTy && SrcVTy) {
      // If the cast is between same-sized registers, then the check is simple.
      if (SrcLT.first == DstLT.first && SrcSize == DstSize) {

        // Assume that Zext is done using AND.
        if (Opcode == Instruction::ZExt)
          return SrcLT.first;

        // Assume that sext is done using SHL and SRA.
        if (Opcode == Instruction::SExt)
          return SrcLT.first * 2;

        // Just check the op cost. If the operation is legal then assume it
        // costs
        // 1 and multiply by the type-legalization overhead.
        if (!TLI->isOperationExpand(ISD, DstLT.second))
          return SrcLT.first * 1;
      }

      // If we are legalizing by splitting, query the concrete TTI for the cost
      // of casting the original vector twice. We also need to factor in the
      // cost of the split itself. Count that as 1, to be consistent with
      // TLI->getTypeLegalizationCost().
      bool SplitSrc =
          TLI->getTypeAction(Src->getContext(), TLI->getValueType(DL, Src)) ==
          TargetLowering::TypeSplitVector;
      bool SplitDst =
          TLI->getTypeAction(Dst->getContext(), TLI->getValueType(DL, Dst)) ==
          TargetLowering::TypeSplitVector;
      if ((SplitSrc || SplitDst) && SrcVTy->getElementCount().isVector() &&
          DstVTy->getElementCount().isVector()) {
        Type *SplitDstTy = VectorType::getHalfElementsVectorType(DstVTy);
        Type *SplitSrcTy = VectorType::getHalfElementsVectorType(SrcVTy);
        T *TTI = static_cast<T *>(this);
        // If both types need to be split then the split is free.
        InstructionCost SplitCost =
            (!SplitSrc || !SplitDst) ? TTI->getVectorSplitCost() : 0;
        return SplitCost +
               (2 * TTI->getCastInstrCost(Opcode, SplitDstTy, SplitSrcTy, CCH,
                                          CostKind, I));
      }

      // Scalarization cost is Invalid, can't assume any num elements.
      if (isa<ScalableVectorType>(DstVTy))
        return InstructionCost::getInvalid();

      // In other cases where the source or destination are illegal, assume
      // the operation will get scalarized.
      unsigned Num = cast<FixedVectorType>(DstVTy)->getNumElements();
      InstructionCost Cost = thisT()->getCastInstrCost(
          Opcode, Dst->getScalarType(), Src->getScalarType(), CCH, CostKind, I);

      // Return the cost of multiple scalar invocation plus the cost of
      // inserting and extracting the values.
      return getScalarizationOverhead(DstVTy, true, true) + Num * Cost;
    }

    // We already handled vector-to-vector and scalar-to-scalar conversions.
    // This
    // is where we handle bitcast between vectors and scalars. We need to assume
    //  that the conversion is scalarized in one way or another.
    if (Opcode == Instruction::BitCast) {
      // Illegal bitcasts are done by storing and loading from a stack slot.
      return (SrcVTy ? getScalarizationOverhead(SrcVTy, false, true) : 0) +
             (DstVTy ? getScalarizationOverhead(DstVTy, true, false) : 0);
    }

    llvm_unreachable("Unhandled cast");
  }

  InstructionCost getExtractWithExtendCost(unsigned Opcode, Type *Dst,
                                           VectorType *VecTy, unsigned Index) {
    return thisT()->getVectorInstrCost(Instruction::ExtractElement, VecTy,
                                       Index) +
           thisT()->getCastInstrCost(Opcode, Dst, VecTy->getElementType(),
                                     TTI::CastContextHint::None,
                                     TTI::TCK_RecipThroughput);
  }

  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr) {
    return BaseT::getCFInstrCost(Opcode, CostKind, I);
  }

  InstructionCost getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                                     CmpInst::Predicate VecPred,
                                     TTI::TargetCostKind CostKind,
                                     const Instruction *I = nullptr) {
    const TargetLoweringBase *TLI = getTLI();
    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    assert(ISD && "Invalid opcode");

    // TODO: Handle other cost kinds.
    if (CostKind != TTI::TCK_RecipThroughput)
      return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                       I);

    // Selects on vectors are actually vector selects.
    if (ISD == ISD::SELECT) {
      assert(CondTy && "CondTy must exist");
      if (CondTy->isVectorTy())
        ISD = ISD::VSELECT;
    }
    std::pair<InstructionCost, MVT> LT =
        TLI->getTypeLegalizationCost(DL, ValTy);

    if (!(ValTy->isVectorTy() && !LT.second.isVector()) &&
        !TLI->isOperationExpand(ISD, LT.second)) {
      // The operation is legal. Assume it costs 1. Multiply
      // by the type-legalization overhead.
      return LT.first * 1;
    }

    // Otherwise, assume that the cast is scalarized.
    // TODO: If one of the types get legalized by splitting, handle this
    // similarly to what getCastInstrCost() does.
    if (auto *ValVTy = dyn_cast<VectorType>(ValTy)) {
      unsigned Num = cast<FixedVectorType>(ValVTy)->getNumElements();
      if (CondTy)
        CondTy = CondTy->getScalarType();
      InstructionCost Cost = thisT()->getCmpSelInstrCost(
          Opcode, ValVTy->getScalarType(), CondTy, VecPred, CostKind, I);

      // Return the cost of multiple scalar invocation plus the cost of
      // inserting and extracting the values.
      return getScalarizationOverhead(ValVTy, true, false) + Num * Cost;
    }

    // Unknown scalar opcode.
    return 1;
  }

  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                     unsigned Index) {
    std::pair<InstructionCost, MVT> LT =
        getTLI()->getTypeLegalizationCost(DL, Val->getScalarType());

    return LT.first;
  }

  InstructionCost getReplicationShuffleCost(Type *EltTy, int ReplicationFactor,
                                            int VF,
                                            const APInt &DemandedDstElts,
                                            TTI::TargetCostKind CostKind) {
    assert(DemandedDstElts.getBitWidth() == (unsigned)VF * ReplicationFactor &&
           "Unexpected size of DemandedDstElts.");

    InstructionCost Cost;

    auto *SrcVT = FixedVectorType::get(EltTy, VF);
    auto *ReplicatedVT = FixedVectorType::get(EltTy, VF * ReplicationFactor);

    // The Mask shuffling cost is extract all the elements of the Mask
    // and insert each of them Factor times into the wide vector:
    //
    // E.g. an interleaved group with factor 3:
    //    %mask = icmp ult <8 x i32> %vec1, %vec2
    //    %interleaved.mask = shufflevector <8 x i1> %mask, <8 x i1> undef,
    //        <24 x i32> <0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7>
    // The cost is estimated as extract all mask elements from the <8xi1> mask
    // vector and insert them factor times into the <24xi1> shuffled mask
    // vector.
    APInt DemandedSrcElts = APIntOps::ScaleBitMask(DemandedDstElts, VF);
    Cost += thisT()->getScalarizationOverhead(SrcVT, DemandedSrcElts,
                                              /*Insert*/ false,
                                              /*Extract*/ true);
    Cost +=
        thisT()->getScalarizationOverhead(ReplicatedVT, DemandedDstElts,
                                          /*Insert*/ true, /*Extract*/ false);

    return Cost;
  }

  InstructionCost getMemoryOpCost(unsigned Opcode, Type *Src,
                                  MaybeAlign Alignment, unsigned AddressSpace,
                                  TTI::TargetCostKind CostKind,
                                  const Instruction *I = nullptr) {
    assert(!Src->isVoidTy() && "Invalid type");
    // Assume types, such as structs, are expensive.
    if (getTLI()->getValueType(DL, Src,  true) == MVT::Other)
      return 4;
    std::pair<InstructionCost, MVT> LT =
        getTLI()->getTypeLegalizationCost(DL, Src);

    // Assuming that all loads of legal types cost 1.
    InstructionCost Cost = LT.first;
    if (CostKind != TTI::TCK_RecipThroughput)
      return Cost;

    if (Src->isVectorTy() &&
        // In practice it's not currently possible to have a change in lane
        // length for extending loads or truncating stores so both types should
        // have the same scalable property.
        TypeSize::isKnownLT(Src->getPrimitiveSizeInBits(),
                            LT.second.getSizeInBits())) {
      // This is a vector load that legalizes to a larger type than the vector
      // itself. Unless the corresponding extending load or truncating store is
      // legal, then this will scalarize.
      TargetLowering::LegalizeAction LA = TargetLowering::Expand;
      EVT MemVT = getTLI()->getValueType(DL, Src);
      if (Opcode == Instruction::Store)
        LA = getTLI()->getTruncStoreAction(LT.second, MemVT);
      else
        LA = getTLI()->getLoadExtAction(ISD::EXTLOAD, LT.second, MemVT);

      if (LA != TargetLowering::Legal && LA != TargetLowering::Custom) {
        // This is a vector load/store for some illegal type that is scalarized.
        // We must account for the cost of building or decomposing the vector.
        Cost += getScalarizationOverhead(cast<VectorType>(Src),
                                         Opcode != Instruction::Store,
                                         Opcode == Instruction::Store);
      }
    }

    return Cost;
  }

  InstructionCost getMaskedMemoryOpCost(unsigned Opcode, Type *DataTy,
                                        Align Alignment, unsigned AddressSpace,
                                        TTI::TargetCostKind CostKind) {
    return getCommonMaskedMemoryOpCost(Opcode, DataTy, Alignment, true, false,
                                       CostKind);
  }

  InstructionCost getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                         const Value *Ptr, bool VariableMask,
                                         Align Alignment,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I = nullptr) {
    return getCommonMaskedMemoryOpCost(Opcode, DataTy, Alignment, VariableMask,
                                       true, CostKind);
  }

  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond = false, bool UseMaskForGaps = false) {
    auto *VT = cast<FixedVectorType>(VecTy);

    unsigned NumElts = VT->getNumElements();
    assert(Factor > 1 && NumElts % Factor == 0 && "Invalid interleave factor");

    unsigned NumSubElts = NumElts / Factor;
    auto *SubVT = FixedVectorType::get(VT->getElementType(), NumSubElts);

    // Firstly, the cost of load/store operation.
    InstructionCost Cost;
    if (UseMaskForCond || UseMaskForGaps)
      Cost = thisT()->getMaskedMemoryOpCost(Opcode, VecTy, Alignment,
                                            AddressSpace, CostKind);
    else
      Cost = thisT()->getMemoryOpCost(Opcode, VecTy, Alignment, AddressSpace,
                                      CostKind);

    // Legalize the vector type, and get the legalized and unlegalized type
    // sizes.
    MVT VecTyLT = getTLI()->getTypeLegalizationCost(DL, VecTy).second;
    unsigned VecTySize = thisT()->getDataLayout().getTypeStoreSize(VecTy);
    unsigned VecTyLTSize = VecTyLT.getStoreSize();

    // Scale the cost of the memory operation by the fraction of legalized
    // instructions that will actually be used. We shouldn't account for the
    // cost of dead instructions since they will be removed.
    //
    // E.g., An interleaved load of factor 8:
    //       %vec = load <16 x i64>, <16 x i64>* %ptr
    //       %v0 = shufflevector %vec, undef, <0, 8>
    //
    // If <16 x i64> is legalized to 8 v2i64 loads, only 2 of the loads will be
    // used (those corresponding to elements [0:1] and [8:9] of the unlegalized
    // type). The other loads are unused.
    //
    // TODO: Note that legalization can turn masked loads/stores into unmasked
    // (legalized) loads/stores. This can be reflected in the cost.
    if (Cost.isValid() && VecTySize > VecTyLTSize) {
      // The number of loads of a legal type it will take to represent a load
      // of the unlegalized vector type.
      unsigned NumLegalInsts = divideCeil(VecTySize, VecTyLTSize);

      // The number of elements of the unlegalized type that correspond to a
      // single legal instruction.
      unsigned NumEltsPerLegalInst = divideCeil(NumElts, NumLegalInsts);

      // Determine which legal instructions will be used.
      BitVector UsedInsts(NumLegalInsts, false);
      for (unsigned Index : Indices)
        for (unsigned Elt = 0; Elt < NumSubElts; ++Elt)
          UsedInsts.set((Index + Elt * Factor) / NumEltsPerLegalInst);

      // Scale the cost of the load by the fraction of legal instructions that
      // will be used.
      Cost = divideCeil(UsedInsts.count() * Cost.getValue().getValue(),
                        NumLegalInsts);
    }

    // Then plus the cost of interleave operation.
    assert(Indices.size() <= Factor &&
           "Interleaved memory op has too many members");

    const APInt DemandedAllSubElts = APInt::getAllOnes(NumSubElts);
    const APInt DemandedAllResultElts = APInt::getAllOnes(NumElts);

    APInt DemandedLoadStoreElts = APInt::getZero(NumElts);
    for (unsigned Index : Indices) {
      assert(Index < Factor && "Invalid index for interleaved memory op");
      for (unsigned Elm = 0; Elm < NumSubElts; Elm++)
        DemandedLoadStoreElts.setBit(Index + Elm * Factor);
    }

    if (Opcode == Instruction::Load) {
      // The interleave cost is similar to extract sub vectors' elements
      // from the wide vector, and insert them into sub vectors.
      //
      // E.g. An interleaved load of factor 2 (with one member of index 0):
      //      %vec = load <8 x i32>, <8 x i32>* %ptr
      //      %v0 = shuffle %vec, undef, <0, 2, 4, 6>         ; Index 0
      // The cost is estimated as extract elements at 0, 2, 4, 6 from the
      // <8 x i32> vector and insert them into a <4 x i32> vector.
      InstructionCost InsSubCost =
          thisT()->getScalarizationOverhead(SubVT, DemandedAllSubElts,
                                            /*Insert*/ true, /*Extract*/ false);
      Cost += Indices.size() * InsSubCost;
      Cost +=
          thisT()->getScalarizationOverhead(VT, DemandedLoadStoreElts,
                                            /*Insert*/ false, /*Extract*/ true);
    } else {
      // The interleave cost is extract elements from sub vectors, and
      // insert them into the wide vector.
      //
      // E.g. An interleaved store of factor 3 with 2 members at indices 0,1:
      // (using VF=4):
      //    %v0_v1 = shuffle %v0, %v1, <0,4,undef,1,5,undef,2,6,undef,3,7,undef>
      //    %gaps.mask = <true, true, false, true, true, false,
      //                  true, true, false, true, true, false>
      //    call llvm.masked.store <12 x i32> %v0_v1, <12 x i32>* %ptr,
      //                           i32 Align, <12 x i1> %gaps.mask
      // The cost is estimated as extract all elements (of actual members,
      // excluding gaps) from both <4 x i32> vectors and insert into the <12 x
      // i32> vector.
      InstructionCost ExtSubCost =
          thisT()->getScalarizationOverhead(SubVT, DemandedAllSubElts,
                                            /*Insert*/ false, /*Extract*/ true);
      Cost += ExtSubCost * Indices.size();
      Cost += thisT()->getScalarizationOverhead(VT, DemandedLoadStoreElts,
                                                /*Insert*/ true,
                                                /*Extract*/ false);
    }

    if (!UseMaskForCond)
      return Cost;

    Type *I8Type = Type::getInt8Ty(VT->getContext());

    Cost += thisT()->getReplicationShuffleCost(
        I8Type, Factor, NumSubElts,
        UseMaskForGaps ? DemandedLoadStoreElts : DemandedAllResultElts,
        CostKind);

    // The Gaps mask is invariant and created outside the loop, therefore the
    // cost of creating it is not accounted for here. However if we have both
    // a MaskForGaps and some other mask that guards the execution of the
    // memory access, we need to account for the cost of And-ing the two masks
    // inside the loop.
    if (UseMaskForGaps) {
      auto *MaskVT = FixedVectorType::get(I8Type, NumElts);
      Cost += thisT()->getArithmeticInstrCost(BinaryOperator::And, MaskVT,
                                              CostKind);
    }

    return Cost;
  }

  /// Get intrinsic cost based on arguments.
  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind) {
    // Check for generically free intrinsics.
    if (BaseT::getIntrinsicInstrCost(ICA, CostKind) == 0)
      return 0;

    // Assume that target intrinsics are cheap.
    Intrinsic::ID IID = ICA.getID();
    if (Function::isTargetIntrinsic(IID))
      return TargetTransformInfo::TCC_Basic;

    if (ICA.isTypeBasedOnly())
      return getTypeBasedIntrinsicInstrCost(ICA, CostKind);

    Type *RetTy = ICA.getReturnType();

    ElementCount RetVF =
        (RetTy->isVectorTy() ? cast<VectorType>(RetTy)->getElementCount()
                             : ElementCount::getFixed(1));
    const IntrinsicInst *I = ICA.getInst();
    const SmallVectorImpl<const Value *> &Args = ICA.getArgs();
    FastMathFlags FMF = ICA.getFlags();
    switch (IID) {
    default:
      break;

    case Intrinsic::cttz:
      // FIXME: If necessary, this should go in target-specific overrides.
      if (RetVF.isScalar() && getTLI()->isCheapToSpeculateCttz())
        return TargetTransformInfo::TCC_Basic;
      break;

    case Intrinsic::ctlz:
      // FIXME: If necessary, this should go in target-specific overrides.
      if (RetVF.isScalar() && getTLI()->isCheapToSpeculateCtlz())
        return TargetTransformInfo::TCC_Basic;
      break;

    case Intrinsic::memcpy:
      return thisT()->getMemcpyCost(ICA.getInst());

    case Intrinsic::masked_scatter: {
      const Value *Mask = Args[3];
      bool VarMask = !isa<Constant>(Mask);
      Align Alignment = cast<ConstantInt>(Args[2])->getAlignValue();
      return thisT()->getGatherScatterOpCost(Instruction::Store,
                                             ICA.getArgTypes()[0], Args[1],
                                             VarMask, Alignment, CostKind, I);
    }
    case Intrinsic::masked_gather: {
      const Value *Mask = Args[2];
      bool VarMask = !isa<Constant>(Mask);
      Align Alignment = cast<ConstantInt>(Args[1])->getAlignValue();
      return thisT()->getGatherScatterOpCost(Instruction::Load, RetTy, Args[0],
                                             VarMask, Alignment, CostKind, I);
    }
    case Intrinsic::experimental_stepvector: {
      if (isa<ScalableVectorType>(RetTy))
        return BaseT::getIntrinsicInstrCost(ICA, CostKind);
      // The cost of materialising a constant integer vector.
      return TargetTransformInfo::TCC_Basic;
    }
    case Intrinsic::experimental_vector_extract: {
      // FIXME: Handle case where a scalable vector is extracted from a scalable
      // vector
      if (isa<ScalableVectorType>(RetTy))
        return BaseT::getIntrinsicInstrCost(ICA, CostKind);
      unsigned Index = cast<ConstantInt>(Args[1])->getZExtValue();
      return thisT()->getShuffleCost(TTI::SK_ExtractSubvector,
                                     cast<VectorType>(Args[0]->getType()), None,
                                     Index, cast<VectorType>(RetTy));
    }
    case Intrinsic::experimental_vector_insert: {
      // FIXME: Handle case where a scalable vector is inserted into a scalable
      // vector
      if (isa<ScalableVectorType>(Args[1]->getType()))
        return BaseT::getIntrinsicInstrCost(ICA, CostKind);
      unsigned Index = cast<ConstantInt>(Args[2])->getZExtValue();
      return thisT()->getShuffleCost(
          TTI::SK_InsertSubvector, cast<VectorType>(Args[0]->getType()), None,
          Index, cast<VectorType>(Args[1]->getType()));
    }
    case Intrinsic::experimental_vector_reverse: {
      return thisT()->getShuffleCost(TTI::SK_Reverse,
                                     cast<VectorType>(Args[0]->getType()), None,
                                     0, cast<VectorType>(RetTy));
    }
    case Intrinsic::experimental_vector_splice: {
      unsigned Index = cast<ConstantInt>(Args[2])->getZExtValue();
      return thisT()->getShuffleCost(TTI::SK_Splice,
                                     cast<VectorType>(Args[0]->getType()), None,
                                     Index, cast<VectorType>(RetTy));
    }
    case Intrinsic::vector_reduce_add:
    case Intrinsic::vector_reduce_mul:
    case Intrinsic::vector_reduce_and:
    case Intrinsic::vector_reduce_or:
    case Intrinsic::vector_reduce_xor:
    case Intrinsic::vector_reduce_smax:
    case Intrinsic::vector_reduce_smin:
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_fmin:
    case Intrinsic::vector_reduce_umax:
    case Intrinsic::vector_reduce_umin: {
      IntrinsicCostAttributes Attrs(IID, RetTy, Args[0]->getType(), FMF, I, 1);
      return getTypeBasedIntrinsicInstrCost(Attrs, CostKind);
    }
    case Intrinsic::vector_reduce_fadd:
    case Intrinsic::vector_reduce_fmul: {
      IntrinsicCostAttributes Attrs(
          IID, RetTy, {Args[0]->getType(), Args[1]->getType()}, FMF, I, 1);
      return getTypeBasedIntrinsicInstrCost(Attrs, CostKind);
    }
    case Intrinsic::fshl:
    case Intrinsic::fshr: {
      if (isa<ScalableVectorType>(RetTy))
        return BaseT::getIntrinsicInstrCost(ICA, CostKind);
      const Value *X = Args[0];
      const Value *Y = Args[1];
      const Value *Z = Args[2];
      TTI::OperandValueProperties OpPropsX, OpPropsY, OpPropsZ, OpPropsBW;
      TTI::OperandValueKind OpKindX = TTI::getOperandInfo(X, OpPropsX);
      TTI::OperandValueKind OpKindY = TTI::getOperandInfo(Y, OpPropsY);
      TTI::OperandValueKind OpKindZ = TTI::getOperandInfo(Z, OpPropsZ);
      TTI::OperandValueKind OpKindBW = TTI::OK_UniformConstantValue;
      OpPropsBW = isPowerOf2_32(RetTy->getScalarSizeInBits()) ? TTI::OP_PowerOf2
                                                              : TTI::OP_None;
      // fshl: (X << (Z % BW)) | (Y >> (BW - (Z % BW)))
      // fshr: (X << (BW - (Z % BW))) | (Y >> (Z % BW))
      InstructionCost Cost = 0;
      Cost +=
          thisT()->getArithmeticInstrCost(BinaryOperator::Or, RetTy, CostKind);
      Cost +=
          thisT()->getArithmeticInstrCost(BinaryOperator::Sub, RetTy, CostKind);
      Cost += thisT()->getArithmeticInstrCost(
          BinaryOperator::Shl, RetTy, CostKind, OpKindX, OpKindZ, OpPropsX);
      Cost += thisT()->getArithmeticInstrCost(
          BinaryOperator::LShr, RetTy, CostKind, OpKindY, OpKindZ, OpPropsY);
      // Non-constant shift amounts requires a modulo.
      if (OpKindZ != TTI::OK_UniformConstantValue &&
          OpKindZ != TTI::OK_NonUniformConstantValue)
        Cost += thisT()->getArithmeticInstrCost(BinaryOperator::URem, RetTy,
                                                CostKind, OpKindZ, OpKindBW,
                                                OpPropsZ, OpPropsBW);
      // For non-rotates (X != Y) we must add shift-by-zero handling costs.
      if (X != Y) {
        Type *CondTy = RetTy->getWithNewBitWidth(1);
        Cost +=
            thisT()->getCmpSelInstrCost(BinaryOperator::ICmp, RetTy, CondTy,
                                        CmpInst::ICMP_EQ, CostKind);
        Cost +=
            thisT()->getCmpSelInstrCost(BinaryOperator::Select, RetTy, CondTy,
                                        CmpInst::ICMP_EQ, CostKind);
      }
      return Cost;
    }
    }

    // Assume that we need to scalarize this intrinsic.
    // Compute the scalarization overhead based on Args for a vector
    // intrinsic.
    InstructionCost ScalarizationCost = InstructionCost::getInvalid();
    if (RetVF.isVector() && !RetVF.isScalable()) {
      ScalarizationCost = 0;
      if (!RetTy->isVoidTy())
        ScalarizationCost +=
            getScalarizationOverhead(cast<VectorType>(RetTy), true, false);
      ScalarizationCost +=
          getOperandsScalarizationOverhead(Args, ICA.getArgTypes());
    }

    IntrinsicCostAttributes Attrs(IID, RetTy, ICA.getArgTypes(), FMF, I,
                                  ScalarizationCost);
    return thisT()->getTypeBasedIntrinsicInstrCost(Attrs, CostKind);
  }

  /// Get intrinsic cost based on argument types.
  /// If ScalarizationCostPassed is std::numeric_limits<unsigned>::max(), the
  /// cost of scalarizing the arguments and the return value will be computed
  /// based on types.
  InstructionCost
  getTypeBasedIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                 TTI::TargetCostKind CostKind) {
    Intrinsic::ID IID = ICA.getID();
    Type *RetTy = ICA.getReturnType();
    const SmallVectorImpl<Type *> &Tys = ICA.getArgTypes();
    FastMathFlags FMF = ICA.getFlags();
    InstructionCost ScalarizationCostPassed = ICA.getScalarizationCost();
    bool SkipScalarizationCost = ICA.skipScalarizationCost();

    VectorType *VecOpTy = nullptr;
    if (!Tys.empty()) {
      // The vector reduction operand is operand 0 except for fadd/fmul.
      // Their operand 0 is a scalar start value, so the vector op is operand 1.
      unsigned VecTyIndex = 0;
      if (IID == Intrinsic::vector_reduce_fadd ||
          IID == Intrinsic::vector_reduce_fmul)
        VecTyIndex = 1;
      assert(Tys.size() > VecTyIndex && "Unexpected IntrinsicCostAttributes");
      VecOpTy = dyn_cast<VectorType>(Tys[VecTyIndex]);
    }

    // Library call cost - other than size, make it expensive.
    unsigned SingleCallCost = CostKind == TTI::TCK_CodeSize ? 1 : 10;
    SmallVector<unsigned, 2> ISDs;
    switch (IID) {
    default: {
      // Scalable vectors cannot be scalarized, so return Invalid.
      if (isa<ScalableVectorType>(RetTy) || any_of(Tys, [](const Type *Ty) {
            return isa<ScalableVectorType>(Ty);
          }))
        return InstructionCost::getInvalid();

      // Assume that we need to scalarize this intrinsic.
      InstructionCost ScalarizationCost =
          SkipScalarizationCost ? ScalarizationCostPassed : 0;
      unsigned ScalarCalls = 1;
      Type *ScalarRetTy = RetTy;
      if (auto *RetVTy = dyn_cast<VectorType>(RetTy)) {
        if (!SkipScalarizationCost)
          ScalarizationCost = getScalarizationOverhead(RetVTy, true, false);
        ScalarCalls = std::max(ScalarCalls,
                               cast<FixedVectorType>(RetVTy)->getNumElements());
        ScalarRetTy = RetTy->getScalarType();
      }
      SmallVector<Type *, 4> ScalarTys;
      for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
        Type *Ty = Tys[i];
        if (auto *VTy = dyn_cast<VectorType>(Ty)) {
          if (!SkipScalarizationCost)
            ScalarizationCost += getScalarizationOverhead(VTy, false, true);
          ScalarCalls = std::max(ScalarCalls,
                                 cast<FixedVectorType>(VTy)->getNumElements());
          Ty = Ty->getScalarType();
        }
        ScalarTys.push_back(Ty);
      }
      if (ScalarCalls == 1)
        return 1; // Return cost of a scalar intrinsic. Assume it to be cheap.

      IntrinsicCostAttributes ScalarAttrs(IID, ScalarRetTy, ScalarTys, FMF);
      InstructionCost ScalarCost =
          thisT()->getIntrinsicInstrCost(ScalarAttrs, CostKind);

      return ScalarCalls * ScalarCost + ScalarizationCost;
    }
    // Look for intrinsics that can be lowered directly or turned into a scalar
    // intrinsic call.
    case Intrinsic::sqrt:
      ISDs.push_back(ISD::FSQRT);
      break;
    case Intrinsic::sin:
      ISDs.push_back(ISD::FSIN);
      break;
    case Intrinsic::cos:
      ISDs.push_back(ISD::FCOS);
      break;
    case Intrinsic::exp:
      ISDs.push_back(ISD::FEXP);
      break;
    case Intrinsic::exp2:
      ISDs.push_back(ISD::FEXP2);
      break;
    case Intrinsic::log:
      ISDs.push_back(ISD::FLOG);
      break;
    case Intrinsic::log10:
      ISDs.push_back(ISD::FLOG10);
      break;
    case Intrinsic::log2:
      ISDs.push_back(ISD::FLOG2);
      break;
    case Intrinsic::fabs:
      ISDs.push_back(ISD::FABS);
      break;
    case Intrinsic::canonicalize:
      ISDs.push_back(ISD::FCANONICALIZE);
      break;
    case Intrinsic::minnum:
      ISDs.push_back(ISD::FMINNUM);
      break;
    case Intrinsic::maxnum:
      ISDs.push_back(ISD::FMAXNUM);
      break;
    case Intrinsic::minimum:
      ISDs.push_back(ISD::FMINIMUM);
      break;
    case Intrinsic::maximum:
      ISDs.push_back(ISD::FMAXIMUM);
      break;
    case Intrinsic::copysign:
      ISDs.push_back(ISD::FCOPYSIGN);
      break;
    case Intrinsic::floor:
      ISDs.push_back(ISD::FFLOOR);
      break;
    case Intrinsic::ceil:
      ISDs.push_back(ISD::FCEIL);
      break;
    case Intrinsic::trunc:
      ISDs.push_back(ISD::FTRUNC);
      break;
    case Intrinsic::nearbyint:
      ISDs.push_back(ISD::FNEARBYINT);
      break;
    case Intrinsic::rint:
      ISDs.push_back(ISD::FRINT);
      break;
    case Intrinsic::round:
      ISDs.push_back(ISD::FROUND);
      break;
    case Intrinsic::roundeven:
      ISDs.push_back(ISD::FROUNDEVEN);
      break;
    case Intrinsic::pow:
      ISDs.push_back(ISD::FPOW);
      break;
    case Intrinsic::fma:
      ISDs.push_back(ISD::FMA);
      break;
    case Intrinsic::fmuladd:
      ISDs.push_back(ISD::FMA);
      break;
    case Intrinsic::experimental_constrained_fmuladd:
      ISDs.push_back(ISD::STRICT_FMA);
      break;
    // FIXME: We should return 0 whenever getIntrinsicCost == TCC_Free.
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::sideeffect:
    case Intrinsic::pseudoprobe:
    case Intrinsic::arithmetic_fence:
      return 0;
    case Intrinsic::masked_store: {
      Type *Ty = Tys[0];
      Align TyAlign = thisT()->DL.getABITypeAlign(Ty);
      return thisT()->getMaskedMemoryOpCost(Instruction::Store, Ty, TyAlign, 0,
                                            CostKind);
    }
    case Intrinsic::masked_load: {
      Type *Ty = RetTy;
      Align TyAlign = thisT()->DL.getABITypeAlign(Ty);
      return thisT()->getMaskedMemoryOpCost(Instruction::Load, Ty, TyAlign, 0,
                                            CostKind);
    }
    case Intrinsic::vector_reduce_add:
      return thisT()->getArithmeticReductionCost(Instruction::Add, VecOpTy,
                                                 None, CostKind);
    case Intrinsic::vector_reduce_mul:
      return thisT()->getArithmeticReductionCost(Instruction::Mul, VecOpTy,
                                                 None, CostKind);
    case Intrinsic::vector_reduce_and:
      return thisT()->getArithmeticReductionCost(Instruction::And, VecOpTy,
                                                 None, CostKind);
    case Intrinsic::vector_reduce_or:
      return thisT()->getArithmeticReductionCost(Instruction::Or, VecOpTy, None,
                                                 CostKind);
    case Intrinsic::vector_reduce_xor:
      return thisT()->getArithmeticReductionCost(Instruction::Xor, VecOpTy,
                                                 None, CostKind);
    case Intrinsic::vector_reduce_fadd:
      return thisT()->getArithmeticReductionCost(Instruction::FAdd, VecOpTy,
                                                 FMF, CostKind);
    case Intrinsic::vector_reduce_fmul:
      return thisT()->getArithmeticReductionCost(Instruction::FMul, VecOpTy,
                                                 FMF, CostKind);
    case Intrinsic::vector_reduce_smax:
    case Intrinsic::vector_reduce_smin:
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_fmin:
      return thisT()->getMinMaxReductionCost(
          VecOpTy, cast<VectorType>(CmpInst::makeCmpResultType(VecOpTy)),
          /*IsUnsigned=*/false, CostKind);
    case Intrinsic::vector_reduce_umax:
    case Intrinsic::vector_reduce_umin:
      return thisT()->getMinMaxReductionCost(
          VecOpTy, cast<VectorType>(CmpInst::makeCmpResultType(VecOpTy)),
          /*IsUnsigned=*/true, CostKind);
    case Intrinsic::abs: {
      // abs(X) = select(icmp(X,0),X,sub(0,X))
      Type *CondTy = RetTy->getWithNewBitWidth(1);
      CmpInst::Predicate Pred = CmpInst::ICMP_SGT;
      InstructionCost Cost = 0;
      Cost += thisT()->getCmpSelInstrCost(BinaryOperator::ICmp, RetTy, CondTy,
                                          Pred, CostKind);
      Cost += thisT()->getCmpSelInstrCost(BinaryOperator::Select, RetTy, CondTy,
                                          Pred, CostKind);
      // TODO: Should we add an OperandValueProperties::OP_Zero property?
      Cost += thisT()->getArithmeticInstrCost(
          BinaryOperator::Sub, RetTy, CostKind, TTI::OK_UniformConstantValue);
      return Cost;
    }
    case Intrinsic::smax:
    case Intrinsic::smin:
    case Intrinsic::umax:
    case Intrinsic::umin: {
      // minmax(X,Y) = select(icmp(X,Y),X,Y)
      Type *CondTy = RetTy->getWithNewBitWidth(1);
      bool IsUnsigned = IID == Intrinsic::umax || IID == Intrinsic::umin;
      CmpInst::Predicate Pred =
          IsUnsigned ? CmpInst::ICMP_UGT : CmpInst::ICMP_SGT;
      InstructionCost Cost = 0;
      Cost += thisT()->getCmpSelInstrCost(BinaryOperator::ICmp, RetTy, CondTy,
                                          Pred, CostKind);
      Cost += thisT()->getCmpSelInstrCost(BinaryOperator::Select, RetTy, CondTy,
                                          Pred, CostKind);
      return Cost;
    }
    case Intrinsic::sadd_sat:
    case Intrinsic::ssub_sat: {
      Type *CondTy = RetTy->getWithNewBitWidth(1);

      Type *OpTy = StructType::create({RetTy, CondTy});
      Intrinsic::ID OverflowOp = IID == Intrinsic::sadd_sat
                                     ? Intrinsic::sadd_with_overflow
                                     : Intrinsic::ssub_with_overflow;
      CmpInst::Predicate Pred = CmpInst::ICMP_SGT;

      // SatMax -> Overflow && SumDiff < 0
      // SatMin -> Overflow && SumDiff >= 0
      InstructionCost Cost = 0;
      IntrinsicCostAttributes Attrs(OverflowOp, OpTy, {RetTy, RetTy}, FMF,
                                    nullptr, ScalarizationCostPassed);
      Cost += thisT()->getIntrinsicInstrCost(Attrs, CostKind);
      Cost += thisT()->getCmpSelInstrCost(BinaryOperator::ICmp, RetTy, CondTy,
                                          Pred, CostKind);
      Cost += 2 * thisT()->getCmpSelInstrCost(BinaryOperator::Select, RetTy,
                                              CondTy, Pred, CostKind);
      return Cost;
    }
    case Intrinsic::uadd_sat:
    case Intrinsic::usub_sat: {
      Type *CondTy = RetTy->getWithNewBitWidth(1);

      Type *OpTy = StructType::create({RetTy, CondTy});
      Intrinsic::ID OverflowOp = IID == Intrinsic::uadd_sat
                                     ? Intrinsic::uadd_with_overflow
                                     : Intrinsic::usub_with_overflow;

      InstructionCost Cost = 0;
      IntrinsicCostAttributes Attrs(OverflowOp, OpTy, {RetTy, RetTy}, FMF,
                                    nullptr, ScalarizationCostPassed);
      Cost += thisT()->getIntrinsicInstrCost(Attrs, CostKind);
      Cost +=
          thisT()->getCmpSelInstrCost(BinaryOperator::Select, RetTy, CondTy,
                                      CmpInst::BAD_ICMP_PREDICATE, CostKind);
      return Cost;
    }
    case Intrinsic::smul_fix:
    case Intrinsic::umul_fix: {
      unsigned ExtSize = RetTy->getScalarSizeInBits() * 2;
      Type *ExtTy = RetTy->getWithNewBitWidth(ExtSize);

      unsigned ExtOp =
          IID == Intrinsic::smul_fix ? Instruction::SExt : Instruction::ZExt;
      TTI::CastContextHint CCH = TTI::CastContextHint::None;

      InstructionCost Cost = 0;
      Cost += 2 * thisT()->getCastInstrCost(ExtOp, ExtTy, RetTy, CCH, CostKind);
      Cost +=
          thisT()->getArithmeticInstrCost(Instruction::Mul, ExtTy, CostKind);
      Cost += 2 * thisT()->getCastInstrCost(Instruction::Trunc, RetTy, ExtTy,
                                            CCH, CostKind);
      Cost += thisT()->getArithmeticInstrCost(Instruction::LShr, RetTy,
                                              CostKind, TTI::OK_AnyValue,
                                              TTI::OK_UniformConstantValue);
      Cost += thisT()->getArithmeticInstrCost(Instruction::Shl, RetTy, CostKind,
                                              TTI::OK_AnyValue,
                                              TTI::OK_UniformConstantValue);
      Cost += thisT()->getArithmeticInstrCost(Instruction::Or, RetTy, CostKind);
      return Cost;
    }
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::ssub_with_overflow: {
      Type *SumTy = RetTy->getContainedType(0);
      Type *OverflowTy = RetTy->getContainedType(1);
      unsigned Opcode = IID == Intrinsic::sadd_with_overflow
                            ? BinaryOperator::Add
                            : BinaryOperator::Sub;

      //   Add:
      //   Overflow -> (Result < LHS) ^ (RHS < 0)
      //   Sub:
      //   Overflow -> (Result < LHS) ^ (RHS > 0)
      InstructionCost Cost = 0;
      Cost += thisT()->getArithmeticInstrCost(Opcode, SumTy, CostKind);
      Cost += 2 * thisT()->getCmpSelInstrCost(
                      Instruction::ICmp, SumTy, OverflowTy,
                      CmpInst::ICMP_SGT, CostKind);
      Cost += thisT()->getArithmeticInstrCost(BinaryOperator::Xor, OverflowTy,
                                              CostKind);
      return Cost;
    }
    case Intrinsic::uadd_with_overflow:
    case Intrinsic::usub_with_overflow: {
      Type *SumTy = RetTy->getContainedType(0);
      Type *OverflowTy = RetTy->getContainedType(1);
      unsigned Opcode = IID == Intrinsic::uadd_with_overflow
                            ? BinaryOperator::Add
                            : BinaryOperator::Sub;
      CmpInst::Predicate Pred = IID == Intrinsic::uadd_with_overflow
                                    ? CmpInst::ICMP_ULT
                                    : CmpInst::ICMP_UGT;

      InstructionCost Cost = 0;
      Cost += thisT()->getArithmeticInstrCost(Opcode, SumTy, CostKind);
      Cost +=
          thisT()->getCmpSelInstrCost(BinaryOperator::ICmp, SumTy, OverflowTy,
                                      Pred, CostKind);
      return Cost;
    }
    case Intrinsic::smul_with_overflow:
    case Intrinsic::umul_with_overflow: {
      Type *MulTy = RetTy->getContainedType(0);
      Type *OverflowTy = RetTy->getContainedType(1);
      unsigned ExtSize = MulTy->getScalarSizeInBits() * 2;
      Type *ExtTy = MulTy->getWithNewBitWidth(ExtSize);
      bool IsSigned = IID == Intrinsic::smul_with_overflow;

      unsigned ExtOp = IsSigned ? Instruction::SExt : Instruction::ZExt;
      TTI::CastContextHint CCH = TTI::CastContextHint::None;

      InstructionCost Cost = 0;
      Cost += 2 * thisT()->getCastInstrCost(ExtOp, ExtTy, MulTy, CCH, CostKind);
      Cost +=
          thisT()->getArithmeticInstrCost(Instruction::Mul, ExtTy, CostKind);
      Cost += 2 * thisT()->getCastInstrCost(Instruction::Trunc, MulTy, ExtTy,
                                            CCH, CostKind);
      Cost += thisT()->getArithmeticInstrCost(Instruction::LShr, ExtTy,
                                              CostKind, TTI::OK_AnyValue,
                                              TTI::OK_UniformConstantValue);

      if (IsSigned)
        Cost += thisT()->getArithmeticInstrCost(Instruction::AShr, MulTy,
                                                CostKind, TTI::OK_AnyValue,
                                                TTI::OK_UniformConstantValue);

      Cost += thisT()->getCmpSelInstrCost(
          BinaryOperator::ICmp, MulTy, OverflowTy, CmpInst::ICMP_NE, CostKind);
      return Cost;
    }
    case Intrinsic::ctpop:
      ISDs.push_back(ISD::CTPOP);
      // In case of legalization use TCC_Expensive. This is cheaper than a
      // library call but still not a cheap instruction.
      SingleCallCost = TargetTransformInfo::TCC_Expensive;
      break;
    case Intrinsic::ctlz:
      ISDs.push_back(ISD::CTLZ);
      break;
    case Intrinsic::cttz:
      ISDs.push_back(ISD::CTTZ);
      break;
    case Intrinsic::bswap:
      ISDs.push_back(ISD::BSWAP);
      break;
    case Intrinsic::bitreverse:
      ISDs.push_back(ISD::BITREVERSE);
      break;
    }

    const TargetLoweringBase *TLI = getTLI();
    std::pair<InstructionCost, MVT> LT =
        TLI->getTypeLegalizationCost(DL, RetTy);

    SmallVector<InstructionCost, 2> LegalCost;
    SmallVector<InstructionCost, 2> CustomCost;
    for (unsigned ISD : ISDs) {
      if (TLI->isOperationLegalOrPromote(ISD, LT.second)) {
        if (IID == Intrinsic::fabs && LT.second.isFloatingPoint() &&
            TLI->isFAbsFree(LT.second)) {
          return 0;
        }

        // The operation is legal. Assume it costs 1.
        // If the type is split to multiple registers, assume that there is some
        // overhead to this.
        // TODO: Once we have extract/insert subvector cost we need to use them.
        if (LT.first > 1)
          LegalCost.push_back(LT.first * 2);
        else
          LegalCost.push_back(LT.first * 1);
      } else if (!TLI->isOperationExpand(ISD, LT.second)) {
        // If the operation is custom lowered then assume
        // that the code is twice as expensive.
        CustomCost.push_back(LT.first * 2);
      }
    }

    auto *MinLegalCostI = std::min_element(LegalCost.begin(), LegalCost.end());
    if (MinLegalCostI != LegalCost.end())
      return *MinLegalCostI;

    auto MinCustomCostI =
        std::min_element(CustomCost.begin(), CustomCost.end());
    if (MinCustomCostI != CustomCost.end())
      return *MinCustomCostI;

    // If we can't lower fmuladd into an FMA estimate the cost as a floating
    // point mul followed by an add.
    if (IID == Intrinsic::fmuladd)
      return thisT()->getArithmeticInstrCost(BinaryOperator::FMul, RetTy,
                                             CostKind) +
             thisT()->getArithmeticInstrCost(BinaryOperator::FAdd, RetTy,
                                             CostKind);
    if (IID == Intrinsic::experimental_constrained_fmuladd) {
      IntrinsicCostAttributes FMulAttrs(
        Intrinsic::experimental_constrained_fmul, RetTy, Tys);
      IntrinsicCostAttributes FAddAttrs(
        Intrinsic::experimental_constrained_fadd, RetTy, Tys);
      return thisT()->getIntrinsicInstrCost(FMulAttrs, CostKind) +
             thisT()->getIntrinsicInstrCost(FAddAttrs, CostKind);
    }

    // Else, assume that we need to scalarize this intrinsic. For math builtins
    // this will emit a costly libcall, adding call overhead and spills. Make it
    // very expensive.
    if (auto *RetVTy = dyn_cast<VectorType>(RetTy)) {
      // Scalable vectors cannot be scalarized, so return Invalid.
      if (isa<ScalableVectorType>(RetTy) || any_of(Tys, [](const Type *Ty) {
            return isa<ScalableVectorType>(Ty);
          }))
        return InstructionCost::getInvalid();

      InstructionCost ScalarizationCost =
          SkipScalarizationCost ? ScalarizationCostPassed
                                : getScalarizationOverhead(RetVTy, true, false);

      unsigned ScalarCalls = cast<FixedVectorType>(RetVTy)->getNumElements();
      SmallVector<Type *, 4> ScalarTys;
      for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
        Type *Ty = Tys[i];
        if (Ty->isVectorTy())
          Ty = Ty->getScalarType();
        ScalarTys.push_back(Ty);
      }
      IntrinsicCostAttributes Attrs(IID, RetTy->getScalarType(), ScalarTys, FMF);
      InstructionCost ScalarCost =
          thisT()->getIntrinsicInstrCost(Attrs, CostKind);
      for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
        if (auto *VTy = dyn_cast<VectorType>(Tys[i])) {
          if (!ICA.skipScalarizationCost())
            ScalarizationCost += getScalarizationOverhead(VTy, false, true);
          ScalarCalls = std::max(ScalarCalls,
                                 cast<FixedVectorType>(VTy)->getNumElements());
        }
      }
      return ScalarCalls * ScalarCost + ScalarizationCost;
    }

    // This is going to be turned into a library call, make it expensive.
    return SingleCallCost;
  }

  /// Compute a cost of the given call instruction.
  ///
  /// Compute the cost of calling function F with return type RetTy and
  /// argument types Tys. F might be nullptr, in this case the cost of an
  /// arbitrary call with the specified signature will be returned.
  /// This is used, for instance,  when we estimate call of a vector
  /// counterpart of the given function.
  /// \param F Called function, might be nullptr.
  /// \param RetTy Return value types.
  /// \param Tys Argument types.
  /// \returns The cost of Call instruction.
  InstructionCost getCallInstrCost(Function *F, Type *RetTy,
                                   ArrayRef<Type *> Tys,
                                   TTI::TargetCostKind CostKind) {
    return 10;
  }

  unsigned getNumberOfParts(Type *Tp) {
    std::pair<InstructionCost, MVT> LT =
        getTLI()->getTypeLegalizationCost(DL, Tp);
    return LT.first.isValid() ? *LT.first.getValue() : 0;
  }

  InstructionCost getAddressComputationCost(Type *Ty, ScalarEvolution *,
                                            const SCEV *) {
    return 0;
  }

  /// Try to calculate arithmetic and shuffle op costs for reduction intrinsics.
  /// We're assuming that reduction operation are performing the following way:
  ///
  /// %val1 = shufflevector<n x t> %val, <n x t> %undef,
  /// <n x i32> <i32 n/2, i32 n/2 + 1, ..., i32 n, i32 undef, ..., i32 undef>
  ///            \----------------v-------------/  \----------v------------/
  ///                            n/2 elements               n/2 elements
  /// %red1 = op <n x t> %val, <n x t> val1
  /// After this operation we have a vector %red1 where only the first n/2
  /// elements are meaningful, the second n/2 elements are undefined and can be
  /// dropped. All other operations are actually working with the vector of
  /// length n/2, not n, though the real vector length is still n.
  /// %val2 = shufflevector<n x t> %red1, <n x t> %undef,
  /// <n x i32> <i32 n/4, i32 n/4 + 1, ..., i32 n/2, i32 undef, ..., i32 undef>
  ///            \----------------v-------------/  \----------v------------/
  ///                            n/4 elements               3*n/4 elements
  /// %red2 = op <n x t> %red1, <n x t> val2  - working with the vector of
  /// length n/2, the resulting vector has length n/4 etc.
  ///
  /// The cost model should take into account that the actual length of the
  /// vector is reduced on each iteration.
  InstructionCost getTreeReductionCost(unsigned Opcode, VectorType *Ty,
                                       TTI::TargetCostKind CostKind) {
    Type *ScalarTy = Ty->getElementType();
    unsigned NumVecElts = cast<FixedVectorType>(Ty)->getNumElements();
    if ((Opcode == Instruction::Or || Opcode == Instruction::And) &&
        ScalarTy == IntegerType::getInt1Ty(Ty->getContext()) &&
        NumVecElts >= 2) {
      // Or reduction for i1 is represented as:
      // %val = bitcast <ReduxWidth x i1> to iReduxWidth
      // %res = cmp ne iReduxWidth %val, 0
      // And reduction for i1 is represented as:
      // %val = bitcast <ReduxWidth x i1> to iReduxWidth
      // %res = cmp eq iReduxWidth %val, 11111
      Type *ValTy = IntegerType::get(Ty->getContext(), NumVecElts);
      return thisT()->getCastInstrCost(Instruction::BitCast, ValTy, Ty,
                                       TTI::CastContextHint::None, CostKind) +
             thisT()->getCmpSelInstrCost(Instruction::ICmp, ValTy,
                                         CmpInst::makeCmpResultType(ValTy),
                                         CmpInst::BAD_ICMP_PREDICATE, CostKind);
    }
    unsigned NumReduxLevels = Log2_32(NumVecElts);
    InstructionCost ArithCost = 0;
    InstructionCost ShuffleCost = 0;
    std::pair<InstructionCost, MVT> LT =
        thisT()->getTLI()->getTypeLegalizationCost(DL, Ty);
    unsigned LongVectorCount = 0;
    unsigned MVTLen =
        LT.second.isVector() ? LT.second.getVectorNumElements() : 1;
    while (NumVecElts > MVTLen) {
      NumVecElts /= 2;
      VectorType *SubTy = FixedVectorType::get(ScalarTy, NumVecElts);
      ShuffleCost += thisT()->getShuffleCost(TTI::SK_ExtractSubvector, Ty, None,
                                             NumVecElts, SubTy);
      ArithCost += thisT()->getArithmeticInstrCost(Opcode, SubTy, CostKind);
      Ty = SubTy;
      ++LongVectorCount;
    }

    NumReduxLevels -= LongVectorCount;

    // The minimal length of the vector is limited by the real length of vector
    // operations performed on the current platform. That's why several final
    // reduction operations are performed on the vectors with the same
    // architecture-dependent length.

    // By default reductions need one shuffle per reduction level.
    ShuffleCost += NumReduxLevels * thisT()->getShuffleCost(
                                     TTI::SK_PermuteSingleSrc, Ty, None, 0, Ty);
    ArithCost +=
        NumReduxLevels * thisT()->getArithmeticInstrCost(Opcode, Ty, CostKind);
    return ShuffleCost + ArithCost +
           thisT()->getVectorInstrCost(Instruction::ExtractElement, Ty, 0);
  }

  /// Try to calculate the cost of performing strict (in-order) reductions,
  /// which involves doing a sequence of floating point additions in lane
  /// order, starting with an initial value. For example, consider a scalar
  /// initial value 'InitVal' of type float and a vector of type <4 x float>:
  ///
  ///   Vector = <float %v0, float %v1, float %v2, float %v3>
  ///
  ///   %add1 = %InitVal + %v0
  ///   %add2 = %add1 + %v1
  ///   %add3 = %add2 + %v2
  ///   %add4 = %add3 + %v3
  ///
  /// As a simple estimate we can say the cost of such a reduction is 4 times
  /// the cost of a scalar FP addition. We can only estimate the costs for
  /// fixed-width vectors here because for scalable vectors we do not know the
  /// runtime number of operations.
  InstructionCost getOrderedReductionCost(unsigned Opcode, VectorType *Ty,
                                          TTI::TargetCostKind CostKind) {
    // Targets must implement a default value for the scalable case, since
    // we don't know how many lanes the vector has.
    if (isa<ScalableVectorType>(Ty))
      return InstructionCost::getInvalid();

    auto *VTy = cast<FixedVectorType>(Ty);
    InstructionCost ExtractCost =
        getScalarizationOverhead(VTy, /*Insert=*/false, /*Extract=*/true);
    InstructionCost ArithCost = thisT()->getArithmeticInstrCost(
        Opcode, VTy->getElementType(), CostKind);
    ArithCost *= VTy->getNumElements();

    return ExtractCost + ArithCost;
  }

  InstructionCost getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                                             Optional<FastMathFlags> FMF,
                                             TTI::TargetCostKind CostKind) {
    if (TTI::requiresOrderedReduction(FMF))
      return getOrderedReductionCost(Opcode, Ty, CostKind);
    return getTreeReductionCost(Opcode, Ty, CostKind);
  }

  /// Try to calculate op costs for min/max reduction operations.
  /// \param CondTy Conditional type for the Select instruction.
  InstructionCost getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                         bool IsUnsigned,
                                         TTI::TargetCostKind CostKind) {
    Type *ScalarTy = Ty->getElementType();
    Type *ScalarCondTy = CondTy->getElementType();
    unsigned NumVecElts = cast<FixedVectorType>(Ty)->getNumElements();
    unsigned NumReduxLevels = Log2_32(NumVecElts);
    unsigned CmpOpcode;
    if (Ty->isFPOrFPVectorTy()) {
      CmpOpcode = Instruction::FCmp;
    } else {
      assert(Ty->isIntOrIntVectorTy() &&
             "expecting floating point or integer type for min/max reduction");
      CmpOpcode = Instruction::ICmp;
    }
    InstructionCost MinMaxCost = 0;
    InstructionCost ShuffleCost = 0;
    std::pair<InstructionCost, MVT> LT =
        thisT()->getTLI()->getTypeLegalizationCost(DL, Ty);
    unsigned LongVectorCount = 0;
    unsigned MVTLen =
        LT.second.isVector() ? LT.second.getVectorNumElements() : 1;
    while (NumVecElts > MVTLen) {
      NumVecElts /= 2;
      auto *SubTy = FixedVectorType::get(ScalarTy, NumVecElts);
      CondTy = FixedVectorType::get(ScalarCondTy, NumVecElts);

      ShuffleCost += thisT()->getShuffleCost(TTI::SK_ExtractSubvector, Ty, None,
                                             NumVecElts, SubTy);
      MinMaxCost +=
          thisT()->getCmpSelInstrCost(CmpOpcode, SubTy, CondTy,
                                      CmpInst::BAD_ICMP_PREDICATE, CostKind) +
          thisT()->getCmpSelInstrCost(Instruction::Select, SubTy, CondTy,
                                      CmpInst::BAD_ICMP_PREDICATE, CostKind);
      Ty = SubTy;
      ++LongVectorCount;
    }

    NumReduxLevels -= LongVectorCount;

    // The minimal length of the vector is limited by the real length of vector
    // operations performed on the current platform. That's why several final
    // reduction opertions are perfomed on the vectors with the same
    // architecture-dependent length.
    ShuffleCost += NumReduxLevels * thisT()->getShuffleCost(
                                     TTI::SK_PermuteSingleSrc, Ty, None, 0, Ty);
    MinMaxCost +=
        NumReduxLevels *
        (thisT()->getCmpSelInstrCost(CmpOpcode, Ty, CondTy,
                                     CmpInst::BAD_ICMP_PREDICATE, CostKind) +
         thisT()->getCmpSelInstrCost(Instruction::Select, Ty, CondTy,
                                     CmpInst::BAD_ICMP_PREDICATE, CostKind));
    // The last min/max should be in vector registers and we counted it above.
    // So just need a single extractelement.
    return ShuffleCost + MinMaxCost +
           thisT()->getVectorInstrCost(Instruction::ExtractElement, Ty, 0);
  }

  InstructionCost getExtendedAddReductionCost(bool IsMLA, bool IsUnsigned,
                                              Type *ResTy, VectorType *Ty,
                                              TTI::TargetCostKind CostKind) {
    // Without any native support, this is equivalent to the cost of
    // vecreduce.add(ext) or if IsMLA vecreduce.add(mul(ext, ext))
    VectorType *ExtTy = VectorType::get(ResTy, Ty);
    InstructionCost RedCost = thisT()->getArithmeticReductionCost(
        Instruction::Add, ExtTy, None, CostKind);
    InstructionCost MulCost = 0;
    InstructionCost ExtCost = thisT()->getCastInstrCost(
        IsUnsigned ? Instruction::ZExt : Instruction::SExt, ExtTy, Ty,
        TTI::CastContextHint::None, CostKind);
    if (IsMLA) {
      MulCost =
          thisT()->getArithmeticInstrCost(Instruction::Mul, ExtTy, CostKind);
      ExtCost *= 2;
    }

    return RedCost + MulCost + ExtCost;
  }

  InstructionCost getVectorSplitCost() { return 1; }

  /// @}
};

/// Concrete BasicTTIImpl that can be used if no further customization
/// is needed.
class BasicTTIImpl : public BasicTTIImplBase<BasicTTIImpl> {
  using BaseT = BasicTTIImplBase<BasicTTIImpl>;

  friend class BasicTTIImplBase<BasicTTIImpl>;

  const TargetSubtargetInfo *ST;
  const TargetLoweringBase *TLI;

  const TargetSubtargetInfo *getST() const { return ST; }
  const TargetLoweringBase *getTLI() const { return TLI; }

public:
  explicit BasicTTIImpl(const TargetMachine *TM, const Function &F);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_BASICTTIIMPL_H
