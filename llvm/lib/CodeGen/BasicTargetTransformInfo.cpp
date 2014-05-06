//===- BasicTargetTransformInfo.cpp - Basic target-independent TTI impl ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the implementation of a basic TargetTransformInfo pass
/// predicated on the target abstractions present in the target independent
/// code generator. It uses these (primarily TargetLowering) to model as much
/// of the TTI query interface as possible. It is included by most targets so
/// that they can specialize only a small subset of the query space.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Target/TargetLowering.h"
#include <utility>
using namespace llvm;

#define DEBUG_TYPE "basictti"

namespace {

class BasicTTI final : public ImmutablePass, public TargetTransformInfo {
  const TargetMachine *TM;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

  const TargetLoweringBase *getTLI() const { return TM->getTargetLowering(); }

public:
  BasicTTI() : ImmutablePass(ID), TM(nullptr) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  BasicTTI(const TargetMachine *TM) : ImmutablePass(ID), TM(TM) {
    initializeBasicTTIPass(*PassRegistry::getPassRegistry());
  }

  void initializePass() override {
    pushTTIStack(this);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  bool hasBranchDivergence() const override;

  /// \name Scalar TTI Implementations
  /// @{

  bool isLegalAddImmediate(int64_t imm) const override;
  bool isLegalICmpImmediate(int64_t imm) const override;
  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                             int64_t BaseOffset, bool HasBaseReg,
                             int64_t Scale) const override;
  int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                           int64_t BaseOffset, bool HasBaseReg,
                           int64_t Scale) const override;
  bool isTruncateFree(Type *Ty1, Type *Ty2) const override;
  bool isTypeLegal(Type *Ty) const override;
  unsigned getJumpBufAlignment() const override;
  unsigned getJumpBufSize() const override;
  bool shouldBuildLookupTables() const override;
  bool haveFastSqrt(Type *Ty) const override;
  void getUnrollingPreferences(Loop *L,
                               UnrollingPreferences &UP) const override;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool Vector) const override;
  unsigned getMaximumUnrollFactor() const override;
  unsigned getRegisterBitWidth(bool Vector) const override;
  unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty, OperandValueKind,
                                  OperandValueKind) const override;
  unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                          int Index, Type *SubTp) const override;
  unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                            Type *Src) const override;
  unsigned getCFInstrCost(unsigned Opcode) const override;
  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                              Type *CondTy) const override;
  unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                              unsigned Index) const override;
  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                           unsigned AddressSpace) const override;
  unsigned getIntrinsicInstrCost(Intrinsic::ID, Type *RetTy,
                                 ArrayRef<Type*> Tys) const override;
  unsigned getNumberOfParts(Type *Tp) const override;
  unsigned getAddressComputationCost( Type *Ty, bool IsComplex) const override;
  unsigned getReductionCost(unsigned Opcode, Type *Ty,
                            bool IsPairwise) const override;

  /// @}
};

}

INITIALIZE_AG_PASS(BasicTTI, TargetTransformInfo, "basictti",
                   "Target independent code generator's TTI", true, true, false)
char BasicTTI::ID = 0;

ImmutablePass *
llvm::createBasicTargetTransformInfoPass(const TargetMachine *TM) {
  return new BasicTTI(TM);
}

bool BasicTTI::hasBranchDivergence() const { return false; }

bool BasicTTI::isLegalAddImmediate(int64_t imm) const {
  return getTLI()->isLegalAddImmediate(imm);
}

bool BasicTTI::isLegalICmpImmediate(int64_t imm) const {
  return getTLI()->isLegalICmpImmediate(imm);
}

bool BasicTTI::isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale) const {
  TargetLoweringBase::AddrMode AM;
  AM.BaseGV = BaseGV;
  AM.BaseOffs = BaseOffset;
  AM.HasBaseReg = HasBaseReg;
  AM.Scale = Scale;
  return getTLI()->isLegalAddressingMode(AM, Ty);
}

int BasicTTI::getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                   int64_t BaseOffset, bool HasBaseReg,
                                   int64_t Scale) const {
  TargetLoweringBase::AddrMode AM;
  AM.BaseGV = BaseGV;
  AM.BaseOffs = BaseOffset;
  AM.HasBaseReg = HasBaseReg;
  AM.Scale = Scale;
  return getTLI()->getScalingFactorCost(AM, Ty);
}

bool BasicTTI::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return getTLI()->isTruncateFree(Ty1, Ty2);
}

bool BasicTTI::isTypeLegal(Type *Ty) const {
  EVT T = getTLI()->getValueType(Ty);
  return getTLI()->isTypeLegal(T);
}

unsigned BasicTTI::getJumpBufAlignment() const {
  return getTLI()->getJumpBufAlignment();
}

unsigned BasicTTI::getJumpBufSize() const {
  return getTLI()->getJumpBufSize();
}

bool BasicTTI::shouldBuildLookupTables() const {
  const TargetLoweringBase *TLI = getTLI();
  return TLI->supportJumpTables() &&
      (TLI->isOperationLegalOrCustom(ISD::BR_JT, MVT::Other) ||
       TLI->isOperationLegalOrCustom(ISD::BRIND, MVT::Other));
}

bool BasicTTI::haveFastSqrt(Type *Ty) const {
  const TargetLoweringBase *TLI = getTLI();
  EVT VT = TLI->getValueType(Ty);
  return TLI->isTypeLegal(VT) && TLI->isOperationLegalOrCustom(ISD::FSQRT, VT);
}

void BasicTTI::getUnrollingPreferences(Loop *, UnrollingPreferences &) const { }

//===----------------------------------------------------------------------===//
//
// Calls used by the vectorizers.
//
//===----------------------------------------------------------------------===//

unsigned BasicTTI::getScalarizationOverhead(Type *Ty, bool Insert,
                                            bool Extract) const {
  assert (Ty->isVectorTy() && "Can only scalarize vectors");
  unsigned Cost = 0;

  for (int i = 0, e = Ty->getVectorNumElements(); i < e; ++i) {
    if (Insert)
      Cost += TopTTI->getVectorInstrCost(Instruction::InsertElement, Ty, i);
    if (Extract)
      Cost += TopTTI->getVectorInstrCost(Instruction::ExtractElement, Ty, i);
  }

  return Cost;
}

unsigned BasicTTI::getNumberOfRegisters(bool Vector) const {
  return 1;
}

unsigned BasicTTI::getRegisterBitWidth(bool Vector) const {
  return 32;
}

unsigned BasicTTI::getMaximumUnrollFactor() const {
  return 1;
}

unsigned BasicTTI::getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                          OperandValueKind,
                                          OperandValueKind) const {
  // Check if any of the operands are vector operands.
  const TargetLoweringBase *TLI = getTLI();
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Ty);

  bool IsFloat = Ty->getScalarType()->isFloatingPointTy();
  // Assume that floating point arithmetic operations cost twice as much as
  // integer operations.
  unsigned OpCost = (IsFloat ? 2 : 1);

  if (TLI->isOperationLegalOrPromote(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1.
    // If the type is split to multiple registers, assume that there is some
    // overhead to this.
    // TODO: Once we have extract/insert subvector cost we need to use them.
    if (LT.first > 1)
      return LT.first * 2 * OpCost;
    return LT.first * 1 * OpCost;
  }

  if (!TLI->isOperationExpand(ISD, LT.second)) {
    // If the operation is custom lowered then assume
    // thare the code is twice as expensive.
    return LT.first * 2 * OpCost;
  }

  // Else, assume that we need to scalarize this op.
  if (Ty->isVectorTy()) {
    unsigned Num = Ty->getVectorNumElements();
    unsigned Cost = TopTTI->getArithmeticInstrCost(Opcode, Ty->getScalarType());
    // return the cost of multiple scalar invocation plus the cost of inserting
    // and extracting the values.
    return getScalarizationOverhead(Ty, true, true) + Num * Cost;
  }

  // We don't know anything about this scalar instruction.
  return OpCost;
}

unsigned BasicTTI::getShuffleCost(ShuffleKind Kind, Type *Tp, int Index,
                                  Type *SubTp) const {
  return 1;
}

unsigned BasicTTI::getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const {
  const TargetLoweringBase *TLI = getTLI();
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  std::pair<unsigned, MVT> SrcLT = TLI->getTypeLegalizationCost(Src);
  std::pair<unsigned, MVT> DstLT = TLI->getTypeLegalizationCost(Dst);

  // Check for NOOP conversions.
  if (SrcLT.first == DstLT.first &&
      SrcLT.second.getSizeInBits() == DstLT.second.getSizeInBits()) {

      // Bitcast between types that are legalized to the same type are free.
      if (Opcode == Instruction::BitCast || Opcode == Instruction::Trunc)
        return 0;
  }

  if (Opcode == Instruction::Trunc &&
      TLI->isTruncateFree(SrcLT.second, DstLT.second))
    return 0;

  if (Opcode == Instruction::ZExt &&
      TLI->isZExtFree(SrcLT.second, DstLT.second))
    return 0;

  // If the cast is marked as legal (or promote) then assume low cost.
  if (SrcLT.first == DstLT.first &&
      TLI->isOperationLegalOrPromote(ISD, DstLT.second))
    return 1;

  // Handle scalar conversions.
  if (!Src->isVectorTy() && !Dst->isVectorTy()) {

    // Scalar bitcasts are usually free.
    if (Opcode == Instruction::BitCast)
      return 0;

    // Just check the op cost. If the operation is legal then assume it costs 1.
    if (!TLI->isOperationExpand(ISD, DstLT.second))
      return  1;

    // Assume that illegal scalar instruction are expensive.
    return 4;
  }

  // Check vector-to-vector casts.
  if (Dst->isVectorTy() && Src->isVectorTy()) {

    // If the cast is between same-sized registers, then the check is simple.
    if (SrcLT.first == DstLT.first &&
        SrcLT.second.getSizeInBits() == DstLT.second.getSizeInBits()) {

      // Assume that Zext is done using AND.
      if (Opcode == Instruction::ZExt)
        return 1;

      // Assume that sext is done using SHL and SRA.
      if (Opcode == Instruction::SExt)
        return 2;

      // Just check the op cost. If the operation is legal then assume it costs
      // 1 and multiply by the type-legalization overhead.
      if (!TLI->isOperationExpand(ISD, DstLT.second))
        return SrcLT.first * 1;
    }

    // If we are converting vectors and the operation is illegal, or
    // if the vectors are legalized to different types, estimate the
    // scalarization costs.
    unsigned Num = Dst->getVectorNumElements();
    unsigned Cost = TopTTI->getCastInstrCost(Opcode, Dst->getScalarType(),
                                             Src->getScalarType());

    // Return the cost of multiple scalar invocation plus the cost of
    // inserting and extracting the values.
    return getScalarizationOverhead(Dst, true, true) + Num * Cost;
  }

  // We already handled vector-to-vector and scalar-to-scalar conversions. This
  // is where we handle bitcast between vectors and scalars. We need to assume
  //  that the conversion is scalarized in one way or another.
  if (Opcode == Instruction::BitCast)
    // Illegal bitcasts are done by storing and loading from a stack slot.
    return (Src->isVectorTy()? getScalarizationOverhead(Src, false, true):0) +
           (Dst->isVectorTy()? getScalarizationOverhead(Dst, true, false):0);

  llvm_unreachable("Unhandled cast");
 }

unsigned BasicTTI::getCFInstrCost(unsigned Opcode) const {
  // Branches are assumed to be predicted.
  return 0;
}

unsigned BasicTTI::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) const {
  const TargetLoweringBase *TLI = getTLI();
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  // Selects on vectors are actually vector selects.
  if (ISD == ISD::SELECT) {
    assert(CondTy && "CondTy must exist");
    if (CondTy->isVectorTy())
      ISD = ISD::VSELECT;
  }

  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(ValTy);

  if (!TLI->isOperationExpand(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1. Multiply
    // by the type-legalization overhead.
    return LT.first * 1;
  }

  // Otherwise, assume that the cast is scalarized.
  if (ValTy->isVectorTy()) {
    unsigned Num = ValTy->getVectorNumElements();
    if (CondTy)
      CondTy = CondTy->getScalarType();
    unsigned Cost = TopTTI->getCmpSelInstrCost(Opcode, ValTy->getScalarType(),
                                               CondTy);

    // Return the cost of multiple scalar invocation plus the cost of inserting
    // and extracting the values.
    return getScalarizationOverhead(ValTy, true, false) + Num * Cost;
  }

  // Unknown scalar opcode.
  return 1;
}

unsigned BasicTTI::getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) const {
  std::pair<unsigned, MVT> LT =  getTLI()->getTypeLegalizationCost(Val->getScalarType());

  return LT.first;
}

unsigned BasicTTI::getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const {
  assert(!Src->isVoidTy() && "Invalid type");
  std::pair<unsigned, MVT> LT = getTLI()->getTypeLegalizationCost(Src);

  // Assuming that all loads of legal types cost 1.
  unsigned Cost = LT.first;

  if (Src->isVectorTy() &&
      Src->getPrimitiveSizeInBits() < LT.second.getSizeInBits()) {
    // This is a vector load that legalizes to a larger type than the vector
    // itself. Unless the corresponding extending load or truncating store is
    // legal, then this will scalarize.
    TargetLowering::LegalizeAction LA = TargetLowering::Expand;
    EVT MemVT = getTLI()->getValueType(Src, true);
    if (MemVT.isSimple() && MemVT != MVT::Other) {
      if (Opcode == Instruction::Store)
        LA = getTLI()->getTruncStoreAction(LT.second, MemVT.getSimpleVT());
      else
        LA = getTLI()->getLoadExtAction(ISD::EXTLOAD, MemVT.getSimpleVT());
    }

    if (LA != TargetLowering::Legal && LA != TargetLowering::Custom) {
      // This is a vector load/store for some illegal type that is scalarized.
      // We must account for the cost of building or decomposing the vector.
      Cost += getScalarizationOverhead(Src, Opcode != Instruction::Store,
                                            Opcode == Instruction::Store);
    }
  }

  return Cost;
}

unsigned BasicTTI::getIntrinsicInstrCost(Intrinsic::ID IID, Type *RetTy,
                                         ArrayRef<Type *> Tys) const {
  unsigned ISD = 0;
  switch (IID) {
  default: {
    // Assume that we need to scalarize this intrinsic.
    unsigned ScalarizationCost = 0;
    unsigned ScalarCalls = 1;
    if (RetTy->isVectorTy()) {
      ScalarizationCost = getScalarizationOverhead(RetTy, true, false);
      ScalarCalls = std::max(ScalarCalls, RetTy->getVectorNumElements());
    }
    for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
      if (Tys[i]->isVectorTy()) {
        ScalarizationCost += getScalarizationOverhead(Tys[i], false, true);
        ScalarCalls = std::max(ScalarCalls, RetTy->getVectorNumElements());
      }
    }

    return ScalarCalls + ScalarizationCost;
  }
  // Look for intrinsics that can be lowered directly or turned into a scalar
  // intrinsic call.
  case Intrinsic::sqrt:    ISD = ISD::FSQRT;  break;
  case Intrinsic::sin:     ISD = ISD::FSIN;   break;
  case Intrinsic::cos:     ISD = ISD::FCOS;   break;
  case Intrinsic::exp:     ISD = ISD::FEXP;   break;
  case Intrinsic::exp2:    ISD = ISD::FEXP2;  break;
  case Intrinsic::log:     ISD = ISD::FLOG;   break;
  case Intrinsic::log10:   ISD = ISD::FLOG10; break;
  case Intrinsic::log2:    ISD = ISD::FLOG2;  break;
  case Intrinsic::fabs:    ISD = ISD::FABS;   break;
  case Intrinsic::copysign: ISD = ISD::FCOPYSIGN; break;
  case Intrinsic::floor:   ISD = ISD::FFLOOR; break;
  case Intrinsic::ceil:    ISD = ISD::FCEIL;  break;
  case Intrinsic::trunc:   ISD = ISD::FTRUNC; break;
  case Intrinsic::nearbyint:
                           ISD = ISD::FNEARBYINT; break;
  case Intrinsic::rint:    ISD = ISD::FRINT;  break;
  case Intrinsic::round:   ISD = ISD::FROUND; break;
  case Intrinsic::pow:     ISD = ISD::FPOW;   break;
  case Intrinsic::fma:     ISD = ISD::FMA;    break;
  case Intrinsic::fmuladd: ISD = ISD::FMA;    break;
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
    return 0;
  }

  const TargetLoweringBase *TLI = getTLI();
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(RetTy);

  if (TLI->isOperationLegalOrPromote(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1.
    // If the type is split to multiple registers, assume that thre is some
    // overhead to this.
    // TODO: Once we have extract/insert subvector cost we need to use them.
    if (LT.first > 1)
      return LT.first * 2;
    return LT.first * 1;
  }

  if (!TLI->isOperationExpand(ISD, LT.second)) {
    // If the operation is custom lowered then assume
    // thare the code is twice as expensive.
    return LT.first * 2;
  }

  // If we can't lower fmuladd into an FMA estimate the cost as a floating
  // point mul followed by an add.
  if (IID == Intrinsic::fmuladd)
    return TopTTI->getArithmeticInstrCost(BinaryOperator::FMul, RetTy) +
           TopTTI->getArithmeticInstrCost(BinaryOperator::FAdd, RetTy);

  // Else, assume that we need to scalarize this intrinsic. For math builtins
  // this will emit a costly libcall, adding call overhead and spills. Make it
  // very expensive.
  if (RetTy->isVectorTy()) {
    unsigned Num = RetTy->getVectorNumElements();
    unsigned Cost = TopTTI->getIntrinsicInstrCost(IID, RetTy->getScalarType(),
                                                  Tys);
    return 10 * Cost * Num;
  }

  // This is going to be turned into a library call, make it expensive.
  return 10;
}

unsigned BasicTTI::getNumberOfParts(Type *Tp) const {
  std::pair<unsigned, MVT> LT = getTLI()->getTypeLegalizationCost(Tp);
  return LT.first;
}

unsigned BasicTTI::getAddressComputationCost(Type *Ty, bool IsComplex) const {
  return 0;
}

unsigned BasicTTI::getReductionCost(unsigned Opcode, Type *Ty,
                                    bool IsPairwise) const {
  assert(Ty->isVectorTy() && "Expect a vector type");
  unsigned NumVecElts = Ty->getVectorNumElements();
  unsigned NumReduxLevels = Log2_32(NumVecElts);
  unsigned ArithCost = NumReduxLevels *
    TopTTI->getArithmeticInstrCost(Opcode, Ty);
  // Assume the pairwise shuffles add a cost.
  unsigned ShuffleCost =
      NumReduxLevels * (IsPairwise + 1) *
      TopTTI->getShuffleCost(SK_ExtractSubvector, Ty, NumVecElts / 2, Ty);
  return ShuffleCost + ArithCost + getScalarizationOverhead(Ty, false, true);
}
