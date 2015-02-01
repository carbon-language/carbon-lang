//===-- AArch64TargetTransformInfo.cpp - AArch64 specific TTI -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetTransformInfo.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "aarch64tti"

/// \brief Calculate the cost of materializing a 64-bit value. This helper
/// method might only calculate a fraction of a larger immediate. Therefore it
/// is valid to return a cost of ZERO.
unsigned AArch64TTIImpl::getIntImmCost(int64_t Val) {
  // Check if the immediate can be encoded within an instruction.
  if (Val == 0 || AArch64_AM::isLogicalImmediate(Val, 64))
    return 0;

  if (Val < 0)
    Val = ~Val;

  // Calculate how many moves we will need to materialize this constant.
  unsigned LZ = countLeadingZeros((uint64_t)Val);
  return (64 - LZ + 15) / 16;
}

/// \brief Calculate the cost of materializing the given constant.
unsigned AArch64TTIImpl::getIntImmCost(const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  // Sign-extend all constants to a multiple of 64-bit.
  APInt ImmVal = Imm;
  if (BitSize & 0x3f)
    ImmVal = Imm.sext((BitSize + 63) & ~0x3fU);

  // Split the constant into 64-bit chunks and calculate the cost for each
  // chunk.
  unsigned Cost = 0;
  for (unsigned ShiftVal = 0; ShiftVal < BitSize; ShiftVal += 64) {
    APInt Tmp = ImmVal.ashr(ShiftVal).sextOrTrunc(64);
    int64_t Val = Tmp.getSExtValue();
    Cost += getIntImmCost(Val);
  }
  // We need at least one instruction to materialze the constant.
  return std::max(1U, Cost);
}

unsigned AArch64TTIImpl::getIntImmCost(unsigned Opcode, unsigned Idx,
                                       const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  // There is no cost model for constants with a bit size of 0. Return TCC_Free
  // here, so that constant hoisting will ignore this constant.
  if (BitSize == 0)
    return TTI::TCC_Free;

  unsigned ImmIdx = ~0U;
  switch (Opcode) {
  default:
    return TTI::TCC_Free;
  case Instruction::GetElementPtr:
    // Always hoist the base address of a GetElementPtr.
    if (Idx == 0)
      return 2 * TTI::TCC_Basic;
    return TTI::TCC_Free;
  case Instruction::Store:
    ImmIdx = 0;
    break;
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::ICmp:
    ImmIdx = 1;
    break;
  // Always return TCC_Free for the shift value of a shift instruction.
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    if (Idx == 1)
      return TTI::TCC_Free;
    break;
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast:
  case Instruction::PHI:
  case Instruction::Call:
  case Instruction::Select:
  case Instruction::Ret:
  case Instruction::Load:
    break;
  }

  if (Idx == ImmIdx) {
    unsigned NumConstants = (BitSize + 63) / 64;
    unsigned Cost = AArch64TTIImpl::getIntImmCost(Imm, Ty);
    return (Cost <= NumConstants * TTI::TCC_Basic)
               ? static_cast<unsigned>(TTI::TCC_Free)
               : Cost;
  }
  return AArch64TTIImpl::getIntImmCost(Imm, Ty);
}

unsigned AArch64TTIImpl::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                       const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  // There is no cost model for constants with a bit size of 0. Return TCC_Free
  // here, so that constant hoisting will ignore this constant.
  if (BitSize == 0)
    return TTI::TCC_Free;

  switch (IID) {
  default:
    return TTI::TCC_Free;
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::umul_with_overflow:
    if (Idx == 1) {
      unsigned NumConstants = (BitSize + 63) / 64;
      unsigned Cost = AArch64TTIImpl::getIntImmCost(Imm, Ty);
      return (Cost <= NumConstants * TTI::TCC_Basic)
                 ? static_cast<unsigned>(TTI::TCC_Free)
                 : Cost;
    }
    break;
  case Intrinsic::experimental_stackmap:
    if ((Idx < 2) || (Imm.getBitWidth() <= 64 && isInt<64>(Imm.getSExtValue())))
      return TTI::TCC_Free;
    break;
  case Intrinsic::experimental_patchpoint_void:
  case Intrinsic::experimental_patchpoint_i64:
    if ((Idx < 4) || (Imm.getBitWidth() <= 64 && isInt<64>(Imm.getSExtValue())))
      return TTI::TCC_Free;
    break;
  }
  return AArch64TTIImpl::getIntImmCost(Imm, Ty);
}

TargetTransformInfo::PopcntSupportKind
AArch64TTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  if (TyWidth == 32 || TyWidth == 64)
    return TTI::PSK_FastHardware;
  // TODO: AArch64TargetLowering::LowerCTPOP() supports 128bit popcount.
  return TTI::PSK_Software;
}

unsigned AArch64TTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst,
                                          Type *Src) {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  EVT SrcTy = TLI->getValueType(Src);
  EVT DstTy = TLI->getValueType(Dst);

  if (!SrcTy.isSimple() || !DstTy.isSimple())
    return BaseT::getCastInstrCost(Opcode, Dst, Src);

  static const TypeConversionCostTblEntry<MVT> ConversionTbl[] = {
    // LowerVectorINT_TO_FP:
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i32, 1 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i32, 1 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i64, 1 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i32, 1 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i32, 1 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i64, 1 },

    // Complex: to v2f32
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i8,  3 },
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i16, 3 },
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i64, 2 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i8,  3 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i16, 3 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i64, 2 },

    // Complex: to v4f32
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i8,  4 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i16, 2 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i8,  3 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i16, 2 },

    // Complex: to v2f64
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i8,  4 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i16, 4 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i32, 2 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i8,  4 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i16, 4 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i32, 2 },


    // LowerVectorFP_TO_INT
    { ISD::FP_TO_SINT, MVT::v2i32, MVT::v2f32, 1 },
    { ISD::FP_TO_SINT, MVT::v4i32, MVT::v4f32, 1 },
    { ISD::FP_TO_SINT, MVT::v2i64, MVT::v2f64, 1 },
    { ISD::FP_TO_UINT, MVT::v2i32, MVT::v2f32, 1 },
    { ISD::FP_TO_UINT, MVT::v4i32, MVT::v4f32, 1 },
    { ISD::FP_TO_UINT, MVT::v2i64, MVT::v2f64, 1 },

    // Complex, from v2f32: legal type is v2i32 (no cost) or v2i64 (1 ext).
    { ISD::FP_TO_SINT, MVT::v2i64, MVT::v2f32, 2 },
    { ISD::FP_TO_SINT, MVT::v2i16, MVT::v2f32, 1 },
    { ISD::FP_TO_SINT, MVT::v2i8,  MVT::v2f32, 1 },
    { ISD::FP_TO_UINT, MVT::v2i64, MVT::v2f32, 2 },
    { ISD::FP_TO_UINT, MVT::v2i16, MVT::v2f32, 1 },
    { ISD::FP_TO_UINT, MVT::v2i8,  MVT::v2f32, 1 },

    // Complex, from v4f32: legal type is v4i16, 1 narrowing => ~2
    { ISD::FP_TO_SINT, MVT::v4i16, MVT::v4f32, 2 },
    { ISD::FP_TO_SINT, MVT::v4i8,  MVT::v4f32, 2 },
    { ISD::FP_TO_UINT, MVT::v4i16, MVT::v4f32, 2 },
    { ISD::FP_TO_UINT, MVT::v4i8,  MVT::v4f32, 2 },

    // Complex, from v2f64: legal type is v2i32, 1 narrowing => ~2.
    { ISD::FP_TO_SINT, MVT::v2i32, MVT::v2f64, 2 },
    { ISD::FP_TO_SINT, MVT::v2i16, MVT::v2f64, 2 },
    { ISD::FP_TO_SINT, MVT::v2i8,  MVT::v2f64, 2 },
    { ISD::FP_TO_UINT, MVT::v2i32, MVT::v2f64, 2 },
    { ISD::FP_TO_UINT, MVT::v2i16, MVT::v2f64, 2 },
    { ISD::FP_TO_UINT, MVT::v2i8,  MVT::v2f64, 2 },
  };

  int Idx = ConvertCostTableLookup<MVT>(
      ConversionTbl, array_lengthof(ConversionTbl), ISD, DstTy.getSimpleVT(),
      SrcTy.getSimpleVT());
  if (Idx != -1)
    return ConversionTbl[Idx].Cost;

  return BaseT::getCastInstrCost(Opcode, Dst, Src);
}

unsigned AArch64TTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                            unsigned Index) {
  assert(Val->isVectorTy() && "This must be a vector type");

  if (Index != -1U) {
    // Legalize the type.
    std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Val);

    // This type is legalized to a scalar type.
    if (!LT.second.isVector())
      return 0;

    // The type may be split. Normalize the index to the new type.
    unsigned Width = LT.second.getVectorNumElements();
    Index = Index % Width;

    // The element at index zero is already inside the vector.
    if (Index == 0)
      return 0;
  }

  // All other insert/extracts cost this much.
  return 2;
}

unsigned AArch64TTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::OperandValueKind Opd1Info,
    TTI::OperandValueKind Opd2Info, TTI::OperandValueProperties Opd1PropInfo,
    TTI::OperandValueProperties Opd2PropInfo) {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Ty);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  if (ISD == ISD::SDIV &&
      Opd2Info == TargetTransformInfo::OK_UniformConstantValue &&
      Opd2PropInfo == TargetTransformInfo::OP_PowerOf2) {
    // On AArch64, scalar signed division by constants power-of-two are
    // normally expanded to the sequence ADD + CMP + SELECT + SRA.
    // The OperandValue properties many not be same as that of previous
    // operation; conservatively assume OP_None.
    unsigned Cost =
      getArithmeticInstrCost(Instruction::Add, Ty, Opd1Info, Opd2Info,
                             TargetTransformInfo::OP_None,
                             TargetTransformInfo::OP_None);
    Cost += getArithmeticInstrCost(Instruction::Sub, Ty, Opd1Info, Opd2Info,
                                   TargetTransformInfo::OP_None,
                                   TargetTransformInfo::OP_None);
    Cost += getArithmeticInstrCost(Instruction::Select, Ty, Opd1Info, Opd2Info,
                                   TargetTransformInfo::OP_None,
                                   TargetTransformInfo::OP_None);
    Cost += getArithmeticInstrCost(Instruction::AShr, Ty, Opd1Info, Opd2Info,
                                   TargetTransformInfo::OP_None,
                                   TargetTransformInfo::OP_None);
    return Cost;
  }

  switch (ISD) {
  default:
    return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                         Opd1PropInfo, Opd2PropInfo);
  case ISD::ADD:
  case ISD::MUL:
  case ISD::XOR:
  case ISD::OR:
  case ISD::AND:
    // These nodes are marked as 'custom' for combining purposes only.
    // We know that they are legal. See LowerAdd in ISelLowering.
    return 1 * LT.first;
  }
}

unsigned AArch64TTIImpl::getAddressComputationCost(Type *Ty, bool IsComplex) {
  // Address computations in vectorized code with non-consecutive addresses will
  // likely result in more instructions compared to scalar code where the
  // computation can more often be merged into the index mode. The resulting
  // extra micro-ops can significantly decrease throughput.
  unsigned NumVectorInstToHideOverhead = 10;

  if (Ty->isVectorTy() && IsComplex)
    return NumVectorInstToHideOverhead;

  // In many cases the address computation is not merged into the instruction
  // addressing mode.
  return 1;
}

unsigned AArch64TTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                            Type *CondTy) {

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  // We don't lower vector selects well that are wider than the register width.
  if (ValTy->isVectorTy() && ISD == ISD::SELECT) {
    // We would need this many instructions to hide the scalarization happening.
    unsigned AmortizationCost = 20;
    static const TypeConversionCostTblEntry<MVT::SimpleValueType>
    VectorSelectTbl[] = {
      { ISD::SELECT, MVT::v16i1, MVT::v16i16, 16 * AmortizationCost },
      { ISD::SELECT, MVT::v8i1, MVT::v8i32, 8 * AmortizationCost },
      { ISD::SELECT, MVT::v16i1, MVT::v16i32, 16 * AmortizationCost },
      { ISD::SELECT, MVT::v4i1, MVT::v4i64, 4 * AmortizationCost },
      { ISD::SELECT, MVT::v8i1, MVT::v8i64, 8 * AmortizationCost },
      { ISD::SELECT, MVT::v16i1, MVT::v16i64, 16 * AmortizationCost }
    };

    EVT SelCondTy = TLI->getValueType(CondTy);
    EVT SelValTy = TLI->getValueType(ValTy);
    if (SelCondTy.isSimple() && SelValTy.isSimple()) {
      int Idx =
          ConvertCostTableLookup(VectorSelectTbl, ISD, SelCondTy.getSimpleVT(),
                                 SelValTy.getSimpleVT());
      if (Idx != -1)
        return VectorSelectTbl[Idx].Cost;
    }
  }
  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned AArch64TTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                         unsigned Alignment,
                                         unsigned AddressSpace) {
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Src);

  if (Opcode == Instruction::Store && Src->isVectorTy() && Alignment != 16 &&
      Src->getVectorElementType()->isIntegerTy(64)) {
    // Unaligned stores are extremely inefficient. We don't split
    // unaligned v2i64 stores because the negative impact that has shown in
    // practice on inlined memcpy code.
    // We make v2i64 stores expensive so that we will only vectorize if there
    // are 6 other instructions getting vectorized.
    unsigned AmortizationCost = 6;

    return LT.first * 2 * AmortizationCost;
  }

  if (Src->isVectorTy() && Src->getVectorElementType()->isIntegerTy(8) &&
      Src->getVectorNumElements() < 8) {
    // We scalarize the loads/stores because there is not v.4b register and we
    // have to promote the elements to v.4h.
    unsigned NumVecElts = Src->getVectorNumElements();
    unsigned NumVectorizableInstsToAmortize = NumVecElts * 2;
    // We generate 2 instructions per vector element.
    return NumVectorizableInstsToAmortize * NumVecElts * 2;
  }

  return LT.first;
}

unsigned AArch64TTIImpl::getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) {
  unsigned Cost = 0;
  for (auto *I : Tys) {
    if (!I->isVectorTy())
      continue;
    if (I->getScalarSizeInBits() * I->getVectorNumElements() == 128)
      Cost += getMemoryOpCost(Instruction::Store, I, 128, 0) +
        getMemoryOpCost(Instruction::Load, I, 128, 0);
  }
  return Cost;
}

unsigned AArch64TTIImpl::getMaxInterleaveFactor() {
  if (ST->isCortexA57())
    return 4;
  return 2;
}

void AArch64TTIImpl::getUnrollingPreferences(Loop *L,
                                             TTI::UnrollingPreferences &UP) {
  // Disable partial & runtime unrolling on -Os.
  UP.PartialOptSizeThreshold = 0;
}

Value *AArch64TTIImpl::getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                                         Type *ExpectedType) {
  switch (Inst->getIntrinsicID()) {
  default:
    return nullptr;
  case Intrinsic::aarch64_neon_st2:
  case Intrinsic::aarch64_neon_st3:
  case Intrinsic::aarch64_neon_st4: {
    // Create a struct type
    StructType *ST = dyn_cast<StructType>(ExpectedType);
    if (!ST)
      return nullptr;
    unsigned NumElts = Inst->getNumArgOperands() - 1;
    if (ST->getNumElements() != NumElts)
      return nullptr;
    for (unsigned i = 0, e = NumElts; i != e; ++i) {
      if (Inst->getArgOperand(i)->getType() != ST->getElementType(i))
        return nullptr;
    }
    Value *Res = UndefValue::get(ExpectedType);
    IRBuilder<> Builder(Inst);
    for (unsigned i = 0, e = NumElts; i != e; ++i) {
      Value *L = Inst->getArgOperand(i);
      Res = Builder.CreateInsertValue(Res, L, i);
    }
    return Res;
  }
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld4:
    if (Inst->getType() == ExpectedType)
      return Inst;
    return nullptr;
  }
}

bool AArch64TTIImpl::getTgtMemIntrinsic(IntrinsicInst *Inst,
                                        MemIntrinsicInfo &Info) {
  switch (Inst->getIntrinsicID()) {
  default:
    break;
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld4:
    Info.ReadMem = true;
    Info.WriteMem = false;
    Info.Vol = false;
    Info.NumMemRefs = 1;
    Info.PtrVal = Inst->getArgOperand(0);
    break;
  case Intrinsic::aarch64_neon_st2:
  case Intrinsic::aarch64_neon_st3:
  case Intrinsic::aarch64_neon_st4:
    Info.ReadMem = false;
    Info.WriteMem = true;
    Info.Vol = false;
    Info.NumMemRefs = 1;
    Info.PtrVal = Inst->getArgOperand(Inst->getNumArgOperands() - 1);
    break;
  }

  switch (Inst->getIntrinsicID()) {
  default:
    return false;
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_st2:
    Info.MatchingId = VECTOR_LDST_TWO_ELEMENTS;
    break;
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_st3:
    Info.MatchingId = VECTOR_LDST_THREE_ELEMENTS;
    break;
  case Intrinsic::aarch64_neon_ld4:
  case Intrinsic::aarch64_neon_st4:
    Info.MatchingId = VECTOR_LDST_FOUR_ELEMENTS;
    break;
  }
  return true;
}
