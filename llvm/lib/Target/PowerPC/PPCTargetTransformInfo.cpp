//===-- PPCTargetTransformInfo.cpp - PPC specific TTI ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "ppctti"

static cl::opt<bool> DisablePPCConstHoist("disable-ppc-constant-hoisting",
cl::desc("disable constant hoisting on PPC"), cl::init(false), cl::Hidden);

//===----------------------------------------------------------------------===//
//
// PPC cost model.
//
//===----------------------------------------------------------------------===//

TargetTransformInfo::PopcntSupportKind
PPCTTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  if (ST->hasPOPCNTD() && TyWidth <= 64)
    return TTI::PSK_FastHardware;
  return TTI::PSK_Software;
}

unsigned PPCTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty) {
  if (DisablePPCConstHoist)
    return BaseT::getIntImmCost(Imm, Ty);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  if (Imm == 0)
    return TTI::TCC_Free;

  if (Imm.getBitWidth() <= 64) {
    if (isInt<16>(Imm.getSExtValue()))
      return TTI::TCC_Basic;

    if (isInt<32>(Imm.getSExtValue())) {
      // A constant that can be materialized using lis.
      if ((Imm.getZExtValue() & 0xFFFF) == 0)
        return TTI::TCC_Basic;

      return 2 * TTI::TCC_Basic;
    }
  }

  return 4 * TTI::TCC_Basic;
}

unsigned PPCTTIImpl::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                   const APInt &Imm, Type *Ty) {
  if (DisablePPCConstHoist)
    return BaseT::getIntImmCost(IID, Idx, Imm, Ty);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  switch (IID) {
  default:
    return TTI::TCC_Free;
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
    if ((Idx == 1) && Imm.getBitWidth() <= 64 && isInt<16>(Imm.getSExtValue()))
      return TTI::TCC_Free;
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
  return PPCTTIImpl::getIntImmCost(Imm, Ty);
}

unsigned PPCTTIImpl::getIntImmCost(unsigned Opcode, unsigned Idx,
                                   const APInt &Imm, Type *Ty) {
  if (DisablePPCConstHoist)
    return BaseT::getIntImmCost(Opcode, Idx, Imm, Ty);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  unsigned ImmIdx = ~0U;
  bool ShiftedFree = false, RunFree = false, UnsignedFree = false,
       ZeroFree = false;
  switch (Opcode) {
  default:
    return TTI::TCC_Free;
  case Instruction::GetElementPtr:
    // Always hoist the base address of a GetElementPtr. This prevents the
    // creation of new constants for every base constant that gets constant
    // folded with the offset.
    if (Idx == 0)
      return 2 * TTI::TCC_Basic;
    return TTI::TCC_Free;
  case Instruction::And:
    RunFree = true; // (for the rotate-and-mask instructions)
    // Fallthrough...
  case Instruction::Add:
  case Instruction::Or:
  case Instruction::Xor:
    ShiftedFree = true;
    // Fallthrough...
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    ImmIdx = 1;
    break;
  case Instruction::ICmp:
    UnsignedFree = true;
    ImmIdx = 1;
    // Fallthrough... (zero comparisons can use record-form instructions)
  case Instruction::Select:
    ZeroFree = true;
    break;
  case Instruction::PHI:
  case Instruction::Call:
  case Instruction::Ret:
  case Instruction::Load:
  case Instruction::Store:
    break;
  }

  if (ZeroFree && Imm == 0)
    return TTI::TCC_Free;

  if (Idx == ImmIdx && Imm.getBitWidth() <= 64) {
    if (isInt<16>(Imm.getSExtValue()))
      return TTI::TCC_Free;

    if (RunFree) {
      if (Imm.getBitWidth() <= 32 &&
          (isShiftedMask_32(Imm.getZExtValue()) ||
           isShiftedMask_32(~Imm.getZExtValue())))
        return TTI::TCC_Free;

      if (ST->isPPC64() &&
          (isShiftedMask_64(Imm.getZExtValue()) ||
           isShiftedMask_64(~Imm.getZExtValue())))
        return TTI::TCC_Free;
    }

    if (UnsignedFree && isUInt<16>(Imm.getZExtValue()))
      return TTI::TCC_Free;

    if (ShiftedFree && (Imm.getZExtValue() & 0xFFFF) == 0)
      return TTI::TCC_Free;
  }

  return PPCTTIImpl::getIntImmCost(Imm, Ty);
}

void PPCTTIImpl::getUnrollingPreferences(Loop *L,
                                         TTI::UnrollingPreferences &UP) {
  if (ST->getDarwinDirective() == PPC::DIR_A2) {
    // The A2 is in-order with a deep pipeline, and concatenation unrolling
    // helps expose latency-hiding opportunities to the instruction scheduler.
    UP.Partial = UP.Runtime = true;
  }

  BaseT::getUnrollingPreferences(L, UP);
}

bool PPCTTIImpl::enableAggressiveInterleaving(bool LoopHasReductions) {
  return LoopHasReductions;
}

unsigned PPCTTIImpl::getNumberOfRegisters(bool Vector) {
  if (Vector && !ST->hasAltivec() && !ST->hasQPX())
    return 0;
  return ST->hasVSX() ? 64 : 32;
}

unsigned PPCTTIImpl::getRegisterBitWidth(bool Vector) {
  if (Vector) {
    if (ST->hasQPX()) return 256;
    if (ST->hasAltivec()) return 128;
    return 0;
  }

  if (ST->isPPC64())
    return 64;
  return 32;

}

unsigned PPCTTIImpl::getMaxInterleaveFactor(unsigned VF) {
  unsigned Directive = ST->getDarwinDirective();
  // The 440 has no SIMD support, but floating-point instructions
  // have a 5-cycle latency, so unroll by 5x for latency hiding.
  if (Directive == PPC::DIR_440)
    return 5;

  // The A2 has no SIMD support, but floating-point instructions
  // have a 6-cycle latency, so unroll by 6x for latency hiding.
  if (Directive == PPC::DIR_A2)
    return 6;

  // FIXME: For lack of any better information, do no harm...
  if (Directive == PPC::DIR_E500mc || Directive == PPC::DIR_E5500)
    return 1;

  // For P7 and P8, floating-point instructions have a 6-cycle latency and
  // there are two execution units, so unroll by 12x for latency hiding.
  if (Directive == PPC::DIR_PWR7 ||
      Directive == PPC::DIR_PWR8)
    return 12;

  // For most things, modern systems have two execution units (and
  // out-of-order execution).
  return 2;
}

unsigned PPCTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::OperandValueKind Op1Info,
    TTI::OperandValueKind Op2Info, TTI::OperandValueProperties Opd1PropInfo,
    TTI::OperandValueProperties Opd2PropInfo) {
  assert(TLI->InstructionOpcodeToISD(Opcode) && "Invalid opcode");

  // Fallback to the default implementation.
  return BaseT::getArithmeticInstrCost(Opcode, Ty, Op1Info, Op2Info,
                                       Opd1PropInfo, Opd2PropInfo);
}

unsigned PPCTTIImpl::getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                                    Type *SubTp) {
  return BaseT::getShuffleCost(Kind, Tp, Index, SubTp);
}

unsigned PPCTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) {
  assert(TLI->InstructionOpcodeToISD(Opcode) && "Invalid opcode");

  return BaseT::getCastInstrCost(Opcode, Dst, Src);
}

unsigned PPCTTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                        Type *CondTy) {
  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned PPCTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                        unsigned Index) {
  assert(Val->isVectorTy() && "This must be a vector type");

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  if (ST->hasVSX() && Val->getScalarType()->isDoubleTy()) {
    // Double-precision scalars are already located in index #0.
    if (Index == 0)
      return 0;

    return BaseT::getVectorInstrCost(Opcode, Val, Index);
  } else if (ST->hasQPX() && Val->getScalarType()->isFloatingPointTy()) {
    // Floating point scalars are already located in index #0.
    if (Index == 0)
      return 0;

    return BaseT::getVectorInstrCost(Opcode, Val, Index);
  }

  // Estimated cost of a load-hit-store delay.  This was obtained
  // experimentally as a minimum needed to prevent unprofitable
  // vectorization for the paq8p benchmark.  It may need to be
  // raised further if other unprofitable cases remain.
  unsigned LHSPenalty = 2;
  if (ISD == ISD::INSERT_VECTOR_ELT)
    LHSPenalty += 7;

  // Vector element insert/extract with Altivec is very expensive,
  // because they require store and reload with the attendant
  // processor stall for load-hit-store.  Until VSX is available,
  // these need to be estimated as very costly.
  if (ISD == ISD::EXTRACT_VECTOR_ELT ||
      ISD == ISD::INSERT_VECTOR_ELT)
    return LHSPenalty + BaseT::getVectorInstrCost(Opcode, Val, Index);

  return BaseT::getVectorInstrCost(Opcode, Val, Index);
}

unsigned PPCTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                     unsigned Alignment,
                                     unsigned AddressSpace) {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Src);
  assert((Opcode == Instruction::Load || Opcode == Instruction::Store) &&
         "Invalid Opcode");

  unsigned Cost = BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace);

  // VSX loads/stores support unaligned access.
  if (ST->hasVSX()) {
    if (LT.second == MVT::v2f64 || LT.second == MVT::v2i64)
      return Cost;
  }

  bool UnalignedAltivec =
    Src->isVectorTy() &&
    Src->getPrimitiveSizeInBits() >= LT.second.getSizeInBits() &&
    LT.second.getSizeInBits() == 128 &&
    Opcode == Instruction::Load;

  // PPC in general does not support unaligned loads and stores. They'll need
  // to be decomposed based on the alignment factor.
  unsigned SrcBytes = LT.second.getStoreSize();
  if (SrcBytes && Alignment && Alignment < SrcBytes && !UnalignedAltivec) {
    Cost += LT.first*(SrcBytes/Alignment-1);

    // For a vector type, there is also scalarization overhead (only for
    // stores, loads are expanded using the vector-load + permutation sequence,
    // which is much less expensive).
    if (Src->isVectorTy() && Opcode == Instruction::Store)
      for (int i = 0, e = Src->getVectorNumElements(); i < e; ++i)
        Cost += getVectorInstrCost(Instruction::ExtractElement, Src, i);
  }

  return Cost;
}

