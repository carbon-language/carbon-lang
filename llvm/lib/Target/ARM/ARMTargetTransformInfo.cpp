//===-- ARMTargetTransformInfo.cpp - ARM specific TTI pass ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// ARM target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "armtti"
#include "ARM.h"
#include "ARMTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/CostTable.h"
using namespace llvm;

// Declare the pass initialization routine locally as target-specific passes
// don't havve a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeARMTTIPass(PassRegistry &);
}

namespace {

class ARMTTI : public ImmutablePass, public TargetTransformInfo {
  const ARMBaseTargetMachine *TM;
  const ARMSubtarget *ST;
  const ARMTargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

public:
  ARMTTI() : ImmutablePass(ID), TM(0), ST(0), TLI(0) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  ARMTTI(const ARMBaseTargetMachine *TM)
      : ImmutablePass(ID), TM(TM), ST(TM->getSubtargetImpl()),
        TLI(TM->getTargetLowering()) {
    initializeARMTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() {
    pushTTIStack(this);
  }

  virtual void finalizePass() {
    popTTIStack();
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  /// \name Scalar TTI Implementations
  /// @{

  virtual unsigned getIntImmCost(const APInt &Imm, Type *Ty) const;

  /// @}


  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool Vector) const {
    if (Vector) {
      if (ST->hasNEON())
        return 16;
      return 0;
    }

    if (ST->isThumb1Only())
      return 8;
    return 16;
  }

  unsigned getRegisterBitWidth(bool Vector) const {
    if (Vector) {
      if (ST->hasNEON())
        return 128;
      return 0;
    }

    return 32;
  }

  unsigned getMaximumUnrollFactor() const {
    // These are out of order CPUs:
    if (ST->isCortexA15() || ST->isSwift())
      return 2;
    return 1;
  }

  unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                          int Index, Type *SubTp) const;

  unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                      Type *Src) const;

  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy) const;

  unsigned getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index) const;

  unsigned getAddressComputationCost(Type *Val) const;
  /// @}
};

} // end anonymous namespace

INITIALIZE_AG_PASS(ARMTTI, TargetTransformInfo, "armtti",
                   "ARM Target Transform Info", true, true, false)
char ARMTTI::ID = 0;

ImmutablePass *
llvm::createARMTargetTransformInfoPass(const ARMBaseTargetMachine *TM) {
  return new ARMTTI(TM);
}


unsigned ARMTTI::getIntImmCost(const APInt &Imm, Type *Ty) const {
  assert(Ty->isIntegerTy());

  unsigned Bits = Ty->getPrimitiveSizeInBits();
  if (Bits == 0 || Bits > 32)
    return 4;

  int32_t SImmVal = Imm.getSExtValue();
  uint32_t ZImmVal = Imm.getZExtValue();
  if (!ST->isThumb()) {
    if ((SImmVal >= 0 && SImmVal < 65536) ||
        (ARM_AM::getSOImmVal(ZImmVal) != -1) ||
        (ARM_AM::getSOImmVal(~ZImmVal) != -1))
      return 1;
    return ST->hasV6T2Ops() ? 2 : 3;
  } else if (ST->isThumb2()) {
    if ((SImmVal >= 0 && SImmVal < 65536) ||
        (ARM_AM::getT2SOImmVal(ZImmVal) != -1) ||
        (ARM_AM::getT2SOImmVal(~ZImmVal) != -1))
      return 1;
    return ST->hasV6T2Ops() ? 2 : 3;
  } else /*Thumb1*/ {
    if (SImmVal >= 0 && SImmVal < 256)
      return 1;
    if ((~ZImmVal < 256) || ARM_AM::isThumbImmShiftedVal(ZImmVal))
      return 2;
    // Load from constantpool.
    return 3;
  }
  return 2;
}

unsigned ARMTTI::getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  EVT SrcTy = TLI->getValueType(Src);
  EVT DstTy = TLI->getValueType(Dst);

  if (!SrcTy.isSimple() || !DstTy.isSimple())
    return TargetTransformInfo::getCastInstrCost(Opcode, Dst, Src);

  // Some arithmetic, load and store operations have specific instructions
  // to cast up/down their types automatically at no extra cost.
  // TODO: Get these tables to know at least what the related operations are.
  static const TypeConversionCostTblEntry<MVT> NEONVectorConversionTbl[] = {
    { ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i16, 0 },
    { ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i16, 0 },
    { ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i32, 1 },
    { ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i32, 1 },
    { ISD::TRUNCATE,    MVT::v4i32, MVT::v4i64, 0 },
    { ISD::TRUNCATE,    MVT::v4i16, MVT::v4i32, 1 },

    // Operations that we legalize using load/stores to the stack.
    { ISD::SIGN_EXTEND, MVT::v16i32, MVT::v16i8, 16*2 + 4*4 },
    { ISD::ZERO_EXTEND, MVT::v16i32, MVT::v16i8, 16*2 + 4*3 },
    { ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i8, 8*2 + 2*4 },
    { ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i8, 8*2 + 2*3 },
    { ISD::TRUNCATE,    MVT::v16i8, MVT::v16i32, 4*1 + 16*2 + 2*1 },
    { ISD::TRUNCATE,    MVT::v8i8, MVT::v8i32, 2*1 + 8*2 + 1 },

    // Vector float <-> i32 conversions.
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i32, 1 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i32, 1 },
    { ISD::FP_TO_SINT,  MVT::v4i32, MVT::v4f32, 1 },
    { ISD::FP_TO_UINT,  MVT::v4i32, MVT::v4f32, 1 },

    // Vector double <-> i32 conversions.
    { ISD::SINT_TO_FP,  MVT::v2f64, MVT::v2i32, 2 },
    { ISD::UINT_TO_FP,  MVT::v2f64, MVT::v2i32, 2 },
    { ISD::FP_TO_SINT,  MVT::v2i32, MVT::v2f64, 2 },
    { ISD::FP_TO_UINT,  MVT::v2i32, MVT::v2f64, 2 }
  };

  if (SrcTy.isVector() && ST->hasNEON()) {
    int Idx = ConvertCostTableLookup<MVT>(NEONVectorConversionTbl,
                                array_lengthof(NEONVectorConversionTbl),
                                ISD, DstTy.getSimpleVT(), SrcTy.getSimpleVT());
    if (Idx != -1)
      return NEONVectorConversionTbl[Idx].Cost;
  }

  // Scalar float to integer conversions.
  static const TypeConversionCostTblEntry<MVT> NEONFloatConversionTbl[] = {
    { ISD::FP_TO_SINT,  MVT::i1, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i1, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i1, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i1, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i8, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i8, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i8, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i8, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i16, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i16, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i16, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i16, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i32, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i32, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i32, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i32, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i64, MVT::f32, 10 },
    { ISD::FP_TO_UINT,  MVT::i64, MVT::f32, 10 },
    { ISD::FP_TO_SINT,  MVT::i64, MVT::f64, 10 },
    { ISD::FP_TO_UINT,  MVT::i64, MVT::f64, 10 }
  };
  if (SrcTy.isFloatingPoint() && ST->hasNEON()) {
    int Idx = ConvertCostTableLookup<MVT>(NEONFloatConversionTbl,
                                        array_lengthof(NEONFloatConversionTbl),
                                        ISD, DstTy.getSimpleVT(),
                                        SrcTy.getSimpleVT());
    if (Idx != -1)
        return NEONFloatConversionTbl[Idx].Cost;
  }


  // Scalar integer to float conversions.
  static const TypeConversionCostTblEntry<MVT> NEONIntegerConversionTbl[] = {
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i1, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i1, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i1, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i1, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i8, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i8, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i8, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i8, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i16, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i16, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i16, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i16, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i32, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i32, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i32, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i32, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i64, 10 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i64, 10 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i64, 10 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i64, 10 }
  };

  if (SrcTy.isInteger() && ST->hasNEON()) {
    int Idx = ConvertCostTableLookup<MVT>(NEONIntegerConversionTbl,
                                       array_lengthof(NEONIntegerConversionTbl),
                                       ISD, DstTy.getSimpleVT(),
                                       SrcTy.getSimpleVT());
    if (Idx != -1)
      return NEONIntegerConversionTbl[Idx].Cost;
  }

  // Scalar integer conversion costs.
  static const TypeConversionCostTblEntry<MVT> ARMIntegerConversionTbl[] = {
    // i16 -> i64 requires two dependent operations.
    { ISD::SIGN_EXTEND, MVT::i64, MVT::i16, 2 },

    // Truncates on i64 are assumed to be free.
    { ISD::TRUNCATE,    MVT::i32, MVT::i64, 0 },
    { ISD::TRUNCATE,    MVT::i16, MVT::i64, 0 },
    { ISD::TRUNCATE,    MVT::i8,  MVT::i64, 0 },
    { ISD::TRUNCATE,    MVT::i1,  MVT::i64, 0 }
  };

  if (SrcTy.isInteger()) {
    int Idx =
      ConvertCostTableLookup<MVT>(ARMIntegerConversionTbl,
                                  array_lengthof(ARMIntegerConversionTbl),
                                  ISD, DstTy.getSimpleVT(),
                                  SrcTy.getSimpleVT());
    if (Idx != -1)
      return ARMIntegerConversionTbl[Idx].Cost;
  }


  return TargetTransformInfo::getCastInstrCost(Opcode, Dst, Src);
}

unsigned ARMTTI::getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                    unsigned Index) const {
  // Penalize inserting into an D-subregister. We end up with a three times
  // lower estimated throughput on swift.
  if (ST->isSwift() &&
      Opcode == Instruction::InsertElement &&
      ValTy->isVectorTy() &&
      ValTy->getScalarSizeInBits() <= 32)
    return 3;

  return TargetTransformInfo::getVectorInstrCost(Opcode, ValTy, Index);
}

unsigned ARMTTI::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                    Type *CondTy) const {

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  // On NEON a a vector select gets lowered to vbsl.
  if (ST->hasNEON() && ValTy->isVectorTy() && ISD == ISD::SELECT) {
    std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(ValTy);
    return LT.first;
  }

  return TargetTransformInfo::getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned ARMTTI::getAddressComputationCost(Type *Ty) const {
  // In many cases the address computation is not merged into the instruction
  // addressing mode.
  return 1;
}

unsigned ARMTTI::getShuffleCost(ShuffleKind Kind, Type *Tp, int Index,
                                Type *SubTp) const {
  // We only handle costs of reverse shuffles for now.
  if (Kind != SK_Reverse)
    return TargetTransformInfo::getShuffleCost(Kind, Tp, Index, SubTp);

  static const CostTblEntry<MVT> NEONShuffleTbl[] = {
    // Reverse shuffle cost one instruction if we are shuffling within a double
    // word (vrev) or two if we shuffle a quad word (vrev, vext).
    { ISD::VECTOR_SHUFFLE, MVT::v2i32, 1 },
    { ISD::VECTOR_SHUFFLE, MVT::v2f32, 1 },
    { ISD::VECTOR_SHUFFLE, MVT::v2i64, 1 },
    { ISD::VECTOR_SHUFFLE, MVT::v2f64, 1 },

    { ISD::VECTOR_SHUFFLE, MVT::v4i32, 2 },
    { ISD::VECTOR_SHUFFLE, MVT::v4f32, 2 },
    { ISD::VECTOR_SHUFFLE, MVT::v8i16, 2 },
    { ISD::VECTOR_SHUFFLE, MVT::v16i8, 2 }
  };

  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Tp);

  int Idx = CostTableLookup<MVT>(NEONShuffleTbl, array_lengthof(NEONShuffleTbl),
                                 ISD::VECTOR_SHUFFLE, LT.second);
  if (Idx == -1)
    return TargetTransformInfo::getShuffleCost(Kind, Tp, Index, SubTp);

  return LT.first * NEONShuffleTbl[Idx].Cost;
}
