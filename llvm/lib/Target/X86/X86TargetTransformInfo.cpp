//===-- X86TargetTransformInfo.cpp - X86 specific TTI pass ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// X86 target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86tti"
#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/CostTable.h"
using namespace llvm;

// Declare the pass initialization routine locally as target-specific passes
// don't havve a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeX86TTIPass(PassRegistry &);
}

namespace {

class X86TTI : public ImmutablePass, public TargetTransformInfo {
  const X86TargetMachine *TM;
  const X86Subtarget *ST;
  const X86TargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

public:
  X86TTI() : ImmutablePass(ID), TM(0), ST(0), TLI(0) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  X86TTI(const X86TargetMachine *TM)
      : ImmutablePass(ID), TM(TM), ST(TM->getSubtargetImpl()),
        TLI(TM->getTargetLowering()) {
    initializeX86TTIPass(*PassRegistry::getPassRegistry());
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
  virtual PopcntSupportKind getPopcntSupport(unsigned TyWidth) const;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  virtual unsigned getNumberOfRegisters(bool Vector) const;
  virtual unsigned getRegisterBitWidth(bool Vector) const;
  virtual unsigned getMaximumUnrollFactor() const;
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const;
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                                  int Index, Type *SubTp) const;
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const;
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) const;
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) const;
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const;

  /// @}
};

} // end anonymous namespace

INITIALIZE_AG_PASS(X86TTI, TargetTransformInfo, "x86tti",
                   "X86 Target Transform Info", true, true, false)
char X86TTI::ID = 0;

ImmutablePass *
llvm::createX86TargetTransformInfoPass(const X86TargetMachine *TM) {
  return new X86TTI(TM);
}


//===----------------------------------------------------------------------===//
//
// X86 cost model.
//
//===----------------------------------------------------------------------===//

X86TTI::PopcntSupportKind X86TTI::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  // TODO: Currently the __builtin_popcount() implementation using SSE3
  //   instructions is inefficient. Once the problem is fixed, we should
  //   call ST->hasSSE3() instead of ST->hasSSE4().
  return ST->hasSSE41() ? PSK_FastHardware : PSK_Software;
}

unsigned X86TTI::getNumberOfRegisters(bool Vector) const {
  if (Vector && !ST->hasSSE1())
    return 0;

  if (ST->is64Bit())
    return 16;
  return 8;
}

unsigned X86TTI::getRegisterBitWidth(bool Vector) const {
  if (Vector) {
    if (ST->hasAVX()) return 256;
    if (ST->hasSSE1()) return 128;
    return 0;
  }

  if (ST->is64Bit())
    return 64;
  return 32;

}

unsigned X86TTI::getMaximumUnrollFactor() const {
  if (ST->isAtom())
    return 1;

  // Sandybridge and Haswell have multiple execution ports and pipelined
  // vector units.
  if (ST->hasAVX())
    return 4;

  return 2;
}

unsigned X86TTI::getArithmeticInstrCost(unsigned Opcode, Type *Ty) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Ty);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  static const CostTblEntry<MVT> AVX2CostTable[] = {
    // Shifts on v4i64/v8i32 on AVX2 is legal even though we declare to
    // customize them to detect the cases where shift amount is a scalar one.
    { ISD::SHL,     MVT::v4i32,    1 },
    { ISD::SRL,     MVT::v4i32,    1 },
    { ISD::SRA,     MVT::v4i32,    1 },
    { ISD::SHL,     MVT::v8i32,    1 },
    { ISD::SRL,     MVT::v8i32,    1 },
    { ISD::SRA,     MVT::v8i32,    1 },
    { ISD::SHL,     MVT::v2i64,    1 },
    { ISD::SRL,     MVT::v2i64,    1 },
    { ISD::SHL,     MVT::v4i64,    1 },
    { ISD::SRL,     MVT::v4i64,    1 },
  };

  // Look for AVX2 lowering tricks.
  if (ST->hasAVX2()) {
    int Idx = CostTableLookup<MVT>(AVX2CostTable, array_lengthof(AVX2CostTable),
                                   ISD, LT.second);
    if (Idx != -1)
      return LT.first * AVX2CostTable[Idx].Cost;
  }

  static const CostTblEntry<MVT> AVX1CostTable[] = {
    // We don't have to scalarize unsupported ops. We can issue two half-sized
    // operations and we only need to extract the upper YMM half.
    // Two ops + 1 extract + 1 insert = 4.
    { ISD::MUL,     MVT::v8i32,    4 },
    { ISD::SUB,     MVT::v8i32,    4 },
    { ISD::ADD,     MVT::v8i32,    4 },
    { ISD::SUB,     MVT::v4i64,    4 },
    { ISD::ADD,     MVT::v4i64,    4 },
    // A v4i64 multiply is custom lowered as two split v2i64 vectors that then
    // are lowered as a series of long multiplies(3), shifts(4) and adds(2)
    // Because we believe v4i64 to be a legal type, we must also include the
    // split factor of two in the cost table. Therefore, the cost here is 18
    // instead of 9.
    { ISD::MUL,     MVT::v4i64,    18 },
  };

  // Look for AVX1 lowering tricks.
  if (ST->hasAVX() && !ST->hasAVX2()) {
    int Idx = CostTableLookup<MVT>(AVX1CostTable, array_lengthof(AVX1CostTable),
                                   ISD, LT.second);
    if (Idx != -1)
      return LT.first * AVX1CostTable[Idx].Cost;
  }

  // Custom lowering of vectors.
  static const CostTblEntry<MVT> CustomLowered[] = {
    // A v2i64/v4i64 and multiply is custom lowered as a series of long
    // multiplies(3), shifts(4) and adds(2).
    { ISD::MUL,     MVT::v2i64,    9 },
    { ISD::MUL,     MVT::v4i64,    9 },
  };
  int Idx = CostTableLookup<MVT>(CustomLowered, array_lengthof(CustomLowered),
                                 ISD, LT.second);
  if (Idx != -1)
    return LT.first * CustomLowered[Idx].Cost;

  // Special lowering of v4i32 mul on sse2, sse3: Lower v4i32 mul as 2x shuffle,
  // 2x pmuludq, 2x shuffle.
  if (ISD == ISD::MUL && LT.second == MVT::v4i32 && ST->hasSSE2() &&
      !ST->hasSSE41())
    return 6;

  // Fallback to the default implementation.
  return TargetTransformInfo::getArithmeticInstrCost(Opcode, Ty);
}

unsigned X86TTI::getShuffleCost(ShuffleKind Kind, Type *Tp, int Index,
                                Type *SubTp) const {
  // We only estimate the cost of reverse shuffles.
  if (Kind != SK_Reverse)
    return TargetTransformInfo::getShuffleCost(Kind, Tp, Index, SubTp);

  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Tp);
  unsigned Cost = 1;
  if (LT.second.getSizeInBits() > 128)
    Cost = 3; // Extract + insert + copy.

  // Multiple by the number of parts.
  return Cost * LT.first;
}

unsigned X86TTI::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) const {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  EVT SrcTy = TLI->getValueType(Src);
  EVT DstTy = TLI->getValueType(Dst);

  if (!SrcTy.isSimple() || !DstTy.isSimple())
    return TargetTransformInfo::getCastInstrCost(Opcode, Dst, Src);

  static const TypeConversionCostTblEntry<MVT> AVXConversionTbl[] = {
    { ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i16, 1 },
    { ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i16, 1 },
    { ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i32, 1 },
    { ISD::ZERO_EXTEND, MVT::v4i64, MVT::v4i32, 1 },
    { ISD::TRUNCATE,    MVT::v4i32, MVT::v4i64, 1 },
    { ISD::TRUNCATE,    MVT::v8i16, MVT::v8i32, 1 },

    { ISD::SINT_TO_FP,  MVT::v8f32, MVT::v8i1,  8 },
    { ISD::SINT_TO_FP,  MVT::v8f32, MVT::v8i8,  8 },
    { ISD::SINT_TO_FP,  MVT::v8f32, MVT::v8i16, 5 },
    { ISD::SINT_TO_FP,  MVT::v8f32, MVT::v8i32, 1 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i1,  3 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i8,  3 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i16, 3 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i32, 1 },
    { ISD::SINT_TO_FP,  MVT::v4f64, MVT::v4i1,  3 },
    { ISD::SINT_TO_FP,  MVT::v4f64, MVT::v4i8,  3 },
    { ISD::SINT_TO_FP,  MVT::v4f64, MVT::v4i16, 3 },
    { ISD::SINT_TO_FP,  MVT::v4f64, MVT::v4i32, 1 },

    { ISD::UINT_TO_FP,  MVT::v8f32, MVT::v8i1,  6 },
    { ISD::UINT_TO_FP,  MVT::v8f32, MVT::v8i8,  5 },
    { ISD::UINT_TO_FP,  MVT::v8f32, MVT::v8i16, 5 },
    { ISD::UINT_TO_FP,  MVT::v8f32, MVT::v8i32, 9 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i1,  7 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i8,  2 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i16, 2 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i32, 6 },
    { ISD::UINT_TO_FP,  MVT::v4f64, MVT::v4i1,  7 },
    { ISD::UINT_TO_FP,  MVT::v4f64, MVT::v4i8,  2 },
    { ISD::UINT_TO_FP,  MVT::v4f64, MVT::v4i16, 2 },
    { ISD::UINT_TO_FP,  MVT::v4f64, MVT::v4i32, 6 },

    { ISD::FP_TO_SINT,  MVT::v8i8,  MVT::v8f32, 1 },
    { ISD::FP_TO_SINT,  MVT::v4i8,  MVT::v4f32, 1 },
    { ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i1,  6 },
    { ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i1,  9 },
    { ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i1,  8 },
    { ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i8,  6 },
    { ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i16, 6 },
    { ISD::TRUNCATE,    MVT::v8i32, MVT::v8i64, 3 },
  };

  if (ST->hasAVX()) {
    int Idx = ConvertCostTableLookup<MVT>(AVXConversionTbl,
                                 array_lengthof(AVXConversionTbl),
                                 ISD, DstTy.getSimpleVT(), SrcTy.getSimpleVT());
    if (Idx != -1)
      return AVXConversionTbl[Idx].Cost;
  }

  return TargetTransformInfo::getCastInstrCost(Opcode, Dst, Src);
}

unsigned X86TTI::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                    Type *CondTy) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(ValTy);

  MVT MTy = LT.second;

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  static const CostTblEntry<MVT> SSE42CostTbl[] = {
    { ISD::SETCC,   MVT::v2f64,   1 },
    { ISD::SETCC,   MVT::v4f32,   1 },
    { ISD::SETCC,   MVT::v2i64,   1 },
    { ISD::SETCC,   MVT::v4i32,   1 },
    { ISD::SETCC,   MVT::v8i16,   1 },
    { ISD::SETCC,   MVT::v16i8,   1 },
  };

  static const CostTblEntry<MVT> AVX1CostTbl[] = {
    { ISD::SETCC,   MVT::v4f64,   1 },
    { ISD::SETCC,   MVT::v8f32,   1 },
    // AVX1 does not support 8-wide integer compare.
    { ISD::SETCC,   MVT::v4i64,   4 },
    { ISD::SETCC,   MVT::v8i32,   4 },
    { ISD::SETCC,   MVT::v16i16,  4 },
    { ISD::SETCC,   MVT::v32i8,   4 },
  };

  static const CostTblEntry<MVT> AVX2CostTbl[] = {
    { ISD::SETCC,   MVT::v4i64,   1 },
    { ISD::SETCC,   MVT::v8i32,   1 },
    { ISD::SETCC,   MVT::v16i16,  1 },
    { ISD::SETCC,   MVT::v32i8,   1 },
  };

  if (ST->hasAVX2()) {
    int Idx = CostTableLookup<MVT>(AVX2CostTbl, array_lengthof(AVX2CostTbl), ISD, MTy);
    if (Idx != -1)
      return LT.first * AVX2CostTbl[Idx].Cost;
  }

  if (ST->hasAVX()) {
    int Idx = CostTableLookup<MVT>(AVX1CostTbl, array_lengthof(AVX1CostTbl), ISD, MTy);
    if (Idx != -1)
      return LT.first * AVX1CostTbl[Idx].Cost;
  }

  if (ST->hasSSE42()) {
    int Idx = CostTableLookup<MVT>(SSE42CostTbl, array_lengthof(SSE42CostTbl), ISD, MTy);
    if (Idx != -1)
      return LT.first * SSE42CostTbl[Idx].Cost;
  }

  return TargetTransformInfo::getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned X86TTI::getVectorInstrCost(unsigned Opcode, Type *Val,
                                    unsigned Index) const {
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

    // Floating point scalars are already located in index #0.
    if (Val->getScalarType()->isFloatingPointTy() && Index == 0)
      return 0;
  }

  return TargetTransformInfo::getVectorInstrCost(Opcode, Val, Index);
}

unsigned X86TTI::getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Src);
  assert((Opcode == Instruction::Load || Opcode == Instruction::Store) &&
         "Invalid Opcode");

  // Each load/store unit costs 1.
  unsigned Cost = LT.first * 1;

  // On Sandybridge 256bit load/stores are double pumped
  // (but not on Haswell).
  if (LT.second.getSizeInBits() > 128 && !ST->hasAVX2())
    Cost*=2;

  return Cost;
}
