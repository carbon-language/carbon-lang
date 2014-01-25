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
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

// Declare the pass initialization routine locally as target-specific passes
// don't havve a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeX86TTIPass(PassRegistry &);
}

namespace {

class X86TTI LLVM_FINAL : public ImmutablePass, public TargetTransformInfo {
  const X86Subtarget *ST;
  const X86TargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

public:
  X86TTI() : ImmutablePass(ID), ST(0), TLI(0) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  X86TTI(const X86TargetMachine *TM)
    : ImmutablePass(ID), ST(TM->getSubtargetImpl()),
      TLI(TM->getTargetLowering()) {
    initializeX86TTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() LLVM_OVERRIDE {
    pushTTIStack(this);
  }

  virtual void finalizePass() {
    popTTIStack();
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const LLVM_OVERRIDE {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) LLVM_OVERRIDE {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  /// \name Scalar TTI Implementations
  /// @{
  virtual PopcntSupportKind
  getPopcntSupport(unsigned TyWidth) const LLVM_OVERRIDE;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  virtual unsigned getNumberOfRegisters(bool Vector) const LLVM_OVERRIDE;
  virtual unsigned getRegisterBitWidth(bool Vector) const LLVM_OVERRIDE;
  virtual unsigned getMaximumUnrollFactor() const LLVM_OVERRIDE;
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                          OperandValueKind,
                                          OperandValueKind) const LLVM_OVERRIDE;
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                                  int Index, Type *SubTp) const LLVM_OVERRIDE;
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const LLVM_OVERRIDE;
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) const LLVM_OVERRIDE;
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) const LLVM_OVERRIDE;
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const LLVM_OVERRIDE;

  virtual unsigned
  getAddressComputationCost(Type *PtrTy, bool IsComplex) const LLVM_OVERRIDE;
  
  virtual unsigned getReductionCost(unsigned Opcode, Type *Ty,
                                    bool IsPairwiseForm) const LLVM_OVERRIDE;

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
  //   call ST->hasSSE3() instead of ST->hasPOPCNT().
  return ST->hasPOPCNT() ? PSK_FastHardware : PSK_Software;
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

unsigned X86TTI::getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                        OperandValueKind Op1Info,
                                        OperandValueKind Op2Info) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Ty);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  static const CostTblEntry<MVT::SimpleValueType> AVX2CostTable[] = {
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

    { ISD::SHL,  MVT::v32i8,  42 }, // cmpeqb sequence.
    { ISD::SHL,  MVT::v16i16,  16*10 }, // Scalarized.

    { ISD::SRL,  MVT::v32i8,  32*10 }, // Scalarized.
    { ISD::SRL,  MVT::v16i16,  8*10 }, // Scalarized.

    { ISD::SRA,  MVT::v32i8,  32*10 }, // Scalarized.
    { ISD::SRA,  MVT::v16i16,  16*10 }, // Scalarized.
    { ISD::SRA,  MVT::v4i64,  4*10 }, // Scalarized.

    // Vectorizing division is a bad idea. See the SSE2 table for more comments.
    { ISD::SDIV,  MVT::v32i8,  32*20 },
    { ISD::SDIV,  MVT::v16i16, 16*20 },
    { ISD::SDIV,  MVT::v8i32,  8*20 },
    { ISD::SDIV,  MVT::v4i64,  4*20 },
    { ISD::UDIV,  MVT::v32i8,  32*20 },
    { ISD::UDIV,  MVT::v16i16, 16*20 },
    { ISD::UDIV,  MVT::v8i32,  8*20 },
    { ISD::UDIV,  MVT::v4i64,  4*20 },
  };

  // Look for AVX2 lowering tricks.
  if (ST->hasAVX2()) {
    int Idx = CostTableLookup(AVX2CostTable, ISD, LT.second);
    if (Idx != -1)
      return LT.first * AVX2CostTable[Idx].Cost;
  }

  static const CostTblEntry<MVT::SimpleValueType>
  SSE2UniformConstCostTable[] = {
    // We don't correctly identify costs of casts because they are marked as
    // custom.
    // Constant splats are cheaper for the following instructions.
    { ISD::SHL,  MVT::v16i8,  1 }, // psllw.
    { ISD::SHL,  MVT::v8i16,  1 }, // psllw.
    { ISD::SHL,  MVT::v4i32,  1 }, // pslld
    { ISD::SHL,  MVT::v2i64,  1 }, // psllq.

    { ISD::SRL,  MVT::v16i8,  1 }, // psrlw.
    { ISD::SRL,  MVT::v8i16,  1 }, // psrlw.
    { ISD::SRL,  MVT::v4i32,  1 }, // psrld.
    { ISD::SRL,  MVT::v2i64,  1 }, // psrlq.

    { ISD::SRA,  MVT::v16i8,  4 }, // psrlw, pand, pxor, psubb.
    { ISD::SRA,  MVT::v8i16,  1 }, // psraw.
    { ISD::SRA,  MVT::v4i32,  1 }, // psrad.
  };

  if (Op2Info == TargetTransformInfo::OK_UniformConstantValue &&
      ST->hasSSE2()) {
    int Idx = CostTableLookup(SSE2UniformConstCostTable, ISD, LT.second);
    if (Idx != -1)
      return LT.first * SSE2UniformConstCostTable[Idx].Cost;
  }


  static const CostTblEntry<MVT::SimpleValueType> SSE2CostTable[] = {
    // We don't correctly identify costs of casts because they are marked as
    // custom.
    // For some cases, where the shift amount is a scalar we would be able
    // to generate better code. Unfortunately, when this is the case the value
    // (the splat) will get hoisted out of the loop, thereby making it invisible
    // to ISel. The cost model must return worst case assumptions because it is
    // used for vectorization and we don't want to make vectorized code worse
    // than scalar code.
    { ISD::SHL,  MVT::v16i8,  30 }, // cmpeqb sequence.
    { ISD::SHL,  MVT::v8i16,  8*10 }, // Scalarized.
    { ISD::SHL,  MVT::v4i32,  2*5 }, // We optimized this using mul.
    { ISD::SHL,  MVT::v2i64,  2*10 }, // Scalarized.

    { ISD::SRL,  MVT::v16i8,  16*10 }, // Scalarized.
    { ISD::SRL,  MVT::v8i16,  8*10 }, // Scalarized.
    { ISD::SRL,  MVT::v4i32,  4*10 }, // Scalarized.
    { ISD::SRL,  MVT::v2i64,  2*10 }, // Scalarized.

    { ISD::SRA,  MVT::v16i8,  16*10 }, // Scalarized.
    { ISD::SRA,  MVT::v8i16,  8*10 }, // Scalarized.
    { ISD::SRA,  MVT::v4i32,  4*10 }, // Scalarized.
    { ISD::SRA,  MVT::v2i64,  2*10 }, // Scalarized.

    // It is not a good idea to vectorize division. We have to scalarize it and
    // in the process we will often end up having to spilling regular
    // registers. The overhead of division is going to dominate most kernels
    // anyways so try hard to prevent vectorization of division - it is
    // generally a bad idea. Assume somewhat arbitrarily that we have to be able
    // to hide "20 cycles" for each lane.
    { ISD::SDIV,  MVT::v16i8,  16*20 },
    { ISD::SDIV,  MVT::v8i16,  8*20 },
    { ISD::SDIV,  MVT::v4i32,  4*20 },
    { ISD::SDIV,  MVT::v2i64,  2*20 },
    { ISD::UDIV,  MVT::v16i8,  16*20 },
    { ISD::UDIV,  MVT::v8i16,  8*20 },
    { ISD::UDIV,  MVT::v4i32,  4*20 },
    { ISD::UDIV,  MVT::v2i64,  2*20 },
  };

  if (ST->hasSSE2()) {
    int Idx = CostTableLookup(SSE2CostTable, ISD, LT.second);
    if (Idx != -1)
      return LT.first * SSE2CostTable[Idx].Cost;
  }

  static const CostTblEntry<MVT::SimpleValueType> AVX1CostTable[] = {
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
    int Idx = CostTableLookup(AVX1CostTable, ISD, LT.second);
    if (Idx != -1)
      return LT.first * AVX1CostTable[Idx].Cost;
  }

  // Custom lowering of vectors.
  static const CostTblEntry<MVT::SimpleValueType> CustomLowered[] = {
    // A v2i64/v4i64 and multiply is custom lowered as a series of long
    // multiplies(3), shifts(4) and adds(2).
    { ISD::MUL,     MVT::v2i64,    9 },
    { ISD::MUL,     MVT::v4i64,    9 },
  };
  int Idx = CostTableLookup(CustomLowered, ISD, LT.second);
  if (Idx != -1)
    return LT.first * CustomLowered[Idx].Cost;

  // Special lowering of v4i32 mul on sse2, sse3: Lower v4i32 mul as 2x shuffle,
  // 2x pmuludq, 2x shuffle.
  if (ISD == ISD::MUL && LT.second == MVT::v4i32 && ST->hasSSE2() &&
      !ST->hasSSE41())
    return 6;

  // Fallback to the default implementation.
  return TargetTransformInfo::getArithmeticInstrCost(Opcode, Ty, Op1Info,
                                                     Op2Info);
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

  std::pair<unsigned, MVT> LTSrc = TLI->getTypeLegalizationCost(Src);
  std::pair<unsigned, MVT> LTDest = TLI->getTypeLegalizationCost(Dst);

  static const TypeConversionCostTblEntry<MVT::SimpleValueType>
  SSE2ConvTbl[] = {
    // These are somewhat magic numbers justified by looking at the output of
    // Intel's IACA, running some kernels and making sure when we take
    // legalization into account the throughput will be overestimated.
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i64, 2*10 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v4i32, 4*10 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v8i16, 8*10 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v16i8, 16*10 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i64, 2*10 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v4i32, 4*10 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v8i16, 8*10 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v16i8, 16*10 },
    // There are faster sequences for float conversions.
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v2i64, 15 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i32, 15 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v8i16, 15 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v16i8, 8 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v2i64, 15 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i32, 15 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v8i16, 15 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v16i8, 8 },
  };

  if (ST->hasSSE2() && !ST->hasAVX()) {
    int Idx =
        ConvertCostTableLookup(SSE2ConvTbl, ISD, LTDest.second, LTSrc.second);
    if (Idx != -1)
      return LTSrc.first * SSE2ConvTbl[Idx].Cost;
  }

  EVT SrcTy = TLI->getValueType(Src);
  EVT DstTy = TLI->getValueType(Dst);

  // The function getSimpleVT only handles simple value types.
  if (!SrcTy.isSimple() || !DstTy.isSimple())
    return TargetTransformInfo::getCastInstrCost(Opcode, Dst, Src);

  static const TypeConversionCostTblEntry<MVT::SimpleValueType>
  AVXConversionTbl[] = {
    { ISD::SIGN_EXTEND, MVT::v16i16, MVT::v16i8, 1 },
    { ISD::ZERO_EXTEND, MVT::v16i16, MVT::v16i8, 1 },
    { ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i16, 1 },
    { ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i16, 1 },
    { ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i32, 1 },
    { ISD::ZERO_EXTEND, MVT::v4i64, MVT::v4i32, 1 },
    { ISD::TRUNCATE,    MVT::v4i32, MVT::v4i64, 1 },
    { ISD::TRUNCATE,    MVT::v8i16, MVT::v8i32, 1 },
    { ISD::TRUNCATE,    MVT::v16i8, MVT::v16i16, 2 },

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
    int Idx = ConvertCostTableLookup(AVXConversionTbl, ISD, DstTy.getSimpleVT(),
                                     SrcTy.getSimpleVT());
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

  static const CostTblEntry<MVT::SimpleValueType> SSE42CostTbl[] = {
    { ISD::SETCC,   MVT::v2f64,   1 },
    { ISD::SETCC,   MVT::v4f32,   1 },
    { ISD::SETCC,   MVT::v2i64,   1 },
    { ISD::SETCC,   MVT::v4i32,   1 },
    { ISD::SETCC,   MVT::v8i16,   1 },
    { ISD::SETCC,   MVT::v16i8,   1 },
  };

  static const CostTblEntry<MVT::SimpleValueType> AVX1CostTbl[] = {
    { ISD::SETCC,   MVT::v4f64,   1 },
    { ISD::SETCC,   MVT::v8f32,   1 },
    // AVX1 does not support 8-wide integer compare.
    { ISD::SETCC,   MVT::v4i64,   4 },
    { ISD::SETCC,   MVT::v8i32,   4 },
    { ISD::SETCC,   MVT::v16i16,  4 },
    { ISD::SETCC,   MVT::v32i8,   4 },
  };

  static const CostTblEntry<MVT::SimpleValueType> AVX2CostTbl[] = {
    { ISD::SETCC,   MVT::v4i64,   1 },
    { ISD::SETCC,   MVT::v8i32,   1 },
    { ISD::SETCC,   MVT::v16i16,  1 },
    { ISD::SETCC,   MVT::v32i8,   1 },
  };

  if (ST->hasAVX2()) {
    int Idx = CostTableLookup(AVX2CostTbl, ISD, MTy);
    if (Idx != -1)
      return LT.first * AVX2CostTbl[Idx].Cost;
  }

  if (ST->hasAVX()) {
    int Idx = CostTableLookup(AVX1CostTbl, ISD, MTy);
    if (Idx != -1)
      return LT.first * AVX1CostTbl[Idx].Cost;
  }

  if (ST->hasSSE42()) {
    int Idx = CostTableLookup(SSE42CostTbl, ISD, MTy);
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

unsigned X86TTI::getScalarizationOverhead(Type *Ty, bool Insert,
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

unsigned X86TTI::getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace) const {
  // Handle non-power-of-two vectors such as <3 x float>
  if (VectorType *VTy = dyn_cast<VectorType>(Src)) {
    unsigned NumElem = VTy->getVectorNumElements();

    // Handle a few common cases:
    // <3 x float>
    if (NumElem == 3 && VTy->getScalarSizeInBits() == 32)
      // Cost = 64 bit store + extract + 32 bit store.
      return 3;

    // <3 x double>
    if (NumElem == 3 && VTy->getScalarSizeInBits() == 64)
      // Cost = 128 bit store + unpack + 64 bit store.
      return 3;

    // Assume that all other non-power-of-two numbers are scalarized.
    if (!isPowerOf2_32(NumElem)) {
      unsigned Cost = TargetTransformInfo::getMemoryOpCost(Opcode,
                                                           VTy->getScalarType(),
                                                           Alignment,
                                                           AddressSpace);
      unsigned SplitCost = getScalarizationOverhead(Src,
                                                    Opcode == Instruction::Load,
                                                    Opcode==Instruction::Store);
      return NumElem * Cost + SplitCost;
    }
  }

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

unsigned X86TTI::getAddressComputationCost(Type *Ty, bool IsComplex) const {
  // Address computations in vectorized code with non-consecutive addresses will
  // likely result in more instructions compared to scalar code where the
  // computation can more often be merged into the index mode. The resulting
  // extra micro-ops can significantly decrease throughput.
  unsigned NumVectorInstToHideOverhead = 10;

  if (Ty->isVectorTy() && IsComplex)
    return NumVectorInstToHideOverhead;

  return TargetTransformInfo::getAddressComputationCost(Ty, IsComplex);
}

unsigned X86TTI::getReductionCost(unsigned Opcode, Type *ValTy,
                                  bool IsPairwise) const {
  
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(ValTy);
  
  MVT MTy = LT.second;
  
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");
 
  // We use the Intel Architecture Code Analyzer(IACA) to measure the throughput 
  // and make it as the cost. 
 
  static const CostTblEntry<MVT::SimpleValueType> SSE42CostTblPairWise[] = {
    { ISD::FADD,  MVT::v2f64,   2 },
    { ISD::FADD,  MVT::v4f32,   4 },
    { ISD::ADD,   MVT::v2i64,   2 },      // The data reported by the IACA tool is "1.6".
    { ISD::ADD,   MVT::v4i32,   3 },      // The data reported by the IACA tool is "3.5".
    { ISD::ADD,   MVT::v8i16,   5 },
  };
 
  static const CostTblEntry<MVT::SimpleValueType> AVX1CostTblPairWise[] = {
    { ISD::FADD,  MVT::v4f32,   4 },
    { ISD::FADD,  MVT::v4f64,   5 },
    { ISD::FADD,  MVT::v8f32,   7 },
    { ISD::ADD,   MVT::v2i64,   1 },      // The data reported by the IACA tool is "1.5".
    { ISD::ADD,   MVT::v4i32,   3 },      // The data reported by the IACA tool is "3.5".
    { ISD::ADD,   MVT::v4i64,   5 },      // The data reported by the IACA tool is "4.8".
    { ISD::ADD,   MVT::v8i16,   5 },
    { ISD::ADD,   MVT::v8i32,   5 },
  };

  static const CostTblEntry<MVT::SimpleValueType> SSE42CostTblNoPairWise[] = {
    { ISD::FADD,  MVT::v2f64,   2 },
    { ISD::FADD,  MVT::v4f32,   4 },
    { ISD::ADD,   MVT::v2i64,   2 },      // The data reported by the IACA tool is "1.6".
    { ISD::ADD,   MVT::v4i32,   3 },      // The data reported by the IACA tool is "3.3".
    { ISD::ADD,   MVT::v8i16,   4 },      // The data reported by the IACA tool is "4.3".
  };
  
  static const CostTblEntry<MVT::SimpleValueType> AVX1CostTblNoPairWise[] = {
    { ISD::FADD,  MVT::v4f32,   3 },
    { ISD::FADD,  MVT::v4f64,   3 },
    { ISD::FADD,  MVT::v8f32,   4 },
    { ISD::ADD,   MVT::v2i64,   1 },      // The data reported by the IACA tool is "1.5".
    { ISD::ADD,   MVT::v4i32,   3 },      // The data reported by the IACA tool is "2.8".
    { ISD::ADD,   MVT::v4i64,   3 },
    { ISD::ADD,   MVT::v8i16,   4 },
    { ISD::ADD,   MVT::v8i32,   5 },
  };
  
  if (IsPairwise) {
    if (ST->hasAVX()) {
      int Idx = CostTableLookup(AVX1CostTblPairWise, ISD, MTy);
      if (Idx != -1)
        return LT.first * AVX1CostTblPairWise[Idx].Cost;
    }
  
    if (ST->hasSSE42()) {
      int Idx = CostTableLookup(SSE42CostTblPairWise, ISD, MTy);
      if (Idx != -1)
        return LT.first * SSE42CostTblPairWise[Idx].Cost;
    }
  } else {
    if (ST->hasAVX()) {
      int Idx = CostTableLookup(AVX1CostTblNoPairWise, ISD, MTy);
      if (Idx != -1)
        return LT.first * AVX1CostTblNoPairWise[Idx].Cost;
    }
    
    if (ST->hasSSE42()) {
      int Idx = CostTableLookup(SSE42CostTblNoPairWise, ISD, MTy);
      if (Idx != -1)
        return LT.first * SSE42CostTblNoPairWise[Idx].Cost;
    }
  }

  return TargetTransformInfo::getReductionCost(Opcode, ValTy, IsPairwise);
}

