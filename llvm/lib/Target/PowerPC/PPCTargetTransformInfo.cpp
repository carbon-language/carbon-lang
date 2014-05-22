//===-- PPCTargetTransformInfo.cpp - PPC specific TTI pass ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// PPC target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "ppctti"

static cl::opt<bool> DisablePPCConstHoist("disable-ppc-constant-hoisting",
cl::desc("disable constant hoisting on PPC"), cl::init(false), cl::Hidden);

// Declare the pass initialization routine locally as target-specific passes
// don't have a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializePPCTTIPass(PassRegistry &);
}

namespace {

class PPCTTI final : public ImmutablePass, public TargetTransformInfo {
  const PPCSubtarget *ST;
  const PPCTargetLowering *TLI;

public:
  PPCTTI() : ImmutablePass(ID), ST(nullptr), TLI(nullptr) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  PPCTTI(const PPCTargetMachine *TM)
      : ImmutablePass(ID), ST(TM->getSubtargetImpl()),
        TLI(TM->getTargetLowering()) {
    initializePPCTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() override {
    pushTTIStack(this);
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  /// \name Scalar TTI Implementations
  /// @{
  unsigned getIntImmCost(const APInt &Imm, Type *Ty) const override;

  unsigned getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                         Type *Ty) const override;
  unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                         Type *Ty) const override;

  virtual PopcntSupportKind
  getPopcntSupport(unsigned TyWidth) const override;
  virtual void getUnrollingPreferences(
    Loop *L, UnrollingPreferences &UP) const override;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  virtual unsigned getNumberOfRegisters(bool Vector) const override;
  virtual unsigned getRegisterBitWidth(bool Vector) const override;
  virtual unsigned getMaximumUnrollFactor() const override;
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                          OperandValueKind,
                                          OperandValueKind) const override;
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                                  int Index, Type *SubTp) const override;
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const override;
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) const override;
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) const override;
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const override;

  /// @}
};

} // end anonymous namespace

INITIALIZE_AG_PASS(PPCTTI, TargetTransformInfo, "ppctti",
                   "PPC Target Transform Info", true, true, false)
char PPCTTI::ID = 0;

ImmutablePass *
llvm::createPPCTargetTransformInfoPass(const PPCTargetMachine *TM) {
  return new PPCTTI(TM);
}


//===----------------------------------------------------------------------===//
//
// PPC cost model.
//
//===----------------------------------------------------------------------===//

PPCTTI::PopcntSupportKind PPCTTI::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  if (ST->hasPOPCNTD() && TyWidth <= 64)
    return PSK_FastHardware;
  return PSK_Software;
}

unsigned PPCTTI::getIntImmCost(const APInt &Imm, Type *Ty) const {
  if (DisablePPCConstHoist)
    return TargetTransformInfo::getIntImmCost(Imm, Ty);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  if (Imm == 0)
    return TCC_Free;

  if (Imm.getBitWidth() <= 64) {
    if (isInt<16>(Imm.getSExtValue()))
      return TCC_Basic;

    if (isInt<32>(Imm.getSExtValue())) {
      // A constant that can be materialized using lis.
      if ((Imm.getZExtValue() & 0xFFFF) == 0)
        return TCC_Basic;

      return 2 * TCC_Basic;
    }
  }

  return 4 * TCC_Basic;
}

unsigned PPCTTI::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                               const APInt &Imm, Type *Ty) const {
  if (DisablePPCConstHoist)
    return TargetTransformInfo::getIntImmCost(IID, Idx, Imm, Ty);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  switch (IID) {
  default: return TCC_Free;
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
    if ((Idx == 1) && Imm.getBitWidth() <= 64 && isInt<16>(Imm.getSExtValue()))
      return TCC_Free;
    break;
  }
  return PPCTTI::getIntImmCost(Imm, Ty);
}

unsigned PPCTTI::getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                               Type *Ty) const {
  if (DisablePPCConstHoist)
    return TargetTransformInfo::getIntImmCost(Opcode, Idx, Imm, Ty);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  unsigned ImmIdx = ~0U;
  bool ShiftedFree = false, RunFree = false, UnsignedFree = false,
       ZeroFree = false;
  switch (Opcode) {
  default: return TCC_Free;
  case Instruction::GetElementPtr:
    // Always hoist the base address of a GetElementPtr. This prevents the
    // creation of new constants for every base constant that gets constant
    // folded with the offset.
    if (Idx == 0)
      return 2 * TCC_Basic;
    return TCC_Free;
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
    return TCC_Free;

  if (Idx == ImmIdx && Imm.getBitWidth() <= 64) {
    if (isInt<16>(Imm.getSExtValue()))
      return TCC_Free;

    if (RunFree) {
      if (Imm.getBitWidth() <= 32 &&
          (isShiftedMask_32(Imm.getZExtValue()) ||
           isShiftedMask_32(~Imm.getZExtValue())))
        return TCC_Free;


      if (ST->isPPC64() &&
          (isShiftedMask_64(Imm.getZExtValue()) ||
           isShiftedMask_64(~Imm.getZExtValue())))
        return TCC_Free;
    }

    if (UnsignedFree && isUInt<16>(Imm.getZExtValue()))
      return TCC_Free;

    if (ShiftedFree && (Imm.getZExtValue() & 0xFFFF) == 0)
      return TCC_Free;
  }

  return PPCTTI::getIntImmCost(Imm, Ty);
}

void PPCTTI::getUnrollingPreferences(Loop *L, UnrollingPreferences &UP) const {
  if (ST->getDarwinDirective() == PPC::DIR_A2) {
    // The A2 is in-order with a deep pipeline, and concatenation unrolling
    // helps expose latency-hiding opportunities to the instruction scheduler.
    UP.Partial = UP.Runtime = true;
  }
}

unsigned PPCTTI::getNumberOfRegisters(bool Vector) const {
  if (Vector && !ST->hasAltivec())
    return 0;
  return ST->hasVSX() ? 64 : 32;
}

unsigned PPCTTI::getRegisterBitWidth(bool Vector) const {
  if (Vector) {
    if (ST->hasAltivec()) return 128;
    return 0;
  }

  if (ST->isPPC64())
    return 64;
  return 32;

}

unsigned PPCTTI::getMaximumUnrollFactor() const {
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

  // For most things, modern systems have two execution units (and
  // out-of-order execution).
  return 2;
}

unsigned PPCTTI::getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                        OperandValueKind Op1Info,
                                        OperandValueKind Op2Info) const {
  assert(TLI->InstructionOpcodeToISD(Opcode) && "Invalid opcode");

  // Fallback to the default implementation.
  return TargetTransformInfo::getArithmeticInstrCost(Opcode, Ty, Op1Info,
                                                     Op2Info);
}

unsigned PPCTTI::getShuffleCost(ShuffleKind Kind, Type *Tp, int Index,
                                Type *SubTp) const {
  return TargetTransformInfo::getShuffleCost(Kind, Tp, Index, SubTp);
}

unsigned PPCTTI::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) const {
  assert(TLI->InstructionOpcodeToISD(Opcode) && "Invalid opcode");

  return TargetTransformInfo::getCastInstrCost(Opcode, Dst, Src);
}

unsigned PPCTTI::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                    Type *CondTy) const {
  return TargetTransformInfo::getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned PPCTTI::getVectorInstrCost(unsigned Opcode, Type *Val,
                                    unsigned Index) const {
  assert(Val->isVectorTy() && "This must be a vector type");

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  if (ST->hasVSX() && Val->getScalarType()->isDoubleTy()) {
    // Double-precision scalars are already located in index #0.
    if (Index == 0)
      return 0;

    return TargetTransformInfo::getVectorInstrCost(Opcode, Val, Index);
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
    return LHSPenalty +
      TargetTransformInfo::getVectorInstrCost(Opcode, Val, Index);

  return TargetTransformInfo::getVectorInstrCost(Opcode, Val, Index);
}

unsigned PPCTTI::getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Src);
  assert((Opcode == Instruction::Load || Opcode == Instruction::Store) &&
         "Invalid Opcode");

  unsigned Cost =
    TargetTransformInfo::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace);

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

