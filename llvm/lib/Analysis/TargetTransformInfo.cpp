//===- llvm/Analysis/TargetTransformInfo.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tti"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// Setup the analysis group to manage the TargetTransformInfo passes.
INITIALIZE_ANALYSIS_GROUP(TargetTransformInfo, "Target Information", NoTTI)
char TargetTransformInfo::ID = 0;

TargetTransformInfo::~TargetTransformInfo() {
}

void TargetTransformInfo::pushTTIStack(Pass *P) {
  TopTTI = this;
  PrevTTI = &P->getAnalysis<TargetTransformInfo>();

  // Walk up the chain and update the top TTI pointer.
  for (TargetTransformInfo *PTTI = PrevTTI; PTTI; PTTI = PTTI->PrevTTI)
    PTTI->TopTTI = this;
}

void TargetTransformInfo::popTTIStack() {
  TopTTI = 0;

  // Walk up the chain and update the top TTI pointer.
  for (TargetTransformInfo *PTTI = PrevTTI; PTTI; PTTI = PTTI->PrevTTI)
    PTTI->TopTTI = PrevTTI;

  PrevTTI = 0;
}

void TargetTransformInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetTransformInfo>();
}

unsigned TargetTransformInfo::getOperationCost(unsigned Opcode, Type *Ty,
                                               Type *OpTy) const {
  return PrevTTI->getOperationCost(Opcode, Ty, OpTy);
}

unsigned TargetTransformInfo::getGEPCost(
    const Value *Ptr, ArrayRef<const Value *> Operands) const {
  return PrevTTI->getGEPCost(Ptr, Operands);
}

unsigned TargetTransformInfo::getUserCost(const User *U) const {
  return PrevTTI->getUserCost(U);
}

bool TargetTransformInfo::isLegalAddImmediate(int64_t Imm) const {
  return PrevTTI->isLegalAddImmediate(Imm);
}

bool TargetTransformInfo::isLegalICmpImmediate(int64_t Imm) const {
  return PrevTTI->isLegalICmpImmediate(Imm);
}

bool TargetTransformInfo::isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                                int64_t BaseOffset,
                                                bool HasBaseReg,
                                                int64_t Scale) const {
  return PrevTTI->isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                                        Scale);
}

bool TargetTransformInfo::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return PrevTTI->isTruncateFree(Ty1, Ty2);
}

bool TargetTransformInfo::isTypeLegal(Type *Ty) const {
  return PrevTTI->isTypeLegal(Ty);
}

unsigned TargetTransformInfo::getJumpBufAlignment() const {
  return PrevTTI->getJumpBufAlignment();
}

unsigned TargetTransformInfo::getJumpBufSize() const {
  return PrevTTI->getJumpBufSize();
}

bool TargetTransformInfo::shouldBuildLookupTables() const {
  return PrevTTI->shouldBuildLookupTables();
}

TargetTransformInfo::PopcntSupportKind
TargetTransformInfo::getPopcntSupport(unsigned IntTyWidthInBit) const {
  return PrevTTI->getPopcntSupport(IntTyWidthInBit);
}

unsigned TargetTransformInfo::getIntImmCost(const APInt &Imm, Type *Ty) const {
  return PrevTTI->getIntImmCost(Imm, Ty);
}

unsigned TargetTransformInfo::getNumberOfRegisters(bool Vector) const {
  return PrevTTI->getNumberOfRegisters(Vector);
}

unsigned TargetTransformInfo::getRegisterBitWidth(bool Vector) const {
  return PrevTTI->getRegisterBitWidth(Vector);
}

unsigned TargetTransformInfo::getMaximumUnrollFactor() const {
  return PrevTTI->getMaximumUnrollFactor();
}

unsigned TargetTransformInfo::getArithmeticInstrCost(unsigned Opcode,
                                                     Type *Ty) const {
  return PrevTTI->getArithmeticInstrCost(Opcode, Ty);
}

unsigned TargetTransformInfo::getShuffleCost(ShuffleKind Kind, Type *Tp,
                                             int Index, Type *SubTp) const {
  return PrevTTI->getShuffleCost(Kind, Tp, Index, SubTp);
}

unsigned TargetTransformInfo::getCastInstrCost(unsigned Opcode, Type *Dst,
                                               Type *Src) const {
  return PrevTTI->getCastInstrCost(Opcode, Dst, Src);
}

unsigned TargetTransformInfo::getCFInstrCost(unsigned Opcode) const {
  return PrevTTI->getCFInstrCost(Opcode);
}

unsigned TargetTransformInfo::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                                 Type *CondTy) const {
  return PrevTTI->getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned TargetTransformInfo::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                 unsigned Index) const {
  return PrevTTI->getVectorInstrCost(Opcode, Val, Index);
}

unsigned TargetTransformInfo::getMemoryOpCost(unsigned Opcode, Type *Src,
                                              unsigned Alignment,
                                              unsigned AddressSpace) const {
  return PrevTTI->getMemoryOpCost(Opcode, Src, Alignment, AddressSpace);
  ;
}

unsigned
TargetTransformInfo::getIntrinsicInstrCost(Intrinsic::ID ID,
                                           Type *RetTy,
                                           ArrayRef<Type *> Tys) const {
  return PrevTTI->getIntrinsicInstrCost(ID, RetTy, Tys);
}

unsigned TargetTransformInfo::getNumberOfParts(Type *Tp) const {
  return PrevTTI->getNumberOfParts(Tp);
}


namespace {

struct NoTTI : ImmutablePass, TargetTransformInfo {
  const DataLayout *DL;

  NoTTI() : ImmutablePass(ID), DL(0) {
    initializeNoTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() {
    // Note that this subclass is special, and must *not* call initializeTTI as
    // it does not chain.
    PrevTTI = 0;
    DL = getAnalysisIfAvailable<DataLayout>();
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    // Note that this subclass is special, and must *not* call
    // TTI::getAnalysisUsage as it breaks the recursion.
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  unsigned getOperationCost(unsigned Opcode, Type *Ty, Type *OpTy) const {
    switch (Opcode) {
    default:
      // By default, just classify everything as 'basic'.
      return TCC_Basic;

    case Instruction::GetElementPtr:
      llvm_unreachable("Use getGEPCost for GEP operations!");

    case Instruction::BitCast:
      assert(OpTy && "Cast instructions must provide the operand type");
      if (Ty == OpTy || (Ty->isPointerTy() && OpTy->isPointerTy()))
        // Identity and pointer-to-pointer casts are free.
        return TCC_Free;

      // Otherwise, the default basic cost is used.
      return TCC_Basic;

    case Instruction::IntToPtr:
      // An inttoptr cast is free so long as the input is a legal integer type
      // which doesn't contain values outside the range of a pointer.
      if (DL && DL->isLegalInteger(OpTy->getScalarSizeInBits()) &&
          OpTy->getScalarSizeInBits() <= DL->getPointerSizeInBits())
        return TCC_Free;

      // Otherwise it's not a no-op.
      return TCC_Basic;

    case Instruction::PtrToInt:
      // A ptrtoint cast is free so long as the result is large enough to store
      // the pointer, and a legal integer type.
      if (DL && DL->isLegalInteger(OpTy->getScalarSizeInBits()) &&
          OpTy->getScalarSizeInBits() >= DL->getPointerSizeInBits())
        return TCC_Free;

      // Otherwise it's not a no-op.
      return TCC_Basic;

    case Instruction::Trunc:
      // trunc to a native type is free (assuming the target has compare and
      // shift-right of the same width).
      if (DL && DL->isLegalInteger(DL->getTypeSizeInBits(Ty)))
        return TCC_Free;

      return TCC_Basic;
    }
  }

  unsigned getGEPCost(const Value *Ptr,
                      ArrayRef<const Value *> Operands) const {
    // In the basic model, we just assume that all-constant GEPs will be folded
    // into their uses via addressing modes.
    for (unsigned Idx = 0, Size = Operands.size(); Idx != Size; ++Idx)
      if (!isa<Constant>(Operands[Idx]))
        return TCC_Basic;

    return TCC_Free;
  }

  unsigned getUserCost(const User *U) const {
    if (const GEPOperator *GEP = dyn_cast<GEPOperator>(U))
      // In the basic model we just assume that all-constant GEPs will be
      // folded into their uses via addressing modes.
      return GEP->hasAllConstantIndices() ? TCC_Free : TCC_Basic;

    // If we have a call of an intrinsic we can provide more detailed analysis
    // by inspecting the particular intrinsic called.
    // FIXME: Hoist this out into a getIntrinsicCost routine.
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
      switch (II->getIntrinsicID()) {
      default:
        return TCC_Basic;
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::invariant_start:
      case Intrinsic::invariant_end:
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::objectsize:
      case Intrinsic::ptr_annotation:
      case Intrinsic::var_annotation:
        // These intrinsics don't count as size.
        return TCC_Free;
      }
    }

    if (const CastInst *CI = dyn_cast<CastInst>(U)) {
      // Result of a cmp instruction is often extended (to be used by other
      // cmp instructions, logical or return instructions). These are usually
      // nop on most sane targets.
      if (isa<CmpInst>(CI->getOperand(0)))
        return TCC_Free;
    }

    // Otherwise delegate to the fully generic implementations.
    return getOperationCost(Operator::getOpcode(U), U->getType(),
                            U->getNumOperands() == 1 ?
                                U->getOperand(0)->getType() : 0);
  }

  bool isLegalAddImmediate(int64_t Imm) const {
    return false;
  }

  bool isLegalICmpImmediate(int64_t Imm) const {
    return false;
  }

  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale) const {
    // Guess that reg+reg addressing is allowed. This heuristic is taken from
    // the implementation of LSR.
    return !BaseGV && BaseOffset == 0 && Scale <= 1;
  }

  bool isTruncateFree(Type *Ty1, Type *Ty2) const {
    return false;
  }

  bool isTypeLegal(Type *Ty) const {
    return false;
  }

  unsigned getJumpBufAlignment() const {
    return 0;
  }

  unsigned getJumpBufSize() const {
    return 0;
  }

  bool shouldBuildLookupTables() const {
    return true;
  }

  PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const {
    return PSK_Software;
  }

  unsigned getIntImmCost(const APInt &Imm, Type *Ty) const {
    return 1;
  }

  unsigned getNumberOfRegisters(bool Vector) const {
    return 8;
  }

  unsigned  getRegisterBitWidth(bool Vector) const {
    return 32;
  }

  unsigned getMaximumUnrollFactor() const {
    return 1;
  }

  unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const {
    return 1;
  }

  unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                          int Index = 0, Type *SubTp = 0) const {
    return 1;
  }

  unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                            Type *Src) const {
    return 1;
  }

  unsigned getCFInstrCost(unsigned Opcode) const {
    return 1;
  }

  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                              Type *CondTy = 0) const {
    return 1;
  }

  unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                              unsigned Index = -1) const {
    return 1;
  }

  unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                           unsigned Alignment,
                           unsigned AddressSpace) const {
    return 1;
  }

  unsigned getIntrinsicInstrCost(Intrinsic::ID ID,
                                 Type *RetTy,
                                 ArrayRef<Type*> Tys) const {
    return 1;
  }

  unsigned getNumberOfParts(Type *Tp) const {
    return 0;
  }
};

} // end anonymous namespace

INITIALIZE_AG_PASS(NoTTI, TargetTransformInfo, "notti",
                   "No target information", true, true, true)
char NoTTI::ID = 0;

ImmutablePass *llvm::createNoTargetTransformInfoPass() {
  return new NoTTI();
}
