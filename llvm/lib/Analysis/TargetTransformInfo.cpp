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
  NoTTI() : ImmutablePass(ID) {
    initializeNoTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() {
    // Note that this subclass is special, and must *not* call initializeTTI as
    // it does not chain.
    PrevTTI = 0;
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
