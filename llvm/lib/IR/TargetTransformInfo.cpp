//===- llvm/IR/TargetTransformInfo.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetTransformInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// Setup the analysis group to manage the TargetTransformInfo passes.
INITIALIZE_ANALYSIS_GROUP(TargetTransformInfo, "Target Information", NoTTI)
char TargetTransformInfo::ID = 0;

TargetTransformInfo::~TargetTransformInfo() {
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

TargetTransformInfo::PopcntHwSupport
TargetTransformInfo::getPopcntHwSupport(unsigned IntTyWidthInBit) const {
  return PrevTTI->getPopcntHwSupport(IntTyWidthInBit);
}

unsigned TargetTransformInfo::getIntImmCost(const APInt &Imm, Type *Ty) const {
  return PrevTTI->getIntImmCost(Imm, Ty);
}

unsigned TargetTransformInfo::getNumberOfRegisters(bool Vector) const {
  return PrevTTI->getNumberOfRegisters(Vector);
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

class NoTTI : public ImmutablePass, public TargetTransformInfo {
  const ScalarTargetTransformInfo *STTI;
  const VectorTargetTransformInfo *VTTI;

public:
  // FIXME: This constructor doesn't work which breaks the use of NoTTI on the
  // commandline. This has to be fixed for NoTTI to be fully usable as an
  // analysis pass.
  NoTTI() : ImmutablePass(ID), TargetTransformInfo(0) {
    llvm_unreachable("Unsupported code path!");
  }

  NoTTI(const ScalarTargetTransformInfo *S, const VectorTargetTransformInfo *V)
      : ImmutablePass(ID),
        TargetTransformInfo(0), // NoTTI is special and doesn't delegate here.
        STTI(S), VTTI(V) {
    initializeNoTTIPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {
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


  // Delegate all predicates through the STTI or VTTI interface.

  bool isLegalAddImmediate(int64_t Imm) const {
    return STTI->isLegalAddImmediate(Imm);
  }

  bool isLegalICmpImmediate(int64_t Imm) const {
    return STTI->isLegalICmpImmediate(Imm);
  }

  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale) const {
    return STTI->isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                                       Scale);
  }

  bool isTruncateFree(Type *Ty1, Type *Ty2) const {
    return STTI->isTruncateFree(Ty1, Ty2);
  }

  bool isTypeLegal(Type *Ty) const {
    return STTI->isTypeLegal(Ty);
  }

  unsigned getJumpBufAlignment() const {
    return STTI->getJumpBufAlignment();
  }

  unsigned getJumpBufSize() const {
    return STTI->getJumpBufSize();
  }

  bool shouldBuildLookupTables() const {
    return STTI->shouldBuildLookupTables();
  }

  PopcntHwSupport getPopcntHwSupport(unsigned IntTyWidthInBit) const {
    return (PopcntHwSupport)STTI->getPopcntHwSupport(IntTyWidthInBit);
  }

  unsigned getIntImmCost(const APInt &Imm, Type *Ty) const {
    return STTI->getIntImmCost(Imm, Ty);
  }

  unsigned getNumberOfRegisters(bool Vector) const {
    return VTTI->getNumberOfRegisters(Vector);
  }

  unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const {
    return VTTI->getArithmeticInstrCost(Opcode, Ty);
  }

  unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                          int Index = 0, Type *SubTp = 0) const {
    return VTTI->getShuffleCost((VectorTargetTransformInfo::ShuffleKind)Kind,
                                Tp, Index, SubTp);
  }

  unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                            Type *Src) const {
    return VTTI->getCastInstrCost(Opcode, Dst, Src);
  }

  unsigned getCFInstrCost(unsigned Opcode) const {
    return VTTI->getCFInstrCost(Opcode);
  }

  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                              Type *CondTy = 0) const {
    return VTTI->getCmpSelInstrCost(Opcode, ValTy, CondTy);
  }

  unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                              unsigned Index = -1) const {
    return VTTI->getVectorInstrCost(Opcode, Val, Index);
  }

  unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                           unsigned Alignment,
                           unsigned AddressSpace) const {
    return VTTI->getMemoryOpCost(Opcode, Src, Alignment, AddressSpace);
  }

  unsigned getIntrinsicInstrCost(Intrinsic::ID ID,
                                 Type *RetTy,
                                 ArrayRef<Type*> Tys) const {
    return VTTI->getIntrinsicInstrCost(ID, RetTy, Tys);
  }

  unsigned getNumberOfParts(Type *Tp) const {
    return VTTI->getNumberOfParts(Tp);
  }
};

} // end anonymous namespace

INITIALIZE_AG_PASS(NoTTI, TargetTransformInfo, "no-tti",
                   "No target information", true, true, true)
char NoTTI::ID = 0;

ImmutablePass *llvm::createNoTTIPass(const ScalarTargetTransformInfo *S,
                                     const VectorTargetTransformInfo *V) {
  return new NoTTI(S, V);
}
