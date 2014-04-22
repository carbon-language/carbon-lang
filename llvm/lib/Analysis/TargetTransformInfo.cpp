//===- llvm/Analysis/TargetTransformInfo.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "tti"

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

unsigned TargetTransformInfo::getCallCost(FunctionType *FTy,
                                          int NumArgs) const {
  return PrevTTI->getCallCost(FTy, NumArgs);
}

unsigned TargetTransformInfo::getCallCost(const Function *F,
                                          int NumArgs) const {
  return PrevTTI->getCallCost(F, NumArgs);
}

unsigned TargetTransformInfo::getCallCost(
    const Function *F, ArrayRef<const Value *> Arguments) const {
  return PrevTTI->getCallCost(F, Arguments);
}

unsigned TargetTransformInfo::getIntrinsicCost(
    Intrinsic::ID IID, Type *RetTy, ArrayRef<Type *> ParamTys) const {
  return PrevTTI->getIntrinsicCost(IID, RetTy, ParamTys);
}

unsigned TargetTransformInfo::getIntrinsicCost(
    Intrinsic::ID IID, Type *RetTy, ArrayRef<const Value *> Arguments) const {
  return PrevTTI->getIntrinsicCost(IID, RetTy, Arguments);
}

unsigned TargetTransformInfo::getUserCost(const User *U) const {
  return PrevTTI->getUserCost(U);
}

bool TargetTransformInfo::hasBranchDivergence() const {
  return PrevTTI->hasBranchDivergence();
}

bool TargetTransformInfo::isLoweredToCall(const Function *F) const {
  return PrevTTI->isLoweredToCall(F);
}

void TargetTransformInfo::getUnrollingPreferences(Loop *L,
                            UnrollingPreferences &UP) const {
  PrevTTI->getUnrollingPreferences(L, UP);
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

int TargetTransformInfo::getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                              int64_t BaseOffset,
                                              bool HasBaseReg,
                                              int64_t Scale) const {
  return PrevTTI->getScalingFactorCost(Ty, BaseGV, BaseOffset, HasBaseReg,
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

bool TargetTransformInfo::haveFastSqrt(Type *Ty) const {
  return PrevTTI->haveFastSqrt(Ty);
}

unsigned TargetTransformInfo::getIntImmCost(const APInt &Imm, Type *Ty) const {
  return PrevTTI->getIntImmCost(Imm, Ty);
}

unsigned TargetTransformInfo::getIntImmCost(unsigned Opc, unsigned Idx,
                                            const APInt &Imm, Type *Ty) const {
  return PrevTTI->getIntImmCost(Opc, Idx, Imm, Ty);
}

unsigned TargetTransformInfo::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                            const APInt &Imm, Type *Ty) const {
  return PrevTTI->getIntImmCost(IID, Idx, Imm, Ty);
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
                                                Type *Ty,
                                                OperandValueKind Op1Info,
                                                OperandValueKind Op2Info) const {
  return PrevTTI->getArithmeticInstrCost(Opcode, Ty, Op1Info, Op2Info);
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

unsigned TargetTransformInfo::getAddressComputationCost(Type *Tp,
                                                        bool IsComplex) const {
  return PrevTTI->getAddressComputationCost(Tp, IsComplex);
}

unsigned TargetTransformInfo::getReductionCost(unsigned Opcode, Type *Ty,
                                               bool IsPairwise) const {
  return PrevTTI->getReductionCost(Opcode, Ty, IsPairwise);
}

namespace {

struct NoTTI final : ImmutablePass, TargetTransformInfo {
  const DataLayout *DL;

  NoTTI() : ImmutablePass(ID), DL(nullptr) {
    initializeNoTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() override {
    // Note that this subclass is special, and must *not* call initializeTTI as
    // it does not chain.
    TopTTI = this;
    PrevTTI = nullptr;
    DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
    DL = DLP ? &DLP->getDataLayout() : nullptr;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    // Note that this subclass is special, and must *not* call
    // TTI::getAnalysisUsage as it breaks the recursion.
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  unsigned getOperationCost(unsigned Opcode, Type *Ty,
                            Type *OpTy) const override {
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

    case Instruction::IntToPtr: {
      if (!DL)
        return TCC_Basic;

      // An inttoptr cast is free so long as the input is a legal integer type
      // which doesn't contain values outside the range of a pointer.
      unsigned OpSize = OpTy->getScalarSizeInBits();
      if (DL->isLegalInteger(OpSize) &&
          OpSize <= DL->getPointerTypeSizeInBits(Ty))
        return TCC_Free;

      // Otherwise it's not a no-op.
      return TCC_Basic;
    }
    case Instruction::PtrToInt: {
      if (!DL)
        return TCC_Basic;

      // A ptrtoint cast is free so long as the result is large enough to store
      // the pointer, and a legal integer type.
      unsigned DestSize = Ty->getScalarSizeInBits();
      if (DL->isLegalInteger(DestSize) &&
          DestSize >= DL->getPointerTypeSizeInBits(OpTy))
        return TCC_Free;

      // Otherwise it's not a no-op.
      return TCC_Basic;
    }
    case Instruction::Trunc:
      // trunc to a native type is free (assuming the target has compare and
      // shift-right of the same width).
      if (DL && DL->isLegalInteger(DL->getTypeSizeInBits(Ty)))
        return TCC_Free;

      return TCC_Basic;
    }
  }

  unsigned getGEPCost(const Value *Ptr,
                      ArrayRef<const Value *> Operands) const override {
    // In the basic model, we just assume that all-constant GEPs will be folded
    // into their uses via addressing modes.
    for (unsigned Idx = 0, Size = Operands.size(); Idx != Size; ++Idx)
      if (!isa<Constant>(Operands[Idx]))
        return TCC_Basic;

    return TCC_Free;
  }

  unsigned getCallCost(FunctionType *FTy, int NumArgs = -1) const override
  {
    assert(FTy && "FunctionType must be provided to this routine.");

    // The target-independent implementation just measures the size of the
    // function by approximating that each argument will take on average one
    // instruction to prepare.

    if (NumArgs < 0)
      // Set the argument number to the number of explicit arguments in the
      // function.
      NumArgs = FTy->getNumParams();

    return TCC_Basic * (NumArgs + 1);
  }

  unsigned getCallCost(const Function *F, int NumArgs = -1) const override
  {
    assert(F && "A concrete function must be provided to this routine.");

    if (NumArgs < 0)
      // Set the argument number to the number of explicit arguments in the
      // function.
      NumArgs = F->arg_size();

    if (Intrinsic::ID IID = (Intrinsic::ID)F->getIntrinsicID()) {
      FunctionType *FTy = F->getFunctionType();
      SmallVector<Type *, 8> ParamTys(FTy->param_begin(), FTy->param_end());
      return TopTTI->getIntrinsicCost(IID, FTy->getReturnType(), ParamTys);
    }

    if (!TopTTI->isLoweredToCall(F))
      return TCC_Basic; // Give a basic cost if it will be lowered directly.

    return TopTTI->getCallCost(F->getFunctionType(), NumArgs);
  }

  unsigned getCallCost(const Function *F,
                       ArrayRef<const Value *> Arguments) const override {
    // Simply delegate to generic handling of the call.
    // FIXME: We should use instsimplify or something else to catch calls which
    // will constant fold with these arguments.
    return TopTTI->getCallCost(F, Arguments.size());
  }

  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<Type *> ParamTys) const override {
    switch (IID) {
    default:
      // Intrinsics rarely (if ever) have normal argument setup constraints.
      // Model them as having a basic instruction cost.
      // FIXME: This is wrong for libc intrinsics.
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
      // These intrinsics don't actually represent code after lowering.
      return TCC_Free;
    }
  }

  unsigned
  getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                   ArrayRef<const Value *> Arguments) const override {
    // Delegate to the generic intrinsic handling code. This mostly provides an
    // opportunity for targets to (for example) special case the cost of
    // certain intrinsics based on constants used as arguments.
    SmallVector<Type *, 8> ParamTys;
    ParamTys.reserve(Arguments.size());
    for (unsigned Idx = 0, Size = Arguments.size(); Idx != Size; ++Idx)
      ParamTys.push_back(Arguments[Idx]->getType());
    return TopTTI->getIntrinsicCost(IID, RetTy, ParamTys);
  }

  unsigned getUserCost(const User *U) const override {
    if (isa<PHINode>(U))
      return TCC_Free; // Model all PHI nodes as free.

    if (const GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
      SmallVector<const Value *, 4> Indices(GEP->idx_begin(), GEP->idx_end());
      return TopTTI->getGEPCost(GEP->getPointerOperand(), Indices);
    }

    if (ImmutableCallSite CS = U) {
      const Function *F = CS.getCalledFunction();
      if (!F) {
        // Just use the called value type.
        Type *FTy = CS.getCalledValue()->getType()->getPointerElementType();
        return TopTTI->getCallCost(cast<FunctionType>(FTy), CS.arg_size());
      }

      SmallVector<const Value *, 8> Arguments(CS.arg_begin(), CS.arg_end());
      return TopTTI->getCallCost(F, Arguments);
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
                                U->getOperand(0)->getType() : nullptr);
  }

  bool hasBranchDivergence() const override { return false; }

  bool isLoweredToCall(const Function *F) const override {
    // FIXME: These should almost certainly not be handled here, and instead
    // handled with the help of TLI or the target itself. This was largely
    // ported from existing analysis heuristics here so that such refactorings
    // can take place in the future.

    if (F->isIntrinsic())
      return false;

    if (F->hasLocalLinkage() || !F->hasName())
      return true;

    StringRef Name = F->getName();

    // These will all likely lower to a single selection DAG node.
    if (Name == "copysign" || Name == "copysignf" || Name == "copysignl" ||
        Name == "fabs" || Name == "fabsf" || Name == "fabsl" || Name == "sin" ||
        Name == "sinf" || Name == "sinl" || Name == "cos" || Name == "cosf" ||
        Name == "cosl" || Name == "sqrt" || Name == "sqrtf" || Name == "sqrtl")
      return false;

    // These are all likely to be optimized into something smaller.
    if (Name == "pow" || Name == "powf" || Name == "powl" || Name == "exp2" ||
        Name == "exp2l" || Name == "exp2f" || Name == "floor" || Name ==
        "floorf" || Name == "ceil" || Name == "round" || Name == "ffs" ||
        Name == "ffsl" || Name == "abs" || Name == "labs" || Name == "llabs")
      return false;

    return true;
  }

  void getUnrollingPreferences(Loop *, UnrollingPreferences &) const override {
  }

  bool isLegalAddImmediate(int64_t Imm) const override {
    return false;
  }

  bool isLegalICmpImmediate(int64_t Imm) const override {
    return false;
  }

  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale) const override
  {
    // Guess that reg+reg addressing is allowed. This heuristic is taken from
    // the implementation of LSR.
    return !BaseGV && BaseOffset == 0 && Scale <= 1;
  }

  int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                           bool HasBaseReg, int64_t Scale) const override {
    // Guess that all legal addressing mode are free.
    if(isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg, Scale))
      return 0;
    return -1;
  }

  bool isTruncateFree(Type *Ty1, Type *Ty2) const override {
    return false;
  }

  bool isTypeLegal(Type *Ty) const override {
    return false;
  }

  unsigned getJumpBufAlignment() const override {
    return 0;
  }

  unsigned getJumpBufSize() const override {
    return 0;
  }

  bool shouldBuildLookupTables() const override {
    return true;
  }

  PopcntSupportKind
  getPopcntSupport(unsigned IntTyWidthInBit) const override {
    return PSK_Software;
  }

  bool haveFastSqrt(Type *Ty) const override {
    return false;
  }

  unsigned getIntImmCost(const APInt &Imm, Type *Ty) const override {
    return TCC_Basic;
  }

  unsigned getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                         Type *Ty) const override {
    return TCC_Free;
  }

  unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                         Type *Ty) const override {
    return TCC_Free;
  }

  unsigned getNumberOfRegisters(bool Vector) const override {
    return 8;
  }

  unsigned  getRegisterBitWidth(bool Vector) const override {
    return 32;
  }

  unsigned getMaximumUnrollFactor() const override {
    return 1;
  }

  unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty, OperandValueKind,
                                  OperandValueKind) const override {
    return 1;
  }

  unsigned getShuffleCost(ShuffleKind Kind, Type *Ty,
                          int Index = 0, Type *SubTp = nullptr) const override {
    return 1;
  }

  unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                            Type *Src) const override {
    return 1;
  }

  unsigned getCFInstrCost(unsigned Opcode) const override {
    return 1;
  }

  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                              Type *CondTy = nullptr) const override {
    return 1;
  }

  unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                              unsigned Index = -1) const override {
    return 1;
  }

  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                           unsigned AddressSpace) const override {
    return 1;
  }

  unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                 ArrayRef<Type*> Tys) const override {
    return 1;
  }

  unsigned getNumberOfParts(Type *Tp) const override {
    return 0;
  }

  unsigned getAddressComputationCost(Type *Tp, bool) const override {
    return 0;
  }

  unsigned getReductionCost(unsigned, Type *, bool) const override {
    return 1;
  }
};

} // end anonymous namespace

INITIALIZE_AG_PASS(NoTTI, TargetTransformInfo, "notti",
                   "No target information", true, true, true)
char NoTTI::ID = 0;

ImmutablePass *llvm::createNoTargetTransformInfoPass() {
  return new NoTTI();
}
