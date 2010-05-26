//===-- Lint.cpp - Check for common errors in LLVM IR ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass statically checks for common and easily-identified constructs
// which produce undefined or likely unintended behavior in LLVM IR.
//
// It is not a guarantee of correctness, in two ways. First, it isn't
// comprehensive. There are checks which could be done statically which are
// not yet implemented. Some of these are indicated by TODO comments, but
// those aren't comprehensive either. Second, many conditions cannot be
// checked statically. This pass does no dynamic instrumentation, so it
// can't check for all possible problems.
// 
// Another limitation is that it assumes all code will be executed. A store
// through a null pointer in a basic block which is never reached is harmless,
// but this pass will warn about it anyway.
//
// Optimization passes may make conditions that this pass checks for more or
// less obvious. If an optimization pass appears to be introducing a warning,
// it may be that the optimization pass is merely exposing an existing
// condition in the code.
// 
// This code may be run before instcombine. In many cases, instcombine checks
// for the same kinds of things and turns instructions with undefined behavior
// into unreachable (or equivalent). Because of this, this pass makes some
// effort to look through bitcasts and so on.
// 
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Lint.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Function.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

namespace {
  namespace MemRef {
    static unsigned Read     = 1;
    static unsigned Write    = 2;
    static unsigned Callee   = 4;
    static unsigned Branchee = 8;
  }

  class Lint : public FunctionPass, public InstVisitor<Lint> {
    friend class InstVisitor<Lint>;

    void visitFunction(Function &F);

    void visitCallSite(CallSite CS);
    void visitMemoryReference(Instruction &I, Value *Ptr, unsigned Align,
                              const Type *Ty, unsigned Flags);

    void visitCallInst(CallInst &I);
    void visitInvokeInst(InvokeInst &I);
    void visitReturnInst(ReturnInst &I);
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);
    void visitXor(BinaryOperator &I);
    void visitSub(BinaryOperator &I);
    void visitLShr(BinaryOperator &I);
    void visitAShr(BinaryOperator &I);
    void visitShl(BinaryOperator &I);
    void visitSDiv(BinaryOperator &I);
    void visitUDiv(BinaryOperator &I);
    void visitSRem(BinaryOperator &I);
    void visitURem(BinaryOperator &I);
    void visitAllocaInst(AllocaInst &I);
    void visitVAArgInst(VAArgInst &I);
    void visitIndirectBrInst(IndirectBrInst &I);
    void visitExtractElementInst(ExtractElementInst &I);
    void visitInsertElementInst(InsertElementInst &I);
    void visitUnreachableInst(UnreachableInst &I);

  public:
    Module *Mod;
    AliasAnalysis *AA;
    TargetData *TD;

    std::string Messages;
    raw_string_ostream MessagesStr;

    static char ID; // Pass identification, replacement for typeid
    Lint() : FunctionPass(&ID), MessagesStr(Messages) {}

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<AliasAnalysis>();
    }
    virtual void print(raw_ostream &O, const Module *M) const {}

    void WriteValue(const Value *V) {
      if (!V) return;
      if (isa<Instruction>(V)) {
        MessagesStr << *V << '\n';
      } else {
        WriteAsOperand(MessagesStr, V, true, Mod);
        MessagesStr << '\n';
      }
    }

    void WriteType(const Type *T) {
      if (!T) return;
      MessagesStr << ' ';
      WriteTypeSymbolic(MessagesStr, T, Mod);
    }

    // CheckFailed - A check failed, so print out the condition and the message
    // that failed.  This provides a nice place to put a breakpoint if you want
    // to see why something is not correct.
    void CheckFailed(const Twine &Message,
                     const Value *V1 = 0, const Value *V2 = 0,
                     const Value *V3 = 0, const Value *V4 = 0) {
      MessagesStr << Message.str() << "\n";
      WriteValue(V1);
      WriteValue(V2);
      WriteValue(V3);
      WriteValue(V4);
    }

    void CheckFailed(const Twine &Message, const Value *V1,
                     const Type *T2, const Value *V3 = 0) {
      MessagesStr << Message.str() << "\n";
      WriteValue(V1);
      WriteType(T2);
      WriteValue(V3);
    }

    void CheckFailed(const Twine &Message, const Type *T1,
                     const Type *T2 = 0, const Type *T3 = 0) {
      MessagesStr << Message.str() << "\n";
      WriteType(T1);
      WriteType(T2);
      WriteType(T3);
    }
  };
}

char Lint::ID = 0;
static RegisterPass<Lint>
X("lint", "Statically lint-checks LLVM IR", false, true);

// Assert - We know that cond should be true, if not print an error message.
#define Assert(C, M) \
    do { if (!(C)) { CheckFailed(M); return; } } while (0)
#define Assert1(C, M, V1) \
    do { if (!(C)) { CheckFailed(M, V1); return; } } while (0)
#define Assert2(C, M, V1, V2) \
    do { if (!(C)) { CheckFailed(M, V1, V2); return; } } while (0)
#define Assert3(C, M, V1, V2, V3) \
    do { if (!(C)) { CheckFailed(M, V1, V2, V3); return; } } while (0)
#define Assert4(C, M, V1, V2, V3, V4) \
    do { if (!(C)) { CheckFailed(M, V1, V2, V3, V4); return; } } while (0)

// Lint::run - This is the main Analysis entry point for a
// function.
//
bool Lint::runOnFunction(Function &F) {
  Mod = F.getParent();
  AA = &getAnalysis<AliasAnalysis>();
  TD = getAnalysisIfAvailable<TargetData>();
  visit(F);
  dbgs() << MessagesStr.str();
  Messages.clear();
  return false;
}

void Lint::visitFunction(Function &F) {
  // This isn't undefined behavior, it's just a little unusual, and it's a
  // fairly common mistake to neglect to name a function.
  Assert1(F.hasName() || F.hasLocalLinkage(),
          "Unusual: Unnamed function with non-local linkage", &F);
}

void Lint::visitCallSite(CallSite CS) {
  Instruction &I = *CS.getInstruction();
  Value *Callee = CS.getCalledValue();

  visitMemoryReference(I, Callee, 0, 0, MemRef::Callee);

  if (Function *F = dyn_cast<Function>(Callee->stripPointerCasts())) {
    Assert1(CS.getCallingConv() == F->getCallingConv(),
            "Undefined behavior: Caller and callee calling convention differ",
            &I);

    const FunctionType *FT = F->getFunctionType();
    unsigned NumActualArgs = unsigned(CS.arg_end()-CS.arg_begin());

    Assert1(FT->isVarArg() ?
              FT->getNumParams() <= NumActualArgs :
              FT->getNumParams() == NumActualArgs,
            "Undefined behavior: Call argument count mismatches callee "
            "argument count", &I);
      
    // TODO: Check argument types (in case the callee was casted)

    // TODO: Check ABI-significant attributes.

    // TODO: Check noalias attribute.

    // TODO: Check sret attribute.
  }

  if (CS.isCall() && cast<CallInst>(CS.getInstruction())->isTailCall())
    for (CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
         AI != AE; ++AI) {
      Value *Obj = (*AI)->getUnderlyingObject();
      Assert1(!isa<AllocaInst>(Obj) && !isa<VAArgInst>(Obj),
              "Undefined behavior: Call with \"tail\" keyword references "
              "alloca or va_arg", &I);
    }


  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
    switch (II->getIntrinsicID()) {
    default: break;

    // TODO: Check more intrinsics

    case Intrinsic::memcpy: {
      MemCpyInst *MCI = cast<MemCpyInst>(&I);
      visitMemoryReference(I, MCI->getSource(), MCI->getAlignment(), 0,
                           MemRef::Write);
      visitMemoryReference(I, MCI->getDest(), MCI->getAlignment(), 0,
                           MemRef::Read);

      // Check that the memcpy arguments don't overlap. The AliasAnalysis API
      // isn't expressive enough for what we really want to do. Known partial
      // overlap is not distinguished from the case where nothing is known.
      unsigned Size = 0;
      if (const ConstantInt *Len =
            dyn_cast<ConstantInt>(MCI->getLength()->stripPointerCasts()))
        if (Len->getValue().isIntN(32))
          Size = Len->getValue().getZExtValue();
      Assert1(AA->alias(MCI->getSource(), Size, MCI->getDest(), Size) !=
              AliasAnalysis::MustAlias,
              "Undefined behavior: memcpy source and destination overlap", &I);
      break;
    }
    case Intrinsic::memmove: {
      MemMoveInst *MMI = cast<MemMoveInst>(&I);
      visitMemoryReference(I, MMI->getSource(), MMI->getAlignment(), 0,
                           MemRef::Write);
      visitMemoryReference(I, MMI->getDest(), MMI->getAlignment(), 0,
                           MemRef::Read);
      break;
    }
    case Intrinsic::memset: {
      MemSetInst *MSI = cast<MemSetInst>(&I);
      visitMemoryReference(I, MSI->getDest(), MSI->getAlignment(), 0,
                           MemRef::Write);
      break;
    }

    case Intrinsic::vastart:
      Assert1(I.getParent()->getParent()->isVarArg(),
              "Undefined behavior: va_start called in a non-varargs function",
              &I);

      visitMemoryReference(I, CS.getArgument(0), 0, 0,
                           MemRef::Read | MemRef::Write);
      break;
    case Intrinsic::vacopy:
      visitMemoryReference(I, CS.getArgument(0), 0, 0, MemRef::Write);
      visitMemoryReference(I, CS.getArgument(1), 0, 0, MemRef::Read);
      break;
    case Intrinsic::vaend:
      visitMemoryReference(I, CS.getArgument(0), 0, 0,
                           MemRef::Read | MemRef::Write);
      break;

    case Intrinsic::stackrestore:
      // Stackrestore doesn't read or write memory, but it sets the
      // stack pointer, which the compiler may read from or write to
      // at any time, so check it for both readability and writeability.
      visitMemoryReference(I, CS.getArgument(0), 0, 0,
                           MemRef::Read | MemRef::Write);
      break;
    }
}

void Lint::visitCallInst(CallInst &I) {
  return visitCallSite(&I);
}

void Lint::visitInvokeInst(InvokeInst &I) {
  return visitCallSite(&I);
}

void Lint::visitReturnInst(ReturnInst &I) {
  Function *F = I.getParent()->getParent();
  Assert1(!F->doesNotReturn(),
          "Unusual: Return statement in function with noreturn attribute",
          &I);
}

// TODO: Add a length argument and check that the reference is in bounds
void Lint::visitMemoryReference(Instruction &I,
                                Value *Ptr, unsigned Align, const Type *Ty,
                                unsigned Flags) {
  Value *UnderlyingObject = Ptr->getUnderlyingObject();
  Assert1(!isa<ConstantPointerNull>(UnderlyingObject),
          "Undefined behavior: Null pointer dereference", &I);
  Assert1(!isa<UndefValue>(UnderlyingObject),
          "Undefined behavior: Undef pointer dereference", &I);

  if (Flags & MemRef::Write) {
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(UnderlyingObject))
      Assert1(!GV->isConstant(),
              "Undefined behavior: Write to read-only memory", &I);
    Assert1(!isa<Function>(UnderlyingObject) &&
            !isa<BlockAddress>(UnderlyingObject),
            "Undefined behavior: Write to text section", &I);
  }
  if (Flags & MemRef::Read) {
    Assert1(!isa<Function>(UnderlyingObject),
            "Unusual: Load from function body", &I);
    Assert1(!isa<BlockAddress>(UnderlyingObject),
            "Undefined behavior: Load from block address", &I);
  }
  if (Flags & MemRef::Callee) {
    Assert1(!isa<BlockAddress>(UnderlyingObject),
            "Undefined behavior: Call to block address", &I);
  }
  if (Flags & MemRef::Branchee) {
    Assert1(!isa<Constant>(UnderlyingObject) ||
            isa<BlockAddress>(UnderlyingObject),
            "Undefined behavior: Branch to non-blockaddress", &I);
  }

  if (TD) {
    if (Align == 0 && Ty) Align = TD->getABITypeAlignment(Ty);

    if (Align != 0) {
      unsigned BitWidth = TD->getTypeSizeInBits(Ptr->getType());
      APInt Mask = APInt::getAllOnesValue(BitWidth),
                   KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
      ComputeMaskedBits(Ptr, Mask, KnownZero, KnownOne, TD);
      Assert1(!(KnownOne & APInt::getLowBitsSet(BitWidth, Log2_32(Align))),
              "Undefined behavior: Memory reference address is misaligned", &I);
    }
  }
}

void Lint::visitLoadInst(LoadInst &I) {
  visitMemoryReference(I, I.getPointerOperand(), I.getAlignment(), I.getType(),
                       MemRef::Read);
}

void Lint::visitStoreInst(StoreInst &I) {
  visitMemoryReference(I, I.getPointerOperand(), I.getAlignment(),
                  I.getOperand(0)->getType(), MemRef::Write);
}

void Lint::visitXor(BinaryOperator &I) {
  Assert1(!isa<UndefValue>(I.getOperand(0)) ||
          !isa<UndefValue>(I.getOperand(1)),
          "Undefined result: xor(undef, undef)", &I);
}

void Lint::visitSub(BinaryOperator &I) {
  Assert1(!isa<UndefValue>(I.getOperand(0)) ||
          !isa<UndefValue>(I.getOperand(1)),
          "Undefined result: sub(undef, undef)", &I);
}

void Lint::visitLShr(BinaryOperator &I) {
  if (ConstantInt *CI =
        dyn_cast<ConstantInt>(I.getOperand(1)->stripPointerCasts()))
    Assert1(CI->getValue().ult(cast<IntegerType>(I.getType())->getBitWidth()),
            "Undefined result: Shift count out of range", &I);
}

void Lint::visitAShr(BinaryOperator &I) {
  if (ConstantInt *CI =
        dyn_cast<ConstantInt>(I.getOperand(1)->stripPointerCasts()))
    Assert1(CI->getValue().ult(cast<IntegerType>(I.getType())->getBitWidth()),
            "Undefined result: Shift count out of range", &I);
}

void Lint::visitShl(BinaryOperator &I) {
  if (ConstantInt *CI =
        dyn_cast<ConstantInt>(I.getOperand(1)->stripPointerCasts()))
    Assert1(CI->getValue().ult(cast<IntegerType>(I.getType())->getBitWidth()),
            "Undefined result: Shift count out of range", &I);
}

static bool isZero(Value *V, TargetData *TD) {
  // Assume undef could be zero.
  if (isa<UndefValue>(V)) return true;

  unsigned BitWidth = cast<IntegerType>(V->getType())->getBitWidth();
  APInt Mask = APInt::getAllOnesValue(BitWidth),
               KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  ComputeMaskedBits(V, Mask, KnownZero, KnownOne, TD);
  return KnownZero.isAllOnesValue();
}

void Lint::visitSDiv(BinaryOperator &I) {
  Assert1(!isZero(I.getOperand(1), TD),
          "Undefined behavior: Division by zero", &I);
}

void Lint::visitUDiv(BinaryOperator &I) {
  Assert1(!isZero(I.getOperand(1), TD),
          "Undefined behavior: Division by zero", &I);
}

void Lint::visitSRem(BinaryOperator &I) {
  Assert1(!isZero(I.getOperand(1), TD),
          "Undefined behavior: Division by zero", &I);
}

void Lint::visitURem(BinaryOperator &I) {
  Assert1(!isZero(I.getOperand(1), TD),
          "Undefined behavior: Division by zero", &I);
}

void Lint::visitAllocaInst(AllocaInst &I) {
  if (isa<ConstantInt>(I.getArraySize()))
    // This isn't undefined behavior, it's just an obvious pessimization.
    Assert1(&I.getParent()->getParent()->getEntryBlock() == I.getParent(),
            "Pessimization: Static alloca outside of entry block", &I);
}

void Lint::visitVAArgInst(VAArgInst &I) {
  visitMemoryReference(I, I.getOperand(0), 0, 0,
                       MemRef::Read | MemRef::Write);
}

void Lint::visitIndirectBrInst(IndirectBrInst &I) {
  visitMemoryReference(I, I.getAddress(), 0, 0, MemRef::Branchee);
}

void Lint::visitExtractElementInst(ExtractElementInst &I) {
  if (ConstantInt *CI =
        dyn_cast<ConstantInt>(I.getIndexOperand()->stripPointerCasts()))
    Assert1(CI->getValue().ult(I.getVectorOperandType()->getNumElements()),
            "Undefined result: extractelement index out of range", &I);
}

void Lint::visitInsertElementInst(InsertElementInst &I) {
  if (ConstantInt *CI =
        dyn_cast<ConstantInt>(I.getOperand(2)->stripPointerCasts()))
    Assert1(CI->getValue().ult(I.getType()->getNumElements()),
            "Undefined result: insertelement index out of range", &I);
}

void Lint::visitUnreachableInst(UnreachableInst &I) {
  // This isn't undefined behavior, it's merely suspicious.
  Assert1(&I == I.getParent()->begin() ||
          prior(BasicBlock::iterator(&I))->mayHaveSideEffects(),
          "Unusual: unreachable immediately preceded by instruction without "
          "side effects", &I);
}

//===----------------------------------------------------------------------===//
//  Implement the public interfaces to this file...
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createLintPass() {
  return new Lint();
}

/// lintFunction - Check a function for errors, printing messages on stderr.
///
void llvm::lintFunction(const Function &f) {
  Function &F = const_cast<Function&>(f);
  assert(!F.isDeclaration() && "Cannot lint external functions");

  FunctionPassManager FPM(F.getParent());
  Lint *V = new Lint();
  FPM.add(V);
  FPM.run(F);
}

/// lintModule - Check a module for errors, printing messages on stderr.
///
void llvm::lintModule(const Module &M) {
  PassManager PM;
  Lint *V = new Lint();
  PM.add(V);
  PM.run(const_cast<Module&>(M));
}
