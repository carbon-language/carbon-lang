//===- Evaluator.cpp - LLVM IR evaluator ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Function evaluator for LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Evaluator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

#define DEBUG_TYPE "evaluator"

using namespace llvm;

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL);

/// Return true if the specified constant can be handled by the code generator.
/// We don't want to generate something like:
///   void *X = &X/42;
/// because the code generator doesn't have a relocation that can handle that.
///
/// This function should be called if C was not found (but just got inserted)
/// in SimpleConstants to avoid having to rescan the same constants all the
/// time.
static bool
isSimpleEnoughValueToCommitHelper(Constant *C,
                                  SmallPtrSetImpl<Constant *> &SimpleConstants,
                                  const DataLayout &DL) {
  // Simple global addresses are supported, do not allow dllimport or
  // thread-local globals.
  if (auto *GV = dyn_cast<GlobalValue>(C))
    return !GV->hasDLLImportStorageClass() && !GV->isThreadLocal();

  // Simple integer, undef, constant aggregate zero, etc are all supported.
  if (C->getNumOperands() == 0 || isa<BlockAddress>(C))
    return true;

  // Aggregate values are safe if all their elements are.
  if (isa<ConstantAggregate>(C)) {
    for (Value *Op : C->operands())
      if (!isSimpleEnoughValueToCommit(cast<Constant>(Op), SimpleConstants, DL))
        return false;
    return true;
  }

  // We don't know exactly what relocations are allowed in constant expressions,
  // so we allow &global+constantoffset, which is safe and uniformly supported
  // across targets.
  ConstantExpr *CE = cast<ConstantExpr>(C);
  switch (CE->getOpcode()) {
  case Instruction::BitCast:
    // Bitcast is fine if the casted value is fine.
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
    // int <=> ptr is fine if the int type is the same size as the
    // pointer type.
    if (DL.getTypeSizeInBits(CE->getType()) !=
        DL.getTypeSizeInBits(CE->getOperand(0)->getType()))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  // GEP is fine if it is simple + constant offset.
  case Instruction::GetElementPtr:
    for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
      if (!isa<ConstantInt>(CE->getOperand(i)))
        return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::Add:
    // We allow simple+cst.
    if (!isa<ConstantInt>(CE->getOperand(1)))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);
  }
  return false;
}

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL) {
  // If we already checked this constant, we win.
  if (!SimpleConstants.insert(C).second)
    return true;
  // Check the constant.
  return isSimpleEnoughValueToCommitHelper(C, SimpleConstants, DL);
}

/// Return true if this constant is simple enough for us to understand.  In
/// particular, if it is a cast to anything other than from one pointer type to
/// another pointer type, we punt.  We basically just support direct accesses to
/// globals and GEP's of globals.  This should be kept up to date with
/// CommitValueTo.
static bool isSimpleEnoughPointerToCommit(Constant *C, const DataLayout &DL) {
  // Conservatively, avoid aggregate types. This is because we don't
  // want to worry about them partially overlapping other stores.
  if (!cast<PointerType>(C->getType())->getElementType()->isSingleValueType())
    return false;

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
    // Do not allow weak/*_odr/linkonce linkage or external globals.
    return GV->hasUniqueInitializer();

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    // Handle a constantexpr gep.
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<GlobalVariable>(CE->getOperand(0)) &&
        cast<GEPOperator>(CE)->isInBounds()) {
      GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
      // Do not allow weak/*_odr/linkonce/dllimport/dllexport linkage or
      // external globals.
      if (!GV->hasUniqueInitializer())
        return false;

      // The first index must be zero.
      ConstantInt *CI = dyn_cast<ConstantInt>(*std::next(CE->op_begin()));
      if (!CI || !CI->isZero()) return false;

      // The remaining indices must be compile-time known integers within the
      // notional bounds of the corresponding static array types.
      if (!CE->isGEPWithNoNotionalOverIndexing())
        return false;

      return ConstantFoldLoadThroughGEPConstantExpr(
          GV->getInitializer(), CE,
          cast<GEPOperator>(CE)->getResultElementType(), DL);
    } else if (CE->getOpcode() == Instruction::BitCast &&
               isa<GlobalVariable>(CE->getOperand(0))) {
      // A constantexpr bitcast from a pointer to another pointer is a no-op,
      // and we know how to evaluate it by moving the bitcast from the pointer
      // operand to the value operand.
      // Do not allow weak/*_odr/linkonce/dllimport/dllexport linkage or
      // external globals.
      return cast<GlobalVariable>(CE->getOperand(0))->hasUniqueInitializer();
    }
  }

  return false;
}

/// Apply \p TryLoad to Ptr. If this returns \p nullptr, introspect the
/// pointer's type and walk down through the initial elements to obtain
/// additional pointers to try. Returns the first non-null return value from
/// \p TryLoad, or \p nullptr if the type can't be introspected further.
static Constant *
evaluateBitcastFromPtr(Constant *Ptr, const DataLayout &DL,
                       const TargetLibraryInfo *TLI,
                       std::function<Constant *(Constant *)> TryLoad) {
  Constant *Val;
  while (!(Val = TryLoad(Ptr))) {
    // If Ty is a non-opaque struct, we can convert the pointer to the struct
    // into a pointer to its first member.
    // FIXME: This could be extended to support arrays as well.
    Type *Ty = cast<PointerType>(Ptr->getType())->getElementType();
    if (!isa<StructType>(Ty) || cast<StructType>(Ty)->isOpaque())
      break;

    IntegerType *IdxTy = IntegerType::get(Ty->getContext(), 32);
    Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
    Constant *const IdxList[] = {IdxZero, IdxZero};

    Ptr = ConstantExpr::getGetElementPtr(Ty, Ptr, IdxList);
    Ptr = ConstantFoldConstant(Ptr, DL, TLI);
  }
  return Val;
}

static Constant *getInitializer(Constant *C) {
  auto *GV = dyn_cast<GlobalVariable>(C);
  return GV && GV->hasDefinitiveInitializer() ? GV->getInitializer() : nullptr;
}

/// Return the value that would be computed by a load from P after the stores
/// reflected by 'memory' have been performed.  If we can't decide, return null.
Constant *Evaluator::ComputeLoadResult(Constant *P, Type *Ty) {
  // If this memory location has been recently stored, use the stored value: it
  // is the most up-to-date.
  auto TryFindMemLoc = [this](Constant *Ptr) {
    return MutatedMemory.lookup(Ptr);
  };

  if (Constant *Val = TryFindMemLoc(P))
    return Val;

  // Access it.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(P)) {
    if (GV->hasDefinitiveInitializer())
      return GV->getInitializer();
    return nullptr;
  }

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(P)) {
    switch (CE->getOpcode()) {
    // Handle a constantexpr getelementptr.
    case Instruction::GetElementPtr:
      if (auto *I = getInitializer(CE->getOperand(0)))
        return ConstantFoldLoadThroughGEPConstantExpr(I, CE, Ty, DL);
      break;
    // Handle a constantexpr bitcast.
    case Instruction::BitCast:
      // We're evaluating a load through a pointer that was bitcast to a
      // different type. See if the "from" pointer has recently been stored.
      // If it hasn't, we may still be able to find a stored pointer by
      // introspecting the type.
      Constant *Val =
          evaluateBitcastFromPtr(CE->getOperand(0), DL, TLI, TryFindMemLoc);
      if (!Val)
        Val = getInitializer(CE->getOperand(0));
      if (Val)
        return ConstantFoldLoadThroughBitcast(
            Val, P->getType()->getPointerElementType(), DL);
      break;
    }
  }

  return nullptr;  // don't know how to evaluate.
}

static Function *getFunction(Constant *C) {
  if (auto *Fn = dyn_cast<Function>(C))
    return Fn;

  if (auto *Alias = dyn_cast<GlobalAlias>(C))
    if (auto *Fn = dyn_cast<Function>(Alias->getAliasee()))
      return Fn;
  return nullptr;
}

Function *
Evaluator::getCalleeWithFormalArgs(CallBase &CB,
                                   SmallVectorImpl<Constant *> &Formals) {
  auto *V = CB.getCalledOperand();
  if (auto *Fn = getFunction(getVal(V)))
    return getFormalParams(CB, Fn, Formals) ? Fn : nullptr;

  auto *CE = dyn_cast<ConstantExpr>(V);
  if (!CE || CE->getOpcode() != Instruction::BitCast ||
      !getFormalParams(CB, getFunction(CE->getOperand(0)), Formals))
    return nullptr;

  return dyn_cast<Function>(
      ConstantFoldLoadThroughBitcast(CE, CE->getOperand(0)->getType(), DL));
}

bool Evaluator::getFormalParams(CallBase &CB, Function *F,
                                SmallVectorImpl<Constant *> &Formals) {
  if (!F)
    return false;

  auto *FTy = F->getFunctionType();
  if (FTy->getNumParams() > CB.getNumArgOperands()) {
    LLVM_DEBUG(dbgs() << "Too few arguments for function.\n");
    return false;
  }

  auto ArgI = CB.arg_begin();
  for (auto ParI = FTy->param_begin(), ParE = FTy->param_end(); ParI != ParE;
       ++ParI) {
    auto *ArgC = ConstantFoldLoadThroughBitcast(getVal(*ArgI), *ParI, DL);
    if (!ArgC) {
      LLVM_DEBUG(dbgs() << "Can not convert function argument.\n");
      return false;
    }
    Formals.push_back(ArgC);
    ++ArgI;
  }
  return true;
}

/// If call expression contains bitcast then we may need to cast
/// evaluated return value to a type of the call expression.
Constant *Evaluator::castCallResultIfNeeded(Value *CallExpr, Constant *RV) {
  ConstantExpr *CE = dyn_cast<ConstantExpr>(CallExpr);
  if (!RV || !CE || CE->getOpcode() != Instruction::BitCast)
    return RV;

  if (auto *FT =
          dyn_cast<FunctionType>(CE->getType()->getPointerElementType())) {
    RV = ConstantFoldLoadThroughBitcast(RV, FT->getReturnType(), DL);
    if (!RV)
      LLVM_DEBUG(dbgs() << "Failed to fold bitcast call expr\n");
  }
  return RV;
}

/// Evaluate all instructions in block BB, returning true if successful, false
/// if we can't evaluate it.  NewBB returns the next BB that control flows into,
/// or null upon return. StrippedPointerCastsForAliasAnalysis is set to true if
/// we looked through pointer casts to evaluate something.
bool Evaluator::EvaluateBlock(BasicBlock::iterator CurInst, BasicBlock *&NextBB,
                              bool &StrippedPointerCastsForAliasAnalysis) {
  // This is the main evaluation loop.
  while (true) {
    Constant *InstResult = nullptr;

    LLVM_DEBUG(dbgs() << "Evaluating Instruction: " << *CurInst << "\n");

    if (StoreInst *SI = dyn_cast<StoreInst>(CurInst)) {
      if (!SI->isSimple()) {
        LLVM_DEBUG(dbgs() << "Store is not simple! Can not evaluate.\n");
        return false;  // no volatile/atomic accesses.
      }
      Constant *Ptr = getVal(SI->getOperand(1));
      Constant *FoldedPtr = ConstantFoldConstant(Ptr, DL, TLI);
      if (Ptr != FoldedPtr) {
        LLVM_DEBUG(dbgs() << "Folding constant ptr expression: " << *Ptr);
        Ptr = FoldedPtr;
        LLVM_DEBUG(dbgs() << "; To: " << *Ptr << "\n");
      }
      if (!isSimpleEnoughPointerToCommit(Ptr, DL)) {
        // If this is too complex for us to commit, reject it.
        LLVM_DEBUG(
            dbgs() << "Pointer is too complex for us to evaluate store.");
        return false;
      }

      Constant *Val = getVal(SI->getOperand(0));

      // If this might be too difficult for the backend to handle (e.g. the addr
      // of one global variable divided by another) then we can't commit it.
      if (!isSimpleEnoughValueToCommit(Val, SimpleConstants, DL)) {
        LLVM_DEBUG(dbgs() << "Store value is too complex to evaluate store. "
                          << *Val << "\n");
        return false;
      }

      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
        if (CE->getOpcode() == Instruction::BitCast) {
          LLVM_DEBUG(dbgs()
                     << "Attempting to resolve bitcast on constant ptr.\n");
          // If we're evaluating a store through a bitcast, then we need
          // to pull the bitcast off the pointer type and push it onto the
          // stored value. In order to push the bitcast onto the stored value,
          // a bitcast from the pointer's element type to Val's type must be
          // legal. If it's not, we can try introspecting the type to find a
          // legal conversion.

          auto TryCastValTy = [&](Constant *P) -> Constant * {
            // The conversion is illegal if the store is wider than the
            // pointee proposed by `evaluateBitcastFromPtr`, since that would
            // drop stores to other struct elements when the caller attempts to
            // look through a struct's 0th element.
            Type *NewTy = cast<PointerType>(P->getType())->getElementType();
            Type *STy = Val->getType();
            if (DL.getTypeSizeInBits(NewTy) < DL.getTypeSizeInBits(STy))
              return nullptr;

            if (Constant *FV = ConstantFoldLoadThroughBitcast(Val, NewTy, DL)) {
              Ptr = P;
              return FV;
            }
            return nullptr;
          };

          Constant *NewVal =
              evaluateBitcastFromPtr(CE->getOperand(0), DL, TLI, TryCastValTy);
          if (!NewVal) {
            LLVM_DEBUG(dbgs() << "Failed to bitcast constant ptr, can not "
                                 "evaluate.\n");
            return false;
          }

          Val = NewVal;
          LLVM_DEBUG(dbgs() << "Evaluated bitcast: " << *Val << "\n");
        }
      }

      MutatedMemory[Ptr] = Val;
    } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(CurInst)) {
      InstResult = ConstantExpr::get(BO->getOpcode(),
                                     getVal(BO->getOperand(0)),
                                     getVal(BO->getOperand(1)));
      LLVM_DEBUG(dbgs() << "Found a BinaryOperator! Simplifying: "
                        << *InstResult << "\n");
    } else if (CmpInst *CI = dyn_cast<CmpInst>(CurInst)) {
      InstResult = ConstantExpr::getCompare(CI->getPredicate(),
                                            getVal(CI->getOperand(0)),
                                            getVal(CI->getOperand(1)));
      LLVM_DEBUG(dbgs() << "Found a CmpInst! Simplifying: " << *InstResult
                        << "\n");
    } else if (CastInst *CI = dyn_cast<CastInst>(CurInst)) {
      InstResult = ConstantExpr::getCast(CI->getOpcode(),
                                         getVal(CI->getOperand(0)),
                                         CI->getType());
      LLVM_DEBUG(dbgs() << "Found a Cast! Simplifying: " << *InstResult
                        << "\n");
    } else if (SelectInst *SI = dyn_cast<SelectInst>(CurInst)) {
      InstResult = ConstantExpr::getSelect(getVal(SI->getOperand(0)),
                                           getVal(SI->getOperand(1)),
                                           getVal(SI->getOperand(2)));
      LLVM_DEBUG(dbgs() << "Found a Select! Simplifying: " << *InstResult
                        << "\n");
    } else if (auto *EVI = dyn_cast<ExtractValueInst>(CurInst)) {
      InstResult = ConstantExpr::getExtractValue(
          getVal(EVI->getAggregateOperand()), EVI->getIndices());
      LLVM_DEBUG(dbgs() << "Found an ExtractValueInst! Simplifying: "
                        << *InstResult << "\n");
    } else if (auto *IVI = dyn_cast<InsertValueInst>(CurInst)) {
      InstResult = ConstantExpr::getInsertValue(
          getVal(IVI->getAggregateOperand()),
          getVal(IVI->getInsertedValueOperand()), IVI->getIndices());
      LLVM_DEBUG(dbgs() << "Found an InsertValueInst! Simplifying: "
                        << *InstResult << "\n");
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurInst)) {
      Constant *P = getVal(GEP->getOperand(0));
      SmallVector<Constant*, 8> GEPOps;
      for (Use &Op : llvm::drop_begin(GEP->operands()))
        GEPOps.push_back(getVal(Op));
      InstResult =
          ConstantExpr::getGetElementPtr(GEP->getSourceElementType(), P, GEPOps,
                                         cast<GEPOperator>(GEP)->isInBounds());
      LLVM_DEBUG(dbgs() << "Found a GEP! Simplifying: " << *InstResult << "\n");
    } else if (LoadInst *LI = dyn_cast<LoadInst>(CurInst)) {
      if (!LI->isSimple()) {
        LLVM_DEBUG(
            dbgs() << "Found a Load! Not a simple load, can not evaluate.\n");
        return false;  // no volatile/atomic accesses.
      }

      Constant *Ptr = getVal(LI->getOperand(0));
      Constant *FoldedPtr = ConstantFoldConstant(Ptr, DL, TLI);
      if (Ptr != FoldedPtr) {
        Ptr = FoldedPtr;
        LLVM_DEBUG(dbgs() << "Found a constant pointer expression, constant "
                             "folding: "
                          << *Ptr << "\n");
      }
      InstResult = ComputeLoadResult(Ptr, LI->getType());
      if (!InstResult) {
        LLVM_DEBUG(
            dbgs() << "Failed to compute load result. Can not evaluate load."
                      "\n");
        return false; // Could not evaluate load.
      }

      LLVM_DEBUG(dbgs() << "Evaluated load: " << *InstResult << "\n");
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(CurInst)) {
      if (AI->isArrayAllocation()) {
        LLVM_DEBUG(dbgs() << "Found an array alloca. Can not evaluate.\n");
        return false;  // Cannot handle array allocs.
      }
      Type *Ty = AI->getAllocatedType();
      AllocaTmps.push_back(std::make_unique<GlobalVariable>(
          Ty, false, GlobalValue::InternalLinkage, UndefValue::get(Ty),
          AI->getName(), /*TLMode=*/GlobalValue::NotThreadLocal,
          AI->getType()->getPointerAddressSpace()));
      InstResult = AllocaTmps.back().get();
      LLVM_DEBUG(dbgs() << "Found an alloca. Result: " << *InstResult << "\n");
    } else if (isa<CallInst>(CurInst) || isa<InvokeInst>(CurInst)) {
      CallBase &CB = *cast<CallBase>(&*CurInst);

      // Debug info can safely be ignored here.
      if (isa<DbgInfoIntrinsic>(CB)) {
        LLVM_DEBUG(dbgs() << "Ignoring debug info.\n");
        ++CurInst;
        continue;
      }

      // Cannot handle inline asm.
      if (CB.isInlineAsm()) {
        LLVM_DEBUG(dbgs() << "Found inline asm, can not evaluate.\n");
        return false;
      }

      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CB)) {
        if (MemSetInst *MSI = dyn_cast<MemSetInst>(II)) {
          if (MSI->isVolatile()) {
            LLVM_DEBUG(dbgs() << "Can not optimize a volatile memset "
                              << "intrinsic.\n");
            return false;
          }
          Constant *Ptr = getVal(MSI->getDest());
          Constant *Val = getVal(MSI->getValue());
          Constant *DestVal =
              ComputeLoadResult(getVal(Ptr), MSI->getValue()->getType());
          if (Val->isNullValue() && DestVal && DestVal->isNullValue()) {
            // This memset is a no-op.
            LLVM_DEBUG(dbgs() << "Ignoring no-op memset.\n");
            ++CurInst;
            continue;
          }
        }

        if (II->isLifetimeStartOrEnd()) {
          LLVM_DEBUG(dbgs() << "Ignoring lifetime intrinsic.\n");
          ++CurInst;
          continue;
        }

        if (II->getIntrinsicID() == Intrinsic::invariant_start) {
          // We don't insert an entry into Values, as it doesn't have a
          // meaningful return value.
          if (!II->use_empty()) {
            LLVM_DEBUG(dbgs()
                       << "Found unused invariant_start. Can't evaluate.\n");
            return false;
          }
          ConstantInt *Size = cast<ConstantInt>(II->getArgOperand(0));
          Value *PtrArg = getVal(II->getArgOperand(1));
          Value *Ptr = PtrArg->stripPointerCasts();
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
            Type *ElemTy = GV->getValueType();
            if (!Size->isMinusOne() &&
                Size->getValue().getLimitedValue() >=
                    DL.getTypeStoreSize(ElemTy)) {
              Invariants.insert(GV);
              LLVM_DEBUG(dbgs() << "Found a global var that is an invariant: "
                                << *GV << "\n");
            } else {
              LLVM_DEBUG(dbgs()
                         << "Found a global var, but can not treat it as an "
                            "invariant.\n");
            }
          }
          // Continue even if we do nothing.
          ++CurInst;
          continue;
        } else if (II->getIntrinsicID() == Intrinsic::assume) {
          LLVM_DEBUG(dbgs() << "Skipping assume intrinsic.\n");
          ++CurInst;
          continue;
        } else if (II->getIntrinsicID() == Intrinsic::sideeffect) {
          LLVM_DEBUG(dbgs() << "Skipping sideeffect intrinsic.\n");
          ++CurInst;
          continue;
        } else if (II->getIntrinsicID() == Intrinsic::pseudoprobe) {
          LLVM_DEBUG(dbgs() << "Skipping pseudoprobe intrinsic.\n");
          ++CurInst;
          continue;
        } else {
          Value *Stripped = CurInst->stripPointerCastsForAliasAnalysis();
          // Only attempt to getVal() if we've actually managed to strip
          // anything away, or else we'll call getVal() on the current
          // instruction.
          if (Stripped != &*CurInst) {
            InstResult = getVal(Stripped);
          }
          if (InstResult) {
            LLVM_DEBUG(dbgs()
                       << "Stripped pointer casts for alias analysis for "
                          "intrinsic call.\n");
            StrippedPointerCastsForAliasAnalysis = true;
            InstResult = ConstantExpr::getBitCast(InstResult, II->getType());
          } else {
            LLVM_DEBUG(dbgs() << "Unknown intrinsic. Cannot evaluate.\n");
            return false;
          }
        }
      }

      if (!InstResult) {
        // Resolve function pointers.
        SmallVector<Constant *, 8> Formals;
        Function *Callee = getCalleeWithFormalArgs(CB, Formals);
        if (!Callee || Callee->isInterposable()) {
          LLVM_DEBUG(dbgs() << "Can not resolve function pointer.\n");
          return false; // Cannot resolve.
        }

        if (Callee->isDeclaration()) {
          // If this is a function we can constant fold, do it.
          if (Constant *C = ConstantFoldCall(&CB, Callee, Formals, TLI)) {
            InstResult = castCallResultIfNeeded(CB.getCalledOperand(), C);
            if (!InstResult)
              return false;
            LLVM_DEBUG(dbgs() << "Constant folded function call. Result: "
                              << *InstResult << "\n");
          } else {
            LLVM_DEBUG(dbgs() << "Can not constant fold function call.\n");
            return false;
          }
        } else {
          if (Callee->getFunctionType()->isVarArg()) {
            LLVM_DEBUG(dbgs()
                       << "Can not constant fold vararg function call.\n");
            return false;
          }

          Constant *RetVal = nullptr;
          // Execute the call, if successful, use the return value.
          ValueStack.emplace_back();
          if (!EvaluateFunction(Callee, RetVal, Formals)) {
            LLVM_DEBUG(dbgs() << "Failed to evaluate function.\n");
            return false;
          }
          ValueStack.pop_back();
          InstResult = castCallResultIfNeeded(CB.getCalledOperand(), RetVal);
          if (RetVal && !InstResult)
            return false;

          if (InstResult) {
            LLVM_DEBUG(dbgs() << "Successfully evaluated function. Result: "
                              << *InstResult << "\n\n");
          } else {
            LLVM_DEBUG(dbgs()
                       << "Successfully evaluated function. Result: 0\n\n");
          }
        }
      }
    } else if (CurInst->isTerminator()) {
      LLVM_DEBUG(dbgs() << "Found a terminator instruction.\n");

      if (BranchInst *BI = dyn_cast<BranchInst>(CurInst)) {
        if (BI->isUnconditional()) {
          NextBB = BI->getSuccessor(0);
        } else {
          ConstantInt *Cond =
            dyn_cast<ConstantInt>(getVal(BI->getCondition()));
          if (!Cond) return false;  // Cannot determine.

          NextBB = BI->getSuccessor(!Cond->getZExtValue());
        }
      } else if (SwitchInst *SI = dyn_cast<SwitchInst>(CurInst)) {
        ConstantInt *Val =
          dyn_cast<ConstantInt>(getVal(SI->getCondition()));
        if (!Val) return false;  // Cannot determine.
        NextBB = SI->findCaseValue(Val)->getCaseSuccessor();
      } else if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(CurInst)) {
        Value *Val = getVal(IBI->getAddress())->stripPointerCasts();
        if (BlockAddress *BA = dyn_cast<BlockAddress>(Val))
          NextBB = BA->getBasicBlock();
        else
          return false;  // Cannot determine.
      } else if (isa<ReturnInst>(CurInst)) {
        NextBB = nullptr;
      } else {
        // invoke, unwind, resume, unreachable.
        LLVM_DEBUG(dbgs() << "Can not handle terminator.");
        return false;  // Cannot handle this terminator.
      }

      // We succeeded at evaluating this block!
      LLVM_DEBUG(dbgs() << "Successfully evaluated block.\n");
      return true;
    } else {
      // Did not know how to evaluate this!
      LLVM_DEBUG(
          dbgs() << "Failed to evaluate block due to unhandled instruction."
                    "\n");
      return false;
    }

    if (!CurInst->use_empty()) {
      InstResult = ConstantFoldConstant(InstResult, DL, TLI);
      setVal(&*CurInst, InstResult);
    }

    // If we just processed an invoke, we finished evaluating the block.
    if (InvokeInst *II = dyn_cast<InvokeInst>(CurInst)) {
      NextBB = II->getNormalDest();
      LLVM_DEBUG(dbgs() << "Found an invoke instruction. Finished Block.\n\n");
      return true;
    }

    // Advance program counter.
    ++CurInst;
  }
}

/// Evaluate a call to function F, returning true if successful, false if we
/// can't evaluate it.  ActualArgs contains the formal arguments for the
/// function.
bool Evaluator::EvaluateFunction(Function *F, Constant *&RetVal,
                                 const SmallVectorImpl<Constant*> &ActualArgs) {
  // Check to see if this function is already executing (recursion).  If so,
  // bail out.  TODO: we might want to accept limited recursion.
  if (is_contained(CallStack, F))
    return false;

  CallStack.push_back(F);

  // Initialize arguments to the incoming values specified.
  unsigned ArgNo = 0;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end(); AI != E;
       ++AI, ++ArgNo)
    setVal(&*AI, ActualArgs[ArgNo]);

  // ExecutedBlocks - We only handle non-looping, non-recursive code.  As such,
  // we can only evaluate any one basic block at most once.  This set keeps
  // track of what we have executed so we can detect recursive cases etc.
  SmallPtrSet<BasicBlock*, 32> ExecutedBlocks;

  // CurBB - The current basic block we're evaluating.
  BasicBlock *CurBB = &F->front();

  BasicBlock::iterator CurInst = CurBB->begin();

  while (true) {
    BasicBlock *NextBB = nullptr; // Initialized to avoid compiler warnings.
    LLVM_DEBUG(dbgs() << "Trying to evaluate BB: " << *CurBB << "\n");

    bool StrippedPointerCastsForAliasAnalysis = false;

    if (!EvaluateBlock(CurInst, NextBB, StrippedPointerCastsForAliasAnalysis))
      return false;

    if (!NextBB) {
      // Successfully running until there's no next block means that we found
      // the return.  Fill it the return value and pop the call stack.
      ReturnInst *RI = cast<ReturnInst>(CurBB->getTerminator());
      if (RI->getNumOperands()) {
        // The Evaluator can look through pointer casts as long as alias
        // analysis holds because it's just a simple interpreter and doesn't
        // skip memory accesses due to invariant group metadata, but we can't
        // let users of Evaluator use a value that's been gleaned looking
        // through stripping pointer casts.
        if (StrippedPointerCastsForAliasAnalysis &&
            !RI->getReturnValue()->getType()->isVoidTy()) {
          return false;
        }
        RetVal = getVal(RI->getOperand(0));
      }
      CallStack.pop_back();
      return true;
    }

    // Okay, we succeeded in evaluating this control flow.  See if we have
    // executed the new block before.  If so, we have a looping function,
    // which we cannot evaluate in reasonable time.
    if (!ExecutedBlocks.insert(NextBB).second)
      return false;  // looped!

    // Okay, we have never been in this block before.  Check to see if there
    // are any PHI nodes.  If so, evaluate them with information about where
    // we came from.
    PHINode *PN = nullptr;
    for (CurInst = NextBB->begin();
         (PN = dyn_cast<PHINode>(CurInst)); ++CurInst)
      setVal(PN, getVal(PN->getIncomingValueForBlock(CurBB)));

    // Advance to the next block.
    CurBB = NextBB;
  }
}
