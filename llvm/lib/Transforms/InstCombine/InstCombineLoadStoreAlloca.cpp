//===- InstCombineLoadStoreAlloca.cpp -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for load, store and alloca.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "instcombine"

STATISTIC(NumDeadStore,    "Number of dead stores eliminated");
STATISTIC(NumGlobalCopies, "Number of allocas copied from constant global");

/// pointsToConstantGlobal - Return true if V (possibly indirectly) points to
/// some part of a constant global variable.  This intentionally only accepts
/// constant expressions because we can't rewrite arbitrary instructions.
static bool pointsToConstantGlobal(Value *V) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return GV->isConstant();

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::BitCast ||
        CE->getOpcode() == Instruction::AddrSpaceCast ||
        CE->getOpcode() == Instruction::GetElementPtr)
      return pointsToConstantGlobal(CE->getOperand(0));
  }
  return false;
}

/// isOnlyCopiedFromConstantGlobal - Recursively walk the uses of a (derived)
/// pointer to an alloca.  Ignore any reads of the pointer, return false if we
/// see any stores or other unknown uses.  If we see pointer arithmetic, keep
/// track of whether it moves the pointer (with IsOffset) but otherwise traverse
/// the uses.  If we see a memcpy/memmove that targets an unoffseted pointer to
/// the alloca, and if the source pointer is a pointer to a constant global, we
/// can optimize this.
static bool
isOnlyCopiedFromConstantGlobal(Value *V, MemTransferInst *&TheCopy,
                               SmallVectorImpl<Instruction *> &ToDelete) {
  // We track lifetime intrinsics as we encounter them.  If we decide to go
  // ahead and replace the value with the global, this lets the caller quickly
  // eliminate the markers.

  SmallVector<std::pair<Value *, bool>, 35> ValuesToInspect;
  ValuesToInspect.push_back(std::make_pair(V, false));
  while (!ValuesToInspect.empty()) {
    auto ValuePair = ValuesToInspect.pop_back_val();
    const bool IsOffset = ValuePair.second;
    for (auto &U : ValuePair.first->uses()) {
      Instruction *I = cast<Instruction>(U.getUser());

      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        // Ignore non-volatile loads, they are always ok.
        if (!LI->isSimple()) return false;
        continue;
      }

      if (isa<BitCastInst>(I) || isa<AddrSpaceCastInst>(I)) {
        // If uses of the bitcast are ok, we are ok.
        ValuesToInspect.push_back(std::make_pair(I, IsOffset));
        continue;
      }
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
        // If the GEP has all zero indices, it doesn't offset the pointer. If it
        // doesn't, it does.
        ValuesToInspect.push_back(
            std::make_pair(I, IsOffset || !GEP->hasAllZeroIndices()));
        continue;
      }

      if (CallSite CS = I) {
        // If this is the function being called then we treat it like a load and
        // ignore it.
        if (CS.isCallee(&U))
          continue;

        // Inalloca arguments are clobbered by the call.
        unsigned ArgNo = CS.getArgumentNo(&U);
        if (CS.isInAllocaArgument(ArgNo))
          return false;

        // If this is a readonly/readnone call site, then we know it is just a
        // load (but one that potentially returns the value itself), so we can
        // ignore it if we know that the value isn't captured.
        if (CS.onlyReadsMemory() &&
            (CS.getInstruction()->use_empty() || CS.doesNotCapture(ArgNo)))
          continue;

        // If this is being passed as a byval argument, the caller is making a
        // copy, so it is only a read of the alloca.
        if (CS.isByValArgument(ArgNo))
          continue;
      }

      // Lifetime intrinsics can be handled by the caller.
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end) {
          assert(II->use_empty() && "Lifetime markers have no result to use!");
          ToDelete.push_back(II);
          continue;
        }
      }

      // If this is isn't our memcpy/memmove, reject it as something we can't
      // handle.
      MemTransferInst *MI = dyn_cast<MemTransferInst>(I);
      if (!MI)
        return false;

      // If the transfer is using the alloca as a source of the transfer, then
      // ignore it since it is a load (unless the transfer is volatile).
      if (U.getOperandNo() == 1) {
        if (MI->isVolatile()) return false;
        continue;
      }

      // If we already have seen a copy, reject the second one.
      if (TheCopy) return false;

      // If the pointer has been offset from the start of the alloca, we can't
      // safely handle this.
      if (IsOffset) return false;

      // If the memintrinsic isn't using the alloca as the dest, reject it.
      if (U.getOperandNo() != 0) return false;

      // If the source of the memcpy/move is not a constant global, reject it.
      if (!pointsToConstantGlobal(MI->getSource()))
        return false;

      // Otherwise, the transform is safe.  Remember the copy instruction.
      TheCopy = MI;
    }
  }
  return true;
}

/// isOnlyCopiedFromConstantGlobal - Return true if the specified alloca is only
/// modified by a copy from a constant global.  If we can prove this, we can
/// replace any uses of the alloca with uses of the global directly.
static MemTransferInst *
isOnlyCopiedFromConstantGlobal(AllocaInst *AI,
                               SmallVectorImpl<Instruction *> &ToDelete) {
  MemTransferInst *TheCopy = nullptr;
  if (isOnlyCopiedFromConstantGlobal(AI, TheCopy, ToDelete))
    return TheCopy;
  return nullptr;
}

Instruction *InstCombiner::visitAllocaInst(AllocaInst &AI) {
  // Ensure that the alloca array size argument has type intptr_t, so that
  // any casting is exposed early.
  if (DL) {
    Type *IntPtrTy = DL->getIntPtrType(AI.getType());
    if (AI.getArraySize()->getType() != IntPtrTy) {
      Value *V = Builder->CreateIntCast(AI.getArraySize(),
                                        IntPtrTy, false);
      AI.setOperand(0, V);
      return &AI;
    }
  }

  // Convert: alloca Ty, C - where C is a constant != 1 into: alloca [C x Ty], 1
  if (AI.isArrayAllocation()) {  // Check C != 1
    if (const ConstantInt *C = dyn_cast<ConstantInt>(AI.getArraySize())) {
      Type *NewTy =
        ArrayType::get(AI.getAllocatedType(), C->getZExtValue());
      AllocaInst *New = Builder->CreateAlloca(NewTy, nullptr, AI.getName());
      New->setAlignment(AI.getAlignment());

      // Scan to the end of the allocation instructions, to skip over a block of
      // allocas if possible...also skip interleaved debug info
      //
      BasicBlock::iterator It = New;
      while (isa<AllocaInst>(*It) || isa<DbgInfoIntrinsic>(*It)) ++It;

      // Now that I is pointing to the first non-allocation-inst in the block,
      // insert our getelementptr instruction...
      //
      Type *IdxTy = DL
                  ? DL->getIntPtrType(AI.getType())
                  : Type::getInt64Ty(AI.getContext());
      Value *NullIdx = Constant::getNullValue(IdxTy);
      Value *Idx[2] = { NullIdx, NullIdx };
      Instruction *GEP =
        GetElementPtrInst::CreateInBounds(New, Idx, New->getName() + ".sub");
      InsertNewInstBefore(GEP, *It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, GEP);
    } else if (isa<UndefValue>(AI.getArraySize())) {
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));
    }
  }

  if (DL && AI.getAllocatedType()->isSized()) {
    // If the alignment is 0 (unspecified), assign it the preferred alignment.
    if (AI.getAlignment() == 0)
      AI.setAlignment(DL->getPrefTypeAlignment(AI.getAllocatedType()));

    // Move all alloca's of zero byte objects to the entry block and merge them
    // together.  Note that we only do this for alloca's, because malloc should
    // allocate and return a unique pointer, even for a zero byte allocation.
    if (DL->getTypeAllocSize(AI.getAllocatedType()) == 0) {
      // For a zero sized alloca there is no point in doing an array allocation.
      // This is helpful if the array size is a complicated expression not used
      // elsewhere.
      if (AI.isArrayAllocation()) {
        AI.setOperand(0, ConstantInt::get(AI.getArraySize()->getType(), 1));
        return &AI;
      }

      // Get the first instruction in the entry block.
      BasicBlock &EntryBlock = AI.getParent()->getParent()->getEntryBlock();
      Instruction *FirstInst = EntryBlock.getFirstNonPHIOrDbg();
      if (FirstInst != &AI) {
        // If the entry block doesn't start with a zero-size alloca then move
        // this one to the start of the entry block.  There is no problem with
        // dominance as the array size was forced to a constant earlier already.
        AllocaInst *EntryAI = dyn_cast<AllocaInst>(FirstInst);
        if (!EntryAI || !EntryAI->getAllocatedType()->isSized() ||
            DL->getTypeAllocSize(EntryAI->getAllocatedType()) != 0) {
          AI.moveBefore(FirstInst);
          return &AI;
        }

        // If the alignment of the entry block alloca is 0 (unspecified),
        // assign it the preferred alignment.
        if (EntryAI->getAlignment() == 0)
          EntryAI->setAlignment(
            DL->getPrefTypeAlignment(EntryAI->getAllocatedType()));
        // Replace this zero-sized alloca with the one at the start of the entry
        // block after ensuring that the address will be aligned enough for both
        // types.
        unsigned MaxAlign = std::max(EntryAI->getAlignment(),
                                     AI.getAlignment());
        EntryAI->setAlignment(MaxAlign);
        if (AI.getType() != EntryAI->getType())
          return new BitCastInst(EntryAI, AI.getType());
        return ReplaceInstUsesWith(AI, EntryAI);
      }
    }
  }

  if (AI.getAlignment()) {
    // Check to see if this allocation is only modified by a memcpy/memmove from
    // a constant global whose alignment is equal to or exceeds that of the
    // allocation.  If this is the case, we can change all users to use
    // the constant global instead.  This is commonly produced by the CFE by
    // constructs like "void foo() { int A[] = {1,2,3,4,5,6,7,8,9...}; }" if 'A'
    // is only subsequently read.
    SmallVector<Instruction *, 4> ToDelete;
    if (MemTransferInst *Copy = isOnlyCopiedFromConstantGlobal(&AI, ToDelete)) {
      unsigned SourceAlign = getOrEnforceKnownAlignment(Copy->getSource(),
                                                        AI.getAlignment(),
                                                        DL, AT, &AI, DT);
      if (AI.getAlignment() <= SourceAlign) {
        DEBUG(dbgs() << "Found alloca equal to global: " << AI << '\n');
        DEBUG(dbgs() << "  memcpy = " << *Copy << '\n');
        for (unsigned i = 0, e = ToDelete.size(); i != e; ++i)
          EraseInstFromFunction(*ToDelete[i]);
        Constant *TheSrc = cast<Constant>(Copy->getSource());
        Constant *Cast
          = ConstantExpr::getPointerBitCastOrAddrSpaceCast(TheSrc, AI.getType());
        Instruction *NewI = ReplaceInstUsesWith(AI, Cast);
        EraseInstFromFunction(*Copy);
        ++NumGlobalCopies;
        return NewI;
      }
    }
  }

  // At last, use the generic allocation site handler to aggressively remove
  // unused allocas.
  return visitAllocSite(AI);
}

/// \brief Helper to combine a load to a new type.
///
/// This just does the work of combining a load to a new type. It handles
/// metadata, etc., and returns the new instruction. The \c NewTy should be the
/// loaded *value* type. This will convert it to a pointer, cast the operand to
/// that pointer type, load it, etc.
///
/// Note that this will create all of the instructions with whatever insert
/// point the \c InstCombiner currently is using.
static LoadInst *combineLoadToNewType(InstCombiner &IC, LoadInst &LI, Type *NewTy) {
  Value *Ptr = LI.getPointerOperand();
  unsigned AS = LI.getPointerAddressSpace();
  SmallVector<std::pair<unsigned, MDNode *>, 8> MD;
  LI.getAllMetadata(MD);

  LoadInst *NewLoad = IC.Builder->CreateAlignedLoad(
      IC.Builder->CreateBitCast(Ptr, NewTy->getPointerTo(AS)),
      LI.getAlignment(), LI.getName());
  for (const auto &MDPair : MD) {
    unsigned ID = MDPair.first;
    MDNode *N = MDPair.second;
    // Note, essentially every kind of metadata should be preserved here! This
    // routine is supposed to clone a load instruction changing *only its type*.
    // The only metadata it makes sense to drop is metadata which is invalidated
    // when the pointer type changes. This should essentially never be the case
    // in LLVM, but we explicitly switch over only known metadata to be
    // conservatively correct. If you are adding metadata to LLVM which pertains
    // to loads, you almost certainly want to add it here.
    switch (ID) {
    case LLVMContext::MD_dbg:
    case LLVMContext::MD_tbaa:
    case LLVMContext::MD_prof:
    case LLVMContext::MD_fpmath:
    case LLVMContext::MD_tbaa_struct:
    case LLVMContext::MD_invariant_load:
    case LLVMContext::MD_alias_scope:
    case LLVMContext::MD_noalias:
    case LLVMContext::MD_nontemporal:
    case LLVMContext::MD_mem_parallel_loop_access:
    case LLVMContext::MD_nonnull:
      // All of these directly apply.
      NewLoad->setMetadata(ID, N);
      break;

    case LLVMContext::MD_range:
      // FIXME: It would be nice to propagate this in some way, but the type
      // conversions make it hard.
      break;
    }
  }
  return NewLoad;
}

/// \brief Combine integer loads to vector stores when the integers bits are
/// just a concatenation of non-integer (and non-vector) types.
///
/// This specifically matches the pattern of loading an integer, right-shifting,
/// trucating, and casting it to a non-integer type. When the shift is an exact
/// multiple of the result non-integer type's size, this is more naturally
/// expressed as a load of a vector and an extractelement. This shows up largely
/// because large integers are sometimes used to represent a "generic" load or
/// store, and only later optimization may uncover that there is a more natural
/// type to represent the load with.
static Instruction *combineIntegerLoadToVectorLoad(InstCombiner &IC,
                                                   LoadInst &LI) {
  // FIXME: This is probably a reasonable transform to make for atomic stores.
  assert(LI.isSimple() && "Do not call for non-simple stores!");

  const DataLayout &DL = *IC.getDataLayout();
  unsigned BaseBits = LI.getType()->getIntegerBitWidth();
  Type *ElementTy = nullptr;
  int ElementSize;

  // We match any number of element extractions from the loaded integer. Each of
  // these should be RAUW'ed with an actual extract element instruction at the
  // given index of a loaded vector.
  struct ExtractedElement {
    Instruction *Element;
    int Index;
  };
  SmallVector<ExtractedElement, 2> Elements;

  // Lambda to match the bit cast in the extracted element (which is the root
  // pattern matched). Accepts the instruction and shifted bits, returns false
  // if at any point we failed to match a suitable bitcast for element
  // extraction.
  auto MatchCast = [&](Instruction *I, unsigned ShiftBits) {
    // The truncate must be casted to some element type. This cast can only be
    // a bitcast or an inttoptr cast which is the same size.
    if (!isa<BitCastInst>(I)) {
      if (auto *PC = dyn_cast<IntToPtrInst>(I)) {
        // Ensure that the pointer and integer have the exact same size.
        if (PC->getOperand(0)->getType()->getIntegerBitWidth() !=
            DL.getTypeSizeInBits(PC->getType()))
          return false;
      } else {
        // We only support bitcast and inttoptr.
        return false;
      }
    }

    // All of the elements inserted need to be the same type. Either capture the
    // first element type or check this element type against the previous
    // element types.
    if (!ElementTy) {
      ElementTy = I->getType();
      // We don't handle integers, sub-vectors, or any aggregate types. We
      // handle pointers and floating ponit types.
      if (!ElementTy->isSingleValueType() || ElementTy->isIntegerTy() ||
          ElementTy->isVectorTy())
        return false;

      ElementSize = DL.getTypeSizeInBits(ElementTy);
      // The base integer size and the shift need to be multiples of the element
      // size in bits.
      if (BaseBits % ElementSize || ShiftBits % ElementSize)
        return false;
    } else if (ElementTy != I->getType()) {
      return false;
    }

    // Compute the vector index and store the element with it.
    int Index =
        (DL.isLittleEndian() ? ShiftBits : BaseBits - ElementSize - ShiftBits) /
        ElementSize;
    ExtractedElement E = {I, Index};
    Elements.push_back(std::move(E));
    return true;
  };

  // Lambda to match the truncate in the extracted element. Accepts the
  // instruction and shifted bits. Returns false if at any point we failed to
  // match a suitable truncate for element extraction.
  auto MatchTruncate = [&](Instruction *I, unsigned ShiftBits) {
    // Handle the truncate to the bit size of the element.
    auto *T = dyn_cast<TruncInst>(I);
    if (!T)
      return false;

    // Walk all the users of the truncate, whuch must all be bitcasts.
    for (User *TU : T->users())
      if (!MatchCast(cast<Instruction>(TU), ShiftBits))
        return false;
    return true;
  };

  for (User *U : LI.users()) {
    Instruction *I = cast<Instruction>(U);

    // Strip off a logical shift right and retain the shifted amount.
    ConstantInt *ShiftC;
    if (!match(I, m_LShr(m_Value(), m_ConstantInt(ShiftC)))) {
      // This must be a direct truncate.
      if (!MatchTruncate(I, 0))
        return nullptr;
      continue;
    }

    unsigned ShiftBits = ShiftC->getLimitedValue(BaseBits);
    // We can't handle shifts of more than the number of bits in the integer.
    if (ShiftBits == BaseBits)
      return nullptr;

    // Match all the element extraction users of the shift.
    for (User *IU : I->users())
      if (!MatchTruncate(cast<Instruction>(IU), ShiftBits))
        return nullptr;
  }

  // If didn't find any extracted elements, there is nothing to do here.
  if (Elements.empty())
    return nullptr;

  // Create a vector load and rewrite all of the elements extracted as
  // extractelement instructions.
  VectorType *VTy = VectorType::get(ElementTy, BaseBits / ElementSize);
  LoadInst *NewLI = combineLoadToNewType(IC, LI, VTy);

  for (const auto &E : Elements) {
    IC.Builder->SetInsertPoint(E.Element);
    E.Element->replaceAllUsesWith(
        IC.Builder->CreateExtractElement(NewLI, IC.Builder->getInt32(E.Index)));
    IC.EraseInstFromFunction(*E.Element);
  }

  // Return the original load to indicate it has been combined away.
  return &LI;
}

/// \brief Combine loads to match the type of value their uses after looking
/// through intervening bitcasts.
///
/// The core idea here is that if the result of a load is used in an operation,
/// we should load the type most conducive to that operation. For example, when
/// loading an integer and converting that immediately to a pointer, we should
/// instead directly load a pointer.
///
/// However, this routine must never change the width of a load or the number of
/// loads as that would introduce a semantic change. This combine is expected to
/// be a semantic no-op which just allows loads to more closely model the types
/// of their consuming operations.
///
/// Currently, we also refuse to change the precise type used for an atomic load
/// or a volatile load. This is debatable, and might be reasonable to change
/// later. However, it is risky in case some backend or other part of LLVM is
/// relying on the exact type loaded to select appropriate atomic operations.
static Instruction *combineLoadToOperationType(InstCombiner &IC, LoadInst &LI) {
  // FIXME: We could probably with some care handle both volatile and atomic
  // loads here but it isn't clear that this is important.
  if (!LI.isSimple())
    return nullptr;

  if (LI.use_empty())
    return nullptr;


  // Fold away bit casts of the loaded value by loading the desired type.
  // FIXME: We should also canonicalize loads of vectors when their elements are
  // cast to other types.
  if (LI.hasOneUse())
    if (auto *BC = dyn_cast<BitCastInst>(LI.user_back())) {
      LoadInst *NewLoad = combineLoadToNewType(IC, LI, BC->getDestTy());
      BC->replaceAllUsesWith(NewLoad);
      IC.EraseInstFromFunction(*BC);
      return &LI;
    }

  // Try to combine integer loads into vector loads when the integer is just
  // loading a bag of bits that are casted into vector element chunks.
  if (LI.getType()->isIntegerTy())
    if (Instruction *R = combineIntegerLoadToVectorLoad(IC, LI))
      return R;

  return nullptr;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);

  // Try to canonicalize the loaded type.
  if (Instruction *Res = combineLoadToOperationType(*this, LI))
    return Res;

  // Attempt to improve the alignment.
  if (DL) {
    unsigned KnownAlign =
      getOrEnforceKnownAlignment(Op, DL->getPrefTypeAlignment(LI.getType()),
                                 DL, AT, &LI, DT);
    unsigned LoadAlign = LI.getAlignment();
    unsigned EffectiveLoadAlign = LoadAlign != 0 ? LoadAlign :
      DL->getABITypeAlignment(LI.getType());

    if (KnownAlign > EffectiveLoadAlign)
      LI.setAlignment(KnownAlign);
    else if (LoadAlign == 0)
      LI.setAlignment(EffectiveLoadAlign);
  }

  // None of the following transforms are legal for volatile/atomic loads.
  // FIXME: Some of it is okay for atomic loads; needs refactoring.
  if (!LI.isSimple()) return nullptr;

  // Do really simple store-to-load forwarding and load CSE, to catch cases
  // where there are several consecutive memory accesses to the same location,
  // separated by a few arithmetic operations.
  BasicBlock::iterator BBI = &LI;
  if (Value *AvailableVal = FindAvailableLoadedValue(Op, LI.getParent(), BBI,6))
    return ReplaceInstUsesWith(
        LI, Builder->CreateBitOrPointerCast(AvailableVal, LI.getType(),
                                            LI.getName() + ".cast"));

  // load(gep null, ...) -> unreachable
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op)) {
    const Value *GEPI0 = GEPI->getOperand(0);
    // TODO: Consider a target hook for valid address spaces for this xform.
    if (isa<ConstantPointerNull>(GEPI0) && GEPI->getPointerAddressSpace() == 0){
      // Insert a new store to null instruction before the load to indicate
      // that this code is not reachable.  We do this instead of inserting
      // an unreachable instruction directly because we cannot modify the
      // CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }
  }

  // load null/undef -> unreachable
  // TODO: Consider a target hook for valid address spaces for this xform.
  if (isa<UndefValue>(Op) ||
      (isa<ConstantPointerNull>(Op) && LI.getPointerAddressSpace() == 0)) {
    // Insert a new store to null instruction before the load to indicate that
    // this code is not reachable.  We do this instead of inserting an
    // unreachable instruction directly because we cannot modify the CFG.
    new StoreInst(UndefValue::get(LI.getType()),
                  Constant::getNullValue(Op->getType()), &LI);
    return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
  }

  if (Op->hasOneUse()) {
    // Change select and PHI nodes to select values instead of addresses: this
    // helps alias analysis out a lot, allows many others simplifications, and
    // exposes redundancy in the code.
    //
    // Note that we cannot do the transformation unless we know that the
    // introduced loads cannot trap!  Something like this is valid as long as
    // the condition is always false: load (select bool %C, int* null, int* %G),
    // but it would not be valid if we transformed it to load from null
    // unconditionally.
    //
    if (SelectInst *SI = dyn_cast<SelectInst>(Op)) {
      // load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2).
      unsigned Align = LI.getAlignment();
      if (isSafeToLoadUnconditionally(SI->getOperand(1), SI, Align, DL) &&
          isSafeToLoadUnconditionally(SI->getOperand(2), SI, Align, DL)) {
        LoadInst *V1 = Builder->CreateLoad(SI->getOperand(1),
                                           SI->getOperand(1)->getName()+".val");
        LoadInst *V2 = Builder->CreateLoad(SI->getOperand(2),
                                           SI->getOperand(2)->getName()+".val");
        V1->setAlignment(Align);
        V2->setAlignment(Align);
        return SelectInst::Create(SI->getCondition(), V1, V2);
      }

      // load (select (cond, null, P)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(1)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(2));
          return &LI;
        }

      // load (select (cond, P, null)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(2)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(1));
          return &LI;
        }
    }
  }
  return nullptr;
}

/// \brief Helper to combine a store to use a new value.
///
/// This just does the work of combining a store to use a new value, potentially
/// of a different type. It handles metadata, etc., and returns the new
/// instruction. The new value is stored to a bitcast of the pointer argument to
/// the original store.
///
/// Note that this will create the instructions with whatever insert point the
/// \c InstCombiner currently is using.
static StoreInst *combineStoreToNewValue(InstCombiner &IC, StoreInst &OldSI,
                                         Value *V) {
  Value *Ptr = OldSI.getPointerOperand();
  unsigned AS = OldSI.getPointerAddressSpace();
  SmallVector<std::pair<unsigned, MDNode *>, 8> MD;
  OldSI.getAllMetadata(MD);

  StoreInst *NewSI = IC.Builder->CreateAlignedStore(
      V, IC.Builder->CreateBitCast(Ptr, V->getType()->getPointerTo(AS)),
      OldSI.getAlignment());
  for (const auto &MDPair : MD) {
    unsigned ID = MDPair.first;
    MDNode *N = MDPair.second;
    // Note, essentially every kind of metadata should be preserved here! This
    // routine is supposed to clone a store instruction changing *only its
    // type*. The only metadata it makes sense to drop is metadata which is
    // invalidated when the pointer type changes. This should essentially
    // never be the case in LLVM, but we explicitly switch over only known
    // metadata to be conservatively correct. If you are adding metadata to
    // LLVM which pertains to stores, you almost certainly want to add it
    // here.
    switch (ID) {
    case LLVMContext::MD_dbg:
    case LLVMContext::MD_tbaa:
    case LLVMContext::MD_prof:
    case LLVMContext::MD_fpmath:
    case LLVMContext::MD_tbaa_struct:
    case LLVMContext::MD_alias_scope:
    case LLVMContext::MD_noalias:
    case LLVMContext::MD_nontemporal:
    case LLVMContext::MD_mem_parallel_loop_access:
    case LLVMContext::MD_nonnull:
      // All of these directly apply.
      NewSI->setMetadata(ID, N);
      break;

    case LLVMContext::MD_invariant_load:
    case LLVMContext::MD_range:
      break;
    }
  }
  return NewSI;
}

/// \brief Combine integer stores to vector stores when the integers bits are
/// just a concatenation of non-integer (and non-vector) types.
///
/// This specifically matches the pattern of taking a sequence of non-integer
/// types, casting them to integers, extending, shifting, and or-ing them
/// together to make a concatenation, and then storing the result. This shows up
/// because large integers are sometimes used to represent a "generic" load or
/// store, and only later optimization may uncover that there is a more natural
/// type to represent the store with.
///
/// \returns true if the store was successfully combined away. This indicates
/// the caller must erase the store instruction. We have to let the caller erase
/// the store instruction sas otherwise there is no way to signal whether it was
/// combined or not: IC.EraseInstFromFunction returns a null pointer.
static bool combineIntegerStoreToVectorStore(InstCombiner &IC, StoreInst &SI) {
  // FIXME: This is probably a reasonable transform to make for atomic stores.
  assert(SI.isSimple() && "Do not call for non-simple stores!");

  Instruction *OrigV = dyn_cast<Instruction>(SI.getValueOperand());
  if (!OrigV)
    return false;

  // We only handle values which are used entirely to store to memory. If the
  // value is used directly as an SSA value, then even if there are matching
  // element insertion and element extraction, we rely on basic integer
  // combining to forward the bits and delete the intermediate math. Here we
  // just need to clean up the places where it actually reaches memory.
  SmallVector<StoreInst *, 2> Stores;
  for (User *U : OrigV->users())
    if (auto *SIU = dyn_cast<StoreInst>(U))
      Stores.push_back(SIU);
    else
      return false;

  const DataLayout &DL = *IC.getDataLayout();
  unsigned BaseBits = OrigV->getType()->getIntegerBitWidth();
  Type *ElementTy = nullptr;
  int ElementSize;

  // We need to match some number of element insertions into an integer. Each
  // insertion takes the form of an element value (and type), index (multiple of
  // the bitwidth of the type) of insertion, and the base it was inserted into.
  struct InsertedElement {
    Value *Base;
    Value *Element;
    int Index;
  };
  auto MatchInsertedElement = [&](Value *V) -> Optional<InsertedElement> {
    // Handle a null input to make it easy to loop over bases.
    if (!V)
      return Optional<InsertedElement>();

    assert(!V->getType()->isVectorTy() && "Must not be a vector.");
    assert(V->getType()->isIntegerTy() && "Must be an integer value.");

    Value *Base = nullptr, *Cast;
    ConstantInt *ShiftC = nullptr;
    auto InsertPattern = m_CombineOr(
        m_Shl(m_OneUse(m_ZExt(m_OneUse(m_Value(Cast)))), m_ConstantInt(ShiftC)),
        m_ZExt(m_OneUse(m_Value(Cast))));
    if (!match(V, m_CombineOr(m_CombineOr(m_Or(m_OneUse(m_Value(Base)),
                                               m_OneUse(InsertPattern)),
                                          m_Or(m_OneUse(InsertPattern),
                                               m_OneUse(m_Value(Base)))),
                              InsertPattern)))
      return Optional<InsertedElement>();

    Value *Element;
    if (auto *BC = dyn_cast<BitCastInst>(Cast)) {
      // Bit casts are trivially correct here.
      Element = BC->getOperand(0);
    } else if (auto *PC = dyn_cast<PtrToIntInst>(Cast)) {
      Element = PC->getOperand(0);
      // If this changes the bit width at all, reject it.
      if (PC->getType()->getIntegerBitWidth() !=
          DL.getTypeSizeInBits(Element->getType()))
        return Optional<InsertedElement>();
    } else {
      // All other casts are rejected.
      return Optional<InsertedElement>();
    }

    // We can't handle shifts wider than the number of bits in the integer.
    unsigned ShiftBits = ShiftC ? ShiftC->getLimitedValue(BaseBits) : 0;
    if (ShiftBits == BaseBits)
      return Optional<InsertedElement>();

    // All of the elements inserted need to be the same type. Either capture the
    // first element type or check this element type against the previous
    // element types.
    if (!ElementTy) {
      ElementTy = Element->getType();
      // The base integer size and the shift need to be multiples of the element
      // size in bits.
      ElementSize = DL.getTypeSizeInBits(ElementTy);
      if (BaseBits % ElementSize || ShiftBits % ElementSize)
        return Optional<InsertedElement>();
    } else if (ElementTy != Element->getType()) {
      return Optional<InsertedElement>();
    }

    // We don't handle integers, sub-vectors, or any aggregate types. We
    // handle pointers and floating ponit types.
    if (!ElementTy->isSingleValueType() || ElementTy->isIntegerTy() ||
        ElementTy->isVectorTy())
      return Optional<InsertedElement>();

    int Index =
        (DL.isLittleEndian() ? ShiftBits : BaseBits - ElementSize - ShiftBits) /
        ElementSize;
    InsertedElement Result = {Base, Element, Index};
    return Result;
  };

  SmallVector<InsertedElement, 2> Elements;
  Value *V = OrigV;
  while (Optional<InsertedElement> E = MatchInsertedElement(V)) {
    V = E->Base;
    Elements.push_back(std::move(*E));
  }
  // If searching for elements found none, or didn't terminate in either an
  // undef or a direct zext, we can't form a vector.
  if (Elements.empty() || (V && !isa<UndefValue>(V)))
    return false;

  // Build a storable vector by looping over the inserted elements.
  VectorType *VTy = VectorType::get(ElementTy, BaseBits / ElementSize);
  V = UndefValue::get(VTy);
  IC.Builder->SetInsertPoint(OrigV);
  for (const auto &E : Elements)
    V = IC.Builder->CreateInsertElement(V, E.Element,
                                        IC.Builder->getInt32(E.Index));

  for (StoreInst *OldSI : Stores) {
    IC.Builder->SetInsertPoint(OldSI);
    combineStoreToNewValue(IC, *OldSI, V);
    if (OldSI != &SI)
      IC.EraseInstFromFunction(*OldSI);
  }
  return true;
}

/// \brief Combine stores to match the type of value being stored.
///
/// The core idea here is that the memory does not have any intrinsic type and
/// where we can we should match the type of a store to the type of value being
/// stored.
///
/// However, this routine must never change the width of a store or the number of
/// stores as that would introduce a semantic change. This combine is expected to
/// be a semantic no-op which just allows stores to more closely model the types
/// of their incoming values.
///
/// Currently, we also refuse to change the precise type used for an atomic or
/// volatile store. This is debatable, and might be reasonable to change later.
/// However, it is risky in case some backend or other part of LLVM is relying
/// on the exact type stored to select appropriate atomic operations.
///
/// \returns true if the store was successfully combined away. This indicates
/// the caller must erase the store instruction. We have to let the caller erase
/// the store instruction sas otherwise there is no way to signal whether it was
/// combined or not: IC.EraseInstFromFunction returns a null pointer.
static bool combineStoreToValueType(InstCombiner &IC, StoreInst &SI) {
  // FIXME: We could probably with some care handle both volatile and atomic
  // stores here but it isn't clear that this is important.
  if (!SI.isSimple())
    return false;

  Value *V = SI.getValueOperand();

  // Fold away bit casts of the stored value by storing the original type.
  if (auto *BC = dyn_cast<BitCastInst>(V)) {
    combineStoreToNewValue(IC, SI, BC->getOperand(0));
    return true;
  }

  // If this is an integer store and we have data layout, look for a pattern of
  // storing a vector as an integer (modeled as a bag of bits).
  if (V->getType()->isIntegerTy() && IC.getDataLayout() &&
      combineIntegerStoreToVectorStore(IC, SI))
    return true;

  // FIXME: We should also canonicalize loads of vectors when their elements are
  // cast to other types.
  return false;
}

/// equivalentAddressValues - Test if A and B will obviously have the same
/// value. This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
///
static bool equivalentAddressValues(Value *A, Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B) return true;

  // Test if the values come form identical arithmetic instructions.
  // This uses isIdenticalToWhenDefined instead of isIdenticalTo because
  // its only used to compare two uses within the same basic block, which
  // means that they'll always either have the same value or one of them
  // will have an undefined value.
  if (isa<BinaryOperator>(A) ||
      isa<CastInst>(A) ||
      isa<PHINode>(A) ||
      isa<GetElementPtrInst>(A))
    if (Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;

  // Otherwise they may not be equivalent.
  return false;
}

Instruction *InstCombiner::visitStoreInst(StoreInst &SI) {
  Value *Val = SI.getOperand(0);
  Value *Ptr = SI.getOperand(1);

  // Try to canonicalize the stored type.
  if (combineStoreToValueType(*this, SI))
    return EraseInstFromFunction(SI);

  // Attempt to improve the alignment.
  if (DL) {
    unsigned KnownAlign =
      getOrEnforceKnownAlignment(Ptr, DL->getPrefTypeAlignment(Val->getType()),
                                 DL, AT, &SI, DT);
    unsigned StoreAlign = SI.getAlignment();
    unsigned EffectiveStoreAlign = StoreAlign != 0 ? StoreAlign :
      DL->getABITypeAlignment(Val->getType());

    if (KnownAlign > EffectiveStoreAlign)
      SI.setAlignment(KnownAlign);
    else if (StoreAlign == 0)
      SI.setAlignment(EffectiveStoreAlign);
  }

  // Don't hack volatile/atomic stores.
  // FIXME: Some bits are legal for atomic stores; needs refactoring.
  if (!SI.isSimple()) return nullptr;

  // If the RHS is an alloca with a single use, zapify the store, making the
  // alloca dead.
  if (Ptr->hasOneUse()) {
    if (isa<AllocaInst>(Ptr))
      return EraseInstFromFunction(SI);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      if (isa<AllocaInst>(GEP->getOperand(0))) {
        if (GEP->getOperand(0)->hasOneUse())
          return EraseInstFromFunction(SI);
      }
    }
  }

  // Do really simple DSE, to catch cases where there are several consecutive
  // stores to the same location, separated by a few arithmetic operations. This
  // situation often occurs with bitfield accesses.
  BasicBlock::iterator BBI = &SI;
  for (unsigned ScanInsts = 6; BBI != SI.getParent()->begin() && ScanInsts;
       --ScanInsts) {
    --BBI;
    // Don't count debug info directives, lest they affect codegen,
    // and we skip pointer-to-pointer bitcasts, which are NOPs.
    if (isa<DbgInfoIntrinsic>(BBI) ||
        (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy())) {
      ScanInsts++;
      continue;
    }

    if (StoreInst *PrevSI = dyn_cast<StoreInst>(BBI)) {
      // Prev store isn't volatile, and stores to the same location?
      if (PrevSI->isSimple() && equivalentAddressValues(PrevSI->getOperand(1),
                                                        SI.getOperand(1))) {
        ++NumDeadStore;
        ++BBI;
        EraseInstFromFunction(*PrevSI);
        continue;
      }
      break;
    }

    // If this is a load, we have to stop.  However, if the loaded value is from
    // the pointer we're loading and is producing the pointer we're storing,
    // then *this* store is dead (X = load P; store X -> P).
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI == Val && equivalentAddressValues(LI->getOperand(0), Ptr) &&
          LI->isSimple())
        return EraseInstFromFunction(SI);

      // Otherwise, this is a load from some other location.  Stores before it
      // may not be dead.
      break;
    }

    // Don't skip over loads or things that can modify memory.
    if (BBI->mayWriteToMemory() || BBI->mayReadFromMemory())
      break;
  }

  // store X, null    -> turns into 'unreachable' in SimplifyCFG
  if (isa<ConstantPointerNull>(Ptr) && SI.getPointerAddressSpace() == 0) {
    if (!isa<UndefValue>(Val)) {
      SI.setOperand(0, UndefValue::get(Val->getType()));
      if (Instruction *U = dyn_cast<Instruction>(Val))
        Worklist.Add(U);  // Dropped a use.
    }
    return nullptr;  // Do not modify these!
  }

  // store undef, Ptr -> noop
  if (isa<UndefValue>(Val))
    return EraseInstFromFunction(SI);

  // If this store is the last instruction in the basic block (possibly
  // excepting debug info instructions), and if the block ends with an
  // unconditional branch, try to move it to the successor block.
  BBI = &SI;
  do {
    ++BBI;
  } while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy()));
  if (BranchInst *BI = dyn_cast<BranchInst>(BBI))
    if (BI->isUnconditional())
      if (SimplifyStoreAtEndOfBlock(SI))
        return nullptr;  // xform done!

  return nullptr;
}

/// SimplifyStoreAtEndOfBlock - Turn things like:
///   if () { *P = v1; } else { *P = v2 }
/// into a phi node with a store in the successor.
///
/// Simplify things like:
///   *P = v1; if () { *P = v2; }
/// into a phi node with a store in the successor.
///
bool InstCombiner::SimplifyStoreAtEndOfBlock(StoreInst &SI) {
  BasicBlock *StoreBB = SI.getParent();

  // Check to see if the successor block has exactly two incoming edges.  If
  // so, see if the other predecessor contains a store to the same location.
  // if so, insert a PHI node (if needed) and move the stores down.
  BasicBlock *DestBB = StoreBB->getTerminator()->getSuccessor(0);

  // Determine whether Dest has exactly two predecessors and, if so, compute
  // the other predecessor.
  pred_iterator PI = pred_begin(DestBB);
  BasicBlock *P = *PI;
  BasicBlock *OtherBB = nullptr;

  if (P != StoreBB)
    OtherBB = P;

  if (++PI == pred_end(DestBB))
    return false;

  P = *PI;
  if (P != StoreBB) {
    if (OtherBB)
      return false;
    OtherBB = P;
  }
  if (++PI != pred_end(DestBB))
    return false;

  // Bail out if all the relevant blocks aren't distinct (this can happen,
  // for example, if SI is in an infinite loop)
  if (StoreBB == DestBB || OtherBB == DestBB)
    return false;

  // Verify that the other block ends in a branch and is not otherwise empty.
  BasicBlock::iterator BBI = OtherBB->getTerminator();
  BranchInst *OtherBr = dyn_cast<BranchInst>(BBI);
  if (!OtherBr || BBI == OtherBB->begin())
    return false;

  // If the other block ends in an unconditional branch, check for the 'if then
  // else' case.  there is an instruction before the branch.
  StoreInst *OtherStore = nullptr;
  if (OtherBr->isUnconditional()) {
    --BBI;
    // Skip over debugging info.
    while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy())) {
      if (BBI==OtherBB->begin())
        return false;
      --BBI;
    }
    // If this isn't a store, isn't a store to the same location, or is not the
    // right kind of store, bail out.
    OtherStore = dyn_cast<StoreInst>(BBI);
    if (!OtherStore || OtherStore->getOperand(1) != SI.getOperand(1) ||
        !SI.isSameOperationAs(OtherStore))
      return false;
  } else {
    // Otherwise, the other block ended with a conditional branch. If one of the
    // destinations is StoreBB, then we have the if/then case.
    if (OtherBr->getSuccessor(0) != StoreBB &&
        OtherBr->getSuccessor(1) != StoreBB)
      return false;

    // Okay, we know that OtherBr now goes to Dest and StoreBB, so this is an
    // if/then triangle.  See if there is a store to the same ptr as SI that
    // lives in OtherBB.
    for (;; --BBI) {
      // Check to see if we find the matching store.
      if ((OtherStore = dyn_cast<StoreInst>(BBI))) {
        if (OtherStore->getOperand(1) != SI.getOperand(1) ||
            !SI.isSameOperationAs(OtherStore))
          return false;
        break;
      }
      // If we find something that may be using or overwriting the stored
      // value, or if we run out of instructions, we can't do the xform.
      if (BBI->mayReadFromMemory() || BBI->mayWriteToMemory() ||
          BBI == OtherBB->begin())
        return false;
    }

    // In order to eliminate the store in OtherBr, we have to
    // make sure nothing reads or overwrites the stored value in
    // StoreBB.
    for (BasicBlock::iterator I = StoreBB->begin(); &*I != &SI; ++I) {
      // FIXME: This should really be AA driven.
      if (I->mayReadFromMemory() || I->mayWriteToMemory())
        return false;
    }
  }

  // Insert a PHI node now if we need it.
  Value *MergedVal = OtherStore->getOperand(0);
  if (MergedVal != SI.getOperand(0)) {
    PHINode *PN = PHINode::Create(MergedVal->getType(), 2, "storemerge");
    PN->addIncoming(SI.getOperand(0), SI.getParent());
    PN->addIncoming(OtherStore->getOperand(0), OtherBB);
    MergedVal = InsertNewInstBefore(PN, DestBB->front());
  }

  // Advance to a place where it is safe to insert the new store and
  // insert it.
  BBI = DestBB->getFirstInsertionPt();
  StoreInst *NewSI = new StoreInst(MergedVal, SI.getOperand(1),
                                   SI.isVolatile(),
                                   SI.getAlignment(),
                                   SI.getOrdering(),
                                   SI.getSynchScope());
  InsertNewInstBefore(NewSI, *BBI);
  NewSI->setDebugLoc(OtherStore->getDebugLoc());

  // If the two stores had AA tags, merge them.
  AAMDNodes AATags;
  SI.getAAMetadata(AATags);
  if (AATags) {
    OtherStore->getAAMetadata(AATags, /* Merge = */ true);
    NewSI->setAAMetadata(AATags);
  }

  // Nuke the old stores.
  EraseInstFromFunction(SI);
  EraseInstFromFunction(*OtherStore);
  return true;
}
