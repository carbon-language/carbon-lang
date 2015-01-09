//===-- Value.cpp - Implement the Value class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Value, ValueHandle, and User classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Value.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                                Value Class
//===----------------------------------------------------------------------===//

static inline Type *checkType(Type *Ty) {
  assert(Ty && "Value defined with a null type: Error!");
  return Ty;
}

Value::Value(Type *ty, unsigned scid)
    : VTy(checkType(ty)), UseList(nullptr), SubclassID(scid), HasValueHandle(0),
      SubclassOptionalData(0), SubclassData(0), NumOperands(0) {
  // FIXME: Why isn't this in the subclass gunk??
  // Note, we cannot call isa<CallInst> before the CallInst has been
  // constructed.
  if (SubclassID == Instruction::Call || SubclassID == Instruction::Invoke)
    assert((VTy->isFirstClassType() || VTy->isVoidTy() || VTy->isStructTy()) &&
           "invalid CallInst type!");
  else if (SubclassID != BasicBlockVal &&
           (SubclassID < ConstantFirstVal || SubclassID > ConstantLastVal))
    assert((VTy->isFirstClassType() || VTy->isVoidTy()) &&
           "Cannot create non-first-class values except for constants!");
}

Value::~Value() {
  // Notify all ValueHandles (if present) that this value is going away.
  if (HasValueHandle)
    ValueHandleBase::ValueIsDeleted(this);
  if (isUsedByMetadata())
    ValueAsMetadata::handleDeletion(this);

#ifndef NDEBUG      // Only in -g mode...
  // Check to make sure that there are no uses of this value that are still
  // around when the value is destroyed.  If there are, then we have a dangling
  // reference and something is wrong.  This code is here to print out what is
  // still being referenced.  The value in question should be printed as
  // a <badref>
  //
  if (!use_empty()) {
    dbgs() << "While deleting: " << *VTy << " %" << getName() << "\n";
    for (use_iterator I = use_begin(), E = use_end(); I != E; ++I)
      dbgs() << "Use still stuck around after Def is destroyed:"
           << **I << "\n";
  }
#endif
  assert(use_empty() && "Uses remain when a value is destroyed!");

  // If this value is named, destroy the name.  This should not be in a symtab
  // at this point.
  destroyValueName();
}

void Value::destroyValueName() {
  ValueName *Name = getValueName();
  if (Name)
    Name->Destroy();
  setValueName(nullptr);
}

bool Value::hasNUses(unsigned N) const {
  const_use_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.
  return UI == E;
}

bool Value::hasNUsesOrMore(unsigned N) const {
  const_use_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.

  return true;
}

bool Value::isUsedInBasicBlock(const BasicBlock *BB) const {
  // This can be computed either by scanning the instructions in BB, or by
  // scanning the use list of this Value. Both lists can be very long, but
  // usually one is quite short.
  //
  // Scan both lists simultaneously until one is exhausted. This limits the
  // search to the shorter list.
  BasicBlock::const_iterator BI = BB->begin(), BE = BB->end();
  const_user_iterator UI = user_begin(), UE = user_end();
  for (; BI != BE && UI != UE; ++BI, ++UI) {
    // Scan basic block: Check if this Value is used by the instruction at BI.
    if (std::find(BI->op_begin(), BI->op_end(), this) != BI->op_end())
      return true;
    // Scan use list: Check if the use at UI is in BB.
    const Instruction *User = dyn_cast<Instruction>(*UI);
    if (User && User->getParent() == BB)
      return true;
  }
  return false;
}

unsigned Value::getNumUses() const {
  return (unsigned)std::distance(use_begin(), use_end());
}

static bool getSymTab(Value *V, ValueSymbolTable *&ST) {
  ST = nullptr;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    if (BasicBlock *P = I->getParent())
      if (Function *PP = P->getParent())
        ST = &PP->getValueSymbolTable();
  } else if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    if (Function *P = BB->getParent())
      ST = &P->getValueSymbolTable();
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (Module *P = GV->getParent())
      ST = &P->getValueSymbolTable();
  } else if (Argument *A = dyn_cast<Argument>(V)) {
    if (Function *P = A->getParent())
      ST = &P->getValueSymbolTable();
  } else {
    assert(isa<Constant>(V) && "Unknown value type!");
    return true;  // no name is setable for this.
  }
  return false;
}

StringRef Value::getName() const {
  // Make sure the empty string is still a C string. For historical reasons,
  // some clients want to call .data() on the result and expect it to be null
  // terminated.
  if (!getValueName())
    return StringRef("", 0);
  return getValueName()->getKey();
}

void Value::setName(const Twine &NewName) {
  // Fast path for common IRBuilder case of setName("") when there is no name.
  if (NewName.isTriviallyEmpty() && !hasName())
    return;

  SmallString<256> NameData;
  StringRef NameRef = NewName.toStringRef(NameData);
  assert(NameRef.find_first_of(0) == StringRef::npos &&
         "Null bytes are not allowed in names");

  // Name isn't changing?
  if (getName() == NameRef)
    return;

  assert(!getType()->isVoidTy() && "Cannot assign a name to void values!");

  // Get the symbol table to update for this object.
  ValueSymbolTable *ST;
  if (getSymTab(this, ST))
    return;  // Cannot set a name on this value (e.g. constant).

  if (Function *F = dyn_cast<Function>(this))
    getContext().pImpl->IntrinsicIDCache.erase(F);

  if (!ST) { // No symbol table to update?  Just do the change.
    if (NameRef.empty()) {
      // Free the name for this value.
      destroyValueName();
      return;
    }

    // NOTE: Could optimize for the case the name is shrinking to not deallocate
    // then reallocated.
    destroyValueName();

    // Create the new name.
    setValueName(ValueName::Create(NameRef));
    getValueName()->setValue(this);
    return;
  }

  // NOTE: Could optimize for the case the name is shrinking to not deallocate
  // then reallocated.
  if (hasName()) {
    // Remove old name.
    ST->removeValueName(getValueName());
    destroyValueName();

    if (NameRef.empty())
      return;
  }

  // Name is changing to something new.
  setValueName(ST->createValueName(NameRef, this));
}

void Value::takeName(Value *V) {
  ValueSymbolTable *ST = nullptr;
  // If this value has a name, drop it.
  if (hasName()) {
    // Get the symtab this is in.
    if (getSymTab(this, ST)) {
      // We can't set a name on this value, but we need to clear V's name if
      // it has one.
      if (V->hasName()) V->setName("");
      return;  // Cannot set a name on this value (e.g. constant).
    }

    // Remove old name.
    if (ST)
      ST->removeValueName(getValueName());
    destroyValueName();
  }

  // Now we know that this has no name.

  // If V has no name either, we're done.
  if (!V->hasName()) return;

  // Get this's symtab if we didn't before.
  if (!ST) {
    if (getSymTab(this, ST)) {
      // Clear V's name.
      V->setName("");
      return;  // Cannot set a name on this value (e.g. constant).
    }
  }

  // Get V's ST, this should always succed, because V has a name.
  ValueSymbolTable *VST;
  bool Failure = getSymTab(V, VST);
  assert(!Failure && "V has a name, so it should have a ST!"); (void)Failure;

  // If these values are both in the same symtab, we can do this very fast.
  // This works even if both values have no symtab yet.
  if (ST == VST) {
    // Take the name!
    setValueName(V->getValueName());
    V->setValueName(nullptr);
    getValueName()->setValue(this);
    return;
  }

  // Otherwise, things are slightly more complex.  Remove V's name from VST and
  // then reinsert it into ST.

  if (VST)
    VST->removeValueName(V->getValueName());
  setValueName(V->getValueName());
  V->setValueName(nullptr);
  getValueName()->setValue(this);

  if (ST)
    ST->reinsertValue(this);
}

#ifndef NDEBUG
static bool contains(SmallPtrSetImpl<ConstantExpr *> &Cache, ConstantExpr *Expr,
                     Constant *C) {
  if (!Cache.insert(Expr).second)
    return false;

  for (auto &O : Expr->operands()) {
    if (O == C)
      return true;
    auto *CE = dyn_cast<ConstantExpr>(O);
    if (!CE)
      continue;
    if (contains(Cache, CE, C))
      return true;
  }
  return false;
}

static bool contains(Value *Expr, Value *V) {
  if (Expr == V)
    return true;

  auto *C = dyn_cast<Constant>(V);
  if (!C)
    return false;

  auto *CE = dyn_cast<ConstantExpr>(Expr);
  if (!CE)
    return false;

  SmallPtrSet<ConstantExpr *, 4> Cache;
  return contains(Cache, CE, C);
}
#endif

void Value::replaceAllUsesWith(Value *New) {
  assert(New && "Value::replaceAllUsesWith(<null>) is invalid!");
  assert(!contains(New, this) &&
         "this->replaceAllUsesWith(expr(this)) is NOT valid!");
  assert(New->getType() == getType() &&
         "replaceAllUses of value with new value of different type!");

  // Notify all ValueHandles (if present) that this value is going away.
  if (HasValueHandle)
    ValueHandleBase::ValueIsRAUWd(this, New);
  if (isUsedByMetadata())
    ValueAsMetadata::handleRAUW(this, New);

  while (!use_empty()) {
    Use &U = *UseList;
    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant because they are uniqued.
    if (auto *C = dyn_cast<Constant>(U.getUser())) {
      if (!isa<GlobalValue>(C)) {
        C->replaceUsesOfWithOnConstant(this, New, &U);
        continue;
      }
    }

    U.set(New);
  }

  if (BasicBlock *BB = dyn_cast<BasicBlock>(this))
    BB->replaceSuccessorsPhiUsesWith(cast<BasicBlock>(New));
}

// Like replaceAllUsesWith except it does not handle constants or basic blocks.
// This routine leaves uses within BB.
void Value::replaceUsesOutsideBlock(Value *New, BasicBlock *BB) {
  assert(New && "Value::replaceUsesOutsideBlock(<null>, BB) is invalid!");
  assert(!contains(New, this) &&
         "this->replaceUsesOutsideBlock(expr(this), BB) is NOT valid!");
  assert(New->getType() == getType() &&
         "replaceUses of value with new value of different type!");
  assert(BB && "Basic block that may contain a use of 'New' must be defined\n");

  use_iterator UI = use_begin(), E = use_end();
  for (; UI != E;) {
    Use &U = *UI;
    ++UI;
    auto *Usr = dyn_cast<Instruction>(U.getUser());
    if (Usr && Usr->getParent() == BB)
      continue;
    U.set(New);
  }
  return;
}

namespace {
// Various metrics for how much to strip off of pointers.
enum PointerStripKind {
  PSK_ZeroIndices,
  PSK_ZeroIndicesAndAliases,
  PSK_InBoundsConstantIndices,
  PSK_InBounds
};

template <PointerStripKind StripKind>
static Value *stripPointerCastsAndOffsets(Value *V) {
  if (!V->getType()->isPointerTy())
    return V;

  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<Value *, 4> Visited;

  Visited.insert(V);
  do {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      switch (StripKind) {
      case PSK_ZeroIndicesAndAliases:
      case PSK_ZeroIndices:
        if (!GEP->hasAllZeroIndices())
          return V;
        break;
      case PSK_InBoundsConstantIndices:
        if (!GEP->hasAllConstantIndices())
          return V;
        // fallthrough
      case PSK_InBounds:
        if (!GEP->isInBounds())
          return V;
        break;
      }
      V = GEP->getPointerOperand();
    } else if (Operator::getOpcode(V) == Instruction::BitCast ||
               Operator::getOpcode(V) == Instruction::AddrSpaceCast) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
      if (StripKind == PSK_ZeroIndices || GA->mayBeOverridden())
        return V;
      V = GA->getAliasee();
    } else {
      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);

  return V;
}
} // namespace

Value *Value::stripPointerCasts() {
  return stripPointerCastsAndOffsets<PSK_ZeroIndicesAndAliases>(this);
}

Value *Value::stripPointerCastsNoFollowAliases() {
  return stripPointerCastsAndOffsets<PSK_ZeroIndices>(this);
}

Value *Value::stripInBoundsConstantOffsets() {
  return stripPointerCastsAndOffsets<PSK_InBoundsConstantIndices>(this);
}

Value *Value::stripAndAccumulateInBoundsConstantOffsets(const DataLayout &DL,
                                                        APInt &Offset) {
  if (!getType()->isPointerTy())
    return this;

  assert(Offset.getBitWidth() == DL.getPointerSizeInBits(cast<PointerType>(
                                     getType())->getAddressSpace()) &&
         "The offset must have exactly as many bits as our pointer.");

  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<Value *, 4> Visited;
  Visited.insert(this);
  Value *V = this;
  do {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      if (!GEP->isInBounds())
        return V;
      APInt GEPOffset(Offset);
      if (!GEP->accumulateConstantOffset(DL, GEPOffset))
        return V;
      Offset = GEPOffset;
      V = GEP->getPointerOperand();
    } else if (Operator::getOpcode(V) == Instruction::BitCast ||
               Operator::getOpcode(V) == Instruction::AddrSpaceCast) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
      V = GA->getAliasee();
    } else {
      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);

  return V;
}

Value *Value::stripInBoundsOffsets() {
  return stripPointerCastsAndOffsets<PSK_InBounds>(this);
}

/// \brief Check if Value is always a dereferenceable pointer.
///
/// Test if V is always a pointer to allocated and suitably aligned memory for
/// a simple load or store.
static bool isDereferenceablePointer(const Value *V, const DataLayout *DL,
                                     SmallPtrSetImpl<const Value *> &Visited) {
  // Note that it is not safe to speculate into a malloc'd region because
  // malloc may return null.

  // These are obviously ok.
  if (isa<AllocaInst>(V)) return true;

  // It's not always safe to follow a bitcast, for example:
  //   bitcast i8* (alloca i8) to i32*
  // would result in a 4-byte load from a 1-byte alloca. However,
  // if we're casting from a pointer from a type of larger size
  // to a type of smaller size (or the same size), and the alignment
  // is at least as large as for the resulting pointer type, then
  // we can look through the bitcast.
  if (DL)
    if (const BitCastInst* BC = dyn_cast<BitCastInst>(V)) {
      Type *STy = BC->getSrcTy()->getPointerElementType(),
           *DTy = BC->getDestTy()->getPointerElementType();
      if (STy->isSized() && DTy->isSized() &&
          (DL->getTypeStoreSize(STy) >=
           DL->getTypeStoreSize(DTy)) &&
          (DL->getABITypeAlignment(STy) >=
           DL->getABITypeAlignment(DTy)))
        return isDereferenceablePointer(BC->getOperand(0), DL, Visited);
    }

  // Global variables which can't collapse to null are ok.
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return !GV->hasExternalWeakLinkage();

  // byval arguments are okay. Arguments specifically marked as
  // dereferenceable are okay too.
  if (const Argument *A = dyn_cast<Argument>(V)) {
    if (A->hasByValAttr())
      return true;
    else if (uint64_t Bytes = A->getDereferenceableBytes()) {
      Type *Ty = V->getType()->getPointerElementType();
      if (Ty->isSized() && DL && DL->getTypeStoreSize(Ty) <= Bytes)
        return true;
    }

    return false;
  }

  // Return values from call sites specifically marked as dereferenceable are
  // also okay.
  if (ImmutableCallSite CS = V) {
    if (uint64_t Bytes = CS.getDereferenceableBytes(0)) {
      Type *Ty = V->getType()->getPointerElementType();
      if (Ty->isSized() && DL && DL->getTypeStoreSize(Ty) <= Bytes)
        return true;
    }
  }

  // For GEPs, determine if the indexing lands within the allocated object.
  if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    // Conservatively require that the base pointer be fully dereferenceable.
    if (!Visited.insert(GEP->getOperand(0)).second)
      return false;
    if (!isDereferenceablePointer(GEP->getOperand(0), DL, Visited))
      return false;
    // Check the indices.
    gep_type_iterator GTI = gep_type_begin(GEP);
    for (User::const_op_iterator I = GEP->op_begin()+1,
         E = GEP->op_end(); I != E; ++I) {
      Value *Index = *I;
      Type *Ty = *GTI++;
      // Struct indices can't be out of bounds.
      if (isa<StructType>(Ty))
        continue;
      ConstantInt *CI = dyn_cast<ConstantInt>(Index);
      if (!CI)
        return false;
      // Zero is always ok.
      if (CI->isZero())
        continue;
      // Check to see that it's within the bounds of an array.
      ArrayType *ATy = dyn_cast<ArrayType>(Ty);
      if (!ATy)
        return false;
      if (CI->getValue().getActiveBits() > 64)
        return false;
      if (CI->getZExtValue() >= ATy->getNumElements())
        return false;
    }
    // Indices check out; this is dereferenceable.
    return true;
  }

  if (const AddrSpaceCastInst *ASC = dyn_cast<AddrSpaceCastInst>(V))
    return isDereferenceablePointer(ASC->getOperand(0), DL, Visited);

  // If we don't know, assume the worst.
  return false;
}

bool Value::isDereferenceablePointer(const DataLayout *DL) const {
  // When dereferenceability information is provided by a dereferenceable
  // attribute, we know exactly how many bytes are dereferenceable. If we can
  // determine the exact offset to the attributed variable, we can use that
  // information here.
  Type *Ty = getType()->getPointerElementType();
  if (Ty->isSized() && DL) {
    APInt Offset(DL->getTypeStoreSizeInBits(getType()), 0);
    const Value *BV = stripAndAccumulateInBoundsConstantOffsets(*DL, Offset);

    APInt DerefBytes(Offset.getBitWidth(), 0);
    if (const Argument *A = dyn_cast<Argument>(BV))
      DerefBytes = A->getDereferenceableBytes();
    else if (ImmutableCallSite CS = BV)
      DerefBytes = CS.getDereferenceableBytes(0);

    if (DerefBytes.getBoolValue() && Offset.isNonNegative()) {
      if (DerefBytes.uge(Offset + DL->getTypeStoreSize(Ty)))
        return true;
    }
  }

  SmallPtrSet<const Value *, 32> Visited;
  return ::isDereferenceablePointer(this, DL, Visited);
}

Value *Value::DoPHITranslation(const BasicBlock *CurBB,
                               const BasicBlock *PredBB) {
  PHINode *PN = dyn_cast<PHINode>(this);
  if (PN && PN->getParent() == CurBB)
    return PN->getIncomingValueForBlock(PredBB);
  return this;
}

LLVMContext &Value::getContext() const { return VTy->getContext(); }

void Value::reverseUseList() {
  if (!UseList || !UseList->Next)
    // No need to reverse 0 or 1 uses.
    return;

  Use *Head = UseList;
  Use *Current = UseList->Next;
  Head->Next = nullptr;
  while (Current) {
    Use *Next = Current->Next;
    Current->Next = Head;
    Head->setPrev(&Current->Next);
    Head = Current;
    Current = Next;
  }
  UseList = Head;
  Head->setPrev(&UseList);
}

//===----------------------------------------------------------------------===//
//                             ValueHandleBase Class
//===----------------------------------------------------------------------===//

void ValueHandleBase::AddToExistingUseList(ValueHandleBase **List) {
  assert(List && "Handle list is null?");

  // Splice ourselves into the list.
  Next = *List;
  *List = this;
  setPrevPtr(List);
  if (Next) {
    Next->setPrevPtr(&Next);
    assert(V == Next->V && "Added to wrong list?");
  }
}

void ValueHandleBase::AddToExistingUseListAfter(ValueHandleBase *List) {
  assert(List && "Must insert after existing node");

  Next = List->Next;
  setPrevPtr(&List->Next);
  List->Next = this;
  if (Next)
    Next->setPrevPtr(&Next);
}

void ValueHandleBase::AddToUseList() {
  assert(V && "Null pointer doesn't have a use list!");

  LLVMContextImpl *pImpl = V->getContext().pImpl;

  if (V->HasValueHandle) {
    // If this value already has a ValueHandle, then it must be in the
    // ValueHandles map already.
    ValueHandleBase *&Entry = pImpl->ValueHandles[V];
    assert(Entry && "Value doesn't have any handles?");
    AddToExistingUseList(&Entry);
    return;
  }

  // Ok, it doesn't have any handles yet, so we must insert it into the
  // DenseMap.  However, doing this insertion could cause the DenseMap to
  // reallocate itself, which would invalidate all of the PrevP pointers that
  // point into the old table.  Handle this by checking for reallocation and
  // updating the stale pointers only if needed.
  DenseMap<Value*, ValueHandleBase*> &Handles = pImpl->ValueHandles;
  const void *OldBucketPtr = Handles.getPointerIntoBucketsArray();

  ValueHandleBase *&Entry = Handles[V];
  assert(!Entry && "Value really did already have handles?");
  AddToExistingUseList(&Entry);
  V->HasValueHandle = true;

  // If reallocation didn't happen or if this was the first insertion, don't
  // walk the table.
  if (Handles.isPointerIntoBucketsArray(OldBucketPtr) ||
      Handles.size() == 1) {
    return;
  }

  // Okay, reallocation did happen.  Fix the Prev Pointers.
  for (DenseMap<Value*, ValueHandleBase*>::iterator I = Handles.begin(),
       E = Handles.end(); I != E; ++I) {
    assert(I->second && I->first == I->second->V &&
           "List invariant broken!");
    I->second->setPrevPtr(&I->second);
  }
}

void ValueHandleBase::RemoveFromUseList() {
  assert(V && V->HasValueHandle &&
         "Pointer doesn't have a use list!");

  // Unlink this from its use list.
  ValueHandleBase **PrevPtr = getPrevPtr();
  assert(*PrevPtr == this && "List invariant broken");

  *PrevPtr = Next;
  if (Next) {
    assert(Next->getPrevPtr() == &Next && "List invariant broken");
    Next->setPrevPtr(PrevPtr);
    return;
  }

  // If the Next pointer was null, then it is possible that this was the last
  // ValueHandle watching VP.  If so, delete its entry from the ValueHandles
  // map.
  LLVMContextImpl *pImpl = V->getContext().pImpl;
  DenseMap<Value*, ValueHandleBase*> &Handles = pImpl->ValueHandles;
  if (Handles.isPointerIntoBucketsArray(PrevPtr)) {
    Handles.erase(V);
    V->HasValueHandle = false;
  }
}


void ValueHandleBase::ValueIsDeleted(Value *V) {
  assert(V->HasValueHandle && "Should only be called if ValueHandles present");

  // Get the linked list base, which is guaranteed to exist since the
  // HasValueHandle flag is set.
  LLVMContextImpl *pImpl = V->getContext().pImpl;
  ValueHandleBase *Entry = pImpl->ValueHandles[V];
  assert(Entry && "Value bit set but no entries exist");

  // We use a local ValueHandleBase as an iterator so that ValueHandles can add
  // and remove themselves from the list without breaking our iteration.  This
  // is not really an AssertingVH; we just have to give ValueHandleBase a kind.
  // Note that we deliberately do not the support the case when dropping a value
  // handle results in a new value handle being permanently added to the list
  // (as might occur in theory for CallbackVH's): the new value handle will not
  // be processed and the checking code will mete out righteous punishment if
  // the handle is still present once we have finished processing all the other
  // value handles (it is fine to momentarily add then remove a value handle).
  for (ValueHandleBase Iterator(Assert, *Entry); Entry; Entry = Iterator.Next) {
    Iterator.RemoveFromUseList();
    Iterator.AddToExistingUseListAfter(Entry);
    assert(Entry->Next == &Iterator && "Loop invariant broken.");

    switch (Entry->getKind()) {
    case Assert:
      break;
    case Tracking:
      // Mark that this value has been deleted by setting it to an invalid Value
      // pointer.
      Entry->operator=(DenseMapInfo<Value *>::getTombstoneKey());
      break;
    case Weak:
      // Weak just goes to null, which will unlink it from the list.
      Entry->operator=(nullptr);
      break;
    case Callback:
      // Forward to the subclass's implementation.
      static_cast<CallbackVH*>(Entry)->deleted();
      break;
    }
  }

  // All callbacks, weak references, and assertingVHs should be dropped by now.
  if (V->HasValueHandle) {
#ifndef NDEBUG      // Only in +Asserts mode...
    dbgs() << "While deleting: " << *V->getType() << " %" << V->getName()
           << "\n";
    if (pImpl->ValueHandles[V]->getKind() == Assert)
      llvm_unreachable("An asserting value handle still pointed to this"
                       " value!");

#endif
    llvm_unreachable("All references to V were not removed?");
  }
}


void ValueHandleBase::ValueIsRAUWd(Value *Old, Value *New) {
  assert(Old->HasValueHandle &&"Should only be called if ValueHandles present");
  assert(Old != New && "Changing value into itself!");
  assert(Old->getType() == New->getType() &&
         "replaceAllUses of value with new value of different type!");

  // Get the linked list base, which is guaranteed to exist since the
  // HasValueHandle flag is set.
  LLVMContextImpl *pImpl = Old->getContext().pImpl;
  ValueHandleBase *Entry = pImpl->ValueHandles[Old];

  assert(Entry && "Value bit set but no entries exist");

  // We use a local ValueHandleBase as an iterator so that
  // ValueHandles can add and remove themselves from the list without
  // breaking our iteration.  This is not really an AssertingVH; we
  // just have to give ValueHandleBase some kind.
  for (ValueHandleBase Iterator(Assert, *Entry); Entry; Entry = Iterator.Next) {
    Iterator.RemoveFromUseList();
    Iterator.AddToExistingUseListAfter(Entry);
    assert(Entry->Next == &Iterator && "Loop invariant broken.");

    switch (Entry->getKind()) {
    case Assert:
      // Asserting handle does not follow RAUW implicitly.
      break;
    case Tracking:
      // Tracking goes to new value like a WeakVH. Note that this may make it
      // something incompatible with its templated type. We don't want to have a
      // virtual (or inline) interface to handle this though, so instead we make
      // the TrackingVH accessors guarantee that a client never sees this value.

      // FALLTHROUGH
    case Weak:
      // Weak goes to the new value, which will unlink it from Old's list.
      Entry->operator=(New);
      break;
    case Callback:
      // Forward to the subclass's implementation.
      static_cast<CallbackVH*>(Entry)->allUsesReplacedWith(New);
      break;
    }
  }

#ifndef NDEBUG
  // If any new tracking or weak value handles were added while processing the
  // list, then complain about it now.
  if (Old->HasValueHandle)
    for (Entry = pImpl->ValueHandles[Old]; Entry; Entry = Entry->Next)
      switch (Entry->getKind()) {
      case Tracking:
      case Weak:
        dbgs() << "After RAUW from " << *Old->getType() << " %"
               << Old->getName() << " to " << *New->getType() << " %"
               << New->getName() << "\n";
        llvm_unreachable("A tracking or weak value handle still pointed to the"
                         " old value!\n");
      default:
        break;
      }
#endif
}

// Pin the vtable to this file.
void CallbackVH::anchor() {}
