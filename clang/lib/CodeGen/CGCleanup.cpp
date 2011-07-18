//===--- CGCleanup.cpp - Bookkeeping and code emission for cleanups -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code dealing with the IR generation for cleanups
// and related information.
//
// A "cleanup" is a piece of code which needs to be executed whenever
// control transfers out of a particular scope.  This can be
// conditionalized to occur only on exceptional control flow, only on
// normal control flow, or both.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCleanup.h"

using namespace clang;
using namespace CodeGen;

bool DominatingValue<RValue>::saved_type::needsSaving(RValue rv) {
  if (rv.isScalar())
    return DominatingLLVMValue::needsSaving(rv.getScalarVal());
  if (rv.isAggregate())
    return DominatingLLVMValue::needsSaving(rv.getAggregateAddr());
  return true;
}

DominatingValue<RValue>::saved_type
DominatingValue<RValue>::saved_type::save(CodeGenFunction &CGF, RValue rv) {
  if (rv.isScalar()) {
    llvm::Value *V = rv.getScalarVal();

    // These automatically dominate and don't need to be saved.
    if (!DominatingLLVMValue::needsSaving(V))
      return saved_type(V, ScalarLiteral);

    // Everything else needs an alloca.
    llvm::Value *addr = CGF.CreateTempAlloca(V->getType(), "saved-rvalue");
    CGF.Builder.CreateStore(V, addr);
    return saved_type(addr, ScalarAddress);
  }

  if (rv.isComplex()) {
    CodeGenFunction::ComplexPairTy V = rv.getComplexVal();
    llvm::Type *ComplexTy =
      llvm::StructType::get(V.first->getType(), V.second->getType(),
                            (void*) 0);
    llvm::Value *addr = CGF.CreateTempAlloca(ComplexTy, "saved-complex");
    CGF.StoreComplexToAddr(V, addr, /*volatile*/ false);
    return saved_type(addr, ComplexAddress);
  }

  assert(rv.isAggregate());
  llvm::Value *V = rv.getAggregateAddr(); // TODO: volatile?
  if (!DominatingLLVMValue::needsSaving(V))
    return saved_type(V, AggregateLiteral);

  llvm::Value *addr = CGF.CreateTempAlloca(V->getType(), "saved-rvalue");
  CGF.Builder.CreateStore(V, addr);
  return saved_type(addr, AggregateAddress);  
}

/// Given a saved r-value produced by SaveRValue, perform the code
/// necessary to restore it to usability at the current insertion
/// point.
RValue DominatingValue<RValue>::saved_type::restore(CodeGenFunction &CGF) {
  switch (K) {
  case ScalarLiteral:
    return RValue::get(Value);
  case ScalarAddress:
    return RValue::get(CGF.Builder.CreateLoad(Value));
  case AggregateLiteral:
    return RValue::getAggregate(Value);
  case AggregateAddress:
    return RValue::getAggregate(CGF.Builder.CreateLoad(Value));
  case ComplexAddress:
    return RValue::getComplex(CGF.LoadComplexFromAddr(Value, false));
  }

  llvm_unreachable("bad saved r-value kind");
  return RValue();
}

/// Push an entry of the given size onto this protected-scope stack.
char *EHScopeStack::allocate(size_t Size) {
  if (!StartOfBuffer) {
    unsigned Capacity = 1024;
    while (Capacity < Size) Capacity *= 2;
    StartOfBuffer = new char[Capacity];
    StartOfData = EndOfBuffer = StartOfBuffer + Capacity;
  } else if (static_cast<size_t>(StartOfData - StartOfBuffer) < Size) {
    unsigned CurrentCapacity = EndOfBuffer - StartOfBuffer;
    unsigned UsedCapacity = CurrentCapacity - (StartOfData - StartOfBuffer);

    unsigned NewCapacity = CurrentCapacity;
    do {
      NewCapacity *= 2;
    } while (NewCapacity < UsedCapacity + Size);

    char *NewStartOfBuffer = new char[NewCapacity];
    char *NewEndOfBuffer = NewStartOfBuffer + NewCapacity;
    char *NewStartOfData = NewEndOfBuffer - UsedCapacity;
    memcpy(NewStartOfData, StartOfData, UsedCapacity);
    delete [] StartOfBuffer;
    StartOfBuffer = NewStartOfBuffer;
    EndOfBuffer = NewEndOfBuffer;
    StartOfData = NewStartOfData;
  }

  assert(StartOfBuffer + Size <= StartOfData);
  StartOfData -= Size;
  return StartOfData;
}

EHScopeStack::stable_iterator
EHScopeStack::getEnclosingEHCleanup(iterator it) const {
  assert(it != end());
  do {
    if (isa<EHCleanupScope>(*it)) {
      if (cast<EHCleanupScope>(*it).isEHCleanup())
        return stabilize(it);
      return cast<EHCleanupScope>(*it).getEnclosingEHCleanup();
    }
    ++it;
  } while (it != end());
  return stable_end();
}


void *EHScopeStack::pushCleanup(CleanupKind Kind, size_t Size) {
  assert(((Size % sizeof(void*)) == 0) && "cleanup type is misaligned");
  char *Buffer = allocate(EHCleanupScope::getSizeForCleanupSize(Size));
  bool IsNormalCleanup = Kind & NormalCleanup;
  bool IsEHCleanup = Kind & EHCleanup;
  bool IsActive = !(Kind & InactiveCleanup);
  EHCleanupScope *Scope =
    new (Buffer) EHCleanupScope(IsNormalCleanup,
                                IsEHCleanup,
                                IsActive,
                                Size,
                                BranchFixups.size(),
                                InnermostNormalCleanup,
                                InnermostEHCleanup);
  if (IsNormalCleanup)
    InnermostNormalCleanup = stable_begin();
  if (IsEHCleanup)
    InnermostEHCleanup = stable_begin();

  return Scope->getCleanupBuffer();
}

void EHScopeStack::popCleanup() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHCleanupScope>(*begin()));
  EHCleanupScope &Cleanup = cast<EHCleanupScope>(*begin());
  InnermostNormalCleanup = Cleanup.getEnclosingNormalCleanup();
  InnermostEHCleanup = Cleanup.getEnclosingEHCleanup();
  StartOfData += Cleanup.getAllocatedSize();

  if (empty()) NextEHDestIndex = FirstEHDestIndex;

  // Destroy the cleanup.
  Cleanup.~EHCleanupScope();

  // Check whether we can shrink the branch-fixups stack.
  if (!BranchFixups.empty()) {
    // If we no longer have any normal cleanups, all the fixups are
    // complete.
    if (!hasNormalCleanups())
      BranchFixups.clear();

    // Otherwise we can still trim out unnecessary nulls.
    else
      popNullFixups();
  }
}

EHFilterScope *EHScopeStack::pushFilter(unsigned NumFilters) {
  char *Buffer = allocate(EHFilterScope::getSizeForNumFilters(NumFilters));
  CatchDepth++;
  return new (Buffer) EHFilterScope(NumFilters);
}

void EHScopeStack::popFilter() {
  assert(!empty() && "popping exception stack when not empty");

  EHFilterScope &Filter = cast<EHFilterScope>(*begin());
  StartOfData += EHFilterScope::getSizeForNumFilters(Filter.getNumFilters());

  if (empty()) NextEHDestIndex = FirstEHDestIndex;

  assert(CatchDepth > 0 && "mismatched filter push/pop");
  CatchDepth--;
}

EHCatchScope *EHScopeStack::pushCatch(unsigned NumHandlers) {
  char *Buffer = allocate(EHCatchScope::getSizeForNumHandlers(NumHandlers));
  CatchDepth++;
  EHCatchScope *Scope = new (Buffer) EHCatchScope(NumHandlers);
  for (unsigned I = 0; I != NumHandlers; ++I)
    Scope->getHandlers()[I].Index = getNextEHDestIndex();
  return Scope;
}

void EHScopeStack::pushTerminate() {
  char *Buffer = allocate(EHTerminateScope::getSize());
  CatchDepth++;
  new (Buffer) EHTerminateScope(getNextEHDestIndex());
}

/// Remove any 'null' fixups on the stack.  However, we can't pop more
/// fixups than the fixup depth on the innermost normal cleanup, or
/// else fixups that we try to add to that cleanup will end up in the
/// wrong place.  We *could* try to shrink fixup depths, but that's
/// actually a lot of work for little benefit.
void EHScopeStack::popNullFixups() {
  // We expect this to only be called when there's still an innermost
  // normal cleanup;  otherwise there really shouldn't be any fixups.
  assert(hasNormalCleanups());

  EHScopeStack::iterator it = find(InnermostNormalCleanup);
  unsigned MinSize = cast<EHCleanupScope>(*it).getFixupDepth();
  assert(BranchFixups.size() >= MinSize && "fixup stack out of order");

  while (BranchFixups.size() > MinSize &&
         BranchFixups.back().Destination == 0)
    BranchFixups.pop_back();
}

void CodeGenFunction::initFullExprCleanup() {
  // Create a variable to decide whether the cleanup needs to be run.
  llvm::AllocaInst *active
    = CreateTempAlloca(Builder.getInt1Ty(), "cleanup.cond");

  // Initialize it to false at a site that's guaranteed to be run
  // before each evaluation.
  llvm::BasicBlock *block = OutermostConditional->getStartingBlock();
  new llvm::StoreInst(Builder.getFalse(), active, &block->back());

  // Initialize it to true at the current location.
  Builder.CreateStore(Builder.getTrue(), active);

  // Set that as the active flag in the cleanup.
  EHCleanupScope &cleanup = cast<EHCleanupScope>(*EHStack.begin());
  assert(cleanup.getActiveFlag() == 0 && "cleanup already has active flag?");
  cleanup.setActiveFlag(active);

  if (cleanup.isNormalCleanup()) cleanup.setTestFlagInNormalCleanup();
  if (cleanup.isEHCleanup()) cleanup.setTestFlagInEHCleanup();
}

void EHScopeStack::Cleanup::anchor() {}

/// All the branch fixups on the EH stack have propagated out past the
/// outermost normal cleanup; resolve them all by adding cases to the
/// given switch instruction.
static void ResolveAllBranchFixups(CodeGenFunction &CGF,
                                   llvm::SwitchInst *Switch,
                                   llvm::BasicBlock *CleanupEntry) {
  llvm::SmallPtrSet<llvm::BasicBlock*, 4> CasesAdded;

  for (unsigned I = 0, E = CGF.EHStack.getNumBranchFixups(); I != E; ++I) {
    // Skip this fixup if its destination isn't set.
    BranchFixup &Fixup = CGF.EHStack.getBranchFixup(I);
    if (Fixup.Destination == 0) continue;

    // If there isn't an OptimisticBranchBlock, then InitialBranch is
    // still pointing directly to its destination; forward it to the
    // appropriate cleanup entry.  This is required in the specific
    // case of
    //   { std::string s; goto lbl; }
    //   lbl:
    // i.e. where there's an unresolved fixup inside a single cleanup
    // entry which we're currently popping.
    if (Fixup.OptimisticBranchBlock == 0) {
      new llvm::StoreInst(CGF.Builder.getInt32(Fixup.DestinationIndex),
                          CGF.getNormalCleanupDestSlot(),
                          Fixup.InitialBranch);
      Fixup.InitialBranch->setSuccessor(0, CleanupEntry);
    }

    // Don't add this case to the switch statement twice.
    if (!CasesAdded.insert(Fixup.Destination)) continue;

    Switch->addCase(CGF.Builder.getInt32(Fixup.DestinationIndex),
                    Fixup.Destination);
  }

  CGF.EHStack.clearFixups();
}

/// Transitions the terminator of the given exit-block of a cleanup to
/// be a cleanup switch.
static llvm::SwitchInst *TransitionToCleanupSwitch(CodeGenFunction &CGF,
                                                   llvm::BasicBlock *Block) {
  // If it's a branch, turn it into a switch whose default
  // destination is its original target.
  llvm::TerminatorInst *Term = Block->getTerminator();
  assert(Term && "can't transition block without terminator");

  if (llvm::BranchInst *Br = dyn_cast<llvm::BranchInst>(Term)) {
    assert(Br->isUnconditional());
    llvm::LoadInst *Load =
      new llvm::LoadInst(CGF.getNormalCleanupDestSlot(), "cleanup.dest", Term);
    llvm::SwitchInst *Switch =
      llvm::SwitchInst::Create(Load, Br->getSuccessor(0), 4, Block);
    Br->eraseFromParent();
    return Switch;
  } else {
    return cast<llvm::SwitchInst>(Term);
  }
}

void CodeGenFunction::ResolveBranchFixups(llvm::BasicBlock *Block) {
  assert(Block && "resolving a null target block");
  if (!EHStack.getNumBranchFixups()) return;

  assert(EHStack.hasNormalCleanups() &&
         "branch fixups exist with no normal cleanups on stack");

  llvm::SmallPtrSet<llvm::BasicBlock*, 4> ModifiedOptimisticBlocks;
  bool ResolvedAny = false;

  for (unsigned I = 0, E = EHStack.getNumBranchFixups(); I != E; ++I) {
    // Skip this fixup if its destination doesn't match.
    BranchFixup &Fixup = EHStack.getBranchFixup(I);
    if (Fixup.Destination != Block) continue;

    Fixup.Destination = 0;
    ResolvedAny = true;

    // If it doesn't have an optimistic branch block, LatestBranch is
    // already pointing to the right place.
    llvm::BasicBlock *BranchBB = Fixup.OptimisticBranchBlock;
    if (!BranchBB)
      continue;

    // Don't process the same optimistic branch block twice.
    if (!ModifiedOptimisticBlocks.insert(BranchBB))
      continue;

    llvm::SwitchInst *Switch = TransitionToCleanupSwitch(*this, BranchBB);

    // Add a case to the switch.
    Switch->addCase(Builder.getInt32(Fixup.DestinationIndex), Block);
  }

  if (ResolvedAny)
    EHStack.popNullFixups();
}

/// Pops cleanup blocks until the given savepoint is reached.
void CodeGenFunction::PopCleanupBlocks(EHScopeStack::stable_iterator Old) {
  assert(Old.isValid());

  while (EHStack.stable_begin() != Old) {
    EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.begin());

    // As long as Old strictly encloses the scope's enclosing normal
    // cleanup, we're going to emit another normal cleanup which
    // fallthrough can propagate through.
    bool FallThroughIsBranchThrough =
      Old.strictlyEncloses(Scope.getEnclosingNormalCleanup());

    PopCleanupBlock(FallThroughIsBranchThrough);
  }
}

static llvm::BasicBlock *CreateNormalEntry(CodeGenFunction &CGF,
                                           EHCleanupScope &Scope) {
  assert(Scope.isNormalCleanup());
  llvm::BasicBlock *Entry = Scope.getNormalBlock();
  if (!Entry) {
    Entry = CGF.createBasicBlock("cleanup");
    Scope.setNormalBlock(Entry);
  }
  return Entry;
}

static llvm::BasicBlock *CreateEHEntry(CodeGenFunction &CGF,
                                       EHCleanupScope &Scope) {
  assert(Scope.isEHCleanup());
  llvm::BasicBlock *Entry = Scope.getEHBlock();
  if (!Entry) {
    Entry = CGF.createBasicBlock("eh.cleanup");
    Scope.setEHBlock(Entry);
  }
  return Entry;
}

/// Attempts to reduce a cleanup's entry block to a fallthrough.  This
/// is basically llvm::MergeBlockIntoPredecessor, except
/// simplified/optimized for the tighter constraints on cleanup blocks.
///
/// Returns the new block, whatever it is.
static llvm::BasicBlock *SimplifyCleanupEntry(CodeGenFunction &CGF,
                                              llvm::BasicBlock *Entry) {
  llvm::BasicBlock *Pred = Entry->getSinglePredecessor();
  if (!Pred) return Entry;

  llvm::BranchInst *Br = dyn_cast<llvm::BranchInst>(Pred->getTerminator());
  if (!Br || Br->isConditional()) return Entry;
  assert(Br->getSuccessor(0) == Entry);

  // If we were previously inserting at the end of the cleanup entry
  // block, we'll need to continue inserting at the end of the
  // predecessor.
  bool WasInsertBlock = CGF.Builder.GetInsertBlock() == Entry;
  assert(!WasInsertBlock || CGF.Builder.GetInsertPoint() == Entry->end());

  // Kill the branch.
  Br->eraseFromParent();

  // Replace all uses of the entry with the predecessor, in case there
  // are phis in the cleanup.
  Entry->replaceAllUsesWith(Pred);

  // Merge the blocks.
  Pred->getInstList().splice(Pred->end(), Entry->getInstList());

  // Kill the entry block.
  Entry->eraseFromParent();

  if (WasInsertBlock)
    CGF.Builder.SetInsertPoint(Pred);

  return Pred;
}

static void EmitCleanup(CodeGenFunction &CGF,
                        EHScopeStack::Cleanup *Fn,
                        EHScopeStack::Cleanup::Flags flags,
                        llvm::Value *ActiveFlag) {
  // EH cleanups always occur within a terminate scope.
  if (flags.isForEHCleanup()) CGF.EHStack.pushTerminate();

  // If there's an active flag, load it and skip the cleanup if it's
  // false.
  llvm::BasicBlock *ContBB = 0;
  if (ActiveFlag) {
    ContBB = CGF.createBasicBlock("cleanup.done");
    llvm::BasicBlock *CleanupBB = CGF.createBasicBlock("cleanup.action");
    llvm::Value *IsActive
      = CGF.Builder.CreateLoad(ActiveFlag, "cleanup.is_active");
    CGF.Builder.CreateCondBr(IsActive, CleanupBB, ContBB);
    CGF.EmitBlock(CleanupBB);
  }

  // Ask the cleanup to emit itself.
  Fn->Emit(CGF, flags);
  assert(CGF.HaveInsertPoint() && "cleanup ended with no insertion point?");

  // Emit the continuation block if there was an active flag.
  if (ActiveFlag)
    CGF.EmitBlock(ContBB);

  // Leave the terminate scope.
  if (flags.isForEHCleanup()) CGF.EHStack.popTerminate();
}

static void ForwardPrebranchedFallthrough(llvm::BasicBlock *Exit,
                                          llvm::BasicBlock *From,
                                          llvm::BasicBlock *To) {
  // Exit is the exit block of a cleanup, so it always terminates in
  // an unconditional branch or a switch.
  llvm::TerminatorInst *Term = Exit->getTerminator();

  if (llvm::BranchInst *Br = dyn_cast<llvm::BranchInst>(Term)) {
    assert(Br->isUnconditional() && Br->getSuccessor(0) == From);
    Br->setSuccessor(0, To);
  } else {
    llvm::SwitchInst *Switch = cast<llvm::SwitchInst>(Term);
    for (unsigned I = 0, E = Switch->getNumSuccessors(); I != E; ++I)
      if (Switch->getSuccessor(I) == From)
        Switch->setSuccessor(I, To);
  }
}

/// Pops a cleanup block.  If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CodeGenFunction::PopCleanupBlock(bool FallthroughIsBranchThrough) {
  assert(!EHStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*EHStack.begin()) && "top not a cleanup!");
  EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.begin());
  assert(Scope.getFixupDepth() <= EHStack.getNumBranchFixups());

  // Remember activation information.
  bool IsActive = Scope.isActive();
  llvm::Value *NormalActiveFlag =
    Scope.shouldTestFlagInNormalCleanup() ? Scope.getActiveFlag() : 0;
  llvm::Value *EHActiveFlag = 
    Scope.shouldTestFlagInEHCleanup() ? Scope.getActiveFlag() : 0;

  // Check whether we need an EH cleanup.  This is only true if we've
  // generated a lazy EH cleanup block.
  bool RequiresEHCleanup = Scope.hasEHBranches();

  // Check the three conditions which might require a normal cleanup:

  // - whether there are branch fix-ups through this cleanup
  unsigned FixupDepth = Scope.getFixupDepth();
  bool HasFixups = EHStack.getNumBranchFixups() != FixupDepth;

  // - whether there are branch-throughs or branch-afters
  bool HasExistingBranches = Scope.hasBranches();

  // - whether there's a fallthrough
  llvm::BasicBlock *FallthroughSource = Builder.GetInsertBlock();
  bool HasFallthrough = (FallthroughSource != 0 && IsActive);

  // Branch-through fall-throughs leave the insertion point set to the
  // end of the last cleanup, which points to the current scope.  The
  // rest of IR gen doesn't need to worry about this; it only happens
  // during the execution of PopCleanupBlocks().
  bool HasPrebranchedFallthrough =
    (FallthroughSource && FallthroughSource->getTerminator());

  // If this is a normal cleanup, then having a prebranched
  // fallthrough implies that the fallthrough source unconditionally
  // jumps here.
  assert(!Scope.isNormalCleanup() || !HasPrebranchedFallthrough ||
         (Scope.getNormalBlock() &&
          FallthroughSource->getTerminator()->getSuccessor(0)
            == Scope.getNormalBlock()));

  bool RequiresNormalCleanup = false;
  if (Scope.isNormalCleanup() &&
      (HasFixups || HasExistingBranches || HasFallthrough)) {
    RequiresNormalCleanup = true;
  }

  EHScopeStack::Cleanup::Flags cleanupFlags;
  if (Scope.isNormalCleanup())
    cleanupFlags.setIsNormalCleanupKind();
  if (Scope.isEHCleanup())
    cleanupFlags.setIsEHCleanupKind();

  // Even if we don't need the normal cleanup, we might still have
  // prebranched fallthrough to worry about.
  if (Scope.isNormalCleanup() && !RequiresNormalCleanup &&
      HasPrebranchedFallthrough) {
    assert(!IsActive);

    llvm::BasicBlock *NormalEntry = Scope.getNormalBlock();

    // If we're branching through this cleanup, just forward the
    // prebranched fallthrough to the next cleanup, leaving the insert
    // point in the old block.
    if (FallthroughIsBranchThrough) {
      EHScope &S = *EHStack.find(Scope.getEnclosingNormalCleanup());
      llvm::BasicBlock *EnclosingEntry = 
        CreateNormalEntry(*this, cast<EHCleanupScope>(S));

      ForwardPrebranchedFallthrough(FallthroughSource,
                                    NormalEntry, EnclosingEntry);
      assert(NormalEntry->use_empty() &&
             "uses of entry remain after forwarding?");
      delete NormalEntry;

    // Otherwise, we're branching out;  just emit the next block.
    } else {
      EmitBlock(NormalEntry);
      SimplifyCleanupEntry(*this, NormalEntry);
    }
  }

  // If we don't need the cleanup at all, we're done.
  if (!RequiresNormalCleanup && !RequiresEHCleanup) {
    EHStack.popCleanup(); // safe because there are no fixups
    assert(EHStack.getNumBranchFixups() == 0 ||
           EHStack.hasNormalCleanups());
    return;
  }

  // Copy the cleanup emission data out.  Note that SmallVector
  // guarantees maximal alignment for its buffer regardless of its
  // type parameter.
  llvm::SmallVector<char, 8*sizeof(void*)> CleanupBuffer;
  CleanupBuffer.reserve(Scope.getCleanupSize());
  memcpy(CleanupBuffer.data(),
         Scope.getCleanupBuffer(), Scope.getCleanupSize());
  CleanupBuffer.set_size(Scope.getCleanupSize());
  EHScopeStack::Cleanup *Fn =
    reinterpret_cast<EHScopeStack::Cleanup*>(CleanupBuffer.data());

  // We want to emit the EH cleanup after the normal cleanup, but go
  // ahead and do the setup for the EH cleanup while the scope is still
  // alive.
  llvm::BasicBlock *EHEntry = 0;
  llvm::SmallVector<llvm::Instruction*, 2> EHInstsToAppend;
  if (RequiresEHCleanup) {
    EHEntry = CreateEHEntry(*this, Scope);

    // Figure out the branch-through dest if necessary.
    llvm::BasicBlock *EHBranchThroughDest = 0;
    if (Scope.hasEHBranchThroughs()) {
      assert(Scope.getEnclosingEHCleanup() != EHStack.stable_end());
      EHScope &S = *EHStack.find(Scope.getEnclosingEHCleanup());
      EHBranchThroughDest = CreateEHEntry(*this, cast<EHCleanupScope>(S));
    }

    // If we have exactly one branch-after and no branch-throughs, we
    // can dispatch it without a switch.
    if (!Scope.hasEHBranchThroughs() &&
        Scope.getNumEHBranchAfters() == 1) {
      assert(!EHBranchThroughDest);

      // TODO: remove the spurious eh.cleanup.dest stores if this edge
      // never went through any switches.
      llvm::BasicBlock *BranchAfterDest = Scope.getEHBranchAfterBlock(0);
      EHInstsToAppend.push_back(llvm::BranchInst::Create(BranchAfterDest));
    
    // Otherwise, if we have any branch-afters, we need a switch.
    } else if (Scope.getNumEHBranchAfters()) {
      // The default of the switch belongs to the branch-throughs if
      // they exist.
      llvm::BasicBlock *Default =
        (EHBranchThroughDest ? EHBranchThroughDest : getUnreachableBlock());

      const unsigned SwitchCapacity = Scope.getNumEHBranchAfters();

      llvm::LoadInst *Load =
        new llvm::LoadInst(getEHCleanupDestSlot(), "cleanup.dest");
      llvm::SwitchInst *Switch =
        llvm::SwitchInst::Create(Load, Default, SwitchCapacity);

      EHInstsToAppend.push_back(Load);
      EHInstsToAppend.push_back(Switch);

      for (unsigned I = 0, E = Scope.getNumEHBranchAfters(); I != E; ++I)
        Switch->addCase(Scope.getEHBranchAfterIndex(I),
                        Scope.getEHBranchAfterBlock(I));

    // Otherwise, we have only branch-throughs; jump to the next EH
    // cleanup.
    } else {
      assert(EHBranchThroughDest);
      EHInstsToAppend.push_back(llvm::BranchInst::Create(EHBranchThroughDest));
    }
  }

  if (!RequiresNormalCleanup) {
    EHStack.popCleanup();
  } else {
    // If we have a fallthrough and no other need for the cleanup,
    // emit it directly.
    if (HasFallthrough && !HasPrebranchedFallthrough &&
        !HasFixups && !HasExistingBranches) {

      // Fixups can cause us to optimistically create a normal block,
      // only to later have no real uses for it.  Just delete it in
      // this case.
      // TODO: we can potentially simplify all the uses after this.
      if (Scope.getNormalBlock()) {
        Scope.getNormalBlock()->replaceAllUsesWith(getUnreachableBlock());
        delete Scope.getNormalBlock();
      }

      EHStack.popCleanup();

      EmitCleanup(*this, Fn, cleanupFlags, NormalActiveFlag);

    // Otherwise, the best approach is to thread everything through
    // the cleanup block and then try to clean up after ourselves.
    } else {
      // Force the entry block to exist.
      llvm::BasicBlock *NormalEntry = CreateNormalEntry(*this, Scope);

      // I.  Set up the fallthrough edge in.

      // If there's a fallthrough, we need to store the cleanup
      // destination index.  For fall-throughs this is always zero.
      if (HasFallthrough) {
        if (!HasPrebranchedFallthrough)
          Builder.CreateStore(Builder.getInt32(0), getNormalCleanupDestSlot());

      // Otherwise, clear the IP if we don't have fallthrough because
      // the cleanup is inactive.  We don't need to save it because
      // it's still just FallthroughSource.
      } else if (FallthroughSource) {
        assert(!IsActive && "source without fallthrough for active cleanup");
        Builder.ClearInsertionPoint();
      }

      // II.  Emit the entry block.  This implicitly branches to it if
      // we have fallthrough.  All the fixups and existing branches
      // should already be branched to it.
      EmitBlock(NormalEntry);

      // III.  Figure out where we're going and build the cleanup
      // epilogue.

      bool HasEnclosingCleanups =
        (Scope.getEnclosingNormalCleanup() != EHStack.stable_end());

      // Compute the branch-through dest if we need it:
      //   - if there are branch-throughs threaded through the scope
      //   - if fall-through is a branch-through
      //   - if there are fixups that will be optimistically forwarded
      //     to the enclosing cleanup
      llvm::BasicBlock *BranchThroughDest = 0;
      if (Scope.hasBranchThroughs() ||
          (FallthroughSource && FallthroughIsBranchThrough) ||
          (HasFixups && HasEnclosingCleanups)) {
        assert(HasEnclosingCleanups);
        EHScope &S = *EHStack.find(Scope.getEnclosingNormalCleanup());
        BranchThroughDest = CreateNormalEntry(*this, cast<EHCleanupScope>(S));
      }

      llvm::BasicBlock *FallthroughDest = 0;
      llvm::SmallVector<llvm::Instruction*, 2> InstsToAppend;

      // If there's exactly one branch-after and no other threads,
      // we can route it without a switch.
      if (!Scope.hasBranchThroughs() && !HasFixups && !HasFallthrough &&
          Scope.getNumBranchAfters() == 1) {
        assert(!BranchThroughDest || !IsActive);

        // TODO: clean up the possibly dead stores to the cleanup dest slot.
        llvm::BasicBlock *BranchAfter = Scope.getBranchAfterBlock(0);
        InstsToAppend.push_back(llvm::BranchInst::Create(BranchAfter));

      // Build a switch-out if we need it:
      //   - if there are branch-afters threaded through the scope
      //   - if fall-through is a branch-after
      //   - if there are fixups that have nowhere left to go and
      //     so must be immediately resolved
      } else if (Scope.getNumBranchAfters() ||
                 (HasFallthrough && !FallthroughIsBranchThrough) ||
                 (HasFixups && !HasEnclosingCleanups)) {

        llvm::BasicBlock *Default =
          (BranchThroughDest ? BranchThroughDest : getUnreachableBlock());

        // TODO: base this on the number of branch-afters and fixups
        const unsigned SwitchCapacity = 10;

        llvm::LoadInst *Load =
          new llvm::LoadInst(getNormalCleanupDestSlot(), "cleanup.dest");
        llvm::SwitchInst *Switch =
          llvm::SwitchInst::Create(Load, Default, SwitchCapacity);

        InstsToAppend.push_back(Load);
        InstsToAppend.push_back(Switch);

        // Branch-after fallthrough.
        if (FallthroughSource && !FallthroughIsBranchThrough) {
          FallthroughDest = createBasicBlock("cleanup.cont");
          if (HasFallthrough)
            Switch->addCase(Builder.getInt32(0), FallthroughDest);
        }

        for (unsigned I = 0, E = Scope.getNumBranchAfters(); I != E; ++I) {
          Switch->addCase(Scope.getBranchAfterIndex(I),
                          Scope.getBranchAfterBlock(I));
        }

        // If there aren't any enclosing cleanups, we can resolve all
        // the fixups now.
        if (HasFixups && !HasEnclosingCleanups)
          ResolveAllBranchFixups(*this, Switch, NormalEntry);
      } else {
        // We should always have a branch-through destination in this case.
        assert(BranchThroughDest);
        InstsToAppend.push_back(llvm::BranchInst::Create(BranchThroughDest));
      }

      // IV.  Pop the cleanup and emit it.
      EHStack.popCleanup();
      assert(EHStack.hasNormalCleanups() == HasEnclosingCleanups);

      EmitCleanup(*this, Fn, cleanupFlags, NormalActiveFlag);

      // Append the prepared cleanup prologue from above.
      llvm::BasicBlock *NormalExit = Builder.GetInsertBlock();
      for (unsigned I = 0, E = InstsToAppend.size(); I != E; ++I)
        NormalExit->getInstList().push_back(InstsToAppend[I]);

      // Optimistically hope that any fixups will continue falling through.
      for (unsigned I = FixupDepth, E = EHStack.getNumBranchFixups();
           I < E; ++I) {
        BranchFixup &Fixup = EHStack.getBranchFixup(I);
        if (!Fixup.Destination) continue;
        if (!Fixup.OptimisticBranchBlock) {
          new llvm::StoreInst(Builder.getInt32(Fixup.DestinationIndex),
                              getNormalCleanupDestSlot(),
                              Fixup.InitialBranch);
          Fixup.InitialBranch->setSuccessor(0, NormalEntry);
        }
        Fixup.OptimisticBranchBlock = NormalExit;
      }

      // V.  Set up the fallthrough edge out.
      
      // Case 1: a fallthrough source exists but shouldn't branch to
      // the cleanup because the cleanup is inactive.
      if (!HasFallthrough && FallthroughSource) {
        assert(!IsActive);

        // If we have a prebranched fallthrough, that needs to be
        // forwarded to the right block.
        if (HasPrebranchedFallthrough) {
          llvm::BasicBlock *Next;
          if (FallthroughIsBranchThrough) {
            Next = BranchThroughDest;
            assert(!FallthroughDest);
          } else {
            Next = FallthroughDest;
          }

          ForwardPrebranchedFallthrough(FallthroughSource, NormalEntry, Next);
        }
        Builder.SetInsertPoint(FallthroughSource);

      // Case 2: a fallthrough source exists and should branch to the
      // cleanup, but we're not supposed to branch through to the next
      // cleanup.
      } else if (HasFallthrough && FallthroughDest) {
        assert(!FallthroughIsBranchThrough);
        EmitBlock(FallthroughDest);

      // Case 3: a fallthrough source exists and should branch to the
      // cleanup and then through to the next.
      } else if (HasFallthrough) {
        // Everything is already set up for this.

      // Case 4: no fallthrough source exists.
      } else {
        Builder.ClearInsertionPoint();
      }

      // VI.  Assorted cleaning.

      // Check whether we can merge NormalEntry into a single predecessor.
      // This might invalidate (non-IR) pointers to NormalEntry.
      llvm::BasicBlock *NewNormalEntry =
        SimplifyCleanupEntry(*this, NormalEntry);

      // If it did invalidate those pointers, and NormalEntry was the same
      // as NormalExit, go back and patch up the fixups.
      if (NewNormalEntry != NormalEntry && NormalEntry == NormalExit)
        for (unsigned I = FixupDepth, E = EHStack.getNumBranchFixups();
               I < E; ++I)
          EHStack.getBranchFixup(I).OptimisticBranchBlock = NewNormalEntry;
    }
  }

  assert(EHStack.hasNormalCleanups() || EHStack.getNumBranchFixups() == 0);

  // Emit the EH cleanup if required.
  if (RequiresEHCleanup) {
    CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

    EmitBlock(EHEntry);

    cleanupFlags.setIsForEHCleanup();
    EmitCleanup(*this, Fn, cleanupFlags, EHActiveFlag);

    // Append the prepared cleanup prologue from above.
    llvm::BasicBlock *EHExit = Builder.GetInsertBlock();
    for (unsigned I = 0, E = EHInstsToAppend.size(); I != E; ++I)
      EHExit->getInstList().push_back(EHInstsToAppend[I]);

    Builder.restoreIP(SavedIP);

    SimplifyCleanupEntry(*this, EHEntry);
  }
}

/// isObviouslyBranchWithoutCleanups - Return true if a branch to the
/// specified destination obviously has no cleanups to run.  'false' is always
/// a conservatively correct answer for this method.
bool CodeGenFunction::isObviouslyBranchWithoutCleanups(JumpDest Dest) const {
  assert(Dest.getScopeDepth().encloses(EHStack.stable_begin())
         && "stale jump destination");
  
  // Calculate the innermost active normal cleanup.
  EHScopeStack::stable_iterator TopCleanup =
    EHStack.getInnermostActiveNormalCleanup();
  
  // If we're not in an active normal cleanup scope, or if the
  // destination scope is within the innermost active normal cleanup
  // scope, we don't need to worry about fixups.
  if (TopCleanup == EHStack.stable_end() ||
      TopCleanup.encloses(Dest.getScopeDepth())) // works for invalid
    return true;

  // Otherwise, we might need some cleanups.
  return false;
}


/// Terminate the current block by emitting a branch which might leave
/// the current cleanup-protected scope.  The target scope may not yet
/// be known, in which case this will require a fixup.
///
/// As a side-effect, this method clears the insertion point.
void CodeGenFunction::EmitBranchThroughCleanup(JumpDest Dest) {
  assert(Dest.getScopeDepth().encloses(EHStack.stable_begin())
         && "stale jump destination");

  if (!HaveInsertPoint())
    return;

  // Create the branch.
  llvm::BranchInst *BI = Builder.CreateBr(Dest.getBlock());

  // Calculate the innermost active normal cleanup.
  EHScopeStack::stable_iterator
    TopCleanup = EHStack.getInnermostActiveNormalCleanup();

  // If we're not in an active normal cleanup scope, or if the
  // destination scope is within the innermost active normal cleanup
  // scope, we don't need to worry about fixups.
  if (TopCleanup == EHStack.stable_end() ||
      TopCleanup.encloses(Dest.getScopeDepth())) { // works for invalid
    Builder.ClearInsertionPoint();
    return;
  }

  // If we can't resolve the destination cleanup scope, just add this
  // to the current cleanup scope as a branch fixup.
  if (!Dest.getScopeDepth().isValid()) {
    BranchFixup &Fixup = EHStack.addBranchFixup();
    Fixup.Destination = Dest.getBlock();
    Fixup.DestinationIndex = Dest.getDestIndex();
    Fixup.InitialBranch = BI;
    Fixup.OptimisticBranchBlock = 0;

    Builder.ClearInsertionPoint();
    return;
  }

  // Otherwise, thread through all the normal cleanups in scope.

  // Store the index at the start.
  llvm::ConstantInt *Index = Builder.getInt32(Dest.getDestIndex());
  new llvm::StoreInst(Index, getNormalCleanupDestSlot(), BI);

  // Adjust BI to point to the first cleanup block.
  {
    EHCleanupScope &Scope =
      cast<EHCleanupScope>(*EHStack.find(TopCleanup));
    BI->setSuccessor(0, CreateNormalEntry(*this, Scope));
  }

  // Add this destination to all the scopes involved.
  EHScopeStack::stable_iterator I = TopCleanup;
  EHScopeStack::stable_iterator E = Dest.getScopeDepth();
  if (E.strictlyEncloses(I)) {
    while (true) {
      EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.find(I));
      assert(Scope.isNormalCleanup());
      I = Scope.getEnclosingNormalCleanup();

      // If this is the last cleanup we're propagating through, tell it
      // that there's a resolved jump moving through it.
      if (!E.strictlyEncloses(I)) {
        Scope.addBranchAfter(Index, Dest.getBlock());
        break;
      }

      // Otherwise, tell the scope that there's a jump propoagating
      // through it.  If this isn't new information, all the rest of
      // the work has been done before.
      if (!Scope.addBranchThrough(Dest.getBlock()))
        break;
    }
  }
  
  Builder.ClearInsertionPoint();
}

void CodeGenFunction::EmitBranchThroughEHCleanup(UnwindDest Dest) {
  // We should never get invalid scope depths for an UnwindDest; that
  // implies that the destination wasn't set up correctly.
  assert(Dest.getScopeDepth().isValid() && "invalid scope depth on EH dest?");

  if (!HaveInsertPoint())
    return;

  // Create the branch.
  llvm::BranchInst *BI = Builder.CreateBr(Dest.getBlock());

  // Calculate the innermost active cleanup.
  EHScopeStack::stable_iterator
    InnermostCleanup = EHStack.getInnermostActiveEHCleanup();

  // If the destination is in the same EH cleanup scope as us, we
  // don't need to thread through anything.
  if (InnermostCleanup.encloses(Dest.getScopeDepth())) {
    Builder.ClearInsertionPoint();
    return;
  }
  assert(InnermostCleanup != EHStack.stable_end());

  // Store the index at the start.
  llvm::ConstantInt *Index = Builder.getInt32(Dest.getDestIndex());
  new llvm::StoreInst(Index, getEHCleanupDestSlot(), BI);

  // Adjust BI to point to the first cleanup block.
  {
    EHCleanupScope &Scope =
      cast<EHCleanupScope>(*EHStack.find(InnermostCleanup));
    BI->setSuccessor(0, CreateEHEntry(*this, Scope));
  }
  
  // Add this destination to all the scopes involved.
  for (EHScopeStack::stable_iterator
         I = InnermostCleanup, E = Dest.getScopeDepth(); ; ) {
    assert(E.strictlyEncloses(I));
    EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.find(I));
    assert(Scope.isEHCleanup());
    I = Scope.getEnclosingEHCleanup();

    // If this is the last cleanup we're propagating through, add this
    // as a branch-after.
    if (I == E) {
      Scope.addEHBranchAfter(Index, Dest.getBlock());
      break;
    }

    // Otherwise, add it as a branch-through.  If this isn't new
    // information, all the rest of the work has been done before.
    if (!Scope.addEHBranchThrough(Dest.getBlock()))
      break;
  }
  
  Builder.ClearInsertionPoint();
}

static bool IsUsedAsNormalCleanup(EHScopeStack &EHStack,
                                  EHScopeStack::stable_iterator C) {
  // If we needed a normal block for any reason, that counts.
  if (cast<EHCleanupScope>(*EHStack.find(C)).getNormalBlock())
    return true;

  // Check whether any enclosed cleanups were needed.
  for (EHScopeStack::stable_iterator
         I = EHStack.getInnermostNormalCleanup();
         I != C; ) {
    assert(C.strictlyEncloses(I));
    EHCleanupScope &S = cast<EHCleanupScope>(*EHStack.find(I));
    if (S.getNormalBlock()) return true;
    I = S.getEnclosingNormalCleanup();
  }

  return false;
}

static bool IsUsedAsEHCleanup(EHScopeStack &EHStack,
                              EHScopeStack::stable_iterator C) {
  // If we needed an EH block for any reason, that counts.
  if (cast<EHCleanupScope>(*EHStack.find(C)).getEHBlock())
    return true;

  // Check whether any enclosed cleanups were needed.
  for (EHScopeStack::stable_iterator
         I = EHStack.getInnermostEHCleanup(); I != C; ) {
    assert(C.strictlyEncloses(I));
    EHCleanupScope &S = cast<EHCleanupScope>(*EHStack.find(I));
    if (S.getEHBlock()) return true;
    I = S.getEnclosingEHCleanup();
  }

  return false;
}

enum ForActivation_t {
  ForActivation,
  ForDeactivation
};

/// The given cleanup block is changing activation state.  Configure a
/// cleanup variable if necessary.
///
/// It would be good if we had some way of determining if there were
/// extra uses *after* the change-over point.
static void SetupCleanupBlockActivation(CodeGenFunction &CGF,
                                        EHScopeStack::stable_iterator C,
                                        ForActivation_t Kind) {
  EHCleanupScope &Scope = cast<EHCleanupScope>(*CGF.EHStack.find(C));

  // We always need the flag if we're activating the cleanup, because
  // we have to assume that the current location doesn't necessarily
  // dominate all future uses of the cleanup.
  bool NeedFlag = (Kind == ForActivation);

  // Calculate whether the cleanup was used:

  //   - as a normal cleanup
  if (Scope.isNormalCleanup() && IsUsedAsNormalCleanup(CGF.EHStack, C)) {
    Scope.setTestFlagInNormalCleanup();
    NeedFlag = true;
  }

  //  - as an EH cleanup
  if (Scope.isEHCleanup() && IsUsedAsEHCleanup(CGF.EHStack, C)) {
    Scope.setTestFlagInEHCleanup();
    NeedFlag = true;
  }

  // If it hasn't yet been used as either, we're done.
  if (!NeedFlag) return;

  llvm::AllocaInst *Var = Scope.getActiveFlag();
  if (!Var) {
    Var = CGF.CreateTempAlloca(CGF.Builder.getInt1Ty(), "cleanup.isactive");
    Scope.setActiveFlag(Var);

    // Initialize to true or false depending on whether it was
    // active up to this point.
    CGF.InitTempAlloca(Var, CGF.Builder.getInt1(Kind == ForDeactivation));
  }

  CGF.Builder.CreateStore(CGF.Builder.getInt1(Kind == ForActivation), Var);
}

/// Activate a cleanup that was created in an inactivated state.
void CodeGenFunction::ActivateCleanupBlock(EHScopeStack::stable_iterator C) {
  assert(C != EHStack.stable_end() && "activating bottom of stack?");
  EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.find(C));
  assert(!Scope.isActive() && "double activation");

  SetupCleanupBlockActivation(*this, C, ForActivation);

  Scope.setActive(true);
}

/// Deactive a cleanup that was created in an active state.
void CodeGenFunction::DeactivateCleanupBlock(EHScopeStack::stable_iterator C) {
  assert(C != EHStack.stable_end() && "deactivating bottom of stack?");
  EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.find(C));
  assert(Scope.isActive() && "double deactivation");

  // If it's the top of the stack, just pop it.
  if (C == EHStack.stable_begin()) {
    // If it's a normal cleanup, we need to pretend that the
    // fallthrough is unreachable.
    CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();
    PopCleanupBlock();
    Builder.restoreIP(SavedIP);
    return;
  }

  // Otherwise, follow the general case.
  SetupCleanupBlockActivation(*this, C, ForDeactivation);

  Scope.setActive(false);
}

llvm::Value *CodeGenFunction::getNormalCleanupDestSlot() {
  if (!NormalCleanupDest)
    NormalCleanupDest =
      CreateTempAlloca(Builder.getInt32Ty(), "cleanup.dest.slot");
  return NormalCleanupDest;
}

llvm::Value *CodeGenFunction::getEHCleanupDestSlot() {
  if (!EHCleanupDest)
    EHCleanupDest =
      CreateTempAlloca(Builder.getInt32Ty(), "eh.cleanup.dest.slot");
  return EHCleanupDest;
}
