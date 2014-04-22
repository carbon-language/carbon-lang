//===- ObjCARCContract.cpp - ObjC ARC Optimization ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines late ObjC ARC optimizations. ARC stands for Automatic
/// Reference Counting and is a system for managing reference counts for objects
/// in Objective C.
///
/// This specific file mainly deals with ``contracting'' multiple lower level
/// operations into singular higher level operations through pattern matching.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

// TODO: ObjCARCContract could insert PHI nodes when uses aren't
// dominated by single calls.

#include "ObjCARC.h"
#include "ARCRuntimeEntryPoints.h"
#include "DependencyAnalysis.h"
#include "ProvenanceAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::objcarc;

#define DEBUG_TYPE "objc-arc-contract"

STATISTIC(NumPeeps,       "Number of calls peephole-optimized");
STATISTIC(NumStoreStrongs, "Number objc_storeStrong calls formed");

namespace {
  /// \brief Late ARC optimizations
  ///
  /// These change the IR in a way that makes it difficult to be analyzed by
  /// ObjCARCOpt, so it's run late.
  class ObjCARCContract : public FunctionPass {
    bool Changed;
    AliasAnalysis *AA;
    DominatorTree *DT;
    ProvenanceAnalysis PA;
    ARCRuntimeEntryPoints EP;

    /// A flag indicating whether this optimization pass should run.
    bool Run;

    /// The inline asm string to insert between calls and RetainRV calls to make
    /// the optimization work on targets which need it.
    const MDString *RetainRVMarker;

    /// The set of inserted objc_storeStrong calls. If at the end of walking the
    /// function we have found no alloca instructions, these calls can be marked
    /// "tail".
    SmallPtrSet<CallInst *, 8> StoreStrongCalls;

    bool OptimizeRetainCall(Function &F, Instruction *Retain);

    bool ContractAutorelease(Function &F, Instruction *Autorelease,
                             InstructionClass Class,
                             SmallPtrSet<Instruction *, 4>
                               &DependingInstructions,
                             SmallPtrSet<const BasicBlock *, 4>
                               &Visited);

    void ContractRelease(Instruction *Release,
                         inst_iterator &Iter);

    void getAnalysisUsage(AnalysisUsage &AU) const override;
    bool doInitialization(Module &M) override;
    bool runOnFunction(Function &F) override;

  public:
    static char ID;
    ObjCARCContract() : FunctionPass(ID) {
      initializeObjCARCContractPass(*PassRegistry::getPassRegistry());
    }
  };
}

char ObjCARCContract::ID = 0;
INITIALIZE_PASS_BEGIN(ObjCARCContract,
                      "objc-arc-contract", "ObjC ARC contraction", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(ObjCARCContract,
                    "objc-arc-contract", "ObjC ARC contraction", false, false)

Pass *llvm::createObjCARCContractPass() {
  return new ObjCARCContract();
}

void ObjCARCContract::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.setPreservesCFG();
}

/// Turn objc_retain into objc_retainAutoreleasedReturnValue if the operand is a
/// return value. We do this late so we do not disrupt the dataflow analysis in
/// ObjCARCOpt.
bool
ObjCARCContract::OptimizeRetainCall(Function &F, Instruction *Retain) {
  ImmutableCallSite CS(GetObjCArg(Retain));
  const Instruction *Call = CS.getInstruction();
  if (!Call)
    return false;
  if (Call->getParent() != Retain->getParent())
    return false;

  // Check that the call is next to the retain.
  BasicBlock::const_iterator I = Call;
  ++I;
  while (IsNoopInstruction(I)) ++I;
  if (&*I != Retain)
    return false;

  // Turn it to an objc_retainAutoreleasedReturnValue.
  Changed = true;
  ++NumPeeps;

  DEBUG(dbgs() << "Transforming objc_retain => "
                  "objc_retainAutoreleasedReturnValue since the operand is a "
                  "return value.\nOld: "<< *Retain << "\n");

  // We do not have to worry about tail calls/does not throw since
  // retain/retainRV have the same properties.
  Constant *Decl = EP.get(ARCRuntimeEntryPoints::EPT_RetainRV);
  cast<CallInst>(Retain)->setCalledFunction(Decl);

  DEBUG(dbgs() << "New: " << *Retain << "\n");
  return true;
}

/// Merge an autorelease with a retain into a fused call.
bool
ObjCARCContract::ContractAutorelease(Function &F, Instruction *Autorelease,
                                     InstructionClass Class,
                                     SmallPtrSet<Instruction *, 4>
                                       &DependingInstructions,
                                     SmallPtrSet<const BasicBlock *, 4>
                                       &Visited) {
  const Value *Arg = GetObjCArg(Autorelease);

  // Check that there are no instructions between the retain and the autorelease
  // (such as an autorelease_pop) which may change the count.
  CallInst *Retain = 0;
  if (Class == IC_AutoreleaseRV)
    FindDependencies(RetainAutoreleaseRVDep, Arg,
                     Autorelease->getParent(), Autorelease,
                     DependingInstructions, Visited, PA);
  else
    FindDependencies(RetainAutoreleaseDep, Arg,
                     Autorelease->getParent(), Autorelease,
                     DependingInstructions, Visited, PA);

  Visited.clear();
  if (DependingInstructions.size() != 1) {
    DependingInstructions.clear();
    return false;
  }

  Retain = dyn_cast_or_null<CallInst>(*DependingInstructions.begin());
  DependingInstructions.clear();

  if (!Retain ||
      GetBasicInstructionClass(Retain) != IC_Retain ||
      GetObjCArg(Retain) != Arg)
    return false;

  Changed = true;
  ++NumPeeps;

  DEBUG(dbgs() << "ObjCARCContract::ContractAutorelease: Fusing "
                  "retain/autorelease. Erasing: " << *Autorelease << "\n"
                  "                                      Old Retain: "
               << *Retain << "\n");

  Constant *Decl = EP.get(Class == IC_AutoreleaseRV ?
                          ARCRuntimeEntryPoints::EPT_RetainAutoreleaseRV :
                          ARCRuntimeEntryPoints::EPT_RetainAutorelease);
  Retain->setCalledFunction(Decl);

  DEBUG(dbgs() << "                                      New Retain: "
               << *Retain << "\n");

  EraseInstruction(Autorelease);
  return true;
}

/// Attempt to merge an objc_release with a store, load, and objc_retain to form
/// an objc_storeStrong. This can be a little tricky because the instructions
/// don't always appear in order, and there may be unrelated intervening
/// instructions.
void ObjCARCContract::ContractRelease(Instruction *Release,
                                      inst_iterator &Iter) {
  LoadInst *Load = dyn_cast<LoadInst>(GetObjCArg(Release));
  if (!Load || !Load->isSimple()) return;

  // For now, require everything to be in one basic block.
  BasicBlock *BB = Release->getParent();
  if (Load->getParent() != BB) return;

  // Walk down to find the store and the release, which may be in either order.
  BasicBlock::iterator I = Load, End = BB->end();
  ++I;
  AliasAnalysis::Location Loc = AA->getLocation(Load);
  StoreInst *Store = 0;
  bool SawRelease = false;
  for (; !Store || !SawRelease; ++I) {
    if (I == End)
      return;

    Instruction *Inst = I;
    if (Inst == Release) {
      SawRelease = true;
      continue;
    }

    InstructionClass Class = GetBasicInstructionClass(Inst);

    // Unrelated retains are harmless.
    if (IsRetain(Class))
      continue;

    if (Store) {
      // The store is the point where we're going to put the objc_storeStrong,
      // so make sure there are no uses after it.
      if (CanUse(Inst, Load, PA, Class))
        return;
    } else if (AA->getModRefInfo(Inst, Loc) & AliasAnalysis::Mod) {
      // We are moving the load down to the store, so check for anything
      // else which writes to the memory between the load and the store.
      Store = dyn_cast<StoreInst>(Inst);
      if (!Store || !Store->isSimple()) return;
      if (Store->getPointerOperand() != Loc.Ptr) return;
    }
  }

  Value *New = StripPointerCastsAndObjCCalls(Store->getValueOperand());

  // Walk up to find the retain.
  I = Store;
  BasicBlock::iterator Begin = BB->begin();
  while (I != Begin && GetBasicInstructionClass(I) != IC_Retain)
    --I;
  Instruction *Retain = I;
  if (GetBasicInstructionClass(Retain) != IC_Retain) return;
  if (GetObjCArg(Retain) != New) return;

  Changed = true;
  ++NumStoreStrongs;

  LLVMContext &C = Release->getContext();
  Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
  Type *I8XX = PointerType::getUnqual(I8X);

  Value *Args[] = { Load->getPointerOperand(), New };
  if (Args[0]->getType() != I8XX)
    Args[0] = new BitCastInst(Args[0], I8XX, "", Store);
  if (Args[1]->getType() != I8X)
    Args[1] = new BitCastInst(Args[1], I8X, "", Store);
  Constant *Decl = EP.get(ARCRuntimeEntryPoints::EPT_StoreStrong);
  CallInst *StoreStrong = CallInst::Create(Decl, Args, "", Store);
  StoreStrong->setDoesNotThrow();
  StoreStrong->setDebugLoc(Store->getDebugLoc());

  // We can't set the tail flag yet, because we haven't yet determined
  // whether there are any escaping allocas. Remember this call, so that
  // we can set the tail flag once we know it's safe.
  StoreStrongCalls.insert(StoreStrong);

  if (&*Iter == Store) ++Iter;
  Store->eraseFromParent();
  Release->eraseFromParent();
  EraseInstruction(Retain);
  if (Load->use_empty())
    Load->eraseFromParent();
}

bool ObjCARCContract::doInitialization(Module &M) {
  // If nothing in the Module uses ARC, don't do anything.
  Run = ModuleHasARC(M);
  if (!Run)
    return false;

  EP.Initialize(&M);

  // Initialize RetainRVMarker.
  RetainRVMarker = 0;
  if (NamedMDNode *NMD =
        M.getNamedMetadata("clang.arc.retainAutoreleasedReturnValueMarker"))
    if (NMD->getNumOperands() == 1) {
      const MDNode *N = NMD->getOperand(0);
      if (N->getNumOperands() == 1)
        if (const MDString *S = dyn_cast<MDString>(N->getOperand(0)))
          RetainRVMarker = S;
    }

  return false;
}

bool ObjCARCContract::runOnFunction(Function &F) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  if (!Run)
    return false;

  Changed = false;
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  PA.setAA(&getAnalysis<AliasAnalysis>());

  // Track whether it's ok to mark objc_storeStrong calls with the "tail"
  // keyword. Be conservative if the function has variadic arguments.
  // It seems that functions which "return twice" are also unsafe for the
  // "tail" argument, because they are setjmp, which could need to
  // return to an earlier stack state.
  bool TailOkForStoreStrongs = !F.isVarArg() &&
                               !F.callsFunctionThatReturnsTwice();

  // For ObjC library calls which return their argument, replace uses of the
  // argument with uses of the call return value, if it dominates the use. This
  // reduces register pressure.
  SmallPtrSet<Instruction *, 4> DependingInstructions;
  SmallPtrSet<const BasicBlock *, 4> Visited;
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;

    DEBUG(dbgs() << "ObjCARCContract: Visiting: " << *Inst << "\n");

    // Only these library routines return their argument. In particular,
    // objc_retainBlock does not necessarily return its argument.
    InstructionClass Class = GetBasicInstructionClass(Inst);
    switch (Class) {
    case IC_FusedRetainAutorelease:
    case IC_FusedRetainAutoreleaseRV:
      break;
    case IC_Autorelease:
    case IC_AutoreleaseRV:
      if (ContractAutorelease(F, Inst, Class, DependingInstructions, Visited))
        continue;
      break;
    case IC_Retain:
      // Attempt to convert retains to retainrvs if they are next to function
      // calls.
      if (!OptimizeRetainCall(F, Inst))
        break;
      // If we succeed in our optimization, fall through.
      // FALLTHROUGH
    case IC_RetainRV: {
      // If we're compiling for a target which needs a special inline-asm
      // marker to do the retainAutoreleasedReturnValue optimization,
      // insert it now.
      if (!RetainRVMarker)
        break;
      BasicBlock::iterator BBI = Inst;
      BasicBlock *InstParent = Inst->getParent();

      // Step up to see if the call immediately precedes the RetainRV call.
      // If it's an invoke, we have to cross a block boundary. And we have
      // to carefully dodge no-op instructions.
      do {
        if (&*BBI == InstParent->begin()) {
          BasicBlock *Pred = InstParent->getSinglePredecessor();
          if (!Pred)
            goto decline_rv_optimization;
          BBI = Pred->getTerminator();
          break;
        }
        --BBI;
      } while (IsNoopInstruction(BBI));

      if (&*BBI == GetObjCArg(Inst)) {
        DEBUG(dbgs() << "ObjCARCContract: Adding inline asm marker for "
                        "retainAutoreleasedReturnValue optimization.\n");
        Changed = true;
        InlineAsm *IA =
          InlineAsm::get(FunctionType::get(Type::getVoidTy(Inst->getContext()),
                                           /*isVarArg=*/false),
                         RetainRVMarker->getString(),
                         /*Constraints=*/"", /*hasSideEffects=*/true);
        CallInst::Create(IA, "", Inst);
      }
    decline_rv_optimization:
      break;
    }
    case IC_InitWeak: {
      // objc_initWeak(p, null) => *p = null
      CallInst *CI = cast<CallInst>(Inst);
      if (IsNullOrUndef(CI->getArgOperand(1))) {
        Value *Null =
          ConstantPointerNull::get(cast<PointerType>(CI->getType()));
        Changed = true;
        new StoreInst(Null, CI->getArgOperand(0), CI);

        DEBUG(dbgs() << "OBJCARCContract: Old = " << *CI << "\n"
                     << "                 New = " << *Null << "\n");

        CI->replaceAllUsesWith(Null);
        CI->eraseFromParent();
      }
      continue;
    }
    case IC_Release:
      ContractRelease(Inst, I);
      continue;
    case IC_User:
      // Be conservative if the function has any alloca instructions.
      // Technically we only care about escaping alloca instructions,
      // but this is sufficient to handle some interesting cases.
      if (isa<AllocaInst>(Inst))
        TailOkForStoreStrongs = false;
      continue;
    case IC_IntrinsicUser:
      // Remove calls to @clang.arc.use(...).
      Inst->eraseFromParent();
      continue;
    default:
      continue;
    }

    DEBUG(dbgs() << "ObjCARCContract: Finished List.\n\n");

    // Don't use GetObjCArg because we don't want to look through bitcasts
    // and such; to do the replacement, the argument must have type i8*.
    Value *Arg = cast<CallInst>(Inst)->getArgOperand(0);
    for (;;) {
      // If we're compiling bugpointed code, don't get in trouble.
      if (!isa<Instruction>(Arg) && !isa<Argument>(Arg))
        break;
      // Look through the uses of the pointer.
      for (Value::use_iterator UI = Arg->use_begin(), UE = Arg->use_end();
           UI != UE; ) {
        // Increment UI now, because we may unlink its element.
        Use &U = *UI++;
        unsigned OperandNo = U.getOperandNo();

        // If the call's return value dominates a use of the call's argument
        // value, rewrite the use to use the return value. We check for
        // reachability here because an unreachable call is considered to
        // trivially dominate itself, which would lead us to rewriting its
        // argument in terms of its return value, which would lead to
        // infinite loops in GetObjCArg.
        if (DT->isReachableFromEntry(U) && DT->dominates(Inst, U)) {
          Changed = true;
          Instruction *Replacement = Inst;
          Type *UseTy = U.get()->getType();
          if (PHINode *PHI = dyn_cast<PHINode>(U.getUser())) {
            // For PHI nodes, insert the bitcast in the predecessor block.
            unsigned ValNo = PHINode::getIncomingValueNumForOperand(OperandNo);
            BasicBlock *BB = PHI->getIncomingBlock(ValNo);
            if (Replacement->getType() != UseTy)
              Replacement = new BitCastInst(Replacement, UseTy, "",
                                            &BB->back());
            // While we're here, rewrite all edges for this PHI, rather
            // than just one use at a time, to minimize the number of
            // bitcasts we emit.
            for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i)
              if (PHI->getIncomingBlock(i) == BB) {
                // Keep the UI iterator valid.
                if (UI != UE &&
                    &PHI->getOperandUse(
                        PHINode::getOperandNumForIncomingValue(i)) == &*UI)
                  ++UI;
                PHI->setIncomingValue(i, Replacement);
              }
          } else {
            if (Replacement->getType() != UseTy)
              Replacement = new BitCastInst(Replacement, UseTy, "",
                                            cast<Instruction>(U.getUser()));
            U.set(Replacement);
          }
        }
      }

      // If Arg is a no-op casted pointer, strip one level of casts and iterate.
      if (const BitCastInst *BI = dyn_cast<BitCastInst>(Arg))
        Arg = BI->getOperand(0);
      else if (isa<GEPOperator>(Arg) &&
               cast<GEPOperator>(Arg)->hasAllZeroIndices())
        Arg = cast<GEPOperator>(Arg)->getPointerOperand();
      else if (isa<GlobalAlias>(Arg) &&
               !cast<GlobalAlias>(Arg)->mayBeOverridden())
        Arg = cast<GlobalAlias>(Arg)->getAliasee();
      else
        break;
    }
  }

  // If this function has no escaping allocas or suspicious vararg usage,
  // objc_storeStrong calls can be marked with the "tail" keyword.
  if (TailOkForStoreStrongs)
    for (SmallPtrSet<CallInst *, 8>::iterator I = StoreStrongCalls.begin(),
         E = StoreStrongCalls.end(); I != E; ++I)
      (*I)->setTailCall();
  StoreStrongCalls.clear();

  return Changed;
}
