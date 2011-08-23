//===- SjLjEHPass.cpp - Eliminate Invoke & Unwind instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation is designed for use by code generators which use SjLj
// based exception handling.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sjljehprepare"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

STATISTIC(NumInvokes, "Number of invokes replaced");
STATISTIC(NumUnwinds, "Number of unwinds replaced");
STATISTIC(NumSpilled, "Number of registers live across unwind edges");

namespace {
  class SjLjEHPass : public FunctionPass {
    const TargetLowering *TLI;
    Type *FunctionContextTy;
    Constant *RegisterFn;
    Constant *UnregisterFn;
    Constant *BuiltinSetjmpFn;
    Constant *FrameAddrFn;
    Constant *StackAddrFn;
    Constant *StackRestoreFn;
    Constant *LSDAAddrFn;
    Value *PersonalityFn;
    Constant *SelectorFn;
    Constant *ExceptionFn;
    Constant *CallSiteFn;
    Constant *DispatchSetupFn;
    Value *CallSite;
    DenseMap<InvokeInst*, BasicBlock*> LPadSuccMap;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit SjLjEHPass(const TargetLowering *tli = NULL)
      : FunctionPass(ID), TLI(tli) { }
    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {}
    const char *getPassName() const {
      return "SJLJ Exception Handling preparation";
    }

  private:
    void insertCallSiteStore(Instruction *I, int Number, Value *CallSite);
    void markInvokeCallSite(InvokeInst *II, int InvokeNo, Value *CallSite,
                            SwitchInst *CatchSwitch);
    void splitLiveRangesAcrossInvokes(SmallVector<InvokeInst*,16> &Invokes);
    bool insertSjLjEHSupport(Function &F);
  };
} // end anonymous namespace

char SjLjEHPass::ID = 0;

// Public Interface To the SjLjEHPass pass.
FunctionPass *llvm::createSjLjEHPass(const TargetLowering *TLI) {
  return new SjLjEHPass(TLI);
}
// doInitialization - Set up decalarations and types needed to process
// exceptions.
bool SjLjEHPass::doInitialization(Module &M) {
  // Build the function context structure.
  // builtin_setjmp uses a five word jbuf
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  FunctionContextTy =
    StructType::get(VoidPtrTy,                        // __prev
                    Int32Ty,                          // call_site
                    ArrayType::get(Int32Ty, 4),       // __data
                    VoidPtrTy,                        // __personality
                    VoidPtrTy,                        // __lsda
                    ArrayType::get(VoidPtrTy, 5),     // __jbuf
                    NULL);
  RegisterFn = M.getOrInsertFunction("_Unwind_SjLj_Register",
                                     Type::getVoidTy(M.getContext()),
                                     PointerType::getUnqual(FunctionContextTy),
                                     (Type *)0);
  UnregisterFn =
    M.getOrInsertFunction("_Unwind_SjLj_Unregister",
                          Type::getVoidTy(M.getContext()),
                          PointerType::getUnqual(FunctionContextTy),
                          (Type *)0);
  FrameAddrFn = Intrinsic::getDeclaration(&M, Intrinsic::frameaddress);
  StackAddrFn = Intrinsic::getDeclaration(&M, Intrinsic::stacksave);
  StackRestoreFn = Intrinsic::getDeclaration(&M, Intrinsic::stackrestore);
  BuiltinSetjmpFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_setjmp);
  LSDAAddrFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_lsda);
  SelectorFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_selector);
  ExceptionFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_exception);
  CallSiteFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_callsite);
  DispatchSetupFn
    = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_dispatch_setup);
  PersonalityFn = 0;

  return true;
}

/// insertCallSiteStore - Insert a store of the call-site value to the
/// function context
void SjLjEHPass::insertCallSiteStore(Instruction *I, int Number,
                                     Value *CallSite) {
  ConstantInt *CallSiteNoC = ConstantInt::get(Type::getInt32Ty(I->getContext()),
                                              Number);
  // Insert a store of the call-site number
  new StoreInst(CallSiteNoC, CallSite, true, I);  // volatile
}

/// markInvokeCallSite - Insert code to mark the call_site for this invoke
void SjLjEHPass::markInvokeCallSite(InvokeInst *II, int InvokeNo,
                                    Value *CallSite,
                                    SwitchInst *CatchSwitch) {
  ConstantInt *CallSiteNoC= ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                              InvokeNo);
  // The runtime comes back to the dispatcher with the call_site - 1 in
  // the context. Odd, but there it is.
  ConstantInt *SwitchValC = ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                            InvokeNo - 1);

  // If the unwind edge has phi nodes, split the edge.
  if (isa<PHINode>(II->getUnwindDest()->begin())) {
    SplitCriticalEdge(II, 1, this);

    // If there are any phi nodes left, they must have a single predecessor.
    while (PHINode *PN = dyn_cast<PHINode>(II->getUnwindDest()->begin())) {
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
      PN->eraseFromParent();
    }
  }

  // Insert the store of the call site value
  insertCallSiteStore(II, InvokeNo, CallSite);

  // Record the call site value for the back end so it stays associated with
  // the invoke.
  CallInst::Create(CallSiteFn, CallSiteNoC, "", II);

  // Add a switch case to our unwind block.
  if (BasicBlock *SuccBB = LPadSuccMap[II]) {
    CatchSwitch->addCase(SwitchValC, SuccBB);
  } else {
    CatchSwitch->addCase(SwitchValC, II->getUnwindDest());
  }

  // We still want this to look like an invoke so we emit the LSDA properly,
  // so we don't transform the invoke into a call here.
}

/// MarkBlocksLiveIn - Insert BB and all of its predescessors into LiveBBs until
/// we reach blocks we've already seen.
static void MarkBlocksLiveIn(BasicBlock *BB, std::set<BasicBlock*> &LiveBBs) {
  if (!LiveBBs.insert(BB).second) return; // already been here.

  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    MarkBlocksLiveIn(*PI, LiveBBs);
}

/// splitLiveRangesAcrossInvokes - Each value that is live across an unwind edge
/// we spill into a stack location, guaranteeing that there is nothing live
/// across the unwind edge.  This process also splits all critical edges
/// coming out of invoke's.
/// FIXME: Move this function to a common utility file (Local.cpp?) so
/// both SjLj and LowerInvoke can use it.
void SjLjEHPass::
splitLiveRangesAcrossInvokes(SmallVector<InvokeInst*,16> &Invokes) {
  // First step, split all critical edges from invoke instructions.
  for (unsigned i = 0, e = Invokes.size(); i != e; ++i) {
    InvokeInst *II = Invokes[i];
    SplitCriticalEdge(II, 0, this);

    // FIXME: New EH - This if-condition will be always true in the new scheme.
    if (II->getUnwindDest()->isLandingPad()) {
      SmallVector<BasicBlock*, 2> NewBBs;
      SplitLandingPadPredecessors(II->getUnwindDest(), II->getParent(),
                                  ".1", ".2", this, NewBBs);
      LPadSuccMap[II] = *succ_begin(NewBBs[0]);
    } else {
      SplitCriticalEdge(II, 1, this);
    }

    assert(!isa<PHINode>(II->getNormalDest()) &&
           !isa<PHINode>(II->getUnwindDest()) &&
           "Critical edge splitting left single entry phi nodes?");
  }

  Function *F = Invokes.back()->getParent()->getParent();

  // To avoid having to handle incoming arguments specially, we lower each arg
  // to a copy instruction in the entry block.  This ensures that the argument
  // value itself cannot be live across the entry block.
  BasicBlock::iterator AfterAllocaInsertPt = F->begin()->begin();
  while (isa<AllocaInst>(AfterAllocaInsertPt) &&
        isa<ConstantInt>(cast<AllocaInst>(AfterAllocaInsertPt)->getArraySize()))
    ++AfterAllocaInsertPt;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
       AI != E; ++AI) {
    Type *Ty = AI->getType();
    // Aggregate types can't be cast, but are legal argument types, so we have
    // to handle them differently. We use an extract/insert pair as a
    // lightweight method to achieve the same goal.
    if (isa<StructType>(Ty) || isa<ArrayType>(Ty) || isa<VectorType>(Ty)) {
      Instruction *EI = ExtractValueInst::Create(AI, 0, "",AfterAllocaInsertPt);
      Instruction *NI = InsertValueInst::Create(AI, EI, 0);
      NI->insertAfter(EI);
      AI->replaceAllUsesWith(NI);
      // Set the operand of the instructions back to the AllocaInst.
      EI->setOperand(0, AI);
      NI->setOperand(0, AI);
    } else {
      // This is always a no-op cast because we're casting AI to AI->getType()
      // so src and destination types are identical. BitCast is the only
      // possibility.
      CastInst *NC = new BitCastInst(
        AI, AI->getType(), AI->getName()+".tmp", AfterAllocaInsertPt);
      AI->replaceAllUsesWith(NC);
      // Set the operand of the cast instruction back to the AllocaInst.
      // Normally it's forbidden to replace a CastInst's operand because it
      // could cause the opcode to reflect an illegal conversion. However,
      // we're replacing it here with the same value it was constructed with.
      // We do this because the above replaceAllUsesWith() clobbered the
      // operand, but we want this one to remain.
      NC->setOperand(0, AI);
    }
  }

  // Finally, scan the code looking for instructions with bad live ranges.
  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ++II) {
      // Ignore obvious cases we don't have to handle.  In particular, most
      // instructions either have no uses or only have a single use inside the
      // current block.  Ignore them quickly.
      Instruction *Inst = II;
      if (Inst->use_empty()) continue;
      if (Inst->hasOneUse() &&
          cast<Instruction>(Inst->use_back())->getParent() == BB &&
          !isa<PHINode>(Inst->use_back())) continue;

      // If this is an alloca in the entry block, it's not a real register
      // value.
      if (AllocaInst *AI = dyn_cast<AllocaInst>(Inst))
        if (isa<ConstantInt>(AI->getArraySize()) && BB == F->begin())
          continue;

      // Avoid iterator invalidation by copying users to a temporary vector.
      SmallVector<Instruction*,16> Users;
      for (Value::use_iterator UI = Inst->use_begin(), E = Inst->use_end();
           UI != E; ++UI) {
        Instruction *User = cast<Instruction>(*UI);
        if (User->getParent() != BB || isa<PHINode>(User))
          Users.push_back(User);
      }

      // Find all of the blocks that this value is live in.
      std::set<BasicBlock*> LiveBBs;
      LiveBBs.insert(Inst->getParent());
      while (!Users.empty()) {
        Instruction *U = Users.back();
        Users.pop_back();

        if (!isa<PHINode>(U)) {
          MarkBlocksLiveIn(U->getParent(), LiveBBs);
        } else {
          // Uses for a PHI node occur in their predecessor block.
          PHINode *PN = cast<PHINode>(U);
          for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
            if (PN->getIncomingValue(i) == Inst)
              MarkBlocksLiveIn(PN->getIncomingBlock(i), LiveBBs);
        }
      }

      // Now that we know all of the blocks that this thing is live in, see if
      // it includes any of the unwind locations.
      bool NeedsSpill = false;
      for (unsigned i = 0, e = Invokes.size(); i != e; ++i) {
        BasicBlock *UnwindBlock = Invokes[i]->getUnwindDest();
        if (UnwindBlock != BB && LiveBBs.count(UnwindBlock)) {
          NeedsSpill = true;
        }
      }

      // If we decided we need a spill, do it.
      // FIXME: Spilling this way is overkill, as it forces all uses of
      // the value to be reloaded from the stack slot, even those that aren't
      // in the unwind blocks. We should be more selective.
      if (NeedsSpill) {
        ++NumSpilled;
        DemoteRegToStack(*Inst, true);
      }
    }
}

/// CreateLandingPadLoad - Load the exception handling values and insert them
/// into a structure.
static Instruction *CreateLandingPadLoad(Function &F, Value *ExnAddr,
                                         Value *SelAddr,
                                         BasicBlock::iterator InsertPt) {
  Value *Exn = new LoadInst(ExnAddr, "exn", false,
                            InsertPt);
  Type *Ty = Type::getInt8PtrTy(F.getContext());
  Exn = CastInst::Create(Instruction::IntToPtr, Exn, Ty, "", InsertPt);
  Value *Sel = new LoadInst(SelAddr, "sel", false, InsertPt);

  Ty = StructType::get(Exn->getType(), Sel->getType(), NULL);
  InsertValueInst *LPadVal = InsertValueInst::Create(llvm::UndefValue::get(Ty),
                                                     Exn, 0,
                                                     "lpad.val", InsertPt);
  return InsertValueInst::Create(LPadVal, Sel, 1, "lpad.val", InsertPt);
}

/// ReplaceLandingPadVal - Replace the landingpad instruction's value with a
/// load from the stored values (via CreateLandingPadLoad). This looks through
/// PHI nodes, and removes them if they are dead.
static void ReplaceLandingPadVal(Function &F, Instruction *Inst, Value *ExnAddr,
                                 Value *SelAddr) {
  if (Inst->use_empty()) return;

  while (!Inst->use_empty()) {
    Instruction *I = cast<Instruction>(Inst->use_back());

    if (PHINode *PN = dyn_cast<PHINode>(I)) {
      ReplaceLandingPadVal(F, PN, ExnAddr, SelAddr);
      if (PN->use_empty()) PN->eraseFromParent();
      continue;
    }

    I->replaceUsesOfWith(Inst, CreateLandingPadLoad(F, ExnAddr, SelAddr, I));
  }
}

bool SjLjEHPass::insertSjLjEHSupport(Function &F) {
  SmallVector<ReturnInst*,16> Returns;
  SmallVector<UnwindInst*,16> Unwinds;
  SmallVector<InvokeInst*,16> Invokes;

  // Look through the terminators of the basic blocks to find invokes, returns
  // and unwinds.
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
      // Remember all return instructions in case we insert an invoke into this
      // function.
      Returns.push_back(RI);
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      Invokes.push_back(II);
    } else if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
      Unwinds.push_back(UI);
    }
  }

  NumInvokes += Invokes.size();
  NumUnwinds += Unwinds.size();

  // If we don't have any invokes, there's nothing to do.
  if (Invokes.empty()) return false;

  // Find the eh.selector.*, eh.exception and alloca calls.
  //
  // Remember any allocas() that aren't in the entry block, as the
  // jmpbuf saved SP will need to be updated for them.
  //
  // We'll use the first eh.selector to determine the right personality
  // function to use. For SJLJ, we always use the same personality for the
  // whole function, not on a per-selector basis.
  // FIXME: That's a bit ugly. Better way?
  SmallVector<CallInst*,16> EH_Selectors;
  SmallVector<CallInst*,16> EH_Exceptions;
  SmallVector<Instruction*,16> JmpbufUpdatePoints;

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    // Note: Skip the entry block since there's nothing there that interests
    // us. eh.selector and eh.exception shouldn't ever be there, and we
    // want to disregard any allocas that are there.
    // 
    // FIXME: This is awkward. The new EH scheme won't need to skip the entry
    //        block.
    if (BB == F.begin()) {
      if (InvokeInst *II = dyn_cast<InvokeInst>(F.begin()->getTerminator())) {
        // FIXME: This will be always non-NULL in the new EH.
        if (LandingPadInst *LPI = II->getUnwindDest()->getLandingPadInst())
          if (!PersonalityFn) PersonalityFn = LPI->getPersonalityFn();
      }

      continue;
    }

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        if (CI->getCalledFunction() == SelectorFn) {
          if (!PersonalityFn) PersonalityFn = CI->getArgOperand(1);
          EH_Selectors.push_back(CI);
        } else if (CI->getCalledFunction() == ExceptionFn) {
          EH_Exceptions.push_back(CI);
        } else if (CI->getCalledFunction() == StackRestoreFn) {
          JmpbufUpdatePoints.push_back(CI);
        }
      } else if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
        JmpbufUpdatePoints.push_back(AI);
      } else if (InvokeInst *II = dyn_cast<InvokeInst>(I)) {
        // FIXME: This will be always non-NULL in the new EH.
        if (LandingPadInst *LPI = II->getUnwindDest()->getLandingPadInst())
          if (!PersonalityFn) PersonalityFn = LPI->getPersonalityFn();
      }
    }
  }

  // If we don't have any eh.selector calls, we can't determine the personality
  // function. Without a personality function, we can't process exceptions.
  if (!PersonalityFn) return false;

  // We have invokes, so we need to add register/unregister calls to get this
  // function onto the global unwind stack.
  //
  // First thing we need to do is scan the whole function for values that are
  // live across unwind edges.  Each value that is live across an unwind edge we
  // spill into a stack location, guaranteeing that there is nothing live across
  // the unwind edge.  This process also splits all critical edges coming out of
  // invoke's.
  splitLiveRangesAcrossInvokes(Invokes);


  SmallVector<LandingPadInst*, 16> LandingPads;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator()))
      // FIXME: This will be always non-NULL in the new EH.
      if (LandingPadInst *LPI = II->getUnwindDest()->getLandingPadInst())
        LandingPads.push_back(LPI);
  }


  BasicBlock *EntryBB = F.begin();
  // Create an alloca for the incoming jump buffer ptr and the new jump buffer
  // that needs to be restored on all exits from the function.  This is an
  // alloca because the value needs to be added to the global context list.
  unsigned Align = 4; // FIXME: Should be a TLI check?
  AllocaInst *FunctionContext =
    new AllocaInst(FunctionContextTy, 0, Align,
                   "fcn_context", F.begin()->begin());

  Value *Idxs[2];
  Type *Int32Ty = Type::getInt32Ty(F.getContext());
  Value *Zero = ConstantInt::get(Int32Ty, 0);
  // We need to also keep around a reference to the call_site field
  Idxs[0] = Zero;
  Idxs[1] = ConstantInt::get(Int32Ty, 1);
  CallSite = GetElementPtrInst::Create(FunctionContext, Idxs, "call_site",
                                       EntryBB->getTerminator());

  // The exception selector comes back in context->data[1]
  Idxs[1] = ConstantInt::get(Int32Ty, 2);
  Value *FCData = GetElementPtrInst::Create(FunctionContext, Idxs, "fc_data",
                                            EntryBB->getTerminator());
  Idxs[1] = ConstantInt::get(Int32Ty, 1);
  Value *SelectorAddr = GetElementPtrInst::Create(FCData, Idxs,
                                                  "exc_selector_gep",
                                                  EntryBB->getTerminator());
  // The exception value comes back in context->data[0]
  Idxs[1] = Zero;
  Value *ExceptionAddr = GetElementPtrInst::Create(FCData, Idxs,
                                                   "exception_gep",
                                                   EntryBB->getTerminator());

  // The result of the eh.selector call will be replaced with a a reference to
  // the selector value returned in the function context. We leave the selector
  // itself so the EH analysis later can use it.
  for (int i = 0, e = EH_Selectors.size(); i < e; ++i) {
    CallInst *I = EH_Selectors[i];
    Value *SelectorVal = new LoadInst(SelectorAddr, "select_val", true, I);
    I->replaceAllUsesWith(SelectorVal);
  }

  // eh.exception calls are replaced with references to the proper location in
  // the context. Unlike eh.selector, the eh.exception calls are removed
  // entirely.
  for (int i = 0, e = EH_Exceptions.size(); i < e; ++i) {
    CallInst *I = EH_Exceptions[i];
    // Possible for there to be duplicates, so check to make sure the
    // instruction hasn't already been removed.
    if (!I->getParent()) continue;
    Value *Val = new LoadInst(ExceptionAddr, "exception", true, I);
    Type *Ty = Type::getInt8PtrTy(F.getContext());
    Val = CastInst::Create(Instruction::IntToPtr, Val, Ty, "", I);

    I->replaceAllUsesWith(Val);
    I->eraseFromParent();
  }

  for (unsigned i = 0, e = LandingPads.size(); i != e; ++i)
    ReplaceLandingPadVal(F, LandingPads[i], ExceptionAddr, SelectorAddr);

  // The entry block changes to have the eh.sjlj.setjmp, with a conditional
  // branch to a dispatch block for non-zero returns. If we return normally,
  // we're not handling an exception and just register the function context and
  // continue.

  // Create the dispatch block.  The dispatch block is basically a big switch
  // statement that goes to all of the invoke landing pads.
  BasicBlock *DispatchBlock =
    BasicBlock::Create(F.getContext(), "eh.sjlj.setjmp.catch", &F);

  // Insert a load of the callsite in the dispatch block, and a switch on its
  // value. By default, we issue a trap statement.
  BasicBlock *TrapBlock =
    BasicBlock::Create(F.getContext(), "trapbb", &F);
  CallInst::Create(Intrinsic::getDeclaration(F.getParent(), Intrinsic::trap),
                   "", TrapBlock);
  new UnreachableInst(F.getContext(), TrapBlock);

  Value *DispatchLoad = new LoadInst(CallSite, "invoke.num", true,
                                     DispatchBlock);
  SwitchInst *DispatchSwitch =
    SwitchInst::Create(DispatchLoad, TrapBlock, Invokes.size(),
                       DispatchBlock);
  // Split the entry block to insert the conditional branch for the setjmp.
  BasicBlock *ContBlock = EntryBB->splitBasicBlock(EntryBB->getTerminator(),
                                                   "eh.sjlj.setjmp.cont");

  // Populate the Function Context
  //   1. LSDA address
  //   2. Personality function address
  //   3. jmpbuf (save SP, FP and call eh.sjlj.setjmp)

  // LSDA address
  Idxs[0] = Zero;
  Idxs[1] = ConstantInt::get(Int32Ty, 4);
  Value *LSDAFieldPtr =
    GetElementPtrInst::Create(FunctionContext, Idxs, "lsda_gep",
                              EntryBB->getTerminator());
  Value *LSDA = CallInst::Create(LSDAAddrFn, "lsda_addr",
                                 EntryBB->getTerminator());
  new StoreInst(LSDA, LSDAFieldPtr, true, EntryBB->getTerminator());

  Idxs[1] = ConstantInt::get(Int32Ty, 3);
  Value *PersonalityFieldPtr =
    GetElementPtrInst::Create(FunctionContext, Idxs, "lsda_gep",
                              EntryBB->getTerminator());
  new StoreInst(PersonalityFn, PersonalityFieldPtr, true,
                EntryBB->getTerminator());

  // Save the frame pointer.
  Idxs[1] = ConstantInt::get(Int32Ty, 5);
  Value *JBufPtr
    = GetElementPtrInst::Create(FunctionContext, Idxs, "jbuf_gep",
                                EntryBB->getTerminator());
  Idxs[1] = ConstantInt::get(Int32Ty, 0);
  Value *FramePtr =
    GetElementPtrInst::Create(JBufPtr, Idxs, "jbuf_fp_gep",
                              EntryBB->getTerminator());

  Value *Val = CallInst::Create(FrameAddrFn,
                                ConstantInt::get(Int32Ty, 0),
                                "fp",
                                EntryBB->getTerminator());
  new StoreInst(Val, FramePtr, true, EntryBB->getTerminator());

  // Save the stack pointer.
  Idxs[1] = ConstantInt::get(Int32Ty, 2);
  Value *StackPtr =
    GetElementPtrInst::Create(JBufPtr, Idxs, "jbuf_sp_gep",
                              EntryBB->getTerminator());

  Val = CallInst::Create(StackAddrFn, "sp", EntryBB->getTerminator());
  new StoreInst(Val, StackPtr, true, EntryBB->getTerminator());

  // Call the setjmp instrinsic. It fills in the rest of the jmpbuf.
  Value *SetjmpArg =
    CastInst::Create(Instruction::BitCast, JBufPtr,
                     Type::getInt8PtrTy(F.getContext()), "",
                     EntryBB->getTerminator());
  Value *DispatchVal = CallInst::Create(BuiltinSetjmpFn, SetjmpArg,
                                        "dispatch",
                                        EntryBB->getTerminator());

  // Add a call to dispatch_setup after the setjmp call. This is expanded to any
  // target-specific setup that needs to be done.
  CallInst::Create(DispatchSetupFn, DispatchVal, "", EntryBB->getTerminator());

  // check the return value of the setjmp. non-zero goes to dispatcher.
  Value *IsNormal = new ICmpInst(EntryBB->getTerminator(),
                                 ICmpInst::ICMP_EQ, DispatchVal, Zero,
                                 "notunwind");
  // Nuke the uncond branch.
  EntryBB->getTerminator()->eraseFromParent();

  // Put in a new condbranch in its place.
  BranchInst::Create(ContBlock, DispatchBlock, IsNormal, EntryBB);

  // Register the function context and make sure it's known to not throw
  CallInst *Register =
    CallInst::Create(RegisterFn, FunctionContext, "",
                     ContBlock->getTerminator());
  Register->setDoesNotThrow();

  // At this point, we are all set up, update the invoke instructions to mark
  // their call_site values, and fill in the dispatch switch accordingly.
  for (unsigned i = 0, e = Invokes.size(); i != e; ++i)
    markInvokeCallSite(Invokes[i], i+1, CallSite, DispatchSwitch);

  // Mark call instructions that aren't nounwind as no-action (call_site ==
  // -1). Skip the entry block, as prior to then, no function context has been
  // created for this function and any unexpected exceptions thrown will go
  // directly to the caller's context, which is what we want anyway, so no need
  // to do anything here.
  for (Function::iterator BB = F.begin(), E = F.end(); ++BB != E;) {
    for (BasicBlock::iterator I = BB->begin(), end = BB->end(); I != end; ++I)
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        // Ignore calls to the EH builtins (eh.selector, eh.exception)
        Constant *Callee = CI->getCalledFunction();
        if (Callee != SelectorFn && Callee != ExceptionFn
            && !CI->doesNotThrow())
          insertCallSiteStore(CI, -1, CallSite);
      }
  }

  // Replace all unwinds with a branch to the unwind handler.
  // ??? Should this ever happen with sjlj exceptions?
  for (unsigned i = 0, e = Unwinds.size(); i != e; ++i) {
    BranchInst::Create(TrapBlock, Unwinds[i]);
    Unwinds[i]->eraseFromParent();
  }

  // Following any allocas not in the entry block, update the saved SP in the
  // jmpbuf to the new value.
  for (unsigned i = 0, e = JmpbufUpdatePoints.size(); i != e; ++i) {
    Instruction *AI = JmpbufUpdatePoints[i];
    Instruction *StackAddr = CallInst::Create(StackAddrFn, "sp");
    StackAddr->insertAfter(AI);
    Instruction *StoreStackAddr = new StoreInst(StackAddr, StackPtr, true);
    StoreStackAddr->insertAfter(StackAddr);
  }

  // Finally, for any returns from this function, if this function contains an
  // invoke, add a call to unregister the function context.
  for (unsigned i = 0, e = Returns.size(); i != e; ++i)
    CallInst::Create(UnregisterFn, FunctionContext, "", Returns[i]);

  return true;
}

bool SjLjEHPass::runOnFunction(Function &F) {
  bool Res = insertSjLjEHSupport(F);
  return Res;
}
