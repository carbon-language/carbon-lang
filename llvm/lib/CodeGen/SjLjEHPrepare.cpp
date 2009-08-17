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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include <set>
using namespace llvm;

STATISTIC(NumInvokes, "Number of invokes replaced");
STATISTIC(NumUnwinds, "Number of unwinds replaced");
STATISTIC(NumSpilled, "Number of registers live across unwind edges");

namespace {
  class VISIBILITY_HIDDEN SjLjEHPass : public FunctionPass {

    const TargetLowering *TLI;

    const Type *FunctionContextTy;
    Constant *RegisterFn;
    Constant *UnregisterFn;
    Constant *ResumeFn;
    Constant *BuiltinSetjmpFn;
    Constant *FrameAddrFn;
    Constant *LSDAAddrFn;
    Value *PersonalityFn;
    Constant *Selector32Fn;
    Constant *Selector64Fn;
    Constant *ExceptionFn;

    Value *CallSite;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit SjLjEHPass(const TargetLowering *tli = NULL)
      : FunctionPass(&ID), TLI(tli) { }
    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const { }
    const char *getPassName() const {
      return "SJLJ Exception Handling preparation";
    }

  private:
    void markInvokeCallSite(InvokeInst *II, unsigned InvokeNo,
                            Value *CallSite,
                            SwitchInst *CatchSwitch);
    void splitLiveRangesLiveAcrossInvokes(std::vector<InvokeInst*> &Invokes);
    bool insertSjLjEHSupport(Function &F);
  };
} // end anonymous namespace

char SjLjEHPass::ID = 0;

// Public Interface To the SjLjEHPass pass.
FunctionPass *llvm::createSjLjEHPass(const TargetLowering *TLI) {
  return new SjLjEHPass(TLI);
}
// doInitialization - Make sure that there is a prototype for abort in the
// current module.
bool SjLjEHPass::doInitialization(Module &M) {
  // Build the function context structure.
  // builtin_setjmp uses a five word jbuf
  const Type *VoidPtrTy =
          PointerType::getUnqual(Type::getInt8Ty(M.getContext()));
  const Type *Int32Ty = Type::getInt32Ty(M.getContext());
  FunctionContextTy =
    StructType::get(M.getContext(),
                    VoidPtrTy,                        // __prev
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
  ResumeFn =
    M.getOrInsertFunction("_Unwind_SjLj_Resume",
                          Type::getVoidTy(M.getContext()),
                          VoidPtrTy,
                          (Type *)0);
  FrameAddrFn = Intrinsic::getDeclaration(&M, Intrinsic::frameaddress);
  BuiltinSetjmpFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_setjmp);
  LSDAAddrFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_lsda);
  Selector32Fn = Intrinsic::getDeclaration(&M, Intrinsic::eh_selector_i32);
  Selector64Fn = Intrinsic::getDeclaration(&M, Intrinsic::eh_selector_i64);
  ExceptionFn = Intrinsic::getDeclaration(&M, Intrinsic::eh_exception);

  return true;
}

/// markInvokeCallSite - Insert code to mark the call_site for this invoke
void SjLjEHPass::markInvokeCallSite(InvokeInst *II, unsigned InvokeNo,
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

  // Insert a store of the invoke num before the invoke and store zero into the
  // location afterward.
  new StoreInst(CallSiteNoC, CallSite, true, II);  // volatile

  // Add a switch case to our unwind block.
  CatchSwitch->addCase(SwitchValC, II->getUnwindDest());
  // We still want this to look like an invoke so we emit the LSDA properly
  // FIXME: ??? Or will this cause strangeness with mis-matched IDs like
  //  when it was in the front end?
}

/// MarkBlocksLiveIn - Insert BB and all of its predescessors into LiveBBs until
/// we reach blocks we've already seen.
static void MarkBlocksLiveIn(BasicBlock *BB, std::set<BasicBlock*> &LiveBBs) {
  if (!LiveBBs.insert(BB).second) return; // already been here.

  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    MarkBlocksLiveIn(*PI, LiveBBs);
}

// live across unwind edges.  Each value that is live across an unwind edge
// we spill into a stack location, guaranteeing that there is nothing live
// across the unwind edge.  This process also splits all critical edges
// coming out of invoke's.
void SjLjEHPass::
splitLiveRangesLiveAcrossInvokes(std::vector<InvokeInst*> &Invokes) {
  // First step, split all critical edges from invoke instructions.
  for (unsigned i = 0, e = Invokes.size(); i != e; ++i) {
    InvokeInst *II = Invokes[i];
    SplitCriticalEdge(II, 0, this);
    SplitCriticalEdge(II, 1, this);
    assert(!isa<PHINode>(II->getNormalDest()) &&
           !isa<PHINode>(II->getUnwindDest()) &&
           "critical edge splitting left single entry phi nodes?");
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
    // This is always a no-op cast because we're casting AI to AI->getType() so
    // src and destination types are identical. BitCast is the only possibility.
    CastInst *NC = new BitCastInst(
      AI, AI->getType(), AI->getName()+".tmp", AfterAllocaInsertPt);
    AI->replaceAllUsesWith(NC);
    // Normally its is forbidden to replace a CastInst's operand because it
    // could cause the opcode to reflect an illegal conversion. However, we're
    // replacing it here with the same value it was constructed with to simply
    // make NC its user.
    NC->setOperand(0, AI);
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
      std::vector<Instruction*> Users;
      for (Value::use_iterator UI = Inst->use_begin(), E = Inst->use_end();
           UI != E; ++UI) {
        Instruction *User = cast<Instruction>(*UI);
        if (User->getParent() != BB || isa<PHINode>(User))
          Users.push_back(User);
      }

      // Scan all of the uses and see if the live range is live across an unwind
      // edge.  If we find a use live across an invoke edge, create an alloca
      // and spill the value.
      std::set<InvokeInst*> InvokesWithStoreInserted;

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
      if (NeedsSpill) {
        ++NumSpilled;
        DemoteRegToStack(*Inst, true);
      }
    }
}

bool SjLjEHPass::insertSjLjEHSupport(Function &F) {
  std::vector<ReturnInst*> Returns;
  std::vector<UnwindInst*> Unwinds;
  std::vector<InvokeInst*> Invokes;

  // Look through the terminators of the basic blocks to find invokes, returns
  // and unwinds
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
      // Remember all return instructions in case we insert an invoke into this
      // function.
      Returns.push_back(RI);
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      Invokes.push_back(II);
    } else if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
      Unwinds.push_back(UI);
    }
  // If we don't have any invokes or unwinds, there's nothing to do.
  if (Unwinds.empty() && Invokes.empty()) return false;

  NumInvokes += Invokes.size();
  NumUnwinds += Unwinds.size();


  if (!Invokes.empty()) {
    // We have invokes, so we need to add register/unregister calls to get
    // this function onto the global unwind stack.
    //
    // First thing we need to do is scan the whole function for values that are
    // live across unwind edges.  Each value that is live across an unwind edge
    // we spill into a stack location, guaranteeing that there is nothing live
    // across the unwind edge.  This process also splits all critical edges
    // coming out of invoke's.
    splitLiveRangesLiveAcrossInvokes(Invokes);

    BasicBlock *EntryBB = F.begin();
    // Create an alloca for the incoming jump buffer ptr and the new jump buffer
    // that needs to be restored on all exits from the function.  This is an
    // alloca because the value needs to be added to the global context list.
    unsigned Align = 4; // FIXME: Should be a TLI check?
    AllocaInst *FunctionContext =
      new AllocaInst(FunctionContextTy, 0, Align,
                     "fcn_context", F.begin()->begin());

    Value *Idxs[2];
    const Type *Int32Ty = Type::getInt32Ty(F.getContext());
    Value *Zero = ConstantInt::get(Int32Ty, 0);
    // We need to also keep around a reference to the call_site field
    Idxs[0] = Zero;
    Idxs[1] = ConstantInt::get(Int32Ty, 1);
    CallSite = GetElementPtrInst::Create(FunctionContext, Idxs, Idxs+2,
                                         "call_site",
                                         EntryBB->getTerminator());

    // The exception selector comes back in context->data[1]
    Idxs[1] = ConstantInt::get(Int32Ty, 2);
    Value *FCData = GetElementPtrInst::Create(FunctionContext, Idxs, Idxs+2,
                                              "fc_data",
                                              EntryBB->getTerminator());
    Idxs[1] = ConstantInt::get(Int32Ty, 1);
    Value *SelectorAddr = GetElementPtrInst::Create(FCData, Idxs, Idxs+2,
                                                    "exc_selector_gep",
                                                    EntryBB->getTerminator());
    // The exception value comes back in context->data[0]
    Idxs[1] = Zero;
    Value *ExceptionAddr = GetElementPtrInst::Create(FCData, Idxs, Idxs+2,
                                                     "exception_gep",
                                                     EntryBB->getTerminator());

    // Find the eh.selector.*  and eh.exception calls. We'll use the first
    // ex.selector to determine the right personality function to use. For
    // SJLJ, we always use the same personality for the whole function,
    // not on a per-selector basis.
    // FIXME: That's a bit ugly. Better way?
    std::vector<CallInst*> EH_Selectors;
    std::vector<CallInst*> EH_Exceptions;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
  //  for (unsigned i = 0, e = Invokes.size(); i != e; ++i) {
//      BasicBlock *Pad = Invokes[0]->getUnwindDest();
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
        if (CallInst *CI = dyn_cast<CallInst>(I)) {
          if (CI->getCalledFunction() == Selector32Fn ||
              CI->getCalledFunction() == Selector64Fn) {
            if (!PersonalityFn) PersonalityFn = CI->getOperand(2);
            EH_Selectors.push_back(CI);
          } else if (CI->getCalledFunction() == ExceptionFn) {
            EH_Exceptions.push_back(CI);
          }
        }
      }
    }
    // The result of the eh.selector call will be replaced with a
    // a reference to the selector value returned in the function
    // context. We leave the selector itself so the EH analysis later
    // can use it.
    for (int i = 0, e = EH_Selectors.size(); i < e; ++i) {
      CallInst *I = EH_Selectors[i];
      Value *SelectorVal = new LoadInst(SelectorAddr, "select_val", true, I);
      I->replaceAllUsesWith(SelectorVal);
    }
    // eh.exception calls are replaced with references to the proper
    // location in the context. Unlike eh.selector, the eh.exception
    // calls are removed entirely.
    for (int i = 0, e = EH_Exceptions.size(); i < e; ++i) {
      CallInst *I = EH_Exceptions[i];
      // Possible for there to be duplicates, so check to make sure
      // the instruction hasn't already been removed.
      if (!I->getParent()) continue;
      Value *Val = new LoadInst(ExceptionAddr, "exception", true, I);
      Val = CastInst::Create(Instruction::IntToPtr, Val,
                             PointerType::getUnqual(Type::getInt8Ty(F.getContext())),
                             "", I);

      I->replaceAllUsesWith(Val);
      I->eraseFromParent();
    }




    // The entry block changes to have the eh.sjlj.setjmp, with a conditional
    // branch to a dispatch block for non-zero returns. If we return normally,
    // we're not handling an exception and just register the function context
    // and continue.

    // Create the dispatch block.  The dispatch block is basically a big switch
    // statement that goes to all of the invoke landing pads.
    BasicBlock *DispatchBlock =
            BasicBlock::Create(F.getContext(), "eh.sjlj.setjmp.catch", &F);

    // Insert a load in the Catch block, and a switch on its value.  By default,
    // we go to a block that just does an unwind (which is the correct action
    // for a standard call).
    BasicBlock *UnwindBlock = BasicBlock::Create(F.getContext(), "unwindbb", &F);
    Unwinds.push_back(new UnwindInst(F.getContext(), UnwindBlock));

    Value *DispatchLoad = new LoadInst(CallSite, "invoke.num", true,
                                       DispatchBlock);
    SwitchInst *DispatchSwitch =
      SwitchInst::Create(DispatchLoad, UnwindBlock, Invokes.size(), DispatchBlock);
    // Split the entry block to insert the conditional branch for the setjmp.
    BasicBlock *ContBlock = EntryBB->splitBasicBlock(EntryBB->getTerminator(),
                                                     "eh.sjlj.setjmp.cont");

    // Populate the Function Context
    //   1. LSDA address
    //   2. Personality function address
    //   3. jmpbuf (save FP and call eh.sjlj.setjmp)

    // LSDA address
    Idxs[0] = Zero;
    Idxs[1] = ConstantInt::get(Int32Ty, 4);
    Value *LSDAFieldPtr =
      GetElementPtrInst::Create(FunctionContext, Idxs, Idxs+2,
                                "lsda_gep",
                                EntryBB->getTerminator());
    Value *LSDA = CallInst::Create(LSDAAddrFn, "lsda_addr",
                                   EntryBB->getTerminator());
    new StoreInst(LSDA, LSDAFieldPtr, true, EntryBB->getTerminator());

    Idxs[1] = ConstantInt::get(Int32Ty, 3);
    Value *PersonalityFieldPtr =
      GetElementPtrInst::Create(FunctionContext, Idxs, Idxs+2,
                                "lsda_gep",
                                EntryBB->getTerminator());
    new StoreInst(PersonalityFn, PersonalityFieldPtr, true,
                  EntryBB->getTerminator());

    //   Save the frame pointer.
    Idxs[1] = ConstantInt::get(Int32Ty, 5);
    Value *FieldPtr
      = GetElementPtrInst::Create(FunctionContext, Idxs, Idxs+2,
                                  "jbuf_gep",
                                  EntryBB->getTerminator());
    Idxs[1] = ConstantInt::get(Int32Ty, 0);
    Value *ElemPtr =
      GetElementPtrInst::Create(FieldPtr, Idxs, Idxs+2, "jbuf_fp_gep",
                                EntryBB->getTerminator());

    Value *Val = CallInst::Create(FrameAddrFn,
                                  ConstantInt::get(Int32Ty, 0),
                                  "fp",
                                  EntryBB->getTerminator());
    new StoreInst(Val, ElemPtr, true, EntryBB->getTerminator());
    // Call the setjmp instrinsic. It fills in the rest of the jmpbuf
    Value *SetjmpArg =
      CastInst::Create(Instruction::BitCast, FieldPtr,
                        Type::getInt8Ty(F.getContext())->getPointerTo(), "",
                        EntryBB->getTerminator());
    Value *DispatchVal = CallInst::Create(BuiltinSetjmpFn, SetjmpArg,
                                          "dispatch",
                                          EntryBB->getTerminator());
    // check the return value of the setjmp. non-zero goes to dispatcher
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

    // At this point, we are all set up, update the invoke instructions
    // to mark their call_site values, and fill in the dispatch switch
    // accordingly.
    for (unsigned i = 0, e = Invokes.size(); i != e; ++i)
      markInvokeCallSite(Invokes[i], i+1, CallSite, DispatchSwitch);

    // The front end has likely added calls to _Unwind_Resume. We need
    // to find those calls and mark the call_site as -1 immediately prior.
    // resume is a noreturn function, so any block that has a call to it
    // should end in an 'unreachable' instruction with the call immediately
    // prior. That's how we'll search.
    // ??? There's got to be a better way. this is fugly.
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      if ((dyn_cast<UnreachableInst>(BB->getTerminator()))) {
        BasicBlock::iterator I = BB->getTerminator();
        // Check the previous instruction and see if it's a resume call
        if (I == BB->begin()) continue;
        if (CallInst *CI = dyn_cast<CallInst>(--I)) {
          if (CI->getCalledFunction() == ResumeFn) {
            Value *NegativeOne = Constant::getAllOnesValue(Int32Ty);
            new StoreInst(NegativeOne, CallSite, true, I);  // volatile
          }
        }
      }

    // Replace all unwinds with a branch to the unwind handler.
    // ??? Should this ever happen with sjlj exceptions?
    for (unsigned i = 0, e = Unwinds.size(); i != e; ++i) {
      BranchInst::Create(UnwindBlock, Unwinds[i]);
      Unwinds[i]->eraseFromParent();
    }

    // Finally, for any returns from this function, if this function contains an
    // invoke, add a call to unregister the function context.
    for (unsigned i = 0, e = Returns.size(); i != e; ++i)
      CallInst::Create(UnregisterFn, FunctionContext, "", Returns[i]);
  }

  return true;
}

bool SjLjEHPass::runOnFunction(Function &F) {
  bool Res = insertSjLjEHSupport(F);
  return Res;
}
