//===- LowerInvoke.cpp - Eliminate Invoke & Unwind instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation is designed for use by code generators which do not yet
// support stack unwinding.  This pass supports two models of exception handling
// lowering, the 'cheap' support and the 'expensive' support.
//
// 'Cheap' exception handling support gives the program the ability to execute
// any program which does not "throw an exception", by turning 'invoke'
// instructions into calls and by turning 'unwind' instructions into calls to
// abort().  If the program does dynamically use the unwind instruction, the
// program will print a message then abort.
//
// 'Expensive' exception handling support gives the full exception handling
// support to the program at the cost of making the 'invoke' instruction
// really expensive.  It basically inserts setjmp/longjmp calls to emulate the
// exception handling as necessary.
//
// Because the 'expensive' support slows down programs a lot, and EH is only
// used for a subset of the programs, it must be specifically enabled by an
// option.
//
// Note that after this pass runs the CFG is not entirely accurate (exceptional
// control flow edges are not correct anymore) so only very simple things should
// be done after the lowerinvoke pass has run (like generation of native code).
// This should not be used as a general purpose "my LLVM-to-LLVM pass doesn't
// support the invoke instruction yet" lowering pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Visibility.h"
#include <csetjmp>
using namespace llvm;

namespace {
  Statistic<> NumInvokes("lowerinvoke", "Number of invokes replaced");
  Statistic<> NumUnwinds("lowerinvoke", "Number of unwinds replaced");
  Statistic<> NumSpilled("lowerinvoke",
                         "Number of registers live across unwind edges");
  cl::opt<bool> ExpensiveEHSupport("enable-correct-eh-support",
 cl::desc("Make the -lowerinvoke pass insert expensive, but correct, EH code"));

  class VISIBILITY_HIDDEN LowerInvoke : public FunctionPass {
    // Used for both models.
    Function *WriteFn;
    Function *AbortFn;
    Value *AbortMessage;
    unsigned AbortMessageLength;

    // Used for expensive EH support.
    const Type *JBLinkTy;
    GlobalVariable *JBListHead;
    Function *SetJmpFn, *LongJmpFn;
  public:
    LowerInvoke(unsigned Size = 200, unsigned Align = 0) : JumpBufSize(Size),
      JumpBufAlign(Align) {}
    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);
 
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // This is a cluster of orthogonal Transforms	
      AU.addPreservedID(PromoteMemoryToRegisterID);
      AU.addPreservedID(LowerSelectID);
      AU.addPreservedID(LowerSwitchID);
      AU.addPreservedID(LowerAllocationsID);
    }
       
  private:
    void createAbortMessage();
    void writeAbortMessage(Instruction *IB);
    bool insertCheapEHSupport(Function &F);
    void splitLiveRangesLiveAcrossInvokes(std::vector<InvokeInst*> &Invokes);
    void rewriteExpensiveInvoke(InvokeInst *II, unsigned InvokeNo,
                                AllocaInst *InvokeNum, SwitchInst *CatchSwitch);
    bool insertExpensiveEHSupport(Function &F);
    
    unsigned JumpBufSize;
    unsigned JumpBufAlign;
  };

  RegisterOpt<LowerInvoke>
  X("lowerinvoke", "Lower invoke and unwind, for unwindless code generators");
}

const PassInfo *llvm::LowerInvokePassID = X.getPassInfo();

// Public Interface To the LowerInvoke pass.
FunctionPass *llvm::createLowerInvokePass(unsigned JumpBufSize, 
                                          unsigned JumpBufAlign) { 
  return new LowerInvoke(JumpBufSize, JumpBufAlign); 
}

// doInitialization - Make sure that there is a prototype for abort in the
// current module.
bool LowerInvoke::doInitialization(Module &M) {
  const Type *VoidPtrTy = PointerType::get(Type::SByteTy);
  AbortMessage = 0;
  if (ExpensiveEHSupport) {
    // Insert a type for the linked list of jump buffers.
    const Type *JmpBufTy = ArrayType::get(VoidPtrTy, JumpBufSize);

    { // The type is recursive, so use a type holder.
      std::vector<const Type*> Elements;
      Elements.push_back(JmpBufTy);
      OpaqueType *OT = OpaqueType::get();
      Elements.push_back(PointerType::get(OT));
      PATypeHolder JBLType(StructType::get(Elements));
      OT->refineAbstractTypeTo(JBLType.get());  // Complete the cycle.
      JBLinkTy = JBLType.get();
      M.addTypeName("llvm.sjljeh.jmpbufty", JBLinkTy);
    }

    const Type *PtrJBList = PointerType::get(JBLinkTy);

    // Now that we've done that, insert the jmpbuf list head global, unless it
    // already exists.
    if (!(JBListHead = M.getGlobalVariable("llvm.sjljeh.jblist", PtrJBList)))
      JBListHead = new GlobalVariable(PtrJBList, false,
                                      GlobalValue::LinkOnceLinkage,
                                      Constant::getNullValue(PtrJBList),
                                      "llvm.sjljeh.jblist", &M);
    SetJmpFn = M.getOrInsertFunction("llvm.setjmp", Type::IntTy,
                                     PointerType::get(JmpBufTy), (Type *)0);
    LongJmpFn = M.getOrInsertFunction("llvm.longjmp", Type::VoidTy,
                                      PointerType::get(JmpBufTy),
                                      Type::IntTy, (Type *)0);
  }

  // We need the 'write' and 'abort' functions for both models.
  AbortFn = M.getOrInsertFunction("abort", Type::VoidTy, (Type *)0);

  // Unfortunately, 'write' can end up being prototyped in several different
  // ways.  If the user defines a three (or more) operand function named 'write'
  // we will use their prototype.  We _do not_ want to insert another instance
  // of a write prototype, because we don't know that the funcresolve pass will
  // run after us.  If there is a definition of a write function, but it's not
  // suitable for our uses, we just don't emit write calls.  If there is no
  // write prototype at all, we just add one.
  if (Function *WF = M.getNamedFunction("write")) {
    if (WF->getFunctionType()->getNumParams() > 3 ||
        WF->getFunctionType()->isVarArg())
      WriteFn = WF;
    else
      WriteFn = 0;
  } else {
    WriteFn = M.getOrInsertFunction("write", Type::VoidTy, Type::IntTy,
                                    VoidPtrTy, Type::IntTy, (Type *)0);
  }
  return true;
}

void LowerInvoke::createAbortMessage() {
  Module &M = *WriteFn->getParent();
  if (ExpensiveEHSupport) {
    // The abort message for expensive EH support tells the user that the
    // program 'unwound' without an 'invoke' instruction.
    Constant *Msg =
      ConstantArray::get("ERROR: Exception thrown, but not caught!\n");
    AbortMessageLength = Msg->getNumOperands()-1;  // don't include \0

    GlobalVariable *MsgGV = new GlobalVariable(Msg->getType(), true,
                                               GlobalValue::InternalLinkage,
                                               Msg, "abortmsg", &M);
    std::vector<Constant*> GEPIdx(2, Constant::getNullValue(Type::IntTy));
    AbortMessage = ConstantExpr::getGetElementPtr(MsgGV, GEPIdx);
  } else {
    // The abort message for cheap EH support tells the user that EH is not
    // enabled.
    Constant *Msg =
      ConstantArray::get("Exception handler needed, but not enabled.  Recompile"
                         " program with -enable-correct-eh-support.\n");
    AbortMessageLength = Msg->getNumOperands()-1;  // don't include \0

    GlobalVariable *MsgGV = new GlobalVariable(Msg->getType(), true,
                                               GlobalValue::InternalLinkage,
                                               Msg, "abortmsg", &M);
    std::vector<Constant*> GEPIdx(2, Constant::getNullValue(Type::IntTy));
    AbortMessage = ConstantExpr::getGetElementPtr(MsgGV, GEPIdx);
  }
}


void LowerInvoke::writeAbortMessage(Instruction *IB) {
  if (WriteFn) {
    if (AbortMessage == 0) createAbortMessage();

    // These are the arguments we WANT...
    std::vector<Value*> Args;
    Args.push_back(ConstantInt::get(Type::IntTy, 2));
    Args.push_back(AbortMessage);
    Args.push_back(ConstantInt::get(Type::IntTy, AbortMessageLength));

    // If the actual declaration of write disagrees, insert casts as
    // appropriate.
    const FunctionType *FT = WriteFn->getFunctionType();
    unsigned NumArgs = FT->getNumParams();
    for (unsigned i = 0; i != 3; ++i)
      if (i < NumArgs && FT->getParamType(i) != Args[i]->getType())
        Args[i] = ConstantExpr::getCast(cast<Constant>(Args[i]),
                                        FT->getParamType(i));

    (new CallInst(WriteFn, Args, "", IB))->setTailCall();
  }
}

bool LowerInvoke::insertCheapEHSupport(Function &F) {
  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      // Insert a normal call instruction...
      std::string Name = II->getName(); II->setName("");
      CallInst *NewCall = new CallInst(II->getCalledValue(),
                                       std::vector<Value*>(II->op_begin()+3,
                                                       II->op_end()), Name, II);
      NewCall->setCallingConv(II->getCallingConv());
      II->replaceAllUsesWith(NewCall);

      // Insert an unconditional branch to the normal destination.
      new BranchInst(II->getNormalDest(), II);

      // Remove any PHI node entries from the exception destination.
      II->getUnwindDest()->removePredecessor(BB);

      // Remove the invoke instruction now.
      BB->getInstList().erase(II);

      ++NumInvokes; Changed = true;
    } else if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
      // Insert a new call to write(2, AbortMessage, AbortMessageLength);
      writeAbortMessage(UI);

      // Insert a call to abort()
      (new CallInst(AbortFn, std::vector<Value*>(), "", UI))->setTailCall();

      // Insert a return instruction.  This really should be a "barrier", as it
      // is unreachable.
      new ReturnInst(F.getReturnType() == Type::VoidTy ? 0 :
                            Constant::getNullValue(F.getReturnType()), UI);

      // Remove the unwind instruction now.
      BB->getInstList().erase(UI);

      ++NumUnwinds; Changed = true;
    }
  return Changed;
}

/// rewriteExpensiveInvoke - Insert code and hack the function to replace the
/// specified invoke instruction with a call.
void LowerInvoke::rewriteExpensiveInvoke(InvokeInst *II, unsigned InvokeNo,
                                         AllocaInst *InvokeNum,
                                         SwitchInst *CatchSwitch) {
  ConstantUInt *InvokeNoC = ConstantUInt::get(Type::UIntTy, InvokeNo);

  // Insert a store of the invoke num before the invoke and store zero into the
  // location afterward.
  new StoreInst(InvokeNoC, InvokeNum, true, II);  // volatile
  
  BasicBlock::iterator NI = II->getNormalDest()->begin();
  while (isa<PHINode>(NI)) ++NI;
  // nonvolatile.
  new StoreInst(Constant::getNullValue(Type::UIntTy), InvokeNum, false, NI);
  
  // Add a switch case to our unwind block.
  CatchSwitch->addCase(InvokeNoC, II->getUnwindDest());
  
  // Insert a normal call instruction.
  std::string Name = II->getName(); II->setName("");
  CallInst *NewCall = new CallInst(II->getCalledValue(),
                                   std::vector<Value*>(II->op_begin()+3,
                                                       II->op_end()), Name,
                                   II);
  NewCall->setCallingConv(II->getCallingConv());
  II->replaceAllUsesWith(NewCall);
  
  // Replace the invoke with an uncond branch.
  new BranchInst(II->getNormalDest(), NewCall->getParent());
  II->eraseFromParent();
}

/// MarkBlocksLiveIn - Insert BB and all of its predescessors into LiveBBs until
/// we reach blocks we've already seen.
static void MarkBlocksLiveIn(BasicBlock *BB, std::set<BasicBlock*> &LiveBBs) {
  if (!LiveBBs.insert(BB).second) return; // already been here.
  
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    MarkBlocksLiveIn(*PI, LiveBBs);  
}

// First thing we need to do is scan the whole function for values that are
// live across unwind edges.  Each value that is live across an unwind edge
// we spill into a stack location, guaranteeing that there is nothing live
// across the unwind edge.  This process also splits all critical edges
// coming out of invoke's.
void LowerInvoke::
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
  // to a copy instruction in the entry block.  This ensure that the argument
  // value itself cannot be live across the entry block.
  BasicBlock::iterator AfterAllocaInsertPt = F->begin()->begin();
  while (isa<AllocaInst>(AfterAllocaInsertPt) &&
        isa<ConstantInt>(cast<AllocaInst>(AfterAllocaInsertPt)->getArraySize()))
    ++AfterAllocaInsertPt;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
       AI != E; ++AI) {
    CastInst *NC = new CastInst(AI, AI->getType(), AI->getName()+".tmp",
                                AfterAllocaInsertPt);
    AI->replaceAllUsesWith(NC);
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
      AllocaInst *SpillLoc = 0;
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

bool LowerInvoke::insertExpensiveEHSupport(Function &F) {
  std::vector<ReturnInst*> Returns;
  std::vector<UnwindInst*> Unwinds;
  std::vector<InvokeInst*> Invokes;

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

  if (Unwinds.empty() && Invokes.empty()) return false;

  NumInvokes += Invokes.size();
  NumUnwinds += Unwinds.size();
  
  // TODO: This is not an optimal way to do this.  In particular, this always
  // inserts setjmp calls into the entries of functions with invoke instructions
  // even though there are possibly paths through the function that do not
  // execute any invokes.  In particular, for functions with early exits, e.g.
  // the 'addMove' method in hexxagon, it would be nice to not have to do the
  // setjmp stuff on the early exit path.  This requires a bit of dataflow, but
  // would not be too hard to do.

  // If we have an invoke instruction, insert a setjmp that dominates all
  // invokes.  After the setjmp, use a cond branch that goes to the original
  // code path on zero, and to a designated 'catch' block of nonzero.
  Value *OldJmpBufPtr = 0;
  if (!Invokes.empty()) {
    // First thing we need to do is scan the whole function for values that are
    // live across unwind edges.  Each value that is live across an unwind edge
    // we spill into a stack location, guaranteeing that there is nothing live
    // across the unwind edge.  This process also splits all critical edges
    // coming out of invoke's.
    splitLiveRangesLiveAcrossInvokes(Invokes);    
    
    BasicBlock *EntryBB = F.begin();
    
    // Create an alloca for the incoming jump buffer ptr and the new jump buffer
    // that needs to be restored on all exits from the function.  This is an
    // alloca because the value needs to be live across invokes.
    AllocaInst *JmpBuf = 
      new AllocaInst(JBLinkTy, 0, JumpBufAlign, "jblink", F.begin()->begin());
    
    std::vector<Value*> Idx;
    Idx.push_back(Constant::getNullValue(Type::IntTy));
    Idx.push_back(ConstantUInt::get(Type::UIntTy, 1));
    OldJmpBufPtr = new GetElementPtrInst(JmpBuf, Idx, "OldBuf",
                                         EntryBB->getTerminator());

    // Copy the JBListHead to the alloca.
    Value *OldBuf = new LoadInst(JBListHead, "oldjmpbufptr", true,
                                 EntryBB->getTerminator());
    new StoreInst(OldBuf, OldJmpBufPtr, true, EntryBB->getTerminator());
    
    // Add the new jumpbuf to the list.
    new StoreInst(JmpBuf, JBListHead, true, EntryBB->getTerminator());

    // Create the catch block.  The catch block is basically a big switch
    // statement that goes to all of the invoke catch blocks.
    BasicBlock *CatchBB = new BasicBlock("setjmp.catch", &F);
    
    // Create an alloca which keeps track of which invoke is currently
    // executing.  For normal calls it contains zero.
    AllocaInst *InvokeNum = new AllocaInst(Type::UIntTy, 0, "invokenum",
                                           EntryBB->begin());
    new StoreInst(ConstantInt::get(Type::UIntTy, 0), InvokeNum, true,
                  EntryBB->getTerminator());
    
    // Insert a load in the Catch block, and a switch on its value.  By default,
    // we go to a block that just does an unwind (which is the correct action
    // for a standard call).
    BasicBlock *UnwindBB = new BasicBlock("unwindbb", &F);
    Unwinds.push_back(new UnwindInst(UnwindBB));
    
    Value *CatchLoad = new LoadInst(InvokeNum, "invoke.num", true, CatchBB);
    SwitchInst *CatchSwitch = 
      new SwitchInst(CatchLoad, UnwindBB, Invokes.size(), CatchBB);

    // Now that things are set up, insert the setjmp call itself.
    
    // Split the entry block to insert the conditional branch for the setjmp.
    BasicBlock *ContBlock = EntryBB->splitBasicBlock(EntryBB->getTerminator(),
                                                     "setjmp.cont");

    Idx[1] = ConstantUInt::get(Type::UIntTy, 0);
    Value *JmpBufPtr = new GetElementPtrInst(JmpBuf, Idx, "TheJmpBuf",
                                             EntryBB->getTerminator());
    Value *SJRet = new CallInst(SetJmpFn, JmpBufPtr, "sjret",
                                EntryBB->getTerminator());
    
    // Compare the return value to zero.
    Value *IsNormal = BinaryOperator::createSetEQ(SJRet,
                                     Constant::getNullValue(SJRet->getType()),
                                        "notunwind", EntryBB->getTerminator());
    // Nuke the uncond branch.
    EntryBB->getTerminator()->eraseFromParent();
    
    // Put in a new condbranch in its place.
    new BranchInst(ContBlock, CatchBB, IsNormal, EntryBB);

    // At this point, we are all set up, rewrite each invoke instruction.
    for (unsigned i = 0, e = Invokes.size(); i != e; ++i)
      rewriteExpensiveInvoke(Invokes[i], i+1, InvokeNum, CatchSwitch);
  }

  // We know that there is at least one unwind.
  
  // Create three new blocks, the block to load the jmpbuf ptr and compare
  // against null, the block to do the longjmp, and the error block for if it
  // is null.  Add them at the end of the function because they are not hot.
  BasicBlock *UnwindHandler = new BasicBlock("dounwind", &F);
  BasicBlock *UnwindBlock = new BasicBlock("unwind", &F);
  BasicBlock *TermBlock = new BasicBlock("unwinderror", &F);

  // If this function contains an invoke, restore the old jumpbuf ptr.
  Value *BufPtr;
  if (OldJmpBufPtr) {
    // Before the return, insert a copy from the saved value to the new value.
    BufPtr = new LoadInst(OldJmpBufPtr, "oldjmpbufptr", UnwindHandler);
    new StoreInst(BufPtr, JBListHead, UnwindHandler);
  } else {
    BufPtr = new LoadInst(JBListHead, "ehlist", UnwindHandler);
  }
  
  // Load the JBList, if it's null, then there was no catch!
  Value *NotNull = BinaryOperator::createSetNE(BufPtr,
                                      Constant::getNullValue(BufPtr->getType()),
                                          "notnull", UnwindHandler);
  new BranchInst(UnwindBlock, TermBlock, NotNull, UnwindHandler);
  
  // Create the block to do the longjmp.
  // Get a pointer to the jmpbuf and longjmp.
  std::vector<Value*> Idx;
  Idx.push_back(Constant::getNullValue(Type::IntTy));
  Idx.push_back(ConstantUInt::get(Type::UIntTy, 0));
  Idx[0] = new GetElementPtrInst(BufPtr, Idx, "JmpBuf", UnwindBlock);
  Idx[1] = ConstantInt::get(Type::IntTy, 1);
  new CallInst(LongJmpFn, Idx, "", UnwindBlock);
  new UnreachableInst(UnwindBlock);
  
  // Set up the term block ("throw without a catch").
  new UnreachableInst(TermBlock);

  // Insert a new call to write(2, AbortMessage, AbortMessageLength);
  writeAbortMessage(TermBlock->getTerminator());
  
  // Insert a call to abort()
  (new CallInst(AbortFn, std::vector<Value*>(), "",
                TermBlock->getTerminator()))->setTailCall();
    
  
  // Replace all unwinds with a branch to the unwind handler.
  for (unsigned i = 0, e = Unwinds.size(); i != e; ++i) {
    new BranchInst(UnwindHandler, Unwinds[i]);
    Unwinds[i]->eraseFromParent();    
  } 
  
  // Finally, for any returns from this function, if this function contains an
  // invoke, restore the old jmpbuf pointer to its input value.
  if (OldJmpBufPtr) {
    for (unsigned i = 0, e = Returns.size(); i != e; ++i) {
      ReturnInst *R = Returns[i];
      
      // Before the return, insert a copy from the saved value to the new value.
      Value *OldBuf = new LoadInst(OldJmpBufPtr, "oldjmpbufptr", true, R);
      new StoreInst(OldBuf, JBListHead, true, R);
    }
  }
  
  return true;
}

bool LowerInvoke::runOnFunction(Function &F) {
  if (ExpensiveEHSupport)
    return insertExpensiveEHSupport(F);
  else
    return insertCheapEHSupport(F);
}
