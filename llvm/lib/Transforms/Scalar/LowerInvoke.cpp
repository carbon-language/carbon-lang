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
// support to the program at making the 'invoke' instruction really expensive.
// It basically inserts setjmp/longjmp calls to emulate the exception handling
// as necessary.
//
// Because the 'expensive' support slows down programs a lot, and EH is only
// used for a subset of the programs, it must be specifically enabled by an
// option.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "Support/Statistic.h"
#include "Support/CommandLine.h"
#include <csetjmp>
using namespace llvm;

namespace {
  Statistic<> NumLowered("lowerinvoke", "Number of invoke & unwinds replaced");
  cl::opt<bool> ExpensiveEHSupport("enable-correct-eh-support", 
 cl::desc("Make the -lowerinvoke pass insert expensive, but correct, EH code"));

  class LowerInvoke : public FunctionPass {
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
    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);
  private:
    void writeAbortMessage(Instruction *IB);
    bool insertCheapEHSupport(Function &F);
    bool insertExpensiveEHSupport(Function &F);
  };

  RegisterOpt<LowerInvoke>
  X("lowerinvoke", "Lower invoke and unwind, for unwindless code generators");
}

// Public Interface To the LowerInvoke pass.
FunctionPass *llvm::createLowerInvokePass() { return new LowerInvoke(); }

// doInitialization - Make sure that there is a prototype for abort in the
// current module.
bool LowerInvoke::doInitialization(Module &M) {
  const Type *VoidPtrTy = PointerType::get(Type::SByteTy);
  if (ExpensiveEHSupport) {
    // Insert a type for the linked list of jump buffers.  Unfortunately, we
    // don't know the size of the target's setjmp buffer, so we make a guess.
    // If this guess turns out to be too small, bad stuff could happen.
    unsigned JmpBufSize = 200;  // PPC has 192 words
    assert(sizeof(jmp_buf) <= JmpBufSize*sizeof(void*) &&
       "LowerInvoke doesn't know about targets with jmp_buf size > 200 words!");
    const Type *JmpBufTy = ArrayType::get(VoidPtrTy, JmpBufSize);

    { // The type is recursive, so use a type holder.
      std::vector<const Type*> Elements;
      OpaqueType *OT = OpaqueType::get();
      Elements.push_back(PointerType::get(OT));
      Elements.push_back(JmpBufTy);
      PATypeHolder JBLType(StructType::get(Elements));
      OT->refineAbstractTypeTo(JBLType.get());  // Complete the cycle.
      JBLinkTy = JBLType.get();
    }

    const Type *PtrJBList = PointerType::get(JBLinkTy);

    // Now that we've done that, insert the jmpbuf list head global, unless it
    // already exists.
    if (!(JBListHead = M.getGlobalVariable("llvm.sjljeh.jblist", PtrJBList)))
      JBListHead = new GlobalVariable(PtrJBList, false,
                                      GlobalValue::LinkOnceLinkage,
                                      Constant::getNullValue(PtrJBList),
                                      "llvm.sjljeh.jblist", &M);
    SetJmpFn = M.getOrInsertFunction("setjmp", Type::IntTy,
                                     PointerType::get(JmpBufTy), 0);
    LongJmpFn = M.getOrInsertFunction("longjmp", Type::VoidTy,
                                      PointerType::get(JmpBufTy),
                                      Type::IntTy, 0);
    
    // The abort message for expensive EH support tells the user that the
    // program 'unwound' without an 'invoke' instruction.
    Constant *Msg =
      ConstantArray::get("ERROR: Exception thrown, but not caught!\n");
    AbortMessageLength = Msg->getNumOperands()-1;  // don't include \0
  
    GlobalVariable *MsgGV = M.getGlobalVariable("abort.msg", Msg->getType());
    if (MsgGV && (!MsgGV->hasInitializer() || MsgGV->getInitializer() != Msg))
      MsgGV = 0;
    if (!MsgGV)
      MsgGV = new GlobalVariable(Msg->getType(), true,
                                 GlobalValue::InternalLinkage,
                                 Msg, "abort.msg", &M);
    std::vector<Constant*> GEPIdx(2, Constant::getNullValue(Type::LongTy));
    AbortMessage =
      ConstantExpr::getGetElementPtr(ConstantPointerRef::get(MsgGV), GEPIdx);

  } else {
    // The abort message for cheap EH support tells the user that EH is not
    // enabled.
    Constant *Msg =
      ConstantArray::get("Exception handler needed, but not enabled.  Recompile"
                         " program with -enable-correct-eh-support.\n");
    AbortMessageLength = Msg->getNumOperands()-1;  // don't include \0
  
    GlobalVariable *MsgGV = M.getGlobalVariable("abort.msg", Msg->getType());
    if (MsgGV && (!MsgGV->hasInitializer() || MsgGV->getInitializer() != Msg))
      MsgGV = 0;

    if (!MsgGV)
      MsgGV = new GlobalVariable(Msg->getType(), true,
                                 GlobalValue::InternalLinkage,
                                 Msg, "abort.msg", &M);
    std::vector<Constant*> GEPIdx(2, Constant::getNullValue(Type::LongTy));
    AbortMessage =
      ConstantExpr::getGetElementPtr(ConstantPointerRef::get(MsgGV), GEPIdx);
  }

  // We need the 'write' and 'abort' functions for both models.
  AbortFn = M.getOrInsertFunction("abort", Type::VoidTy, 0);

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
                                    VoidPtrTy, Type::IntTy, 0);
  }
  return true;
}

void LowerInvoke::writeAbortMessage(Instruction *IB) {
  if (WriteFn) {
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

    new CallInst(WriteFn, Args, "", IB);
  }
}

bool LowerInvoke::insertCheapEHSupport(Function &F) {
  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      // Insert a normal call instruction...
      std::string Name = II->getName(); II->setName("");
      Value *NewCall = new CallInst(II->getCalledValue(),
                                    std::vector<Value*>(II->op_begin()+3,
                                                        II->op_end()), Name,II);
      II->replaceAllUsesWith(NewCall);
      
      // Insert an unconditional branch to the normal destination.
      new BranchInst(II->getNormalDest(), II);

      // Remove any PHI node entries from the exception destination.
      II->getUnwindDest()->removePredecessor(BB);

      // Remove the invoke instruction now.
      BB->getInstList().erase(II);

      ++NumLowered; Changed = true;
    } else if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
      // Insert a new call to write(2, AbortMessage, AbortMessageLength);
      writeAbortMessage(UI);

      // Insert a call to abort()
      new CallInst(AbortFn, std::vector<Value*>(), "", UI);

      // Insert a return instruction.  This really should be a "barrier", as it
      // is unreachable.
      new ReturnInst(F.getReturnType() == Type::VoidTy ? 0 :
                            Constant::getNullValue(F.getReturnType()), UI);

      // Remove the unwind instruction now.
      BB->getInstList().erase(UI);

      ++NumLowered; Changed = true;
    }
  return Changed;
}

bool LowerInvoke::insertExpensiveEHSupport(Function &F) {
  bool Changed = false;

  // If a function uses invoke, we have an alloca for the jump buffer.
  AllocaInst *JmpBuf = 0;

  // If this function contains an unwind instruction, two blocks get added: one
  // to actually perform the longjmp, and one to terminate the program if there
  // is no handler.
  BasicBlock *UnwindBlock = 0, *TermBlock = 0;
  std::vector<LoadInst*> JBPtrs;

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      if (JmpBuf == 0)
        JmpBuf = new AllocaInst(JBLinkTy, 0, "jblink", F.begin()->begin());

      // On the entry to the invoke, we must install our JmpBuf as the top of
      // the stack.
      LoadInst *OldEntry = new LoadInst(JBListHead, "oldehlist", II);

      // Store this old value as our 'next' field, and store our alloca as the
      // current jblist.
      std::vector<Value*> Idx;
      Idx.push_back(Constant::getNullValue(Type::LongTy));
      Idx.push_back(ConstantUInt::get(Type::UByteTy, 0));
      Value *NextFieldPtr = new GetElementPtrInst(JmpBuf, Idx, "NextField", II);
      new StoreInst(OldEntry, NextFieldPtr, II);
      new StoreInst(JmpBuf, JBListHead, II);
      
      // Call setjmp, passing in the address of the jmpbuffer.
      Idx[1] = ConstantUInt::get(Type::UByteTy, 1);
      Value *JmpBufPtr = new GetElementPtrInst(JmpBuf, Idx, "TheJmpBuf", II);
      Value *SJRet = new CallInst(SetJmpFn, JmpBufPtr, "sjret", II);

      // Compare the return value to zero.
      Value *IsNormal = BinaryOperator::create(Instruction::SetEQ, SJRet,
                                       Constant::getNullValue(SJRet->getType()),
                                               "notunwind", II);
      // Create the receiver block if there is a critical edge to the normal
      // destination.
      SplitCriticalEdge(II, 0, this);
      Instruction *InsertLoc = II->getNormalDest()->begin();
      
      // Insert a normal call instruction on the normal execution path.
      std::string Name = II->getName(); II->setName("");
      Value *NewCall = new CallInst(II->getCalledValue(),
                                    std::vector<Value*>(II->op_begin()+3,
                                                        II->op_end()), Name,
                                    InsertLoc);
      II->replaceAllUsesWith(NewCall);
      
      // If we got this far, then no exception was thrown and we can pop our
      // jmpbuf entry off.
      new StoreInst(OldEntry, JBListHead, InsertLoc);

      // Now we change the invoke into a branch instruction.
      new BranchInst(II->getNormalDest(), II->getUnwindDest(), IsNormal, II);

      // Remove the InvokeInst now.
      BB->getInstList().erase(II);
      ++NumLowered; Changed = true;      
      
    } else if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
      if (UnwindBlock == 0) {
        // Create two new blocks, the unwind block and the terminate block.  Add
        // them at the end of the function because they are not hot.
        UnwindBlock = new BasicBlock("unwind", &F);
        TermBlock = new BasicBlock("unwinderror", &F);

        // Insert return instructions.  These really should be "barrier"s, as
        // they are unreachable.
        new ReturnInst(F.getReturnType() == Type::VoidTy ? 0 :
                       Constant::getNullValue(F.getReturnType()), UnwindBlock);
        new ReturnInst(F.getReturnType() == Type::VoidTy ? 0 :
                       Constant::getNullValue(F.getReturnType()), TermBlock);
      }

      // Load the JBList, if it's null, then there was no catch!
      LoadInst *Ptr = new LoadInst(JBListHead, "ehlist", UI);
      Value *NotNull = BinaryOperator::create(Instruction::SetNE, Ptr,
                                        Constant::getNullValue(Ptr->getType()),
                                              "notnull", UI);
      new BranchInst(UnwindBlock, TermBlock, NotNull, UI);

      // Remember the loaded value so we can insert the PHI node as needed.
      JBPtrs.push_back(Ptr);

      // Remove the UnwindInst now.
      BB->getInstList().erase(UI);
      ++NumLowered; Changed = true;      
    }

  // If an unwind instruction was inserted, we need to set up the Unwind and
  // term blocks.
  if (UnwindBlock) {
    // In the unwind block, we know that the pointer coming in on the JBPtrs
    // list are non-null.
    Instruction *RI = UnwindBlock->getTerminator();

    Value *RecPtr;
    if (JBPtrs.size() == 1)
      RecPtr = JBPtrs[0];
    else {
      // If there is more than one unwind in this function, make a PHI node to
      // merge in all of the loaded values.
      PHINode *PN = new PHINode(JBPtrs[0]->getType(), "jbptrs", RI);
      for (unsigned i = 0, e = JBPtrs.size(); i != e; ++i)
        PN->addIncoming(JBPtrs[i], JBPtrs[i]->getParent());
      RecPtr = PN;
    }

    // Now that we have a pointer to the whole record, remove the entry from the
    // JBList.
    std::vector<Value*> Idx;
    Idx.push_back(Constant::getNullValue(Type::LongTy));
    Idx.push_back(ConstantUInt::get(Type::UByteTy, 0));
    Value *NextFieldPtr = new GetElementPtrInst(RecPtr, Idx, "NextField", RI);
    Value *NextRec = new LoadInst(NextFieldPtr, "NextRecord", RI);
    new StoreInst(NextRec, JBListHead, RI);

    // Now that we popped the top of the JBList, get a pointer to the jmpbuf and
    // longjmp.
    Idx[1] = ConstantUInt::get(Type::UByteTy, 1);
    Idx[0] = new GetElementPtrInst(RecPtr, Idx, "JmpBuf", RI);
    Idx[1] = ConstantInt::get(Type::IntTy, 1);
    new CallInst(LongJmpFn, Idx, "", RI);

    // Now we set up the terminate block.
    RI = TermBlock->getTerminator();
    
    // Insert a new call to write(2, AbortMessage, AbortMessageLength);
    writeAbortMessage(RI);

    // Insert a call to abort()
    new CallInst(AbortFn, std::vector<Value*>(), "", RI);
  }

  return Changed;
}

bool LowerInvoke::runOnFunction(Function &F) {
  if (ExpensiveEHSupport)
    return insertExpensiveEHSupport(F);
  else
    return insertCheapEHSupport(F);
}
