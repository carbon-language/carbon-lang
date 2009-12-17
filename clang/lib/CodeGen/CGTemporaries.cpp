//===--- CGTemporaries.cpp - Emit LLVM Code for C++ temporaries -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of temporaries
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
using namespace clang;
using namespace CodeGen;

void CodeGenFunction::PushCXXTemporary(const CXXTemporary *Temporary,
                                       llvm::Value *Ptr) {
  llvm::BasicBlock *DtorBlock = createBasicBlock("temp.dtor");

  llvm::Value *CondPtr = 0;

  // Check if temporaries need to be conditional. If so, we'll create a
  // condition boolean, initialize it to 0 and
  if (ConditionalBranchLevel != 0) {
    CondPtr = CreateTempAlloca(llvm::Type::getInt1Ty(VMContext), "cond");

    // Initialize it to false. This initialization takes place right after
    // the alloca insert point.
    llvm::StoreInst *SI =
      new llvm::StoreInst(llvm::ConstantInt::getFalse(VMContext), CondPtr);
    llvm::BasicBlock *Block = AllocaInsertPt->getParent();
    Block->getInstList().insertAfter((llvm::Instruction *)AllocaInsertPt, SI);

    // Now set it to true.
    Builder.CreateStore(llvm::ConstantInt::getTrue(VMContext), CondPtr);
  }

  LiveTemporaries.push_back(CXXLiveTemporaryInfo(Temporary, Ptr, DtorBlock,
                                                 CondPtr));

  PushCleanupBlock(DtorBlock);

  if (Exceptions) {
    const CXXLiveTemporaryInfo& Info = LiveTemporaries.back();
    llvm::BasicBlock *CondEnd = 0;
    
    EHCleanupBlock Cleanup(*this);

    // If this is a conditional temporary, we need to check the condition
    // boolean and only call the destructor if it's true.
    if (Info.CondPtr) {
      llvm::BasicBlock *CondBlock = createBasicBlock("cond.dtor.call");
      CondEnd = createBasicBlock("cond.dtor.end");

      llvm::Value *Cond = Builder.CreateLoad(Info.CondPtr);
      Builder.CreateCondBr(Cond, CondBlock, CondEnd);
      EmitBlock(CondBlock);
    }

    EmitCXXDestructorCall(Info.Temporary->getDestructor(),
                          Dtor_Complete, Info.ThisPtr);

    if (CondEnd) {
      // Reset the condition. to false.
      Builder.CreateStore(llvm::ConstantInt::getFalse(VMContext), Info.CondPtr);
      EmitBlock(CondEnd);
    }
  }
}

void CodeGenFunction::PopCXXTemporary() {
  const CXXLiveTemporaryInfo& Info = LiveTemporaries.back();

  CleanupBlockInfo CleanupInfo = PopCleanupBlock();
  assert(CleanupInfo.CleanupBlock == Info.DtorBlock &&
         "Cleanup block mismatch!");
  assert(!CleanupInfo.SwitchBlock &&
         "Should not have a switch block for temporary cleanup!");
  assert(!CleanupInfo.EndBlock &&
         "Should not have an end block for temporary cleanup!");

  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();
  if (CurBB && !CurBB->getTerminator() &&
      Info.DtorBlock->getNumUses() == 0) {
    CurBB->getInstList().splice(CurBB->end(), Info.DtorBlock->getInstList());
    delete Info.DtorBlock;
  } else
    EmitBlock(Info.DtorBlock);

  llvm::BasicBlock *CondEnd = 0;

  // If this is a conditional temporary, we need to check the condition
  // boolean and only call the destructor if it's true.
  if (Info.CondPtr) {
    llvm::BasicBlock *CondBlock = createBasicBlock("cond.dtor.call");
    CondEnd = createBasicBlock("cond.dtor.end");

    llvm::Value *Cond = Builder.CreateLoad(Info.CondPtr);
    Builder.CreateCondBr(Cond, CondBlock, CondEnd);
    EmitBlock(CondBlock);
  }

  EmitCXXDestructorCall(Info.Temporary->getDestructor(),
                        Dtor_Complete, Info.ThisPtr);

  if (CondEnd) {
    // Reset the condition. to false.
    Builder.CreateStore(llvm::ConstantInt::getFalse(VMContext), Info.CondPtr);
    EmitBlock(CondEnd);
  }

  LiveTemporaries.pop_back();
}

RValue
CodeGenFunction::EmitCXXExprWithTemporaries(const CXXExprWithTemporaries *E,
                                            llvm::Value *AggLoc,
                                            bool IsAggLocVolatile,
                                            bool IsInitializer) {
  // Keep track of the current cleanup stack depth.
  size_t CleanupStackDepth = CleanupEntries.size();
  (void) CleanupStackDepth;

  unsigned OldNumLiveTemporaries = LiveTemporaries.size();

  RValue RV = EmitAnyExpr(E->getSubExpr(), AggLoc, IsAggLocVolatile,
                          /*IgnoreResult=*/false, IsInitializer);

  // Pop temporaries.
  while (LiveTemporaries.size() > OldNumLiveTemporaries)
    PopCXXTemporary();

  assert(CleanupEntries.size() == CleanupStackDepth &&
         "Cleanup size mismatch!");

  return RV;
}

LValue CodeGenFunction::EmitCXXExprWithTemporariesLValue(
                                              const CXXExprWithTemporaries *E) {
  // Keep track of the current cleanup stack depth.
  size_t CleanupStackDepth = CleanupEntries.size();
  (void) CleanupStackDepth;

  unsigned OldNumLiveTemporaries = LiveTemporaries.size();

  LValue LV = EmitLValue(E->getSubExpr());

  // Pop temporaries.
  while (LiveTemporaries.size() > OldNumLiveTemporaries)
    PopCXXTemporary();

  assert(CleanupEntries.size() == CleanupStackDepth &&
         "Cleanup size mismatch!");

  return LV;
}
