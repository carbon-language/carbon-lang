//===--- CGCXXTemp.cpp - Emit LLVM Code for C++ temporaries ---------------===//
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
    
  LiveTemporaries.push_back(CXXLiveTemporaryInfo(Temporary, Ptr, DtorBlock, 0));

  PushCleanupBlock(DtorBlock);
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
  
  EmitBlock(Info.DtorBlock);

  EmitCXXDestructorCall(Info.Temporary->getDestructor(),
                        Dtor_Complete, Info.ThisPtr);

  LiveTemporaries.pop_back();
}

RValue
CodeGenFunction::EmitCXXExprWithTemporaries(const CXXExprWithTemporaries *E,
                                            llvm::Value *AggLoc,
                                            bool isAggLocVolatile) {
  // Keep track of the current cleanup stack depth.
  size_t CleanupStackDepth = CleanupEntries.size();

  unsigned OldNumLiveTemporaries = LiveTemporaries.size();
  
  RValue RV = EmitAnyExpr(E->getSubExpr(), AggLoc, isAggLocVolatile);
  
  // Go through the temporaries backwards.
  for (unsigned i = E->getNumTemporaries(); i != 0; --i) {
    assert(LiveTemporaries.back().Temporary == E->getTemporary(i - 1));
    LiveTemporaries.pop_back();
  }

  assert(OldNumLiveTemporaries == LiveTemporaries.size() &&
         "Live temporary stack mismatch!");
  
  EmitCleanupBlocks(CleanupStackDepth);

  return RV;
}

void 
CodeGenFunction::PushConditionalTempDestruction() {
  // Store the current number of live temporaries.
  ConditionalTempDestructionStack.push_back(LiveTemporaries.size());
}

void CodeGenFunction::PopConditionalTempDestruction() {
  ConditionalTempDestructionStack.pop_back();
}
  
