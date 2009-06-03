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
  LiveTemporaries.push_back(CXXLiveTemporaryInfo(Temporary, Ptr, 0, 0));
  
  // Make a cleanup scope and emit the destructor.
  {
    CleanupScope Scope(*this);
   
    EmitCXXDestructorCall(Temporary->getDestructor(), Dtor_Complete, Ptr);
  }
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
