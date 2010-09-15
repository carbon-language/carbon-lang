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

namespace {
  struct DestroyTemporary : EHScopeStack::Cleanup {
    const CXXTemporary *Temporary;
    llvm::Value *Addr;
    llvm::Value *CondPtr;

    DestroyTemporary(const CXXTemporary *Temporary, llvm::Value *Addr,
                     llvm::Value *CondPtr)
      : Temporary(Temporary), Addr(Addr), CondPtr(CondPtr) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      llvm::BasicBlock *CondEnd = 0;
    
      // If this is a conditional temporary, we need to check the condition
      // boolean and only call the destructor if it's true.
      if (CondPtr) {
        llvm::BasicBlock *CondBlock =
          CGF.createBasicBlock("temp.cond-dtor.call");
        CondEnd = CGF.createBasicBlock("temp.cond-dtor.cont");

        llvm::Value *Cond = CGF.Builder.CreateLoad(CondPtr);
        CGF.Builder.CreateCondBr(Cond, CondBlock, CondEnd);
        CGF.EmitBlock(CondBlock);
      }

      CGF.EmitCXXDestructorCall(Temporary->getDestructor(),
                                Dtor_Complete, /*ForVirtualBase=*/false,
                                Addr);

      if (CondPtr) {
        // Reset the condition to false.
        CGF.Builder.CreateStore(CGF.Builder.getFalse(), CondPtr);
        CGF.EmitBlock(CondEnd);
      }
    }
  };
}

/// Emits all the code to cause the given temporary to be cleaned up.
void CodeGenFunction::EmitCXXTemporary(const CXXTemporary *Temporary,
                                       llvm::Value *Ptr) {
  llvm::AllocaInst *CondPtr = 0;

  // Check if temporaries need to be conditional. If so, we'll create a
  // condition boolean, initialize it to 0 and
  if (ConditionalBranchLevel != 0) {
    CondPtr = CreateTempAlloca(llvm::Type::getInt1Ty(VMContext), "cond");

    // Initialize it to false. This initialization takes place right after
    // the alloca insert point.
    InitTempAlloca(CondPtr, llvm::ConstantInt::getFalse(VMContext));

    // Now set it to true.
    Builder.CreateStore(Builder.getTrue(), CondPtr);
  }

  EHStack.pushCleanup<DestroyTemporary>(NormalAndEHCleanup,
                                        Temporary, Ptr, CondPtr);
}

RValue
CodeGenFunction::EmitCXXExprWithTemporaries(const CXXExprWithTemporaries *E,
                                            AggValueSlot Slot) {
  RunCleanupsScope Scope(*this);
  return EmitAnyExpr(E->getSubExpr(), Slot);
}

LValue CodeGenFunction::EmitCXXExprWithTemporariesLValue(
                                              const CXXExprWithTemporaries *E) {
  RunCleanupsScope Scope(*this);
  return EmitLValue(E->getSubExpr());
}
