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
  struct DestroyTemporary {
    static void Emit(CodeGenFunction &CGF, bool forEH,
                     const CXXDestructorDecl *dtor, llvm::Value *addr) {
      CGF.EmitCXXDestructorCall(dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                                addr);
    }
  };
}

/// Emits all the code to cause the given temporary to be cleaned up.
void CodeGenFunction::EmitCXXTemporary(const CXXTemporary *Temporary,
                                       llvm::Value *Ptr) {
  pushFullExprCleanup<DestroyTemporary>(NormalAndEHCleanup,
                                        Temporary->getDestructor(),
                                        Ptr);
}

RValue
CodeGenFunction::EmitExprWithCleanups(const ExprWithCleanups *E,
                                      AggValueSlot Slot) {
  RunCleanupsScope Scope(*this);
  return EmitAnyExpr(E->getSubExpr(), Slot);
}

LValue CodeGenFunction::EmitExprWithCleanupsLValue(const ExprWithCleanups *E) {
  RunCleanupsScope Scope(*this);
  return EmitLValue(E->getSubExpr());
}
