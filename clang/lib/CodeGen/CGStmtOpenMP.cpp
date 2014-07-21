//===--- CGStmtOpenMP.cpp - Emit LLVM Code from Statements ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OpenMP nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                              OpenMP Directive Emission
//===----------------------------------------------------------------------===//

void CodeGenFunction::EmitOMPParallelDirective(const OMPParallelDirective &S) {
  const CapturedStmt *CS = cast<CapturedStmt>(S.getAssociatedStmt());
  llvm::Value *CapturedStruct = GenerateCapturedStmtArgument(*CS);

  llvm::Value *OutlinedFn;
  {
    CodeGenFunction CGF(CGM, true);
    CGCapturedStmtInfo CGInfo(*CS, CS->getCapturedRegionKind());
    CGF.CapturedStmtInfo = &CGInfo;
    OutlinedFn = CGF.GenerateCapturedStmtFunction(*CS);
  }

  // Build call __kmpc_fork_call(loc, 1, microtask, captured_struct/*context*/)
  llvm::Value *Args[] = {
      CGM.getOpenMPRuntime().EmitOpenMPUpdateLocation(*this, S.getLocStart()),
      Builder.getInt32(1), // Number of arguments after 'microtask' argument
      // (there is only one additional argument - 'context')
      Builder.CreateBitCast(OutlinedFn,
                            CGM.getOpenMPRuntime().getKmpc_MicroPointerTy()),
      EmitCastToVoidPtr(CapturedStruct)};
  llvm::Constant *RTLFn = CGM.getOpenMPRuntime().CreateRuntimeFunction(
      CGOpenMPRuntime::OMPRTL__kmpc_fork_call);
  EmitRuntimeCall(RTLFn, Args);
}

void CodeGenFunction::EmitOMPSimdDirective(const OMPSimdDirective &S) {
  const CapturedStmt *CS = cast<CapturedStmt>(S.getAssociatedStmt());
  const Stmt *Body = CS->getCapturedStmt();
  LoopStack.setParallel();
  LoopStack.setVectorizerEnable(true);
  for (auto C : S.clauses()) {
    switch (C->getClauseKind()) {
    case OMPC_safelen: {
      RValue Len = EmitAnyExpr(cast<OMPSafelenClause>(C)->getSafelen(),
                               AggValueSlot::ignored(), true);
      llvm::ConstantInt *Val = cast<llvm::ConstantInt>(Len.getScalarVal());
      LoopStack.setVectorizerWidth(Val->getZExtValue());
      // In presence of finite 'safelen', it may be unsafe to mark all
      // the memory instructions parallel, because loop-carried
      // dependences of 'safelen' iterations are possible.
      LoopStack.setParallel(false);
      break;
    }
    default:
      // Not handled yet
      ;
    }
  }
  EmitStmt(Body);
}

void CodeGenFunction::EmitOMPForDirective(const OMPForDirective &) {
  llvm_unreachable("CodeGen for 'omp for' is not supported yet.");
}

void CodeGenFunction::EmitOMPSectionsDirective(const OMPSectionsDirective &) {
  llvm_unreachable("CodeGen for 'omp sections' is not supported yet.");
}

void CodeGenFunction::EmitOMPSectionDirective(const OMPSectionDirective &) {
  llvm_unreachable("CodeGen for 'omp section' is not supported yet.");
}

void CodeGenFunction::EmitOMPSingleDirective(const OMPSingleDirective &) {
  llvm_unreachable("CodeGen for 'omp single' is not supported yet.");
}

void CodeGenFunction::EmitOMPMasterDirective(const OMPMasterDirective &) {
  llvm_unreachable("CodeGen for 'omp master' is not supported yet.");
}

void CodeGenFunction::EmitOMPCriticalDirective(const OMPCriticalDirective &) {
  llvm_unreachable("CodeGen for 'omp critical' is not supported yet.");
}

void
CodeGenFunction::EmitOMPParallelForDirective(const OMPParallelForDirective &) {
  llvm_unreachable("CodeGen for 'omp parallel for' is not supported yet.");
}

void CodeGenFunction::EmitOMPParallelSectionsDirective(
    const OMPParallelSectionsDirective &) {
  llvm_unreachable("CodeGen for 'omp parallel sections' is not supported yet.");
}

void CodeGenFunction::EmitOMPTaskDirective(const OMPTaskDirective &) {
  llvm_unreachable("CodeGen for 'omp task' is not supported yet.");
}

void CodeGenFunction::EmitOMPTaskyieldDirective(const OMPTaskyieldDirective &) {
  llvm_unreachable("CodeGen for 'omp taskyield' is not supported yet.");
}

void CodeGenFunction::EmitOMPBarrierDirective(const OMPBarrierDirective &) {
  llvm_unreachable("CodeGen for 'omp barrier' is not supported yet.");
}

void CodeGenFunction::EmitOMPTaskwaitDirective(const OMPTaskwaitDirective &) {
  llvm_unreachable("CodeGen for 'omp taskwait' is not supported yet.");
}

void CodeGenFunction::EmitOMPFlushDirective(const OMPFlushDirective &) {
  llvm_unreachable("CodeGen for 'omp flush' is not supported yet.");
}

