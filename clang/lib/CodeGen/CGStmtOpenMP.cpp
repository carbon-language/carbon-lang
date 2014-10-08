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
#include "TargetInfo.h"
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

void CodeGenFunction::EmitOMPLoopBody(const OMPLoopDirective &S,
                                      bool SeparateIter) {
  RunCleanupsScope BodyScope(*this);
  // Update counters values on current iteration.
  for (auto I : S.updates()) {
    EmitIgnoredExpr(I);
  }
  // On a continue in the body, jump to the end.
  auto Continue = getJumpDestInCurrentScope("omp.body.continue");
  BreakContinueStack.push_back(BreakContinue(JumpDest(), Continue));
  // Emit loop body.
  EmitStmt(S.getBody());
  // The end (updates/cleanups).
  EmitBlock(Continue.getBlock());
  BreakContinueStack.pop_back();
  if (SeparateIter) {
    // TODO: Update lastprivates if the SeparateIter flag is true.
    // This will be implemented in a follow-up OMPLastprivateClause patch, but
    // result should be still correct without it, as we do not make these
    // variables private yet.
  }
}

void CodeGenFunction::EmitOMPInnerLoop(const OMPLoopDirective &S,
                                       OMPPrivateScope &LoopScope,
                                       bool SeparateIter) {
  auto LoopExit = getJumpDestInCurrentScope("omp.inner.for.end");
  auto Cnt = getPGORegionCounter(&S);

  // Start the loop with a block that tests the condition.
  auto CondBlock = createBasicBlock("omp.inner.for.cond");
  EmitBlock(CondBlock);
  LoopStack.push(CondBlock);

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  auto ExitBlock = LoopExit.getBlock();
  if (LoopScope.requiresCleanups())
    ExitBlock = createBasicBlock("omp.inner.for.cond.cleanup");

  auto LoopBody = createBasicBlock("omp.inner.for.body");

  // Emit condition: "IV < LastIteration + 1 [ - 1]"
  // ("- 1" when lastprivate clause is present - separate one iteration).
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond(SeparateIter));
  Builder.CreateCondBr(BoolCondVal, LoopBody, ExitBlock,
                       PGO.createLoopWeights(S.getCond(SeparateIter), Cnt));

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  EmitBlock(LoopBody);
  Cnt.beginRegion(Builder);

  // Create a block for the increment.
  auto Continue = getJumpDestInCurrentScope("omp.inner.for.inc");
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  EmitOMPLoopBody(S);
  EmitStopPoint(&S);

  // Emit "IV = IV + 1" and a back-edge to the condition block.
  EmitBlock(Continue.getBlock());
  EmitIgnoredExpr(S.getInc());
  BreakContinueStack.pop_back();
  EmitBranch(CondBlock);
  LoopStack.pop();
  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock());
}

void CodeGenFunction::EmitOMPSimdFinal(const OMPLoopDirective &S) {
  auto IC = S.counters().begin();
  for (auto F : S.finals()) {
    if (LocalDeclMap.lookup(cast<DeclRefExpr>((*IC))->getDecl())) {
      EmitIgnoredExpr(F);
    }
    ++IC;
  }
}

static void EmitOMPAlignedClause(CodeGenFunction &CGF, CodeGenModule &CGM,
                                 const OMPAlignedClause &Clause) {
  unsigned ClauseAlignment = 0;
  if (auto AlignmentExpr = Clause.getAlignment()) {
    auto AlignmentCI =
        cast<llvm::ConstantInt>(CGF.EmitScalarExpr(AlignmentExpr));
    ClauseAlignment = static_cast<unsigned>(AlignmentCI->getZExtValue());
  }
  for (auto E : Clause.varlists()) {
    unsigned Alignment = ClauseAlignment;
    if (Alignment == 0) {
      // OpenMP [2.8.1, Description]
      // If no optional parameter isspecified, implementation-defined default
      // alignments for SIMD instructions on the target platforms are assumed.
      Alignment = CGM.getTargetCodeGenInfo().getOpenMPSimdDefaultAlignment(
          E->getType());
    }
    assert((Alignment == 0 || llvm::isPowerOf2_32(Alignment)) &&
           "alignment is not power of 2");
    if (Alignment != 0) {
      llvm::Value *PtrValue = CGF.EmitScalarExpr(E);
      CGF.EmitAlignmentAssumption(PtrValue, Alignment);
    }
  }
}

void CodeGenFunction::EmitOMPSimdDirective(const OMPSimdDirective &S) {
  // Pragma 'simd' code depends on presence of 'lastprivate'.
  // If present, we have to separate last iteration of the loop:
  //
  // if (LastIteration != 0) {
  //   for (IV in 0..LastIteration-1) BODY;
  //   BODY with updates of lastprivate vars;
  //   <Final counter/linear vars updates>;
  // }
  //
  // otherwise (when there's no lastprivate):
  //
  //   for (IV in 0..LastIteration) BODY;
  //   <Final counter/linear vars updates>;
  //

  // Walk clauses and process safelen/lastprivate.
  bool SeparateIter = false;
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
    case OMPC_aligned:
      EmitOMPAlignedClause(*this, CGM, cast<OMPAlignedClause>(*C));
      break;
    case OMPC_lastprivate:
      SeparateIter = true;
      break;
    default:
      // Not handled yet
      ;
    }
  }

  RunCleanupsScope DirectiveScope(*this);

  CGDebugInfo *DI = getDebugInfo();
  if (DI)
    DI->EmitLexicalBlockStart(Builder, S.getSourceRange().getBegin());

  // Emit the loop iteration variable.
  const Expr *IVExpr = S.getIterationVariable();
  const VarDecl *IVDecl = cast<VarDecl>(cast<DeclRefExpr>(IVExpr)->getDecl());
  EmitVarDecl(*IVDecl);
  EmitIgnoredExpr(S.getInit());

  // Emit the iterations count variable.
  // If it is not a variable, Sema decided to calculate iterations count on each
  // iteration (e.g., it is foldable into a constant).
  if (auto LIExpr = dyn_cast<DeclRefExpr>(S.getLastIteration())) {
    EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
    // Emit calculation of the iterations count.
    EmitIgnoredExpr(S.getCalcLastIteration());
  }

  if (SeparateIter) {
    // Emit: if (LastIteration > 0) - begin.
    RegionCounter Cnt = getPGORegionCounter(&S);
    auto ThenBlock = createBasicBlock("simd.if.then");
    auto ContBlock = createBasicBlock("simd.if.end");
    EmitBranchOnBoolExpr(S.getPreCond(), ThenBlock, ContBlock, Cnt.getCount());
    EmitBlock(ThenBlock);
    Cnt.beginRegion(Builder);
    // Emit 'then' code.
    {
      OMPPrivateScope LoopScope(*this);
      LoopScope.addPrivates(S.counters());
      EmitOMPInnerLoop(S, LoopScope, /* SeparateIter */ true);
      EmitOMPLoopBody(S, /* SeparateIter */ true);
    }
    EmitOMPSimdFinal(S);
    // Emit: if (LastIteration != 0) - end.
    EmitBranch(ContBlock);
    EmitBlock(ContBlock, true);
  } else {
    {
      OMPPrivateScope LoopScope(*this);
      LoopScope.addPrivates(S.counters());
      EmitOMPInnerLoop(S, LoopScope);
    }
    EmitOMPSimdFinal(S);
  }

  if (DI)
    DI->EmitLexicalBlockEnd(Builder, S.getSourceRange().getEnd());
}

void CodeGenFunction::EmitOMPForDirective(const OMPForDirective &) {
  llvm_unreachable("CodeGen for 'omp for' is not supported yet.");
}

void CodeGenFunction::EmitOMPForSimdDirective(const OMPForSimdDirective &) {
  llvm_unreachable("CodeGen for 'omp for simd' is not supported yet.");
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

void CodeGenFunction::EmitOMPCriticalDirective(const OMPCriticalDirective &S) {
  // __kmpc_critical();
  // <captured_body>
  // __kmpc_end_critical();
  //

  auto Lock = CGM.getOpenMPRuntime().GetCriticalRegionLock(
      S.getDirectiveName().getAsString());
  CGM.getOpenMPRuntime().EmitOMPCriticalRegionStart(*this, Lock,
                                                    S.getLocStart());
  {
    RunCleanupsScope Scope(*this);
    EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    EnsureInsertPoint();
  }
  CGM.getOpenMPRuntime().EmitOMPCriticalRegionEnd(*this, Lock, S.getLocEnd());
}

void
CodeGenFunction::EmitOMPParallelForDirective(const OMPParallelForDirective &) {
  llvm_unreachable("CodeGen for 'omp parallel for' is not supported yet.");
}

void CodeGenFunction::EmitOMPParallelForSimdDirective(
    const OMPParallelForSimdDirective &) {
  llvm_unreachable("CodeGen for 'omp parallel for simd' is not supported yet.");
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

void CodeGenFunction::EmitOMPOrderedDirective(const OMPOrderedDirective &) {
  llvm_unreachable("CodeGen for 'omp ordered' is not supported yet.");
}

void CodeGenFunction::EmitOMPAtomicDirective(const OMPAtomicDirective &) {
  llvm_unreachable("CodeGen for 'omp atomic' is not supported yet.");
}

void CodeGenFunction::EmitOMPTargetDirective(const OMPTargetDirective &) {
  llvm_unreachable("CodeGen for 'omp target' is not supported yet.");
}

