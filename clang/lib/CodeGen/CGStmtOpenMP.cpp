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
#include "TargetInfo.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                              OpenMP Directive Emission
//===----------------------------------------------------------------------===//
/// \brief Emits code for OpenMP 'if' clause using specified \a CodeGen
/// function. Here is the logic:
/// if (Cond) {
///   CodeGen(true);
/// } else {
///   CodeGen(false);
/// }
static void EmitOMPIfClause(CodeGenFunction &CGF, const Expr *Cond,
                            const std::function<void(bool)> &CodeGen) {
  CodeGenFunction::LexicalScope ConditionScope(CGF, Cond->getSourceRange());

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  bool CondConstant;
  if (CGF.ConstantFoldsToSimpleInteger(Cond, CondConstant)) {
    CodeGen(CondConstant);
    return;
  }

  // Otherwise, the condition did not fold, or we couldn't elide it.  Just
  // emit the conditional branch.
  auto ThenBlock = CGF.createBasicBlock(/*name*/ "omp_if.then");
  auto ElseBlock = CGF.createBasicBlock(/*name*/ "omp_if.else");
  auto ContBlock = CGF.createBasicBlock(/*name*/ "omp_if.end");
  CGF.EmitBranchOnBoolExpr(Cond, ThenBlock, ElseBlock, /*TrueCount*/ 0);

  // Emit the 'then' code.
  CGF.EmitBlock(ThenBlock);
  CodeGen(/*ThenBlock*/ true);
  CGF.EmitBranch(ContBlock);
  // Emit the 'else' code if present.
  {
    // There is no need to emit line number for unconditional branch.
    auto NL = ApplyDebugLocation::CreateEmpty(CGF);
    CGF.EmitBlock(ElseBlock);
  }
  CodeGen(/*ThenBlock*/ false);
  {
    // There is no need to emit line number for unconditional branch.
    auto NL = ApplyDebugLocation::CreateEmpty(CGF);
    CGF.EmitBranch(ContBlock);
  }
  // Emit the continuation block for code after the if.
  CGF.EmitBlock(ContBlock, /*IsFinished*/ true);
}

void CodeGenFunction::EmitOMPAggregateAssign(
    llvm::Value *DestAddr, llvm::Value *SrcAddr, QualType OriginalType,
    const llvm::function_ref<void(llvm::Value *, llvm::Value *)> &CopyGen) {
  // Perform element-by-element initialization.
  QualType ElementTy;
  auto SrcBegin = SrcAddr;
  auto DestBegin = DestAddr;
  auto ArrayTy = OriginalType->getAsArrayTypeUnsafe();
  auto NumElements = emitArrayLength(ArrayTy, ElementTy, DestBegin);
  // Cast from pointer to array type to pointer to single element.
  SrcBegin = Builder.CreatePointerBitCastOrAddrSpaceCast(SrcBegin,
                                                         DestBegin->getType());
  auto DestEnd = Builder.CreateGEP(DestBegin, NumElements);
  // The basic structure here is a while-do loop.
  auto BodyBB = createBasicBlock("omp.arraycpy.body");
  auto DoneBB = createBasicBlock("omp.arraycpy.done");
  auto IsEmpty =
      Builder.CreateICmpEQ(DestBegin, DestEnd, "omp.arraycpy.isempty");
  Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);

  // Enter the loop body, making that address the current address.
  auto EntryBB = Builder.GetInsertBlock();
  EmitBlock(BodyBB);
  auto SrcElementCurrent =
      Builder.CreatePHI(SrcBegin->getType(), 2, "omp.arraycpy.srcElementPast");
  SrcElementCurrent->addIncoming(SrcBegin, EntryBB);
  auto DestElementCurrent = Builder.CreatePHI(DestBegin->getType(), 2,
                                              "omp.arraycpy.destElementPast");
  DestElementCurrent->addIncoming(DestBegin, EntryBB);

  // Emit copy.
  CopyGen(DestElementCurrent, SrcElementCurrent);

  // Shift the address forward by one element.
  auto DestElementNext = Builder.CreateConstGEP1_32(
      DestElementCurrent, /*Idx0=*/1, "omp.arraycpy.dest.element");
  auto SrcElementNext = Builder.CreateConstGEP1_32(
      SrcElementCurrent, /*Idx0=*/1, "omp.arraycpy.src.element");
  // Check whether we've reached the end.
  auto Done =
      Builder.CreateICmpEQ(DestElementNext, DestEnd, "omp.arraycpy.done");
  Builder.CreateCondBr(Done, DoneBB, BodyBB);
  DestElementCurrent->addIncoming(DestElementNext, Builder.GetInsertBlock());
  SrcElementCurrent->addIncoming(SrcElementNext, Builder.GetInsertBlock());

  // Done.
  EmitBlock(DoneBB, /*IsFinished=*/true);
}

void CodeGenFunction::EmitOMPCopy(CodeGenFunction &CGF,
                                  QualType OriginalType, llvm::Value *DestAddr,
                                  llvm::Value *SrcAddr, const VarDecl *DestVD,
                                  const VarDecl *SrcVD, const Expr *Copy) {
  if (OriginalType->isArrayType()) {
    auto *BO = dyn_cast<BinaryOperator>(Copy);
    if (BO && BO->getOpcode() == BO_Assign) {
      // Perform simple memcpy for simple copying.
      CGF.EmitAggregateAssign(DestAddr, SrcAddr, OriginalType);
    } else {
      // For arrays with complex element types perform element by element
      // copying.
      CGF.EmitOMPAggregateAssign(
          DestAddr, SrcAddr, OriginalType,
          [&CGF, Copy, SrcVD, DestVD](llvm::Value *DestElement,
                                          llvm::Value *SrcElement) {
            // Working with the single array element, so have to remap
            // destination and source variables to corresponding array
            // elements.
            CodeGenFunction::OMPPrivateScope Remap(CGF);
            Remap.addPrivate(DestVD, [DestElement]() -> llvm::Value *{
              return DestElement;
            });
            Remap.addPrivate(
                SrcVD, [SrcElement]() -> llvm::Value *{ return SrcElement; });
            (void)Remap.Privatize();
            CGF.EmitIgnoredExpr(Copy);
          });
    }
  } else {
    // Remap pseudo source variable to private copy.
    CodeGenFunction::OMPPrivateScope Remap(CGF);
    Remap.addPrivate(SrcVD, [SrcAddr]() -> llvm::Value *{ return SrcAddr; });
    Remap.addPrivate(DestVD, [DestAddr]() -> llvm::Value *{ return DestAddr; });
    (void)Remap.Privatize();
    // Emit copying of the whole variable.
    CGF.EmitIgnoredExpr(Copy);
  }
}

bool CodeGenFunction::EmitOMPFirstprivateClause(const OMPExecutableDirective &D,
                                                OMPPrivateScope &PrivateScope) {
  auto FirstprivateFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_firstprivate;
  };
  llvm::DenseSet<const VarDecl *> EmittedAsFirstprivate;
  for (OMPExecutableDirective::filtered_clause_iterator<decltype(
           FirstprivateFilter)> I(D.clauses(), FirstprivateFilter);
       I; ++I) {
    auto *C = cast<OMPFirstprivateClause>(*I);
    auto IRef = C->varlist_begin();
    auto InitsRef = C->inits().begin();
    for (auto IInit : C->private_copies()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      if (EmittedAsFirstprivate.count(OrigVD) == 0) {
        EmittedAsFirstprivate.insert(OrigVD);
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(IInit)->getDecl());
        auto *VDInit = cast<VarDecl>(cast<DeclRefExpr>(*InitsRef)->getDecl());
        bool IsRegistered;
        DeclRefExpr DRE(
            const_cast<VarDecl *>(OrigVD),
            /*RefersToEnclosingVariableOrCapture=*/CapturedStmtInfo->lookup(
                OrigVD) != nullptr,
            (*IRef)->getType(), VK_LValue, (*IRef)->getExprLoc());
        auto *OriginalAddr = EmitLValue(&DRE).getAddress();
        if (OrigVD->getType()->isArrayType()) {
          // Emit VarDecl with copy init for arrays.
          // Get the address of the original variable captured in current
          // captured region.
          IsRegistered = PrivateScope.addPrivate(OrigVD, [&]() -> llvm::Value *{
            auto Emission = EmitAutoVarAlloca(*VD);
            auto *Init = VD->getInit();
            if (!isa<CXXConstructExpr>(Init) || isTrivialInitializer(Init)) {
              // Perform simple memcpy.
              EmitAggregateAssign(Emission.getAllocatedAddress(), OriginalAddr,
                                  (*IRef)->getType());
            } else {
              EmitOMPAggregateAssign(
                  Emission.getAllocatedAddress(), OriginalAddr,
                  (*IRef)->getType(),
                  [this, VDInit, Init](llvm::Value *DestElement,
                                       llvm::Value *SrcElement) {
                    // Clean up any temporaries needed by the initialization.
                    RunCleanupsScope InitScope(*this);
                    // Emit initialization for single element.
                    LocalDeclMap[VDInit] = SrcElement;
                    EmitAnyExprToMem(Init, DestElement,
                                     Init->getType().getQualifiers(),
                                     /*IsInitializer*/ false);
                    LocalDeclMap.erase(VDInit);
                  });
            }
            EmitAutoVarCleanups(Emission);
            return Emission.getAllocatedAddress();
          });
        } else {
          IsRegistered = PrivateScope.addPrivate(OrigVD, [&]() -> llvm::Value *{
            // Emit private VarDecl with copy init.
            // Remap temp VDInit variable to the address of the original
            // variable
            // (for proper handling of captured global variables).
            LocalDeclMap[VDInit] = OriginalAddr;
            EmitDecl(*VD);
            LocalDeclMap.erase(VDInit);
            return GetAddrOfLocalVar(VD);
          });
        }
        assert(IsRegistered &&
               "firstprivate var already registered as private");
        // Silence the warning about unused variable.
        (void)IsRegistered;
      }
      ++IRef, ++InitsRef;
    }
  }
  return !EmittedAsFirstprivate.empty();
}

void CodeGenFunction::EmitOMPPrivateClause(
    const OMPExecutableDirective &D,
    CodeGenFunction::OMPPrivateScope &PrivateScope) {
  auto PrivateFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_private;
  };
  for (OMPExecutableDirective::filtered_clause_iterator<decltype(PrivateFilter)>
           I(D.clauses(), PrivateFilter); I; ++I) {
    auto *C = cast<OMPPrivateClause>(*I);
    auto IRef = C->varlist_begin();
    for (auto IInit : C->private_copies()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      auto VD = cast<VarDecl>(cast<DeclRefExpr>(IInit)->getDecl());
      bool IsRegistered =
          PrivateScope.addPrivate(OrigVD, [&]() -> llvm::Value * {
            // Emit private VarDecl with copy init.
            EmitDecl(*VD);
            return GetAddrOfLocalVar(VD);
          });
      assert(IsRegistered && "private var already registered as private");
      // Silence the warning about unused variable.
      (void)IsRegistered;
      ++IRef;
    }
  }
}

bool CodeGenFunction::EmitOMPCopyinClause(const OMPExecutableDirective &D) {
  // threadprivate_var1 = master_threadprivate_var1;
  // operator=(threadprivate_var2, master_threadprivate_var2);
  // ...
  // __kmpc_barrier(&loc, global_tid);
  auto CopyinFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_copyin;
  };
  llvm::DenseSet<const VarDecl *> CopiedVars;
  llvm::BasicBlock *CopyBegin = nullptr, *CopyEnd = nullptr;
  for (OMPExecutableDirective::filtered_clause_iterator<decltype(CopyinFilter)>
           I(D.clauses(), CopyinFilter);
       I; ++I) {
    auto *C = cast<OMPCopyinClause>(*I);
    auto IRef = C->varlist_begin();
    auto ISrcRef = C->source_exprs().begin();
    auto IDestRef = C->destination_exprs().begin();
    for (auto *AssignOp : C->assignment_ops()) {
      auto *VD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      if (CopiedVars.insert(VD->getCanonicalDecl()).second) {
        // Get the address of the master variable.
        auto *MasterAddr = VD->isStaticLocal()
                               ? CGM.getStaticLocalDeclAddress(VD)
                               : CGM.GetAddrOfGlobal(VD);
        // Get the address of the threadprivate variable.
        auto *PrivateAddr = EmitLValue(*IRef).getAddress();
        if (CopiedVars.size() == 1) {
          // At first check if current thread is a master thread. If it is, no
          // need to copy data.
          CopyBegin = createBasicBlock("copyin.not.master");
          CopyEnd = createBasicBlock("copyin.not.master.end");
          Builder.CreateCondBr(
              Builder.CreateICmpNE(
                  Builder.CreatePtrToInt(MasterAddr, CGM.IntPtrTy),
                  Builder.CreatePtrToInt(PrivateAddr, CGM.IntPtrTy)),
              CopyBegin, CopyEnd);
          EmitBlock(CopyBegin);
        }
        auto *SrcVD = cast<VarDecl>(cast<DeclRefExpr>(*ISrcRef)->getDecl());
        auto *DestVD = cast<VarDecl>(cast<DeclRefExpr>(*IDestRef)->getDecl());
        EmitOMPCopy(*this, (*IRef)->getType(), PrivateAddr, MasterAddr, DestVD,
                    SrcVD, AssignOp);
      }
      ++IRef;
      ++ISrcRef;
      ++IDestRef;
    }
  }
  if (CopyEnd) {
    // Exit out of copying procedure for non-master thread.
    EmitBlock(CopyEnd, /*IsFinished=*/true);
    return true;
  }
  return false;
}

bool CodeGenFunction::EmitOMPLastprivateClauseInit(
    const OMPExecutableDirective &D, OMPPrivateScope &PrivateScope) {
  auto LastprivateFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_lastprivate;
  };
  bool HasAtLeastOneLastprivate = false;
  llvm::DenseSet<const VarDecl *> AlreadyEmittedVars;
  for (OMPExecutableDirective::filtered_clause_iterator<decltype(
           LastprivateFilter)> I(D.clauses(), LastprivateFilter);
       I; ++I) {
    auto *C = cast<OMPLastprivateClause>(*I);
    auto IRef = C->varlist_begin();
    auto IDestRef = C->destination_exprs().begin();
    for (auto *IInit : C->private_copies()) {
      // Keep the address of the original variable for future update at the end
      // of the loop.
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      if (AlreadyEmittedVars.insert(OrigVD->getCanonicalDecl()).second) {
        auto *DestVD = cast<VarDecl>(cast<DeclRefExpr>(*IDestRef)->getDecl());
        PrivateScope.addPrivate(DestVD, [this, OrigVD, IRef]() -> llvm::Value *{
          DeclRefExpr DRE(
              const_cast<VarDecl *>(OrigVD),
              /*RefersToEnclosingVariableOrCapture=*/CapturedStmtInfo->lookup(
                  OrigVD) != nullptr,
              (*IRef)->getType(), VK_LValue, (*IRef)->getExprLoc());
          return EmitLValue(&DRE).getAddress();
        });
        // Check if the variable is also a firstprivate: in this case IInit is
        // not generated. Initialization of this variable will happen in codegen
        // for 'firstprivate' clause.
        if (!IInit)
          continue;
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(IInit)->getDecl());
        bool IsRegistered =
            PrivateScope.addPrivate(OrigVD, [&]() -> llvm::Value *{
              // Emit private VarDecl with copy init.
              EmitDecl(*VD);
              return GetAddrOfLocalVar(VD);
            });
        assert(IsRegistered && "lastprivate var already registered as private");
        HasAtLeastOneLastprivate = HasAtLeastOneLastprivate || IsRegistered;
      }
      ++IRef, ++IDestRef;
    }
  }
  return HasAtLeastOneLastprivate;
}

void CodeGenFunction::EmitOMPLastprivateClauseFinal(
    const OMPExecutableDirective &D, llvm::Value *IsLastIterCond) {
  // Emit following code:
  // if (<IsLastIterCond>) {
  //   orig_var1 = private_orig_var1;
  //   ...
  //   orig_varn = private_orig_varn;
  // }
  auto *ThenBB = createBasicBlock(".omp.lastprivate.then");
  auto *DoneBB = createBasicBlock(".omp.lastprivate.done");
  Builder.CreateCondBr(IsLastIterCond, ThenBB, DoneBB);
  EmitBlock(ThenBB);
  {
    auto LastprivateFilter = [](const OMPClause *C) -> bool {
      return C->getClauseKind() == OMPC_lastprivate;
    };
    llvm::DenseSet<const VarDecl *> AlreadyEmittedVars;
    for (OMPExecutableDirective::filtered_clause_iterator<decltype(
             LastprivateFilter)> I(D.clauses(), LastprivateFilter);
         I; ++I) {
      auto *C = cast<OMPLastprivateClause>(*I);
      auto IRef = C->varlist_begin();
      auto ISrcRef = C->source_exprs().begin();
      auto IDestRef = C->destination_exprs().begin();
      for (auto *AssignOp : C->assignment_ops()) {
        auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
        if (AlreadyEmittedVars.insert(PrivateVD->getCanonicalDecl()).second) {
          auto *SrcVD = cast<VarDecl>(cast<DeclRefExpr>(*ISrcRef)->getDecl());
          auto *DestVD = cast<VarDecl>(cast<DeclRefExpr>(*IDestRef)->getDecl());
          // Get the address of the original variable.
          auto *OriginalAddr = GetAddrOfLocalVar(DestVD);
          // Get the address of the private variable.
          auto *PrivateAddr = GetAddrOfLocalVar(PrivateVD);
          EmitOMPCopy(*this, (*IRef)->getType(), OriginalAddr, PrivateAddr,
                      DestVD, SrcVD, AssignOp);
        }
        ++IRef;
        ++ISrcRef;
        ++IDestRef;
      }
    }
  }
  EmitBlock(DoneBB, /*IsFinished=*/true);
}

void CodeGenFunction::EmitOMPReductionClauseInit(
    const OMPExecutableDirective &D,
    CodeGenFunction::OMPPrivateScope &PrivateScope) {
  auto ReductionFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_reduction;
  };
  for (OMPExecutableDirective::filtered_clause_iterator<decltype(
           ReductionFilter)> I(D.clauses(), ReductionFilter);
       I; ++I) {
    auto *C = cast<OMPReductionClause>(*I);
    auto ILHS = C->lhs_exprs().begin();
    auto IRHS = C->rhs_exprs().begin();
    for (auto IRef : C->varlists()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(IRef)->getDecl());
      auto *LHSVD = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
      auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
      // Store the address of the original variable associated with the LHS
      // implicit variable.
      PrivateScope.addPrivate(LHSVD, [this, OrigVD, IRef]() -> llvm::Value *{
        DeclRefExpr DRE(const_cast<VarDecl *>(OrigVD),
                        CapturedStmtInfo->lookup(OrigVD) != nullptr,
                        IRef->getType(), VK_LValue, IRef->getExprLoc());
        return EmitLValue(&DRE).getAddress();
      });
      // Emit reduction copy.
      bool IsRegistered =
          PrivateScope.addPrivate(OrigVD, [this, PrivateVD]() -> llvm::Value *{
            // Emit private VarDecl with reduction init.
            EmitDecl(*PrivateVD);
            return GetAddrOfLocalVar(PrivateVD);
          });
      assert(IsRegistered && "private var already registered as private");
      // Silence the warning about unused variable.
      (void)IsRegistered;
      ++ILHS, ++IRHS;
    }
  }
}

void CodeGenFunction::EmitOMPReductionClauseFinal(
    const OMPExecutableDirective &D) {
  llvm::SmallVector<const Expr *, 8> LHSExprs;
  llvm::SmallVector<const Expr *, 8> RHSExprs;
  llvm::SmallVector<const Expr *, 8> ReductionOps;
  auto ReductionFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_reduction;
  };
  bool HasAtLeastOneReduction = false;
  for (OMPExecutableDirective::filtered_clause_iterator<decltype(
           ReductionFilter)> I(D.clauses(), ReductionFilter);
       I; ++I) {
    HasAtLeastOneReduction = true;
    auto *C = cast<OMPReductionClause>(*I);
    LHSExprs.append(C->lhs_exprs().begin(), C->lhs_exprs().end());
    RHSExprs.append(C->rhs_exprs().begin(), C->rhs_exprs().end());
    ReductionOps.append(C->reduction_ops().begin(), C->reduction_ops().end());
  }
  if (HasAtLeastOneReduction) {
    // Emit nowait reduction if nowait clause is present or directive is a
    // parallel directive (it always has implicit barrier).
    CGM.getOpenMPRuntime().emitReduction(
        *this, D.getLocEnd(), LHSExprs, RHSExprs, ReductionOps,
        D.getSingleClause(OMPC_nowait) ||
            isOpenMPParallelDirective(D.getDirectiveKind()));
  }
}

/// \brief Emits code for OpenMP parallel directive in the parallel region.
static void emitOMPParallelCall(CodeGenFunction &CGF,
                                const OMPExecutableDirective &S,
                                llvm::Value *OutlinedFn,
                                llvm::Value *CapturedStruct) {
  if (auto C = S.getSingleClause(/*K*/ OMPC_num_threads)) {
    CodeGenFunction::RunCleanupsScope NumThreadsScope(CGF);
    auto NumThreadsClause = cast<OMPNumThreadsClause>(C);
    auto NumThreads = CGF.EmitScalarExpr(NumThreadsClause->getNumThreads(),
                                         /*IgnoreResultAssign*/ true);
    CGF.CGM.getOpenMPRuntime().emitNumThreadsClause(
        CGF, NumThreads, NumThreadsClause->getLocStart());
  }
  CGF.CGM.getOpenMPRuntime().emitParallelCall(CGF, S.getLocStart(), OutlinedFn,
                                              CapturedStruct);
}

static void emitCommonOMPParallelDirective(CodeGenFunction &CGF,
                                           const OMPExecutableDirective &S,
                                           const RegionCodeGenTy &CodeGen) {
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto CapturedStruct = CGF.GenerateCapturedStmtArgument(*CS);
  auto OutlinedFn = CGF.CGM.getOpenMPRuntime().emitParallelOutlinedFunction(
      S, *CS->getCapturedDecl()->param_begin(), CodeGen);
  if (auto C = S.getSingleClause(/*K*/ OMPC_if)) {
    auto Cond = cast<OMPIfClause>(C)->getCondition();
    EmitOMPIfClause(CGF, Cond, [&](bool ThenBlock) {
      if (ThenBlock)
        emitOMPParallelCall(CGF, S, OutlinedFn, CapturedStruct);
      else
        CGF.CGM.getOpenMPRuntime().emitSerialCall(CGF, S.getLocStart(),
                                                  OutlinedFn, CapturedStruct);
    });
  } else
    emitOMPParallelCall(CGF, S, OutlinedFn, CapturedStruct);
}

void CodeGenFunction::EmitOMPParallelDirective(const OMPParallelDirective &S) {
  LexicalScope Scope(*this, S.getSourceRange());
  // Emit parallel region as a standalone region.
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
    OMPPrivateScope PrivateScope(CGF);
    bool Copyins = CGF.EmitOMPCopyinClause(S);
    bool Firstprivates = CGF.EmitOMPFirstprivateClause(S, PrivateScope);
    if (Copyins || Firstprivates) {
      // Emit implicit barrier to synchronize threads and avoid data races on
      // initialization of firstprivate variables or propagation master's thread
      // values of threadprivate variables to local instances of that variables
      // of all other implicit threads.
      CGF.CGM.getOpenMPRuntime().emitBarrierCall(CGF, S.getLocStart(),
                                                 OMPD_unknown);
    }
    CGF.EmitOMPPrivateClause(S, PrivateScope);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    (void)PrivateScope.Privatize();
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    CGF.EmitOMPReductionClauseFinal(S);
    // Emit implicit barrier at the end of the 'parallel' directive.
    CGF.CGM.getOpenMPRuntime().emitBarrierCall(CGF, S.getLocStart(),
                                               OMPD_unknown);
  };
  emitCommonOMPParallelDirective(*this, S, CodeGen);
}

void CodeGenFunction::EmitOMPLoopBody(const OMPLoopDirective &S,
                                      bool SeparateIter) {
  RunCleanupsScope BodyScope(*this);
  // Update counters values on current iteration.
  for (auto I : S.updates()) {
    EmitIgnoredExpr(I);
  }
  // Update the linear variables.
  for (auto C : OMPExecutableDirective::linear_filter(S.clauses())) {
    for (auto U : C->updates()) {
      EmitIgnoredExpr(U);
    }
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

void CodeGenFunction::EmitOMPInnerLoop(
    const Stmt &S, bool RequiresCleanup, const Expr *LoopCond,
    const Expr *IncExpr,
    const llvm::function_ref<void(CodeGenFunction &)> &BodyGen) {
  auto LoopExit = getJumpDestInCurrentScope("omp.inner.for.end");
  auto Cnt = getPGORegionCounter(&S);

  // Start the loop with a block that tests the condition.
  auto CondBlock = createBasicBlock("omp.inner.for.cond");
  EmitBlock(CondBlock);
  LoopStack.push(CondBlock);

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  auto ExitBlock = LoopExit.getBlock();
  if (RequiresCleanup)
    ExitBlock = createBasicBlock("omp.inner.for.cond.cleanup");

  auto LoopBody = createBasicBlock("omp.inner.for.body");

  // Emit condition.
  EmitBranchOnBoolExpr(LoopCond, LoopBody, ExitBlock, Cnt.getCount());
  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  EmitBlock(LoopBody);
  Cnt.beginRegion(Builder);

  // Create a block for the increment.
  auto Continue = getJumpDestInCurrentScope("omp.inner.for.inc");
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  BodyGen(*this);

  // Emit "IV = IV + 1" and a back-edge to the condition block.
  EmitBlock(Continue.getBlock());
  EmitIgnoredExpr(IncExpr);
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
  // Emit the final values of the linear variables.
  for (auto C : OMPExecutableDirective::linear_filter(S.clauses())) {
    for (auto F : C->finals()) {
      EmitIgnoredExpr(F);
    }
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
      // If no optional parameter is specified, implementation-defined default
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

static void EmitPrivateLoopCounters(CodeGenFunction &CGF,
                                    CodeGenFunction::OMPPrivateScope &LoopScope,
                                    ArrayRef<Expr *> Counters) {
  for (auto *E : Counters) {
    auto VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    bool IsRegistered = LoopScope.addPrivate(VD, [&]() -> llvm::Value * {
      // Emit var without initialization.
      auto VarEmission = CGF.EmitAutoVarAlloca(*VD);
      CGF.EmitAutoVarCleanups(VarEmission);
      return VarEmission.getAllocatedAddress();
    });
    assert(IsRegistered && "counter already registered as private");
    // Silence the warning about unused variable.
    (void)IsRegistered;
  }
}

static void
EmitPrivateLinearVars(CodeGenFunction &CGF, const OMPExecutableDirective &D,
                      CodeGenFunction::OMPPrivateScope &PrivateScope) {
  for (auto Clause : OMPExecutableDirective::linear_filter(D.clauses())) {
    for (auto *E : Clause->varlists()) {
      auto VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
      bool IsRegistered = PrivateScope.addPrivate(VD, [&]()->llvm::Value * {
        // Emit var without initialization.
        auto VarEmission = CGF.EmitAutoVarAlloca(*VD);
        CGF.EmitAutoVarCleanups(VarEmission);
        return VarEmission.getAllocatedAddress();
      });
      assert(IsRegistered && "linear var already registered as private");
      // Silence the warning about unused variable.
      (void)IsRegistered;
    }
  }
}

void CodeGenFunction::EmitOMPSimdDirective(const OMPSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
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
    CGF.LoopStack.setParallel();
    CGF.LoopStack.setVectorizerEnable(true);
    for (auto C : S.clauses()) {
      switch (C->getClauseKind()) {
      case OMPC_safelen: {
        RValue Len = CGF.EmitAnyExpr(cast<OMPSafelenClause>(C)->getSafelen(),
                                     AggValueSlot::ignored(), true);
        llvm::ConstantInt *Val = cast<llvm::ConstantInt>(Len.getScalarVal());
        CGF.LoopStack.setVectorizerWidth(Val->getZExtValue());
        // In presence of finite 'safelen', it may be unsafe to mark all
        // the memory instructions parallel, because loop-carried
        // dependences of 'safelen' iterations are possible.
        CGF.LoopStack.setParallel(false);
        break;
      }
      case OMPC_aligned:
        EmitOMPAlignedClause(CGF, CGF.CGM, cast<OMPAlignedClause>(*C));
        break;
      case OMPC_lastprivate:
        SeparateIter = true;
        break;
      default:
        // Not handled yet
        ;
      }
    }

    // Emit inits for the linear variables.
    for (auto C : OMPExecutableDirective::linear_filter(S.clauses())) {
      for (auto Init : C->inits()) {
        auto *D = cast<VarDecl>(cast<DeclRefExpr>(Init)->getDecl());
        CGF.EmitVarDecl(*D);
      }
    }

    // Emit the loop iteration variable.
    const Expr *IVExpr = S.getIterationVariable();
    const VarDecl *IVDecl = cast<VarDecl>(cast<DeclRefExpr>(IVExpr)->getDecl());
    CGF.EmitVarDecl(*IVDecl);
    CGF.EmitIgnoredExpr(S.getInit());

    // Emit the iterations count variable.
    // If it is not a variable, Sema decided to calculate iterations count on
    // each
    // iteration (e.g., it is foldable into a constant).
    if (auto LIExpr = dyn_cast<DeclRefExpr>(S.getLastIteration())) {
      CGF.EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
      // Emit calculation of the iterations count.
      CGF.EmitIgnoredExpr(S.getCalcLastIteration());
    }

    // Emit the linear steps for the linear clauses.
    // If a step is not constant, it is pre-calculated before the loop.
    for (auto C : OMPExecutableDirective::linear_filter(S.clauses())) {
      if (auto CS = cast_or_null<BinaryOperator>(C->getCalcStep()))
        if (auto SaveRef = cast<DeclRefExpr>(CS->getLHS())) {
          CGF.EmitVarDecl(*cast<VarDecl>(SaveRef->getDecl()));
          // Emit calculation of the linear step.
          CGF.EmitIgnoredExpr(CS);
        }
    }

    if (SeparateIter) {
      // Emit: if (LastIteration > 0) - begin.
      RegionCounter Cnt = CGF.getPGORegionCounter(&S);
      auto ThenBlock = CGF.createBasicBlock("simd.if.then");
      auto ContBlock = CGF.createBasicBlock("simd.if.end");
      CGF.EmitBranchOnBoolExpr(S.getPreCond(), ThenBlock, ContBlock,
                               Cnt.getCount());
      CGF.EmitBlock(ThenBlock);
      Cnt.beginRegion(CGF.Builder);
      // Emit 'then' code.
      {
        OMPPrivateScope LoopScope(CGF);
        EmitPrivateLoopCounters(CGF, LoopScope, S.counters());
        EmitPrivateLinearVars(CGF, S, LoopScope);
        CGF.EmitOMPPrivateClause(S, LoopScope);
        (void)LoopScope.Privatize();
        CGF.EmitOMPInnerLoop(S, LoopScope.requiresCleanups(),
                             S.getCond(/*SeparateIter=*/true), S.getInc(),
                             [&S](CodeGenFunction &CGF) {
                               CGF.EmitOMPLoopBody(S);
                               CGF.EmitStopPoint(&S);
                             });
        CGF.EmitOMPLoopBody(S, /* SeparateIter */ true);
      }
      CGF.EmitOMPSimdFinal(S);
      // Emit: if (LastIteration != 0) - end.
      CGF.EmitBranch(ContBlock);
      CGF.EmitBlock(ContBlock, true);
    } else {
      {
        OMPPrivateScope LoopScope(CGF);
        EmitPrivateLoopCounters(CGF, LoopScope, S.counters());
        EmitPrivateLinearVars(CGF, S, LoopScope);
        CGF.EmitOMPPrivateClause(S, LoopScope);
        (void)LoopScope.Privatize();
        CGF.EmitOMPInnerLoop(S, LoopScope.requiresCleanups(),
                             S.getCond(/*SeparateIter=*/false), S.getInc(),
                             [&S](CodeGenFunction &CGF) {
                               CGF.EmitOMPLoopBody(S);
                               CGF.EmitStopPoint(&S);
                             });
      }
      CGF.EmitOMPSimdFinal(S);
    }
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, CodeGen);
}

void CodeGenFunction::EmitOMPForOuterLoop(OpenMPScheduleClauseKind ScheduleKind,
                                          const OMPLoopDirective &S,
                                          OMPPrivateScope &LoopScope,
                                          llvm::Value *LB, llvm::Value *UB,
                                          llvm::Value *ST, llvm::Value *IL,
                                          llvm::Value *Chunk) {
  auto &RT = CGM.getOpenMPRuntime();

  // Dynamic scheduling of the outer loop (dynamic, guided, auto, runtime).
  const bool Dynamic = RT.isDynamic(ScheduleKind);

  assert(!RT.isStaticNonchunked(ScheduleKind, /* Chunked */ Chunk != nullptr) &&
         "static non-chunked schedule does not need outer loop");

  // Emit outer loop.
  //
  // OpenMP [2.7.1, Loop Construct, Description, table 2-1]
  // When schedule(dynamic,chunk_size) is specified, the iterations are
  // distributed to threads in the team in chunks as the threads request them.
  // Each thread executes a chunk of iterations, then requests another chunk,
  // until no chunks remain to be distributed. Each chunk contains chunk_size
  // iterations, except for the last chunk to be distributed, which may have
  // fewer iterations. When no chunk_size is specified, it defaults to 1.
  //
  // When schedule(guided,chunk_size) is specified, the iterations are assigned
  // to threads in the team in chunks as the executing threads request them.
  // Each thread executes a chunk of iterations, then requests another chunk,
  // until no chunks remain to be assigned. For a chunk_size of 1, the size of
  // each chunk is proportional to the number of unassigned iterations divided
  // by the number of threads in the team, decreasing to 1. For a chunk_size
  // with value k (greater than 1), the size of each chunk is determined in the
  // same way, with the restriction that the chunks do not contain fewer than k
  // iterations (except for the last chunk to be assigned, which may have fewer
  // than k iterations).
  //
  // When schedule(auto) is specified, the decision regarding scheduling is
  // delegated to the compiler and/or runtime system. The programmer gives the
  // implementation the freedom to choose any possible mapping of iterations to
  // threads in the team.
  //
  // When schedule(runtime) is specified, the decision regarding scheduling is
  // deferred until run time, and the schedule and chunk size are taken from the
  // run-sched-var ICV. If the ICV is set to auto, the schedule is
  // implementation defined
  //
  // while(__kmpc_dispatch_next(&LB, &UB)) {
  //   idx = LB;
  //   while (idx <= UB) { BODY; ++idx; } // inner loop
  // }
  //
  // OpenMP [2.7.1, Loop Construct, Description, table 2-1]
  // When schedule(static, chunk_size) is specified, iterations are divided into
  // chunks of size chunk_size, and the chunks are assigned to the threads in
  // the team in a round-robin fashion in the order of the thread number.
  //
  // while(UB = min(UB, GlobalUB), idx = LB, idx < UB) {
  //   while (idx <= UB) { BODY; ++idx; } // inner loop
  //   LB = LB + ST;
  //   UB = UB + ST;
  // }
  //

  const Expr *IVExpr = S.getIterationVariable();
  const unsigned IVSize = getContext().getTypeSize(IVExpr->getType());
  const bool IVSigned = IVExpr->getType()->hasSignedIntegerRepresentation();

  RT.emitForInit(
      *this, S.getLocStart(), ScheduleKind, IVSize, IVSigned, IL, LB,
      (Dynamic ? EmitAnyExpr(S.getLastIteration()).getScalarVal() : UB), ST,
      Chunk);

  auto LoopExit = getJumpDestInCurrentScope("omp.dispatch.end");

  // Start the loop with a block that tests the condition.
  auto CondBlock = createBasicBlock("omp.dispatch.cond");
  EmitBlock(CondBlock);
  LoopStack.push(CondBlock);

  llvm::Value *BoolCondVal = nullptr;
  if (!Dynamic) {
    // UB = min(UB, GlobalUB)
    EmitIgnoredExpr(S.getEnsureUpperBound());
    // IV = LB
    EmitIgnoredExpr(S.getInit());
    // IV < UB
    BoolCondVal = EvaluateExprAsBool(S.getCond(false));
  } else {
    BoolCondVal = RT.emitForNext(*this, S.getLocStart(), IVSize, IVSigned,
                                    IL, LB, UB, ST);
  }

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  auto ExitBlock = LoopExit.getBlock();
  if (LoopScope.requiresCleanups())
    ExitBlock = createBasicBlock("omp.dispatch.cleanup");

  auto LoopBody = createBasicBlock("omp.dispatch.body");
  Builder.CreateCondBr(BoolCondVal, LoopBody, ExitBlock);
  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }
  EmitBlock(LoopBody);

  // Emit "IV = LB" (in case of static schedule, we have already calculated new
  // LB for loop condition and emitted it above).
  if (Dynamic)
    EmitIgnoredExpr(S.getInit());

  // Create a block for the increment.
  auto Continue = getJumpDestInCurrentScope("omp.dispatch.inc");
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  EmitOMPInnerLoop(S, LoopScope.requiresCleanups(),
                   S.getCond(/*SeparateIter=*/false), S.getInc(),
                   [&S](CodeGenFunction &CGF) {
                     CGF.EmitOMPLoopBody(S);
                     CGF.EmitStopPoint(&S);
                   });

  EmitBlock(Continue.getBlock());
  BreakContinueStack.pop_back();
  if (!Dynamic) {
    // Emit "LB = LB + Stride", "UB = UB + Stride".
    EmitIgnoredExpr(S.getNextLowerBound());
    EmitIgnoredExpr(S.getNextUpperBound());
  }

  EmitBranch(CondBlock);
  LoopStack.pop();
  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock());

  // Tell the runtime we are done.
  // FIXME: Also call fini for ordered loops with dynamic scheduling.
  if (!Dynamic)
    RT.emitForFinish(*this, S.getLocStart(), ScheduleKind);
}

/// \brief Emit a helper variable and return corresponding lvalue.
static LValue EmitOMPHelperVar(CodeGenFunction &CGF,
                               const DeclRefExpr *Helper) {
  auto VDecl = cast<VarDecl>(Helper->getDecl());
  CGF.EmitVarDecl(*VDecl);
  return CGF.EmitLValue(Helper);
}

bool CodeGenFunction::EmitOMPWorksharingLoop(const OMPLoopDirective &S) {
  // Emit the loop iteration variable.
  auto IVExpr = cast<DeclRefExpr>(S.getIterationVariable());
  auto IVDecl = cast<VarDecl>(IVExpr->getDecl());
  EmitVarDecl(*IVDecl);

  // Emit the iterations count variable.
  // If it is not a variable, Sema decided to calculate iterations count on each
  // iteration (e.g., it is foldable into a constant).
  if (auto LIExpr = dyn_cast<DeclRefExpr>(S.getLastIteration())) {
    EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
    // Emit calculation of the iterations count.
    EmitIgnoredExpr(S.getCalcLastIteration());
  }

  auto &RT = CGM.getOpenMPRuntime();

  bool HasLastprivateClause;
  // Check pre-condition.
  {
    // Skip the entire loop if we don't meet the precondition.
    RegionCounter Cnt = getPGORegionCounter(&S);
    auto ThenBlock = createBasicBlock("omp.precond.then");
    auto ContBlock = createBasicBlock("omp.precond.end");
    EmitBranchOnBoolExpr(S.getPreCond(), ThenBlock, ContBlock, Cnt.getCount());
    EmitBlock(ThenBlock);
    Cnt.beginRegion(Builder);
    // Emit 'then' code.
    {
      // Emit helper vars inits.
      LValue LB =
          EmitOMPHelperVar(*this, cast<DeclRefExpr>(S.getLowerBoundVariable()));
      LValue UB =
          EmitOMPHelperVar(*this, cast<DeclRefExpr>(S.getUpperBoundVariable()));
      LValue ST =
          EmitOMPHelperVar(*this, cast<DeclRefExpr>(S.getStrideVariable()));
      LValue IL =
          EmitOMPHelperVar(*this, cast<DeclRefExpr>(S.getIsLastIterVariable()));

      OMPPrivateScope LoopScope(*this);
      if (EmitOMPFirstprivateClause(S, LoopScope)) {
        // Emit implicit barrier to synchronize threads and avoid data races on
        // initialization of firstprivate variables.
        CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(),
                                               OMPD_unknown);
      }
      HasLastprivateClause = EmitOMPLastprivateClauseInit(S, LoopScope);
      EmitPrivateLoopCounters(*this, LoopScope, S.counters());
      (void)LoopScope.Privatize();

      // Detect the loop schedule kind and chunk.
      auto ScheduleKind = OMPC_SCHEDULE_unknown;
      llvm::Value *Chunk = nullptr;
      if (auto C = cast_or_null<OMPScheduleClause>(
              S.getSingleClause(OMPC_schedule))) {
        ScheduleKind = C->getScheduleKind();
        if (auto Ch = C->getChunkSize()) {
          Chunk = EmitScalarExpr(Ch);
          Chunk = EmitScalarConversion(Chunk, Ch->getType(),
                                       S.getIterationVariable()->getType());
        }
      }
      const unsigned IVSize = getContext().getTypeSize(IVExpr->getType());
      const bool IVSigned = IVExpr->getType()->hasSignedIntegerRepresentation();
      if (RT.isStaticNonchunked(ScheduleKind,
                                /* Chunked */ Chunk != nullptr)) {
        // OpenMP [2.7.1, Loop Construct, Description, table 2-1]
        // When no chunk_size is specified, the iteration space is divided into
        // chunks that are approximately equal in size, and at most one chunk is
        // distributed to each thread. Note that the size of the chunks is
        // unspecified in this case.
        RT.emitForInit(*this, S.getLocStart(), ScheduleKind, IVSize, IVSigned,
                       IL.getAddress(), LB.getAddress(), UB.getAddress(),
                       ST.getAddress());
        // UB = min(UB, GlobalUB);
        EmitIgnoredExpr(S.getEnsureUpperBound());
        // IV = LB;
        EmitIgnoredExpr(S.getInit());
        // while (idx <= UB) { BODY; ++idx; }
        EmitOMPInnerLoop(S, LoopScope.requiresCleanups(),
                         S.getCond(/*SeparateIter=*/false), S.getInc(),
                         [&S](CodeGenFunction &CGF) {
                           CGF.EmitOMPLoopBody(S);
                           CGF.EmitStopPoint(&S);
                         });
        // Tell the runtime we are done.
        RT.emitForFinish(*this, S.getLocStart(), ScheduleKind);
      } else {
        // Emit the outer loop, which requests its work chunk [LB..UB] from
        // runtime and runs the inner loop to process it.
        EmitOMPForOuterLoop(ScheduleKind, S, LoopScope, LB.getAddress(),
                            UB.getAddress(), ST.getAddress(), IL.getAddress(),
                            Chunk);
      }
      // Emit final copy of the lastprivate variables if IsLastIter != 0.
      if (HasLastprivateClause)
        EmitOMPLastprivateClauseFinal(
            S, Builder.CreateIsNotNull(EmitLoadOfScalar(IL, S.getLocStart())));
    }
    // We're now done with the loop, so jump to the continuation block.
    EmitBranch(ContBlock);
    EmitBlock(ContBlock, true);
  }
  return HasLastprivateClause;
}

void CodeGenFunction::EmitOMPForDirective(const OMPForDirective &S) {
  LexicalScope Scope(*this, S.getSourceRange());
  bool HasLastprivates = false;
  auto &&CodeGen = [&S, &HasLastprivates](CodeGenFunction &CGF) {
    HasLastprivates = CGF.EmitOMPWorksharingLoop(S);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, CodeGen);

  // Emit an implicit barrier at the end.
  if (!S.getSingleClause(OMPC_nowait) || HasLastprivates) {
    CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(), OMPD_for);
  }
}

void CodeGenFunction::EmitOMPForSimdDirective(const OMPForSimdDirective &) {
  llvm_unreachable("CodeGen for 'omp for simd' is not supported yet.");
}

static LValue createSectionLVal(CodeGenFunction &CGF, QualType Ty,
                                const Twine &Name,
                                llvm::Value *Init = nullptr) {
  auto LVal = CGF.MakeNaturalAlignAddrLValue(CGF.CreateMemTemp(Ty, Name), Ty);
  if (Init)
    CGF.EmitScalarInit(Init, LVal);
  return LVal;
}

static OpenMPDirectiveKind emitSections(CodeGenFunction &CGF,
                                        const OMPExecutableDirective &S) {
  auto *Stmt = cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt();
  auto *CS = dyn_cast<CompoundStmt>(Stmt);
  if (CS && CS->size() > 1) {
    auto &&CodeGen = [&S, CS](CodeGenFunction &CGF) {
      auto &C = CGF.CGM.getContext();
      auto KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
      // Emit helper vars inits.
      LValue LB = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.lb.",
                                    CGF.Builder.getInt32(0));
      auto *GlobalUBVal = CGF.Builder.getInt32(CS->size() - 1);
      LValue UB =
          createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.ub.", GlobalUBVal);
      LValue ST = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.st.",
                                    CGF.Builder.getInt32(1));
      LValue IL = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.il.",
                                    CGF.Builder.getInt32(0));
      // Loop counter.
      LValue IV = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.iv.");
      OpaqueValueExpr IVRefExpr(S.getLocStart(), KmpInt32Ty, VK_LValue);
      CodeGenFunction::OpaqueValueMapping OpaqueIV(CGF, &IVRefExpr, IV);
      OpaqueValueExpr UBRefExpr(S.getLocStart(), KmpInt32Ty, VK_LValue);
      CodeGenFunction::OpaqueValueMapping OpaqueUB(CGF, &UBRefExpr, UB);
      // Generate condition for loop.
      BinaryOperator Cond(&IVRefExpr, &UBRefExpr, BO_LE, C.BoolTy, VK_RValue,
                          OK_Ordinary, S.getLocStart(),
                          /*fpContractable=*/false);
      // Increment for loop counter.
      UnaryOperator Inc(&IVRefExpr, UO_PreInc, KmpInt32Ty, VK_RValue,
                        OK_Ordinary, S.getLocStart());
      auto BodyGen = [CS, &S, &IV](CodeGenFunction &CGF) {
        // Iterate through all sections and emit a switch construct:
        // switch (IV) {
        //   case 0:
        //     <SectionStmt[0]>;
        //     break;
        // ...
        //   case <NumSection> - 1:
        //     <SectionStmt[<NumSection> - 1]>;
        //     break;
        // }
        // .omp.sections.exit:
        auto *ExitBB = CGF.createBasicBlock(".omp.sections.exit");
        auto *SwitchStmt = CGF.Builder.CreateSwitch(
            CGF.EmitLoadOfLValue(IV, S.getLocStart()).getScalarVal(), ExitBB,
            CS->size());
        unsigned CaseNumber = 0;
        for (auto C = CS->children(); C; ++C, ++CaseNumber) {
          auto CaseBB = CGF.createBasicBlock(".omp.sections.case");
          CGF.EmitBlock(CaseBB);
          SwitchStmt->addCase(CGF.Builder.getInt32(CaseNumber), CaseBB);
          CGF.EmitStmt(*C);
          CGF.EmitBranch(ExitBB);
        }
        CGF.EmitBlock(ExitBB, /*IsFinished=*/true);
      };
      // Emit static non-chunked loop.
      CGF.CGM.getOpenMPRuntime().emitForInit(
          CGF, S.getLocStart(), OMPC_SCHEDULE_static, /*IVSize=*/32,
          /*IVSigned=*/true, IL.getAddress(), LB.getAddress(), UB.getAddress(),
          ST.getAddress());
      // UB = min(UB, GlobalUB);
      auto *UBVal = CGF.EmitLoadOfScalar(UB, S.getLocStart());
      auto *MinUBGlobalUB = CGF.Builder.CreateSelect(
          CGF.Builder.CreateICmpSLT(UBVal, GlobalUBVal), UBVal, GlobalUBVal);
      CGF.EmitStoreOfScalar(MinUBGlobalUB, UB);
      // IV = LB;
      CGF.EmitStoreOfScalar(CGF.EmitLoadOfScalar(LB, S.getLocStart()), IV);
      // while (idx <= UB) { BODY; ++idx; }
      CGF.EmitOMPInnerLoop(S, /*RequiresCleanup=*/false, &Cond, &Inc, BodyGen);
      // Tell the runtime we are done.
      CGF.CGM.getOpenMPRuntime().emitForFinish(CGF, S.getLocStart(),
                                               OMPC_SCHEDULE_static);
    };

    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, CodeGen);
    return OMPD_sections;
  }
  // If only one section is found - no need to generate loop, emit as a single
  // region.
  auto &&CodeGen = [Stmt](CodeGenFunction &CGF) {
    CGF.EmitStmt(Stmt);
    CGF.EnsureInsertPoint();
  };
  CGF.CGM.getOpenMPRuntime().emitSingleRegion(CGF, CodeGen, S.getLocStart(),
                                              llvm::None, llvm::None,
                                              llvm::None, llvm::None);
  return OMPD_single;
}

void CodeGenFunction::EmitOMPSectionsDirective(const OMPSectionsDirective &S) {
  LexicalScope Scope(*this, S.getSourceRange());
  OpenMPDirectiveKind EmittedAs = emitSections(*this, S);
  // Emit an implicit barrier at the end.
  if (!S.getSingleClause(OMPC_nowait)) {
    CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(), EmittedAs);
  }
}

void CodeGenFunction::EmitOMPSectionDirective(const OMPSectionDirective &S) {
  LexicalScope Scope(*this, S.getSourceRange());
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    CGF.EnsureInsertPoint();
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, CodeGen);
}

void CodeGenFunction::EmitOMPSingleDirective(const OMPSingleDirective &S) {
  llvm::SmallVector<const Expr *, 8> CopyprivateVars;
  llvm::SmallVector<const Expr *, 8> DestExprs;
  llvm::SmallVector<const Expr *, 8> SrcExprs;
  llvm::SmallVector<const Expr *, 8> AssignmentOps;
  // Check if there are any 'copyprivate' clauses associated with this
  // 'single'
  // construct.
  auto CopyprivateFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_copyprivate;
  };
  // Build a list of copyprivate variables along with helper expressions
  // (<source>, <destination>, <destination>=<source> expressions)
  typedef OMPExecutableDirective::filtered_clause_iterator<decltype(
      CopyprivateFilter)> CopyprivateIter;
  for (CopyprivateIter I(S.clauses(), CopyprivateFilter); I; ++I) {
    auto *C = cast<OMPCopyprivateClause>(*I);
    CopyprivateVars.append(C->varlists().begin(), C->varlists().end());
    DestExprs.append(C->destination_exprs().begin(),
                     C->destination_exprs().end());
    SrcExprs.append(C->source_exprs().begin(), C->source_exprs().end());
    AssignmentOps.append(C->assignment_ops().begin(),
                         C->assignment_ops().end());
  }
  LexicalScope Scope(*this, S.getSourceRange());
  // Emit code for 'single' region along with 'copyprivate' clauses
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    CGF.EnsureInsertPoint();
  };
  CGM.getOpenMPRuntime().emitSingleRegion(*this, CodeGen, S.getLocStart(),
                                          CopyprivateVars, DestExprs, SrcExprs,
                                          AssignmentOps);
  // Emit an implicit barrier at the end.
  if (!S.getSingleClause(OMPC_nowait)) {
    CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(), OMPD_single);
  }
}

void CodeGenFunction::EmitOMPMasterDirective(const OMPMasterDirective &S) {
  LexicalScope Scope(*this, S.getSourceRange());
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    CGF.EnsureInsertPoint();
  };
  CGM.getOpenMPRuntime().emitMasterRegion(*this, CodeGen, S.getLocStart());
}

void CodeGenFunction::EmitOMPCriticalDirective(const OMPCriticalDirective &S) {
  LexicalScope Scope(*this, S.getSourceRange());
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    CGF.EnsureInsertPoint();
  };
  CGM.getOpenMPRuntime().emitCriticalRegion(
      *this, S.getDirectiveName().getAsString(), CodeGen, S.getLocStart());
}

void CodeGenFunction::EmitOMPParallelForDirective(
    const OMPParallelForDirective &S) {
  // Emit directive as a combined directive that consists of two implicit
  // directives: 'parallel' with 'for' directive.
  LexicalScope Scope(*this, S.getSourceRange());
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
    CGF.EmitOMPWorksharingLoop(S);
    // Emit implicit barrier at the end of parallel region, but this barrier
    // is at the end of 'for' directive, so emit it as the implicit barrier for
    // this 'for' directive.
    CGF.CGM.getOpenMPRuntime().emitBarrierCall(CGF, S.getLocStart(),
                                               OMPD_parallel);
  };
  emitCommonOMPParallelDirective(*this, S, CodeGen);
}

void CodeGenFunction::EmitOMPParallelForSimdDirective(
    const OMPParallelForSimdDirective &) {
  llvm_unreachable("CodeGen for 'omp parallel for simd' is not supported yet.");
}

void CodeGenFunction::EmitOMPParallelSectionsDirective(
    const OMPParallelSectionsDirective &S) {
  // Emit directive as a combined directive that consists of two implicit
  // directives: 'parallel' with 'sections' directive.
  LexicalScope Scope(*this, S.getSourceRange());
  auto &&CodeGen = [&S](CodeGenFunction &CGF) {
    (void)emitSections(CGF, S);
    // Emit implicit barrier at the end of parallel region.
    CGF.CGM.getOpenMPRuntime().emitBarrierCall(CGF, S.getLocStart(),
                                               OMPD_parallel);
  };
  emitCommonOMPParallelDirective(*this, S, CodeGen);
}

void CodeGenFunction::EmitOMPTaskDirective(const OMPTaskDirective &S) {
  // Emit outlined function for task construct.
  LexicalScope Scope(*this, S.getSourceRange());
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto CapturedStruct = GenerateCapturedStmtArgument(*CS);
  auto *I = CS->getCapturedDecl()->param_begin();
  auto *PartId = std::next(I);
  // The first function argument for tasks is a thread id, the second one is a
  // part id (0 for tied tasks, >=0 for untied task).
  auto &&CodeGen = [PartId, &S](CodeGenFunction &CGF) {
    if (*PartId) {
      // TODO: emit code for untied tasks.
    }
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
  };
  auto OutlinedFn =
      CGM.getOpenMPRuntime().emitTaskOutlinedFunction(S, *I, CodeGen);
  // Check if we should emit tied or untied task.
  bool Tied = !S.getSingleClause(OMPC_untied);
  // Check if the task is final
  llvm::PointerIntPair<llvm::Value *, 1, bool> Final;
  if (auto *Clause = S.getSingleClause(OMPC_final)) {
    // If the condition constant folds and can be elided, try to avoid emitting
    // the condition and the dead arm of the if/else.
    auto *Cond = cast<OMPFinalClause>(Clause)->getCondition();
    bool CondConstant;
    if (ConstantFoldsToSimpleInteger(Cond, CondConstant))
      Final.setInt(CondConstant);
    else
      Final.setPointer(EvaluateExprAsBool(Cond));
  } else {
    // By default the task is not final.
    Final.setInt(/*IntVal=*/false);
  }
  auto SharedsTy = getContext().getRecordType(CS->getCapturedRecordDecl());
  CGM.getOpenMPRuntime().emitTaskCall(*this, S.getLocStart(), Tied, Final,
                                      OutlinedFn, SharedsTy, CapturedStruct);
}

void CodeGenFunction::EmitOMPTaskyieldDirective(
    const OMPTaskyieldDirective &S) {
  CGM.getOpenMPRuntime().emitTaskyieldCall(*this, S.getLocStart());
}

void CodeGenFunction::EmitOMPBarrierDirective(const OMPBarrierDirective &S) {
  CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(), OMPD_barrier);
}

void CodeGenFunction::EmitOMPTaskwaitDirective(const OMPTaskwaitDirective &) {
  llvm_unreachable("CodeGen for 'omp taskwait' is not supported yet.");
}

void CodeGenFunction::EmitOMPFlushDirective(const OMPFlushDirective &S) {
  CGM.getOpenMPRuntime().emitFlush(*this, [&]() -> ArrayRef<const Expr *> {
    if (auto C = S.getSingleClause(/*K*/ OMPC_flush)) {
      auto FlushClause = cast<OMPFlushClause>(C);
      return llvm::makeArrayRef(FlushClause->varlist_begin(),
                                FlushClause->varlist_end());
    }
    return llvm::None;
  }(), S.getLocStart());
}

void CodeGenFunction::EmitOMPOrderedDirective(const OMPOrderedDirective &) {
  llvm_unreachable("CodeGen for 'omp ordered' is not supported yet.");
}

static llvm::Value *convertToScalarValue(CodeGenFunction &CGF, RValue Val,
                                         QualType SrcType, QualType DestType) {
  assert(CGF.hasScalarEvaluationKind(DestType) &&
         "DestType must have scalar evaluation kind.");
  assert(!Val.isAggregate() && "Must be a scalar or complex.");
  return Val.isScalar()
             ? CGF.EmitScalarConversion(Val.getScalarVal(), SrcType, DestType)
             : CGF.EmitComplexToScalarConversion(Val.getComplexVal(), SrcType,
                                                 DestType);
}

static CodeGenFunction::ComplexPairTy
convertToComplexValue(CodeGenFunction &CGF, RValue Val, QualType SrcType,
                      QualType DestType) {
  assert(CGF.getEvaluationKind(DestType) == TEK_Complex &&
         "DestType must have complex evaluation kind.");
  CodeGenFunction::ComplexPairTy ComplexVal;
  if (Val.isScalar()) {
    // Convert the input element to the element type of the complex.
    auto DestElementType = DestType->castAs<ComplexType>()->getElementType();
    auto ScalarVal =
        CGF.EmitScalarConversion(Val.getScalarVal(), SrcType, DestElementType);
    ComplexVal = CodeGenFunction::ComplexPairTy(
        ScalarVal, llvm::Constant::getNullValue(ScalarVal->getType()));
  } else {
    assert(Val.isComplex() && "Must be a scalar or complex.");
    auto SrcElementType = SrcType->castAs<ComplexType>()->getElementType();
    auto DestElementType = DestType->castAs<ComplexType>()->getElementType();
    ComplexVal.first = CGF.EmitScalarConversion(
        Val.getComplexVal().first, SrcElementType, DestElementType);
    ComplexVal.second = CGF.EmitScalarConversion(
        Val.getComplexVal().second, SrcElementType, DestElementType);
  }
  return ComplexVal;
}

static void EmitOMPAtomicReadExpr(CodeGenFunction &CGF, bool IsSeqCst,
                                  const Expr *X, const Expr *V,
                                  SourceLocation Loc) {
  // v = x;
  assert(V->isLValue() && "V of 'omp atomic read' is not lvalue");
  assert(X->isLValue() && "X of 'omp atomic read' is not lvalue");
  LValue XLValue = CGF.EmitLValue(X);
  LValue VLValue = CGF.EmitLValue(V);
  RValue Res = XLValue.isGlobalReg()
                   ? CGF.EmitLoadOfLValue(XLValue, Loc)
                   : CGF.EmitAtomicLoad(XLValue, Loc,
                                        IsSeqCst ? llvm::SequentiallyConsistent
                                                 : llvm::Monotonic,
                                        XLValue.isVolatile());
  // OpenMP, 2.12.6, atomic Construct
  // Any atomic construct with a seq_cst clause forces the atomically
  // performed operation to include an implicit flush operation without a
  // list.
  if (IsSeqCst)
    CGF.CGM.getOpenMPRuntime().emitFlush(CGF, llvm::None, Loc);
  switch (CGF.getEvaluationKind(V->getType())) {
  case TEK_Scalar:
    CGF.EmitStoreOfScalar(
        convertToScalarValue(CGF, Res, X->getType(), V->getType()), VLValue);
    break;
  case TEK_Complex:
    CGF.EmitStoreOfComplex(
        convertToComplexValue(CGF, Res, X->getType(), V->getType()), VLValue,
        /*isInit=*/false);
    break;
  case TEK_Aggregate:
    llvm_unreachable("Must be a scalar or complex.");
  }
}

static void EmitOMPAtomicWriteExpr(CodeGenFunction &CGF, bool IsSeqCst,
                                   const Expr *X, const Expr *E,
                                   SourceLocation Loc) {
  // x = expr;
  assert(X->isLValue() && "X of 'omp atomic write' is not lvalue");
  LValue XLValue = CGF.EmitLValue(X);
  RValue ExprRValue = CGF.EmitAnyExpr(E);
  if (XLValue.isGlobalReg())
    CGF.EmitStoreThroughGlobalRegLValue(ExprRValue, XLValue);
  else
    CGF.EmitAtomicStore(ExprRValue, XLValue,
                        IsSeqCst ? llvm::SequentiallyConsistent
                                 : llvm::Monotonic,
                        XLValue.isVolatile(), /*IsInit=*/false);
  // OpenMP, 2.12.6, atomic Construct
  // Any atomic construct with a seq_cst clause forces the atomically
  // performed operation to include an implicit flush operation without a
  // list.
  if (IsSeqCst)
    CGF.CGM.getOpenMPRuntime().emitFlush(CGF, llvm::None, Loc);
}

bool emitOMPAtomicRMW(CodeGenFunction &CGF, LValue X, RValue Update,
                      BinaryOperatorKind BO, llvm::AtomicOrdering AO,
                      bool IsXLHSInRHSPart) {
  auto &Context = CGF.CGM.getContext();
  // Allow atomicrmw only if 'x' and 'update' are integer values, lvalue for 'x'
  // expression is simple and atomic is allowed for the given type for the
  // target platform.
  if (BO == BO_Comma || !Update.isScalar() ||
      !Update.getScalarVal()->getType()->isIntegerTy() || !X.isSimple() ||
      (!isa<llvm::ConstantInt>(Update.getScalarVal()) &&
       (Update.getScalarVal()->getType() !=
        X.getAddress()->getType()->getPointerElementType())) ||
      !Context.getTargetInfo().hasBuiltinAtomic(
          Context.getTypeSize(X.getType()), Context.toBits(X.getAlignment())))
    return false;

  llvm::AtomicRMWInst::BinOp RMWOp;
  switch (BO) {
  case BO_Add:
    RMWOp = llvm::AtomicRMWInst::Add;
    break;
  case BO_Sub:
    if (!IsXLHSInRHSPart)
      return false;
    RMWOp = llvm::AtomicRMWInst::Sub;
    break;
  case BO_And:
    RMWOp = llvm::AtomicRMWInst::And;
    break;
  case BO_Or:
    RMWOp = llvm::AtomicRMWInst::Or;
    break;
  case BO_Xor:
    RMWOp = llvm::AtomicRMWInst::Xor;
    break;
  case BO_LT:
    RMWOp = X.getType()->hasSignedIntegerRepresentation()
                ? (IsXLHSInRHSPart ? llvm::AtomicRMWInst::Min
                                   : llvm::AtomicRMWInst::Max)
                : (IsXLHSInRHSPart ? llvm::AtomicRMWInst::UMin
                                   : llvm::AtomicRMWInst::UMax);
    break;
  case BO_GT:
    RMWOp = X.getType()->hasSignedIntegerRepresentation()
                ? (IsXLHSInRHSPart ? llvm::AtomicRMWInst::Max
                                   : llvm::AtomicRMWInst::Min)
                : (IsXLHSInRHSPart ? llvm::AtomicRMWInst::UMax
                                   : llvm::AtomicRMWInst::UMin);
    break;
  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Shl:
  case BO_Shr:
  case BO_LAnd:
  case BO_LOr:
    return false;
  case BO_PtrMemD:
  case BO_PtrMemI:
  case BO_LE:
  case BO_GE:
  case BO_EQ:
  case BO_NE:
  case BO_Assign:
  case BO_AddAssign:
  case BO_SubAssign:
  case BO_AndAssign:
  case BO_OrAssign:
  case BO_XorAssign:
  case BO_MulAssign:
  case BO_DivAssign:
  case BO_RemAssign:
  case BO_ShlAssign:
  case BO_ShrAssign:
  case BO_Comma:
    llvm_unreachable("Unsupported atomic update operation");
  }
  auto *UpdateVal = Update.getScalarVal();
  if (auto *IC = dyn_cast<llvm::ConstantInt>(UpdateVal)) {
    UpdateVal = CGF.Builder.CreateIntCast(
        IC, X.getAddress()->getType()->getPointerElementType(),
        X.getType()->hasSignedIntegerRepresentation());
  }
  CGF.Builder.CreateAtomicRMW(RMWOp, X.getAddress(), UpdateVal, AO);
  return true;
}

void CodeGenFunction::EmitOMPAtomicSimpleUpdateExpr(
    LValue X, RValue E, BinaryOperatorKind BO, bool IsXLHSInRHSPart,
    llvm::AtomicOrdering AO, SourceLocation Loc,
    const llvm::function_ref<RValue(RValue)> &CommonGen) {
  // Update expressions are allowed to have the following forms:
  // x binop= expr; -> xrval + expr;
  // x++, ++x -> xrval + 1;
  // x--, --x -> xrval - 1;
  // x = x binop expr; -> xrval binop expr
  // x = expr Op x; - > expr binop xrval;
  if (!emitOMPAtomicRMW(*this, X, E, BO, AO, IsXLHSInRHSPart)) {
    if (X.isGlobalReg()) {
      // Emit an update expression: 'xrval' binop 'expr' or 'expr' binop
      // 'xrval'.
      EmitStoreThroughLValue(CommonGen(EmitLoadOfLValue(X, Loc)), X);
    } else {
      // Perform compare-and-swap procedure.
      EmitAtomicUpdate(X, AO, CommonGen, X.getType().isVolatileQualified());
    }
  }
}

static void EmitOMPAtomicUpdateExpr(CodeGenFunction &CGF, bool IsSeqCst,
                                    const Expr *X, const Expr *E,
                                    const Expr *UE, bool IsXLHSInRHSPart,
                                    SourceLocation Loc) {
  assert(isa<BinaryOperator>(UE->IgnoreImpCasts()) &&
         "Update expr in 'atomic update' must be a binary operator.");
  auto *BOUE = cast<BinaryOperator>(UE->IgnoreImpCasts());
  // Update expressions are allowed to have the following forms:
  // x binop= expr; -> xrval + expr;
  // x++, ++x -> xrval + 1;
  // x--, --x -> xrval - 1;
  // x = x binop expr; -> xrval binop expr
  // x = expr Op x; - > expr binop xrval;
  assert(X->isLValue() && "X of 'omp atomic update' is not lvalue");
  LValue XLValue = CGF.EmitLValue(X);
  RValue ExprRValue = CGF.EmitAnyExpr(E);
  auto AO = IsSeqCst ? llvm::SequentiallyConsistent : llvm::Monotonic;
  auto *LHS = cast<OpaqueValueExpr>(BOUE->getLHS()->IgnoreImpCasts());
  auto *RHS = cast<OpaqueValueExpr>(BOUE->getRHS()->IgnoreImpCasts());
  auto *XRValExpr = IsXLHSInRHSPart ? LHS : RHS;
  auto *ERValExpr = IsXLHSInRHSPart ? RHS : LHS;
  auto Gen =
      [&CGF, UE, ExprRValue, XRValExpr, ERValExpr](RValue XRValue) -> RValue {
        CodeGenFunction::OpaqueValueMapping MapExpr(CGF, ERValExpr, ExprRValue);
        CodeGenFunction::OpaqueValueMapping MapX(CGF, XRValExpr, XRValue);
        return CGF.EmitAnyExpr(UE);
      };
  CGF.EmitOMPAtomicSimpleUpdateExpr(XLValue, ExprRValue, BOUE->getOpcode(),
                                    IsXLHSInRHSPart, AO, Loc, Gen);
  // OpenMP, 2.12.6, atomic Construct
  // Any atomic construct with a seq_cst clause forces the atomically
  // performed operation to include an implicit flush operation without a
  // list.
  if (IsSeqCst)
    CGF.CGM.getOpenMPRuntime().emitFlush(CGF, llvm::None, Loc);
}

static void EmitOMPAtomicExpr(CodeGenFunction &CGF, OpenMPClauseKind Kind,
                              bool IsSeqCst, const Expr *X, const Expr *V,
                              const Expr *E, const Expr *UE,
                              bool IsXLHSInRHSPart, SourceLocation Loc) {
  switch (Kind) {
  case OMPC_read:
    EmitOMPAtomicReadExpr(CGF, IsSeqCst, X, V, Loc);
    break;
  case OMPC_write:
    EmitOMPAtomicWriteExpr(CGF, IsSeqCst, X, E, Loc);
    break;
  case OMPC_unknown:
  case OMPC_update:
    EmitOMPAtomicUpdateExpr(CGF, IsSeqCst, X, E, UE, IsXLHSInRHSPart, Loc);
    break;
  case OMPC_capture:
    llvm_unreachable("CodeGen for 'omp atomic clause' is not supported yet.");
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_reduction:
  case OMPC_safelen:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_seq_cst:
  case OMPC_shared:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_flush:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_threadprivate:
  case OMPC_mergeable:
    llvm_unreachable("Clause is not allowed in 'omp atomic'.");
  }
}

void CodeGenFunction::EmitOMPAtomicDirective(const OMPAtomicDirective &S) {
  bool IsSeqCst = S.getSingleClause(/*K=*/OMPC_seq_cst);
  OpenMPClauseKind Kind = OMPC_unknown;
  for (auto *C : S.clauses()) {
    // Find first clause (skip seq_cst clause, if it is first).
    if (C->getClauseKind() != OMPC_seq_cst) {
      Kind = C->getClauseKind();
      break;
    }
  }

  const auto *CS =
      S.getAssociatedStmt()->IgnoreContainers(/*IgnoreCaptured=*/true);
  if (const auto *EWC = dyn_cast<ExprWithCleanups>(CS))
    enterFullExpression(EWC);

  LexicalScope Scope(*this, S.getSourceRange());
  auto &&CodeGen = [&S, Kind, IsSeqCst](CodeGenFunction &CGF) {
    EmitOMPAtomicExpr(CGF, Kind, IsSeqCst, S.getX(), S.getV(), S.getExpr(),
                      S.getUpdateExpr(), S.isXLHSInRHSPart(), S.getLocStart());
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, CodeGen);
}

void CodeGenFunction::EmitOMPTargetDirective(const OMPTargetDirective &) {
  llvm_unreachable("CodeGen for 'omp target' is not supported yet.");
}

void CodeGenFunction::EmitOMPTeamsDirective(const OMPTeamsDirective &) {
  llvm_unreachable("CodeGen for 'omp teams' is not supported yet.");
}

