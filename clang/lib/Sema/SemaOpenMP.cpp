//===--- SemaOpenMP.cpp - Semantic Analysis for OpenMP constructs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements semantic analysis for OpenMP directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/OpenMPKinds.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
using namespace clang;

namespace {

class VarDeclFilterCCC : public CorrectionCandidateCallback {
private:
  Sema &Actions;
public:
  VarDeclFilterCCC(Sema &S) : Actions(S) { }
  virtual bool ValidateCandidate(const TypoCorrection &Candidate) {
    NamedDecl *ND = Candidate.getCorrectionDecl();
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(ND)) {
      return VD->hasGlobalStorage() &&
             Actions.isDeclInScope(ND, Actions.getCurLexicalContext(),
                                   Actions.getCurScope());
    }
    return false;
  }
};
}

ExprResult Sema::ActOnOpenMPIdExpression(Scope *CurScope,
                                         CXXScopeSpec &ScopeSpec,
                                         const DeclarationNameInfo &Id) {
  LookupResult Lookup(*this, Id, LookupOrdinaryName);
  LookupParsedName(Lookup, CurScope, &ScopeSpec, true);

  if (Lookup.isAmbiguous())
    return ExprError();

  VarDecl *VD;
  if (!Lookup.isSingleResult()) {
    VarDeclFilterCCC Validator(*this);
    TypoCorrection Corrected = CorrectTypo(Id, LookupOrdinaryName, CurScope,
                                           0, Validator);
    std::string CorrectedStr = Corrected.getAsString(getLangOpts());
    std::string CorrectedQuotedStr = Corrected.getQuoted(getLangOpts());
    if (Lookup.empty()) {
      if (Corrected.isResolved()) {
        Diag(Id.getLoc(), diag::err_undeclared_var_use_suggest)
          << Id.getName() << CorrectedQuotedStr
          << FixItHint::CreateReplacement(Id.getLoc(), CorrectedStr);
      } else {
        Diag(Id.getLoc(), diag::err_undeclared_var_use)
          << Id.getName();
      }
    } else {
      Diag(Id.getLoc(), diag::err_omp_expected_var_arg_suggest)
        << Id.getName() << Corrected.isResolved() << CorrectedQuotedStr
        << FixItHint::CreateReplacement(Id.getLoc(), CorrectedStr);
    }
    if (!Corrected.isResolved()) return ExprError();
    VD = Corrected.getCorrectionDeclAs<VarDecl>();
  } else {
    if (!(VD = Lookup.getAsSingle<VarDecl>())) {
      Diag(Id.getLoc(), diag::err_omp_expected_var_arg_suggest)
        << Id.getName() << 0;
      Diag(Lookup.getFoundDecl()->getLocation(), diag::note_declared_at);
      return ExprError();
    }
  }
  Lookup.suppressDiagnostics();

  // OpenMP [2.9.2, Syntax, C/C++]
  //   Variables must be file-scope, namespace-scope, or static block-scope.
  if (!VD->hasGlobalStorage()) {
    Diag(Id.getLoc(), diag::err_omp_global_var_arg)
      << getOpenMPDirectiveName(OMPD_threadprivate)
      << !VD->isStaticLocal();
    bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                  VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here) << VD;
    return ExprError();
  }

  // OpenMP [2.9.2, Restrictions, C/C++, p.2]
  //   A threadprivate directive for file-scope variables must appear outside
  //   any definition or declaration.
  // OpenMP [2.9.2, Restrictions, C/C++, p.3]
  //   A threadprivate directive for static class member variables must appear
  //   in the class definition, in the same scope in which the member
  //   variables are declared.
  // OpenMP [2.9.2, Restrictions, C/C++, p.4]
  //   A threadprivate directive for namespace-scope variables must appear
  //   outside any definition or declaration other than the namespace
  //   definition itself.
  // OpenMP [2.9.2, Restrictions, C/C++, p.6]
  //   A threadprivate directive for static block-scope variables must appear
  //   in the scope of the variable and not in a nested scope.
  NamedDecl *ND = cast<NamedDecl>(VD);
  if (!isDeclInScope(ND, getCurLexicalContext(), CurScope)) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
      << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                  VarDecl::DeclarationOnly;
    Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                     diag::note_defined_here) << VD;
    return ExprError();
  }

  // OpenMP [2.9.2, Restrictions, C/C++, p.2-6]
  //   A threadprivate directive must lexically precede all references to any
  //   of the variables in its list.
  if (VD->isUsed()) {
    Diag(Id.getLoc(), diag::err_omp_var_used)
      << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    return ExprError();
  }

  QualType ExprType = VD->getType().getNonReferenceType();
  ExprResult DE = BuildDeclRefExpr(VD, ExprType, VK_RValue, Id.getLoc());
  return DE;
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPThreadprivateDirective(
                                SourceLocation Loc,
                                ArrayRef<Expr *> VarList) {
  if (OMPThreadPrivateDecl *D = CheckOMPThreadPrivateDecl(Loc, VarList)) {
    CurContext->addDecl(D);
    return DeclGroupPtrTy::make(DeclGroupRef(D));
  }
  return DeclGroupPtrTy();
}

OMPThreadPrivateDecl *Sema::CheckOMPThreadPrivateDecl(
                                 SourceLocation Loc,
                                 ArrayRef<Expr *> VarList) {
  SmallVector<Expr *, 8> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(),
                                         E = VarList.end();
       I != E; ++I) {
    DeclRefExpr *DE = cast<DeclRefExpr>(*I);
    VarDecl *VD = cast<VarDecl>(DE->getDecl());
    SourceLocation ILoc = DE->getExprLoc();

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have an incomplete type.
    if (RequireCompleteType(ILoc, VD->getType(),
                            diag::err_omp_threadprivate_incomplete_type)) {
      continue;
    }

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have a reference type.
    if (VD->getType()->isReferenceType()) {
      Diag(ILoc, diag::err_omp_ref_type_arg)
        << getOpenMPDirectiveName(OMPD_threadprivate)
        << VD->getType();
      bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                    VarDecl::DeclarationOnly;
      Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                       diag::note_defined_here) << VD;
      continue;
    }

    // Check if this is a TLS variable.
    if (VD->getTLSKind()) {
      Diag(ILoc, diag::err_omp_var_thread_local) << VD;
      bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                    VarDecl::DeclarationOnly;
      Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                       diag::note_defined_here) << VD;
      continue;
    }

    Vars.push_back(*I);
  }
  return Vars.empty() ?
              0 : OMPThreadPrivateDecl::Create(Context,
                                               getCurLexicalContext(),
                                               Loc, Vars);
}

StmtResult Sema::ActOnOpenMPExecutableDirective(OpenMPDirectiveKind Kind,
                                                ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  StmtResult Res = StmtError();
  switch (Kind) {
  case OMPD_parallel:
    Res = ActOnOpenMPParallelDirective(Clauses, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_threadprivate:
  case OMPD_task:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
  case NUM_OPENMP_DIRECTIVES:
    llvm_unreachable("Unknown OpenMP directive");
  }
  return Res;
}

StmtResult Sema::ActOnOpenMPParallelDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return Owned(OMPParallelDirective::Create(Context, StartLoc, EndLoc,
                                            Clauses, AStmt));
}

OMPClause *Sema::ActOnOpenMPSimpleClause(OpenMPClauseKind Kind,
                                         unsigned Argument,
                                         SourceLocation ArgumentLoc,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_default:
    Res = ActOnOpenMPDefaultClause(
                             static_cast<OpenMPDefaultClauseKind>(Argument),
                             ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_private:
  case OMPC_threadprivate:
  case OMPC_unknown:
  case NUM_OPENMP_CLAUSES:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPDefaultClause(OpenMPDefaultClauseKind Kind,
                                          SourceLocation KindKwLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  if (Kind == OMPC_DEFAULT_unknown) {
    std::string Values;
    std::string Sep(NUM_OPENMP_DEFAULT_KINDS > 1 ? ", " : "");
    for (unsigned i = OMPC_DEFAULT_unknown + 1;
         i < NUM_OPENMP_DEFAULT_KINDS; ++i) {
      Values += "'";
      Values += getOpenMPSimpleClauseTypeName(OMPC_default, i);
      Values += "'";
      switch (i) {
      case NUM_OPENMP_DEFAULT_KINDS - 2:
        Values += " or ";
        break;
      case NUM_OPENMP_DEFAULT_KINDS - 1:
        break;
      default:
        Values += Sep;
        break;
      }
    }
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
      << Values << getOpenMPClauseName(OMPC_default);
    return 0;
  }
  return new (Context) OMPDefaultClause(Kind, KindKwLoc, StartLoc, LParenLoc,
                                        EndLoc);
}

OMPClause *Sema::ActOnOpenMPVarListClause(OpenMPClauseKind Kind,
                                          ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_private:
    Res = ActOnOpenMPPrivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_default:
  case OMPC_threadprivate:
  case OMPC_unknown:
  case NUM_OPENMP_CLAUSES:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPPrivateClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    if (*I && isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name)
        << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_private_incomplete_type)) {
      continue;
    }
    if (Type->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
        << getOpenMPClauseName(OMPC_private) << Type;
      bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                    VarDecl::DeclarationOnly;
      Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                       diag::note_defined_here) << VD;
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accesible, unambiguous default constructor for the
    //  class type.
    while (Type.getNonReferenceType()->isArrayType()) {
      Type = cast<ArrayType>(
                 Type.getNonReferenceType().getTypePtr())->getElementType();
    }
    CXXRecordDecl *RD = getLangOpts().CPlusPlus ?
                          Type.getNonReferenceType()->getAsCXXRecordDecl() : 0;
    if (RD) {
      CXXConstructorDecl *CD = LookupDefaultConstructor(RD);
      PartialDiagnostic PD =
        PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
      if (!CD ||
          CheckConstructorAccess(ELoc, CD,
                                 InitializedEntity::InitializeTemporary(Type),
                                 CD->getAccess(), PD) == AR_inaccessible ||
          CD->isDeleted()) {
        Diag(ELoc, diag::err_omp_required_method)
             << getOpenMPClauseName(OMPC_private) << 0;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                         diag::note_defined_here) << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      }
      MarkFunctionReferenced(ELoc, CD);
      DiagnoseUseOfDecl(CD, ELoc);

      CXXDestructorDecl *DD = RD->getDestructor();
      if (DD) {
        if (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
            DD->isDeleted()) {
          Diag(ELoc, diag::err_omp_required_method)
               << getOpenMPClauseName(OMPC_private) << 4;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                           diag::note_defined_here) << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        }
        MarkFunctionReferenced(ELoc, DD);
        DiagnoseUseOfDecl(DD, ELoc);
      }
    }

    Vars.push_back(DE);
  }

  if (Vars.empty()) return 0;

  return OMPPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

