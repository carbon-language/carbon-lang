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
#include "clang/AST/DeclOpenMP.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
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
        << getOpenMPDirectiveName(OMPD_threadprivate);
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

