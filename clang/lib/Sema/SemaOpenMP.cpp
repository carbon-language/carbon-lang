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
/// clauses
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
Sema::DeclGroupPtrTy Sema::ActOnOpenMPThreadprivateDirective(
                              SourceLocation Loc,
                              Scope *CurScope,
                              ArrayRef<DeclarationNameInfo> IdList) {
  SmallVector<DeclRefExpr *, 5> Vars;
  for (ArrayRef<DeclarationNameInfo>::iterator I = IdList.begin(),
                                               E = IdList.end();
       I != E; ++I) {
    LookupResult Lookup(*this, *I, LookupOrdinaryName);
    LookupParsedName(Lookup, CurScope, NULL, true);

    if (Lookup.isAmbiguous())
      continue;

    VarDecl *VD;
    if (!Lookup.isSingleResult()) {
      VarDeclFilterCCC Validator(*this);
      TypoCorrection Corrected = CorrectTypo(*I, LookupOrdinaryName, CurScope,
                                             0, Validator);
      std::string CorrectedStr = Corrected.getAsString(getLangOpts());
      std::string CorrectedQuotedStr = Corrected.getQuoted(getLangOpts());
      if (Lookup.empty()) {
        if (Corrected.isResolved()) {
          Diag(I->getLoc(), diag::err_undeclared_var_use_suggest)
            << I->getName() << CorrectedQuotedStr
            << FixItHint::CreateReplacement(I->getLoc(), CorrectedStr);
        } else {
          Diag(I->getLoc(), diag::err_undeclared_var_use)
            << I->getName();
        }
      } else {
        Diag(I->getLoc(), diag::err_omp_expected_var_arg_suggest)
          << I->getName() << Corrected.isResolved() << CorrectedQuotedStr
          << FixItHint::CreateReplacement(I->getLoc(), CorrectedStr);
      }
      if (!Corrected.isResolved()) continue;
      VD = Corrected.getCorrectionDeclAs<VarDecl>();
    } else {
      if (!(VD = Lookup.getAsSingle<VarDecl>())) {
        Diag(I->getLoc(), diag::err_omp_expected_var_arg_suggest)
          << I->getName() << 0;
        Diag(Lookup.getFoundDecl()->getLocation(), diag::note_declared_at);
        continue;
      }
    }

    // OpenMP [2.9.2, Syntax, C/C++]
    //   Variables must be file-scope, namespace-scope, or static block-scope.
    if (!VD->hasGlobalStorage()) {
      Diag(I->getLoc(), diag::err_omp_global_var_arg)
        << getOpenMPDirectiveName(OMPD_threadprivate)
        << !VD->isStaticLocal();
      Diag(VD->getLocation(), diag::note_forward_declaration) << VD;
      continue;
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
      Diag(I->getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
      Diag(VD->getLocation(), diag::note_forward_declaration) << VD;
      continue;
    }

    // OpenMP [2.9.2, Restrictions, C/C++, p.2-6]
    //   A threadprivate directive must lexically precede all references to any
    //   of the variables in its list.
    if (VD->isUsed()) {
      Diag(I->getLoc(), diag::err_omp_var_used)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
      continue;
    }

    QualType ExprType = VD->getType().getNonReferenceType();
    DeclRefExpr *Var = cast<DeclRefExpr>(BuildDeclRefExpr(VD,
                                                          ExprType,
                                                          VK_RValue,
                                                          I->getLoc()).take());
    Vars.push_back(Var);
  }
  if (OMPThreadPrivateDecl *D = CheckOMPThreadPrivateDecl(Loc, Vars)) {
    CurContext->addDecl(D);
    return DeclGroupPtrTy::make(DeclGroupRef(D));
  }
  return DeclGroupPtrTy();
}

OMPThreadPrivateDecl *Sema::CheckOMPThreadPrivateDecl(
                                 SourceLocation Loc,
                                 ArrayRef<DeclRefExpr *> VarList) {
  SmallVector<DeclRefExpr *, 5> Vars;
  for (ArrayRef<DeclRefExpr *>::iterator I = VarList.begin(),
                                         E = VarList.end();
       I != E; ++I) {
    VarDecl *VD = cast<VarDecl>((*I)->getDecl());
    SourceLocation ILoc = (*I)->getLocation();

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have an incomplete type.
    if (RequireCompleteType(ILoc, VD->getType(),
                            diag::err_omp_incomplete_type)) {
      continue;
    }

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have a reference type.
    if (VD->getType()->isReferenceType()) {
      Diag(ILoc, diag::err_omp_ref_type_arg)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD->getType();
      Diag(VD->getLocation(), diag::note_forward_declaration) << VD;
      continue;
    }

    // Check if this is a TLS variable.
    if (VD->getTLSKind()) {
      Diag(ILoc, diag::err_omp_var_thread_local) << VD;
      Diag(VD->getLocation(), diag::note_forward_declaration) << VD;
      continue;
    }

    Vars.push_back(*I);
  }
  return Vars.empty() ?
              0 : OMPThreadPrivateDecl::Create(Context,
                                               getCurLexicalContext(),
                                               Loc, Vars);
}
