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
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Stack of data-sharing attributes for variables
//===----------------------------------------------------------------------===//

namespace {
/// \brief Default data sharing attributes, which can be applied to directive.
enum DefaultDataSharingAttributes {
  DSA_unspecified = 0,   /// \brief Data sharing attribute not specified.
  DSA_none = 1 << 0,     /// \brief Default data sharing attribute 'none'.
  DSA_shared = 1 << 1    /// \brief Default data sharing attribute 'shared'.
};

/// \brief Stack for tracking declarations used in OpenMP directives and
/// clauses and their data-sharing attributes.
class DSAStackTy {
public:
  struct DSAVarData {
    OpenMPDirectiveKind DKind;
    OpenMPClauseKind CKind;
    DeclRefExpr *RefExpr;
    DSAVarData() : DKind(OMPD_unknown), CKind(OMPC_unknown), RefExpr(0) { }
  };
private:
  struct DSAInfo {
    OpenMPClauseKind Attributes;
    DeclRefExpr *RefExpr;
  };
  typedef llvm::SmallDenseMap<VarDecl *, DSAInfo, 64> DeclSAMapTy;

  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    DefaultDataSharingAttributes DefaultAttr;
    OpenMPDirectiveKind Directive;
    DeclarationNameInfo DirectiveName;
    Scope *CurScope;
    SharingMapTy(OpenMPDirectiveKind DKind,
                 const DeclarationNameInfo &Name,
                 Scope *CurScope)
      : SharingMap(), DefaultAttr(DSA_unspecified), Directive(DKind),
        DirectiveName(Name), CurScope(CurScope) { }
    SharingMapTy()
      : SharingMap(), DefaultAttr(DSA_unspecified),
        Directive(OMPD_unknown), DirectiveName(),
        CurScope(0) { }
  };

  typedef SmallVector<SharingMapTy, 64> StackTy;

  /// \brief Stack of used declaration and their data-sharing attributes.
  StackTy Stack;
  Sema &Actions;

  typedef SmallVector<SharingMapTy, 8>::reverse_iterator reverse_iterator;

  DSAVarData getDSA(StackTy::reverse_iterator Iter, VarDecl *D);

  /// \brief Checks if the variable is a local for OpenMP region.
  bool isOpenMPLocal(VarDecl *D, StackTy::reverse_iterator Iter);
public:
  explicit DSAStackTy(Sema &S) : Stack(1), Actions(S) { }

  void push(OpenMPDirectiveKind DKind, const DeclarationNameInfo &DirName,
            Scope *CurScope) {
    Stack.push_back(SharingMapTy(DKind, DirName, CurScope));
  }

  void pop() {
    assert(Stack.size() > 1 && "Data-sharing attributes stack is empty!");
    Stack.pop_back();
  }

  /// \brief Adds explicit data sharing attribute to the specified declaration.
  void addDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A);

  /// \brief Returns data sharing attributes from top of the stack for the
  /// specified declaration.
  DSAVarData getTopDSA(VarDecl *D);
  /// \brief Returns data-sharing attributes for the specified declaration.
  DSAVarData getImplicitDSA(VarDecl *D);
  /// \brief Checks if the specified variables has \a CKind data-sharing
  /// attribute in \a DKind directive.
  DSAVarData hasDSA(VarDecl *D, OpenMPClauseKind CKind,
                    OpenMPDirectiveKind DKind = OMPD_unknown);


  /// \brief Returns currently analyzed directive.
  OpenMPDirectiveKind getCurrentDirective() const {
    return Stack.back().Directive;
  }

  /// \brief Set default data sharing attribute to none.
  void setDefaultDSANone() { Stack.back().DefaultAttr = DSA_none; }
  /// \brief Set default data sharing attribute to shared.
  void setDefaultDSAShared() { Stack.back().DefaultAttr = DSA_shared; }

  DefaultDataSharingAttributes getDefaultDSA() const {
    return Stack.back().DefaultAttr;
  }

  Scope *getCurScope() { return Stack.back().CurScope; }
};
} // end anonymous namespace.

DSAStackTy::DSAVarData DSAStackTy::getDSA(StackTy::reverse_iterator Iter,
                                          VarDecl *D) {
  DSAVarData DVar;
  if (Iter == Stack.rend() - 1) {
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  File-scope or namespace-scope variables referenced in called routines
    //  in the region are shared unless they appear in a threadprivate
    //  directive.
    // TODO
    if (!D->isFunctionOrMethodVarDecl())
      DVar.CKind = OMPC_shared;

    // OpenMP [2.9.1.2, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  Variables with static storage duration that are declared in called
    //  routines in the region are shared.
    if (D->hasGlobalStorage())
      DVar.CKind = OMPC_shared;

    return DVar;
  }

  DVar.DKind = Iter->Directive;
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  // Variables with automatic storage duration that are declared in a scope
  // inside the construct are private.
  if (DVar.DKind != OMPD_parallel) {
    if (isOpenMPLocal(D, Iter) && D->isLocalVarDecl() &&
        (D->getStorageClass() == SC_Auto ||
         D->getStorageClass() == SC_None)) {
      DVar.CKind = OMPC_private;
      return DVar;
    }
  }

  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  if (Iter->SharingMap.count(D)) {
    DVar.RefExpr = Iter->SharingMap[D].RefExpr;
    DVar.CKind = Iter->SharingMap[D].Attributes;
    return DVar;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, implicitly determined, p.1]
  //  In a parallel or task construct, the data-sharing attributes of these
  //  variables are determined by the default clause, if present.
  switch (Iter->DefaultAttr) {
  case DSA_shared:
    DVar.CKind = OMPC_shared;
    return DVar;
  case DSA_none:
    return DVar;
  case DSA_unspecified:
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.2]
    //  In a parallel construct, if no default clause is present, these
    //  variables are shared.
    if (DVar.DKind == OMPD_parallel) {
      DVar.CKind = OMPC_shared;
      return DVar;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.4]
    //  In a task construct, if no default clause is present, a variable that in
    //  the enclosing context is determined to be shared by all implicit tasks
    //  bound to the current team is shared.
    // TODO
    if (DVar.DKind == OMPD_task) {
      DSAVarData DVarTemp;
      for (StackTy::reverse_iterator I = std::next(Iter),
                                     EE = std::prev(Stack.rend());
           I != EE; ++I) {
        // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
        // in a Construct, implicitly determined, p.6]
        //  In a task construct, if no default clause is present, a variable
        //  whose data-sharing attribute is not determined by the rules above is
        //  firstprivate.
        DVarTemp = getDSA(I, D);
        if (DVarTemp.CKind != OMPC_shared) {
          DVar.RefExpr = 0;
          DVar.DKind = OMPD_task;
          DVar.CKind = OMPC_firstprivate;
          return DVar;
        }
        if (I->Directive == OMPD_parallel) break;
      }
      DVar.DKind = OMPD_task;
      DVar.CKind =
        (DVarTemp.CKind == OMPC_unknown) ? OMPC_firstprivate : OMPC_shared;
      return DVar;
    }
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, implicitly determined, p.3]
  //  For constructs other than task, if no default clause is present, these
  //  variables inherit their data-sharing attributes from the enclosing
  //  context.
  return getDSA(std::next(Iter), D);
}

void DSAStackTy::addDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A) {
  if (A == OMPC_threadprivate) {
    Stack[0].SharingMap[D].Attributes = A;
    Stack[0].SharingMap[D].RefExpr = E;
  } else {
    assert(Stack.size() > 1 && "Data-sharing attributes stack is empty");
    Stack.back().SharingMap[D].Attributes = A;
    Stack.back().SharingMap[D].RefExpr = E;
  }
}

bool DSAStackTy::isOpenMPLocal(VarDecl *D, StackTy::reverse_iterator Iter) {
  if (Stack.size() > 2) {
    reverse_iterator I = Iter, E = Stack.rend() - 1;
    Scope *TopScope = 0;
    while (I != E &&
           I->Directive != OMPD_parallel) {
      ++I;
    }
    if (I == E) return false;
    TopScope = I->CurScope ? I->CurScope->getParent() : 0;
    Scope *CurScope = getCurScope();
    while (CurScope != TopScope && !CurScope->isDeclScope(D)) {
      CurScope = CurScope->getParent();
    }
    return CurScope != TopScope;
  }
  return false;
}

DSAStackTy::DSAVarData DSAStackTy::getTopDSA(VarDecl *D) {
  DSAVarData DVar;

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  //  Variables appearing in threadprivate directives are threadprivate.
  if (D->getTLSKind() != VarDecl::TLS_None) {
    DVar.CKind = OMPC_threadprivate;
    return DVar;
  }
  if (Stack[0].SharingMap.count(D)) {
    DVar.RefExpr = Stack[0].SharingMap[D].RefExpr;
    DVar.CKind = OMPC_threadprivate;
    return DVar;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  // Variables with automatic storage duration that are declared in a scope
  // inside the construct are private.
  OpenMPDirectiveKind Kind = getCurrentDirective();
  if (Kind != OMPD_parallel) {
    if (isOpenMPLocal(D, std::next(Stack.rbegin())) && D->isLocalVarDecl() &&
        (D->getStorageClass() == SC_Auto ||
         D->getStorageClass() == SC_None))
      DVar.CKind = OMPC_private;
      return DVar;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.4]
  //  Static data memebers are shared.
  if (D->isStaticDataMember()) {
    // Variables with const-qualified type having no mutable member may be listed
    // in a firstprivate clause, even if they are static data members.
    DSAVarData DVarTemp = hasDSA(D, OMPC_firstprivate);
    if (DVarTemp.CKind == OMPC_firstprivate && DVarTemp.RefExpr)
      return DVar;

    DVar.CKind = OMPC_shared;
    return DVar;
  }

  QualType Type = D->getType().getNonReferenceType().getCanonicalType();
  bool IsConstant = Type.isConstant(Actions.getASTContext());
  while (Type->isArrayType()) {
    QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
    Type = ElemType.getNonReferenceType().getCanonicalType();
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.6]
  //  Variables with const qualified type having no mutable member are
  //  shared.
  CXXRecordDecl *RD = Actions.getLangOpts().CPlusPlus ?
                                Type->getAsCXXRecordDecl() : 0;
  if (IsConstant &&
      !(Actions.getLangOpts().CPlusPlus && RD && RD->hasMutableFields())) {
    // Variables with const-qualified type having no mutable member may be
    // listed in a firstprivate clause, even if they are static data members.
    DSAVarData DVarTemp = hasDSA(D, OMPC_firstprivate);
    if (DVarTemp.CKind == OMPC_firstprivate && DVarTemp.RefExpr)
      return DVar;

    DVar.CKind = OMPC_shared;
    return DVar;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.7]
  //  Variables with static storage duration that are declared in a scope
  //  inside the construct are shared.
  if (D->isStaticLocal()) {
    DVar.CKind = OMPC_shared;
    return DVar;
  }

  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  if (Stack.back().SharingMap.count(D)) {
    DVar.RefExpr = Stack.back().SharingMap[D].RefExpr;
    DVar.CKind = Stack.back().SharingMap[D].Attributes;
  }

  return DVar;
}

DSAStackTy::DSAVarData DSAStackTy::getImplicitDSA(VarDecl *D) {
  return getDSA(std::next(Stack.rbegin()), D);
}

DSAStackTy::DSAVarData DSAStackTy::hasDSA(VarDecl *D, OpenMPClauseKind CKind,
                                          OpenMPDirectiveKind DKind) {
  for (StackTy::reverse_iterator I = std::next(Stack.rbegin()),
                                 E = std::prev(Stack.rend());
       I != E; ++I) {
    if (DKind != OMPD_unknown && DKind != I->Directive) continue;
    DSAVarData DVar = getDSA(I, D);
    if (DVar.CKind == CKind)
      return DVar;
  }
  return DSAVarData();
}

void Sema::InitDataSharingAttributesStack() {
  VarDataSharingAttributesStack = new DSAStackTy(*this);
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStack)

void Sema::DestroyDataSharingAttributesStack() {
  delete DSAStack;
}

void Sema::StartOpenMPDSABlock(OpenMPDirectiveKind DKind,
                               const DeclarationNameInfo &DirName,
                               Scope *CurScope) {
  DSAStack->push(DKind, DirName, CurScope);
  PushExpressionEvaluationContext(PotentiallyEvaluated);
}

void Sema::EndOpenMPDSABlock(Stmt *CurDirective) {
  DSAStack->pop();
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

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
    if (TypoCorrection Corrected = CorrectTypo(Id, LookupOrdinaryName, CurScope,
                                               0, Validator)) {
      diagnoseTypo(Corrected,
                   PDiag(Lookup.empty()? diag::err_undeclared_var_use_suggest
                                       : diag::err_omp_expected_var_arg_suggest)
                     << Id.getName());
      VD = Corrected.getCorrectionDeclAs<VarDecl>();
    } else {
      Diag(Id.getLoc(), Lookup.empty() ? diag::err_undeclared_var_use
                                       : diag::err_omp_expected_var_arg)
          << Id.getName();
      return ExprError();
    }
  } else {
    if (!(VD = Lookup.getAsSingle<VarDecl>())) {
      Diag(Id.getLoc(), diag::err_omp_expected_var_arg)
        << Id.getName();
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

  VarDecl *CanonicalVD = VD->getCanonicalDecl();
  NamedDecl *ND = cast<NamedDecl>(CanonicalVD);
  // OpenMP [2.9.2, Restrictions, C/C++, p.2]
  //   A threadprivate directive for file-scope variables must appear outside
  //   any definition or declaration.
  if (CanonicalVD->getDeclContext()->isTranslationUnit() &&
      !getCurLexicalContext()->isTranslationUnit()) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
      << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                  VarDecl::DeclarationOnly;
    Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                     diag::note_defined_here) << VD;
    return ExprError();
  }
  // OpenMP [2.9.2, Restrictions, C/C++, p.3]
  //   A threadprivate directive for static class member variables must appear
  //   in the class definition, in the same scope in which the member
  //   variables are declared.
  if (CanonicalVD->isStaticDataMember() &&
      !CanonicalVD->getDeclContext()->Equals(getCurLexicalContext())) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
      << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                  VarDecl::DeclarationOnly;
    Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                     diag::note_defined_here) << VD;
    return ExprError();
  }
  // OpenMP [2.9.2, Restrictions, C/C++, p.4]
  //   A threadprivate directive for namespace-scope variables must appear
  //   outside any definition or declaration other than the namespace
  //   definition itself.
  if (CanonicalVD->getDeclContext()->isNamespace() &&
      (!getCurLexicalContext()->isFileContext() ||
       !getCurLexicalContext()->Encloses(CanonicalVD->getDeclContext()))) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
      << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                  VarDecl::DeclarationOnly;
    Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                     diag::note_defined_here) << VD;
    return ExprError();
  }
  // OpenMP [2.9.2, Restrictions, C/C++, p.6]
  //   A threadprivate directive for static block-scope variables must appear
  //   in the scope of the variable and not in a nested scope.
  if (CanonicalVD->isStaticLocal() && CurScope &&
      !isDeclInScope(ND, getCurLexicalContext(), CurScope)) {
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
  DSAStack->addDSA(VD, cast<DeclRefExpr>(DE.get()), OMPC_threadprivate);
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
  OMPThreadPrivateDecl *D = 0;
  if (!Vars.empty()) {
    D = OMPThreadPrivateDecl::Create(Context, getCurLexicalContext(), Loc,
                                     Vars);
    D->setAccess(AS_public);
  }
  return D;
}

namespace {
class DSAAttrChecker : public StmtVisitor<DSAAttrChecker, void> {
  DSAStackTy *Stack;
  Sema &Actions;
  bool ErrorFound;
  CapturedStmt *CS;
  llvm::SmallVector<Expr *, 8> ImplicitFirstprivate;
public:
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if(VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      // Skip internally declared variables.
      if (VD->isLocalVarDecl() && !CS->capturesVariable(VD)) return;

      SourceLocation ELoc = E->getExprLoc();

      OpenMPDirectiveKind DKind = Stack->getCurrentDirective();
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(VD);
      if (DVar.CKind != OMPC_unknown) {
        if (DKind == OMPD_task && DVar.CKind != OMPC_shared &&
            DVar.CKind != OMPC_threadprivate && !DVar.RefExpr)
          ImplicitFirstprivate.push_back(DVar.RefExpr);
        return;
      }
      // The default(none) clause requires that each variable that is referenced
      // in the construct, and does not have a predetermined data-sharing
      // attribute, must have its data-sharing attribute explicitly determined
      // by being listed in a data-sharing attribute clause.
      if (DVar.CKind == OMPC_unknown && Stack->getDefaultDSA() == DSA_none &&
          (DKind == OMPD_parallel || DKind == OMPD_task)) {
        ErrorFound = true;
        Actions.Diag(ELoc, diag::err_omp_no_dsa_for_variable) << VD;
        return;
      }

      // OpenMP [2.9.3.6, Restrictions, p.2]
      //  A list item that appears in a reduction clause of the innermost
      //  enclosing worksharing or parallel construct may not be accessed in an
      //  explicit task.
      // TODO:

      // Define implicit data-sharing attributes for task.
      DVar = Stack->getImplicitDSA(VD);
      if (DKind == OMPD_task && DVar.CKind != OMPC_shared)
        ImplicitFirstprivate.push_back(DVar.RefExpr);
    }
  }
  void VisitOMPExecutableDirective(OMPExecutableDirective *S) {
    for (ArrayRef<OMPClause *>::iterator I = S->clauses().begin(),
                                         E = S->clauses().end();
         I != E; ++I)
      if (OMPClause *C = *I)
        for (StmtRange R = C->children(); R; ++R)
          if (Stmt *Child = *R)
            Visit(Child);
  }
  void VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end();
         I != E; ++I)
      if (Stmt *Child = *I)
        if (!isa<OMPExecutableDirective>(Child))
          Visit(Child);
    }

  bool isErrorFound() { return ErrorFound; }
  ArrayRef<Expr *> getImplicitFirstprivate() { return ImplicitFirstprivate; }

  DSAAttrChecker(DSAStackTy *S, Sema &Actions, CapturedStmt *CS)
    : Stack(S), Actions(Actions), ErrorFound(false), CS(CS) { }
};
}

StmtResult Sema::ActOnOpenMPExecutableDirective(OpenMPDirectiveKind Kind,
                                                ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");

  StmtResult Res = StmtError();

  // Check default data sharing attributes for referenced variables.
  DSAAttrChecker DSAChecker(DSAStack, *this, cast<CapturedStmt>(AStmt));
  DSAChecker.Visit(cast<CapturedStmt>(AStmt)->getCapturedStmt());
  if (DSAChecker.isErrorFound())
    return StmtError();
  // Generate list of implicitly defined firstprivate variables.
  llvm::SmallVector<OMPClause *, 8> ClausesWithImplicit;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());

  bool ErrorFound = false;
  if (!DSAChecker.getImplicitFirstprivate().empty()) {
    if (OMPClause *Implicit =
         ActOnOpenMPFirstprivateClause(DSAChecker.getImplicitFirstprivate(),
                                       SourceLocation(), SourceLocation(),
                                       SourceLocation())) {
      ClausesWithImplicit.push_back(Implicit);
      ErrorFound = cast<OMPFirstprivateClause>(Implicit)->varlist_size() !=
                                    DSAChecker.getImplicitFirstprivate().size();
    } else
      ErrorFound = true;
  }

  switch (Kind) {
  case OMPD_parallel:
    Res = ActOnOpenMPParallelDirective(ClausesWithImplicit, AStmt,
                                       StartLoc, EndLoc);
    break;
  case OMPD_simd:
    Res = ActOnOpenMPSimdDirective(ClausesWithImplicit, AStmt,
                                   StartLoc, EndLoc);
    break;
  case OMPD_threadprivate:
  case OMPD_task:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
  case NUM_OPENMP_DIRECTIVES:
    llvm_unreachable("Unknown OpenMP directive");
  }

  if (ErrorFound) return StmtError();
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

StmtResult Sema::ActOnOpenMPSimdDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  Stmt *CStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(CStmt))
    CStmt = CS->getCapturedStmt();
  while (AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(CStmt))
    CStmt = AS->getSubStmt();
  ForStmt *For = dyn_cast<ForStmt>(CStmt);
  if (!For) {
    Diag(CStmt->getLocStart(), diag::err_omp_not_for)
      << getOpenMPDirectiveName(OMPD_simd);
    return StmtError();
  }

  // FIXME: Checking loop canonical form, collapsing etc.

  getCurFunction()->setHasBranchProtectedScope();
  return Owned(OMPSimdDirective::Create(Context, StartLoc, EndLoc,
                                        Clauses, AStmt));
}

OMPClause *Sema::ActOnOpenMPSingleExprClause(OpenMPClauseKind Kind,
                                             Expr *Expr,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_if:
    Res = ActOnOpenMPIfClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_default:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_shared:
  case OMPC_threadprivate:
  case OMPC_unknown:
  case NUM_OPENMP_CLAUSES:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPIfClause(Expr *Condition,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = ActOnBooleanCondition(DSAStack->getCurScope(),
                                           Condition->getExprLoc(),
                                           Condition);
    if (Val.isInvalid())
      return 0;

    ValExpr = Val.take();
  }

  return new (Context) OMPIfClause(ValExpr, StartLoc, LParenLoc, EndLoc);
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
    Res =
      ActOnOpenMPDefaultClause(static_cast<OpenMPDefaultClauseKind>(Argument),
                               ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_shared:
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
  switch (Kind) {
  case OMPC_DEFAULT_none:
    DSAStack->setDefaultDSANone();
    break;
  case OMPC_DEFAULT_shared:
    DSAStack->setDefaultDSAShared();
    break;
  case OMPC_DEFAULT_unknown:
  case NUM_OPENMP_DEFAULT_KINDS:
    llvm_unreachable("Clause kind is not allowed.");
    break;
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
  case OMPC_firstprivate:
    Res = ActOnOpenMPFirstprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_shared:
    Res = ActOnOpenMPSharedClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_if:
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
    assert(*I && "NULL expr in OpenMP private clause.");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
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

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_private) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
         << getOpenMPClauseName(DVar.CKind)
         << getOpenMPClauseName(OMPC_private);
      if (DVar.RefExpr) {
        Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_explicit_dsa)
             << getOpenMPClauseName(DVar.CKind);
      } else {
        Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
             << getOpenMPClauseName(DVar.CKind);
      }
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_private);
    Vars.push_back(DE);
  }

  if (Vars.empty()) return 0;

  return OMPPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPFirstprivateClause(ArrayRef<Expr *> VarList,
                                               SourceLocation StartLoc,
                                               SourceLocation LParenLoc,
                                               SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "NULL expr in OpenMP firstprivate clause.");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
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
                            diag::err_omp_firstprivate_incomplete_type)) {
      continue;
    }
    if (Type->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
        << getOpenMPClauseName(OMPC_firstprivate) << Type;
      bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                    VarDecl::DeclarationOnly;
      Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl :
                                       diag::note_defined_here) << VD;
      continue;
    }

    // OpenMP [2.9.3.4, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accesible, unambiguous copy constructor for the
    //  class type.
    Type = Context.getBaseElementType(Type);
    CXXRecordDecl *RD = getLangOpts().CPlusPlus ?
                          Type.getNonReferenceType()->getAsCXXRecordDecl() : 0;
    if (RD) {
      CXXConstructorDecl *CD = LookupCopyingConstructor(RD, 0);
      PartialDiagnostic PD =
        PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
      if (!CD ||
          CheckConstructorAccess(ELoc, CD,
                                 InitializedEntity::InitializeTemporary(Type),
                                 CD->getAccess(), PD) == AR_inaccessible ||
          CD->isDeleted()) {
        Diag(ELoc, diag::err_omp_required_method)
             << getOpenMPClauseName(OMPC_firstprivate) << 1;
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
               << getOpenMPClauseName(OMPC_firstprivate) << 4;
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

    // If StartLoc and EndLoc are invalid - this is an implicit firstprivate
    // variable and it was checked already.
    if (StartLoc.isValid() && EndLoc.isValid()) {
      DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD);
      Type = Type.getNonReferenceType().getCanonicalType();
      bool IsConstant = Type.isConstant(Context);
      Type = Context.getBaseElementType(Type);
      // OpenMP [2.4.13, Data-sharing Attribute Clauses]
      //  A list item that specifies a given variable may not appear in more
      // than one clause on the same directive, except that a variable may be
      //  specified in both firstprivate and lastprivate clauses.
      //  TODO: add processing for lastprivate.
      if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_firstprivate &&
          DVar.RefExpr) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
           << getOpenMPClauseName(DVar.CKind)
           << getOpenMPClauseName(OMPC_firstprivate);
        Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_explicit_dsa)
           << getOpenMPClauseName(DVar.CKind);
        continue;
      }

      // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
      // in a Construct]
      //  Variables with the predetermined data-sharing attributes may not be
      //  listed in data-sharing attributes clauses, except for the cases
      //  listed below. For these exceptions only, listing a predetermined
      //  variable in a data-sharing attribute clause is allowed and overrides
      //  the variable's predetermined data-sharing attributes.
      // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
      // in a Construct, C/C++, p.2]
      //  Variables with const-qualified type having no mutable member may be
      //  listed in a firstprivate clause, even if they are static data members.
      if (!(IsConstant || VD->isStaticDataMember()) && !DVar.RefExpr &&
          DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
           << getOpenMPClauseName(DVar.CKind)
           << getOpenMPClauseName(OMPC_firstprivate);
        Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
           << getOpenMPClauseName(DVar.CKind);
        continue;
      }

      // OpenMP [2.9.3.4, Restrictions, p.2]
      //  A list item that is private within a parallel region must not appear
      //  in a firstprivate clause on a worksharing construct if any of the
      //  worksharing regions arising from the worksharing construct ever bind
      //  to any of the parallel regions arising from the parallel construct.
      // OpenMP [2.9.3.4, Restrictions, p.3]
      //  A list item that appears in a reduction clause of a parallel construct
      //  must not appear in a firstprivate clause on a worksharing or task
      //  construct if any of the worksharing or task regions arising from the
      //  worksharing or task construct ever bind to any of the parallel regions
      //  arising from the parallel construct.
      // OpenMP [2.9.3.4, Restrictions, p.4]
      //  A list item that appears in a reduction clause in worksharing
      //  construct must not appear in a firstprivate clause in a task construct
      //  encountered during execution of any of the worksharing regions arising
      //  from the worksharing construct.
      // TODO:
    }

    DSAStack->addDSA(VD, DE, OMPC_firstprivate);
    Vars.push_back(DE);
  }

  if (Vars.empty()) return 0;

  return OMPFirstprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                       Vars);
}

OMPClause *Sema::ActOnOpenMPSharedClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "NULL expr in OpenMP shared clause.");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.4, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(*I);
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

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_shared && DVar.RefExpr) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
         << getOpenMPClauseName(DVar.CKind)
         << getOpenMPClauseName(OMPC_shared);
      Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_explicit_dsa)
           << getOpenMPClauseName(DVar.CKind);
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_shared);
    Vars.push_back(DE);
  }

  if (Vars.empty()) return 0;

  return OMPSharedClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

#undef DSAStack
