//===--- SemaOpenMP.cpp - Semantic Analysis for OpenMP constructs ---------===//
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

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/OpenMPKinds.h"
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
  DSA_unspecified = 0, /// \brief Data sharing attribute not specified.
  DSA_none = 1 << 0,   /// \brief Default data sharing attribute 'none'.
  DSA_shared = 1 << 1  /// \brief Default data sharing attribute 'shared'.
};

template <class T> struct MatchesAny {
  explicit MatchesAny(ArrayRef<T> Arr) : Arr(std::move(Arr)) {}
  bool operator()(T Kind) {
    for (auto KindEl : Arr)
      if (KindEl == Kind)
        return true;
    return false;
  }

private:
  ArrayRef<T> Arr;
};
struct MatchesAlways {
  MatchesAlways() {}
  template <class T> bool operator()(T) { return true; }
};

typedef MatchesAny<OpenMPClauseKind> MatchesAnyClause;
typedef MatchesAny<OpenMPDirectiveKind> MatchesAnyDirective;

/// \brief Stack for tracking declarations used in OpenMP directives and
/// clauses and their data-sharing attributes.
class DSAStackTy {
public:
  struct DSAVarData {
    OpenMPDirectiveKind DKind;
    OpenMPClauseKind CKind;
    DeclRefExpr *RefExpr;
    SourceLocation ImplicitDSALoc;
    DSAVarData()
        : DKind(OMPD_unknown), CKind(OMPC_unknown), RefExpr(nullptr),
          ImplicitDSALoc() {}
  };

private:
  struct DSAInfo {
    OpenMPClauseKind Attributes;
    DeclRefExpr *RefExpr;
  };
  typedef llvm::SmallDenseMap<VarDecl *, DSAInfo, 64> DeclSAMapTy;
  typedef llvm::SmallDenseMap<VarDecl *, DeclRefExpr *, 64> AlignedMapTy;

  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    AlignedMapTy AlignedMap;
    DefaultDataSharingAttributes DefaultAttr;
    SourceLocation DefaultAttrLoc;
    OpenMPDirectiveKind Directive;
    DeclarationNameInfo DirectiveName;
    Scope *CurScope;
    SourceLocation ConstructLoc;
    SharingMapTy(OpenMPDirectiveKind DKind, DeclarationNameInfo Name,
                 Scope *CurScope, SourceLocation Loc)
        : SharingMap(), AlignedMap(), DefaultAttr(DSA_unspecified),
          Directive(DKind), DirectiveName(std::move(Name)), CurScope(CurScope),
          ConstructLoc(Loc) {}
    SharingMapTy()
        : SharingMap(), AlignedMap(), DefaultAttr(DSA_unspecified),
          Directive(OMPD_unknown), DirectiveName(), CurScope(nullptr),
          ConstructLoc() {}
  };

  typedef SmallVector<SharingMapTy, 64> StackTy;

  /// \brief Stack of used declaration and their data-sharing attributes.
  StackTy Stack;
  Sema &SemaRef;

  typedef SmallVector<SharingMapTy, 8>::reverse_iterator reverse_iterator;

  DSAVarData getDSA(StackTy::reverse_iterator Iter, VarDecl *D);

  /// \brief Checks if the variable is a local for OpenMP region.
  bool isOpenMPLocal(VarDecl *D, StackTy::reverse_iterator Iter);

public:
  explicit DSAStackTy(Sema &S) : Stack(1), SemaRef(S) {}

  void push(OpenMPDirectiveKind DKind, const DeclarationNameInfo &DirName,
            Scope *CurScope, SourceLocation Loc) {
    Stack.push_back(SharingMapTy(DKind, DirName, CurScope, Loc));
    Stack.back().DefaultAttrLoc = Loc;
  }

  void pop() {
    assert(Stack.size() > 1 && "Data-sharing attributes stack is empty!");
    Stack.pop_back();
  }

  /// \brief If 'aligned' declaration for given variable \a D was not seen yet,
  /// add it and return NULL; otherwise return previous occurrence's expression
  /// for diagnostics.
  DeclRefExpr *addUniqueAligned(VarDecl *D, DeclRefExpr *NewDE);

  /// \brief Adds explicit data sharing attribute to the specified declaration.
  void addDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A);

  /// \brief Returns data sharing attributes from top of the stack for the
  /// specified declaration.
  DSAVarData getTopDSA(VarDecl *D);
  /// \brief Returns data-sharing attributes for the specified declaration.
  DSAVarData getImplicitDSA(VarDecl *D);
  /// \brief Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any directive which matches \a DPred
  /// predicate.
  template <class ClausesPredicate, class DirectivesPredicate>
  DSAVarData hasDSA(VarDecl *D, ClausesPredicate CPred,
                    DirectivesPredicate DPred);
  /// \brief Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any innermost directive which
  /// matches \a DPred predicate.
  template <class ClausesPredicate, class DirectivesPredicate>
  DSAVarData hasInnermostDSA(VarDecl *D, ClausesPredicate CPred,
                             DirectivesPredicate DPred);

  /// \brief Returns currently analyzed directive.
  OpenMPDirectiveKind getCurrentDirective() const {
    return Stack.back().Directive;
  }
  /// \brief Returns parent directive.
  OpenMPDirectiveKind getParentDirective() const {
    if (Stack.size() > 2)
      return Stack[Stack.size() - 2].Directive;
    return OMPD_unknown;
  }

  /// \brief Set default data sharing attribute to none.
  void setDefaultDSANone(SourceLocation Loc) {
    Stack.back().DefaultAttr = DSA_none;
    Stack.back().DefaultAttrLoc = Loc;
  }
  /// \brief Set default data sharing attribute to shared.
  void setDefaultDSAShared(SourceLocation Loc) {
    Stack.back().DefaultAttr = DSA_shared;
    Stack.back().DefaultAttrLoc = Loc;
  }

  DefaultDataSharingAttributes getDefaultDSA() const {
    return Stack.back().DefaultAttr;
  }
  SourceLocation getDefaultDSALocation() const {
    return Stack.back().DefaultAttrLoc;
  }

  /// \brief Checks if the specified variable is a threadprivate.
  bool isThreadPrivate(VarDecl *D) {
    DSAVarData DVar = getTopDSA(D);
    return isOpenMPThreadPrivate(DVar.CKind);
  }

  Scope *getCurScope() const { return Stack.back().CurScope; }
  Scope *getCurScope() { return Stack.back().CurScope; }
  SourceLocation getConstructLoc() { return Stack.back().ConstructLoc; }
};
} // namespace

DSAStackTy::DSAVarData DSAStackTy::getDSA(StackTy::reverse_iterator Iter,
                                          VarDecl *D) {
  DSAVarData DVar;
  if (Iter == std::prev(Stack.rend())) {
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  File-scope or namespace-scope variables referenced in called routines
    //  in the region are shared unless they appear in a threadprivate
    //  directive.
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
  if (isOpenMPLocal(D, Iter) && D->isLocalVarDecl() &&
      (D->getStorageClass() == SC_Auto || D->getStorageClass() == SC_None)) {
    DVar.CKind = OMPC_private;
    return DVar;
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
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    return DVar;
  case DSA_none:
    return DVar;
  case DSA_unspecified:
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.2]
    //  In a parallel construct, if no default clause is present, these
    //  variables are shared.
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    if (isOpenMPParallelDirective(DVar.DKind)) {
      DVar.CKind = OMPC_shared;
      return DVar;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.4]
    //  In a task construct, if no default clause is present, a variable that in
    //  the enclosing context is determined to be shared by all implicit tasks
    //  bound to the current team is shared.
    if (DVar.DKind == OMPD_task) {
      DSAVarData DVarTemp;
      for (StackTy::reverse_iterator I = std::next(Iter),
                                     EE = std::prev(Stack.rend());
           I != EE; ++I) {
        // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables
        // Referenced
        // in a Construct, implicitly determined, p.6]
        //  In a task construct, if no default clause is present, a variable
        //  whose data-sharing attribute is not determined by the rules above is
        //  firstprivate.
        DVarTemp = getDSA(I, D);
        if (DVarTemp.CKind != OMPC_shared) {
          DVar.RefExpr = nullptr;
          DVar.DKind = OMPD_task;
          DVar.CKind = OMPC_firstprivate;
          return DVar;
        }
        if (isOpenMPParallelDirective(I->Directive))
          break;
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

DeclRefExpr *DSAStackTy::addUniqueAligned(VarDecl *D, DeclRefExpr *NewDE) {
  assert(Stack.size() > 1 && "Data sharing attributes stack is empty");
  auto It = Stack.back().AlignedMap.find(D);
  if (It == Stack.back().AlignedMap.end()) {
    assert(NewDE && "Unexpected nullptr expr to be added into aligned map");
    Stack.back().AlignedMap[D] = NewDE;
    return nullptr;
  } else {
    assert(It->second && "Unexpected nullptr expr in the aligned map");
    return It->second;
  }
  return nullptr;
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
    reverse_iterator I = Iter, E = std::prev(Stack.rend());
    Scope *TopScope = nullptr;
    while (I != E && !isOpenMPParallelDirective(I->Directive)) {
      ++I;
    }
    if (I == E)
      return false;
    TopScope = I->CurScope ? I->CurScope->getParent() : nullptr;
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
  if (!isOpenMPParallelDirective(Kind)) {
    if (isOpenMPLocal(D, std::next(Stack.rbegin())) && D->isLocalVarDecl() &&
        (D->getStorageClass() == SC_Auto || D->getStorageClass() == SC_None)) {
      DVar.CKind = OMPC_private;
      return DVar;
    }
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.4]
  //  Static data members are shared.
  if (D->isStaticDataMember()) {
    // Variables with const-qualified type having no mutable member may be
    // listed in a firstprivate clause, even if they are static data members.
    DSAVarData DVarTemp =
        hasDSA(D, MatchesAnyClause(OMPC_firstprivate), MatchesAlways());
    if (DVarTemp.CKind == OMPC_firstprivate && DVarTemp.RefExpr)
      return DVar;

    DVar.CKind = OMPC_shared;
    return DVar;
  }

  QualType Type = D->getType().getNonReferenceType().getCanonicalType();
  bool IsConstant = Type.isConstant(SemaRef.getASTContext());
  while (Type->isArrayType()) {
    QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
    Type = ElemType.getNonReferenceType().getCanonicalType();
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.6]
  //  Variables with const qualified type having no mutable member are
  //  shared.
  CXXRecordDecl *RD =
      SemaRef.getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : nullptr;
  if (IsConstant &&
      !(SemaRef.getLangOpts().CPlusPlus && RD && RD->hasMutableFields())) {
    // Variables with const-qualified type having no mutable member may be
    // listed in a firstprivate clause, even if they are static data members.
    DSAVarData DVarTemp =
        hasDSA(D, MatchesAnyClause(OMPC_firstprivate), MatchesAlways());
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

template <class ClausesPredicate, class DirectivesPredicate>
DSAStackTy::DSAVarData DSAStackTy::hasDSA(VarDecl *D, ClausesPredicate CPred,
                                          DirectivesPredicate DPred) {
  for (StackTy::reverse_iterator I = std::next(Stack.rbegin()),
                                 E = std::prev(Stack.rend());
       I != E; ++I) {
    if (!DPred(I->Directive))
      continue;
    DSAVarData DVar = getDSA(I, D);
    if (CPred(DVar.CKind))
      return DVar;
  }
  return DSAVarData();
}

template <class ClausesPredicate, class DirectivesPredicate>
DSAStackTy::DSAVarData DSAStackTy::hasInnermostDSA(VarDecl *D,
                                                   ClausesPredicate CPred,
                                                   DirectivesPredicate DPred) {
  for (auto I = Stack.rbegin(), EE = std::prev(Stack.rend()); I != EE; ++I) {
    if (!DPred(I->Directive))
      continue;
    DSAVarData DVar = getDSA(I, D);
    if (CPred(DVar.CKind))
      return DVar;
    return DSAVarData();
  }
  return DSAVarData();
}

void Sema::InitDataSharingAttributesStack() {
  VarDataSharingAttributesStack = new DSAStackTy(*this);
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStack)

void Sema::DestroyDataSharingAttributesStack() { delete DSAStack; }

void Sema::StartOpenMPDSABlock(OpenMPDirectiveKind DKind,
                               const DeclarationNameInfo &DirName,
                               Scope *CurScope, SourceLocation Loc) {
  DSAStack->push(DKind, DirName, CurScope, Loc);
  PushExpressionEvaluationContext(PotentiallyEvaluated);
}

void Sema::EndOpenMPDSABlock(Stmt *CurDirective) {
  // OpenMP [2.14.3.5, Restrictions, C/C++, p.1]
  //  A variable of class type (or array thereof) that appears in a lastprivate
  //  clause requires an accessible, unambiguous default constructor for the
  //  class type, unless the list item is also specified in a firstprivate
  //  clause.
  if (auto D = dyn_cast_or_null<OMPExecutableDirective>(CurDirective)) {
    for (auto C : D->clauses()) {
      if (auto Clause = dyn_cast<OMPLastprivateClause>(C)) {
        for (auto VarRef : Clause->varlists()) {
          if (VarRef->isValueDependent() || VarRef->isTypeDependent())
            continue;
          auto VD = cast<VarDecl>(cast<DeclRefExpr>(VarRef)->getDecl());
          auto DVar = DSAStack->getTopDSA(VD);
          if (DVar.CKind == OMPC_lastprivate) {
            SourceLocation ELoc = VarRef->getExprLoc();
            auto Type = VarRef->getType();
            if (Type->isArrayType())
              Type = QualType(Type->getArrayElementTypeNoTypeQual(), 0);
            CXXRecordDecl *RD =
                getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : nullptr;
            // FIXME This code must be replaced by actual constructing of the
            // lastprivate variable.
            if (RD) {
              CXXConstructorDecl *CD = LookupDefaultConstructor(RD);
              PartialDiagnostic PD =
                  PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
              if (!CD ||
                  CheckConstructorAccess(
                      ELoc, CD, InitializedEntity::InitializeTemporary(Type),
                      CD->getAccess(), PD) == AR_inaccessible ||
                  CD->isDeleted()) {
                Diag(ELoc, diag::err_omp_required_method)
                    << getOpenMPClauseName(OMPC_lastprivate) << 0;
                bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                              VarDecl::DeclarationOnly;
                Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl
                                               : diag::note_defined_here)
                    << VD;
                Diag(RD->getLocation(), diag::note_previous_decl) << RD;
                continue;
              }
              MarkFunctionReferenced(ELoc, CD);
              DiagnoseUseOfDecl(CD, ELoc);
            }
          }
        }
      }
    }
  }

  DSAStack->pop();
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

namespace {

class VarDeclFilterCCC : public CorrectionCandidateCallback {
private:
  Sema &SemaRef;

public:
  explicit VarDeclFilterCCC(Sema &S) : SemaRef(S) {}
  bool ValidateCandidate(const TypoCorrection &Candidate) override {
    NamedDecl *ND = Candidate.getCorrectionDecl();
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(ND)) {
      return VD->hasGlobalStorage() &&
             SemaRef.isDeclInScope(ND, SemaRef.getCurLexicalContext(),
                                   SemaRef.getCurScope());
    }
    return false;
  }
};
} // namespace

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
    if (TypoCorrection Corrected =
            CorrectTypo(Id, LookupOrdinaryName, CurScope, nullptr, Validator,
                        CTK_ErrorRecovery)) {
      diagnoseTypo(Corrected,
                   PDiag(Lookup.empty()
                             ? diag::err_undeclared_var_use_suggest
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
      Diag(Id.getLoc(), diag::err_omp_expected_var_arg) << Id.getName();
      Diag(Lookup.getFoundDecl()->getLocation(), diag::note_declared_at);
      return ExprError();
    }
  }
  Lookup.suppressDiagnostics();

  // OpenMP [2.9.2, Syntax, C/C++]
  //   Variables must be file-scope, namespace-scope, or static block-scope.
  if (!VD->hasGlobalStorage()) {
    Diag(Id.getLoc(), diag::err_omp_global_var_arg)
        << getOpenMPDirectiveName(OMPD_threadprivate) << !VD->isStaticLocal();
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
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
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
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
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
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
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }
  // OpenMP [2.9.2, Restrictions, C/C++, p.6]
  //   A threadprivate directive for static block-scope variables must appear
  //   in the scope of the variable and not in a nested scope.
  if (CanonicalVD->isStaticLocal() && CurScope &&
      !isDeclInScope(ND, getCurLexicalContext(), CurScope)) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
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
  ExprResult DE = BuildDeclRefExpr(VD, ExprType, VK_LValue, Id.getLoc());
  return DE;
}

Sema::DeclGroupPtrTy
Sema::ActOnOpenMPThreadprivateDirective(SourceLocation Loc,
                                        ArrayRef<Expr *> VarList) {
  if (OMPThreadPrivateDecl *D = CheckOMPThreadPrivateDecl(Loc, VarList)) {
    CurContext->addDecl(D);
    return DeclGroupPtrTy::make(DeclGroupRef(D));
  }
  return DeclGroupPtrTy();
}

namespace {
class LocalVarRefChecker : public ConstStmtVisitor<LocalVarRefChecker, bool> {
  Sema &SemaRef;

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (auto VD = dyn_cast<VarDecl>(E->getDecl())) {
      if (VD->hasLocalStorage()) {
        SemaRef.Diag(E->getLocStart(),
                     diag::err_omp_local_var_in_threadprivate_init)
            << E->getSourceRange();
        SemaRef.Diag(VD->getLocation(), diag::note_defined_here)
            << VD << VD->getSourceRange();
        return true;
      }
    }
    return false;
  }
  bool VisitStmt(const Stmt *S) {
    for (auto Child : S->children()) {
      if (Child && Visit(Child))
        return true;
    }
    return false;
  }
  explicit LocalVarRefChecker(Sema &SemaRef) : SemaRef(SemaRef) {}
};
} // namespace

OMPThreadPrivateDecl *
Sema::CheckOMPThreadPrivateDecl(SourceLocation Loc, ArrayRef<Expr *> VarList) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    DeclRefExpr *DE = cast<DeclRefExpr>(RefExpr);
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
          << getOpenMPDirectiveName(OMPD_threadprivate) << VD->getType();
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // Check if this is a TLS variable.
    if (VD->getTLSKind()) {
      Diag(ILoc, diag::err_omp_var_thread_local) << VD;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // Check if initial value of threadprivate variable reference variable with
    // local storage (it is not supported by runtime).
    if (auto Init = VD->getAnyInitializer()) {
      LocalVarRefChecker Checker(*this);
      if (Checker.Visit(Init))
        continue;
    }

    Vars.push_back(RefExpr);
    DSAStack->addDSA(VD, DE, OMPC_threadprivate);
  }
  OMPThreadPrivateDecl *D = nullptr;
  if (!Vars.empty()) {
    D = OMPThreadPrivateDecl::Create(Context, getCurLexicalContext(), Loc,
                                     Vars);
    D->setAccess(AS_public);
  }
  return D;
}

static void ReportOriginalDSA(Sema &SemaRef, DSAStackTy *Stack,
                              const VarDecl *VD, DSAStackTy::DSAVarData DVar,
                              bool IsLoopIterVar = false) {
  if (DVar.RefExpr) {
    SemaRef.Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_explicit_dsa)
        << getOpenMPClauseName(DVar.CKind);
    return;
  }
  enum {
    PDSA_StaticMemberShared,
    PDSA_StaticLocalVarShared,
    PDSA_LoopIterVarPrivate,
    PDSA_LoopIterVarLinear,
    PDSA_LoopIterVarLastprivate,
    PDSA_ConstVarShared,
    PDSA_GlobalVarShared,
    PDSA_LocalVarPrivate,
    PDSA_Implicit
  } Reason = PDSA_Implicit;
  bool ReportHint = false;
  if (IsLoopIterVar) {
    if (DVar.CKind == OMPC_private)
      Reason = PDSA_LoopIterVarPrivate;
    else if (DVar.CKind == OMPC_lastprivate)
      Reason = PDSA_LoopIterVarLastprivate;
    else
      Reason = PDSA_LoopIterVarLinear;
  } else if (VD->isStaticLocal())
    Reason = PDSA_StaticLocalVarShared;
  else if (VD->isStaticDataMember())
    Reason = PDSA_StaticMemberShared;
  else if (VD->isFileVarDecl())
    Reason = PDSA_GlobalVarShared;
  else if (VD->getType().isConstant(SemaRef.getASTContext()))
    Reason = PDSA_ConstVarShared;
  else if (VD->isLocalVarDecl() && DVar.CKind == OMPC_private) {
    ReportHint = true;
    Reason = PDSA_LocalVarPrivate;
  }
  if (Reason != PDSA_Implicit) {
    SemaRef.Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
        << Reason << ReportHint
        << getOpenMPDirectiveName(Stack->getCurrentDirective());
  } else if (DVar.ImplicitDSALoc.isValid()) {
    SemaRef.Diag(DVar.ImplicitDSALoc, diag::note_omp_implicit_dsa)
        << getOpenMPClauseName(DVar.CKind);
  }
}

namespace {
class DSAAttrChecker : public StmtVisitor<DSAAttrChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  bool ErrorFound;
  CapturedStmt *CS;
  llvm::SmallVector<Expr *, 8> ImplicitFirstprivate;
  llvm::DenseMap<VarDecl *, Expr *> VarsWithInheritedDSA;

public:
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      // Skip internally declared variables.
      if (VD->isLocalVarDecl() && !CS->capturesVariable(VD))
        return;

      SourceLocation ELoc = E->getExprLoc();

      OpenMPDirectiveKind DKind = Stack->getCurrentDirective();
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(VD);
      if (DVar.CKind != OMPC_unknown) {
        if (DKind == OMPD_task && DVar.CKind != OMPC_shared &&
            !Stack->isThreadPrivate(VD) && !DVar.RefExpr)
          ImplicitFirstprivate.push_back(DVar.RefExpr);
        return;
      }
      // The default(none) clause requires that each variable that is referenced
      // in the construct, and does not have a predetermined data-sharing
      // attribute, must have its data-sharing attribute explicitly determined
      // by being listed in a data-sharing attribute clause.
      if (DVar.CKind == OMPC_unknown && Stack->getDefaultDSA() == DSA_none &&
          (isOpenMPParallelDirective(DKind) || DKind == OMPD_task) &&
          VarsWithInheritedDSA.count(VD) == 0) {
        VarsWithInheritedDSA[VD] = E;
        return;
      }

      // OpenMP [2.9.3.6, Restrictions, p.2]
      //  A list item that appears in a reduction clause of the innermost
      //  enclosing worksharing or parallel construct may not be accessed in an
      //  explicit task.
      DVar = Stack->hasInnermostDSA(VD, MatchesAnyClause(OMPC_reduction),
                                    MatchesAlways());
      if (DKind == OMPD_task && DVar.CKind == OMPC_reduction) {
        ErrorFound = true;
        SemaRef.Diag(ELoc, diag::err_omp_reduction_in_task);
        ReportOriginalDSA(SemaRef, Stack, VD, DVar);
        return;
      }

      // Define implicit data-sharing attributes for task.
      DVar = Stack->getImplicitDSA(VD);
      if (DKind == OMPD_task && DVar.CKind != OMPC_shared)
        ImplicitFirstprivate.push_back(DVar.RefExpr);
    }
  }
  void VisitOMPExecutableDirective(OMPExecutableDirective *S) {
    for (auto C : S->clauses())
      if (C)
        for (StmtRange R = C->children(); R; ++R)
          if (Stmt *Child = *R)
            Visit(Child);
  }
  void VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E;
         ++I)
      if (Stmt *Child = *I)
        if (!isa<OMPExecutableDirective>(Child))
          Visit(Child);
  }

  bool isErrorFound() { return ErrorFound; }
  ArrayRef<Expr *> getImplicitFirstprivate() { return ImplicitFirstprivate; }
  llvm::DenseMap<VarDecl *, Expr *> &getVarsWithInheritedDSA() {
    return VarsWithInheritedDSA;
  }

  DSAAttrChecker(DSAStackTy *S, Sema &SemaRef, CapturedStmt *CS)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false), CS(CS) {}
};
} // namespace

void Sema::ActOnOpenMPRegionStart(OpenMPDirectiveKind DKind, Scope *CurScope) {
  switch (DKind) {
  case OMPD_parallel: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType KmpInt32PtrTy = Context.getPointerType(KmpInt32Ty);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_simd: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_for: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_sections: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_section: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_single: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_parallel_for: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType KmpInt32PtrTy = Context.getPointerType(KmpInt32Ty);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_parallel_sections: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_threadprivate:
  case OMPD_task:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
    llvm_unreachable("Unknown OpenMP directive");
  }
}

bool CheckNestingOfRegions(Sema &SemaRef, DSAStackTy *Stack,
                           OpenMPDirectiveKind CurrentRegion,
                           SourceLocation StartLoc) {
  // Allowed nesting of constructs
  // +------------------+-----------------+------------------------------------+
  // | Parent directive | Child directive | Closely (!), No-Closely(+), Both(*)|
  // +------------------+-----------------+------------------------------------+
  // | parallel         | parallel        | *                                  |
  // | parallel         | for             | *                                  |
  // | parallel         | simd            | *                                  |
  // | parallel         | sections        | *                                  |
  // | parallel         | section         | +                                  | 
  // | parallel         | single          | *                                  |
  // | parallel         | parallel for    | *                                  |
  // | parallel         |parallel sections| *                                  |
  // +------------------+-----------------+------------------------------------+
  // | for              | parallel        | *                                  |
  // | for              | for             | +                                  |
  // | for              | simd            | *                                  |
  // | for              | sections        | +                                  |
  // | for              | section         | +                                  |
  // | for              | single          | +                                  |
  // | for              | parallel for    | *                                  |
  // | for              |parallel sections| *                                  |
  // +------------------+-----------------+------------------------------------+
  // | simd             | parallel        |                                    |
  // | simd             | for             |                                    |
  // | simd             | simd            |                                    |
  // | simd             | sections        |                                    |
  // | simd             | section         |                                    |
  // | simd             | single          |                                    |
  // | simd             | parallel for    |                                    |
  // | simd             |parallel sections|                                    |
  // +------------------+-----------------+------------------------------------+
  // | sections         | parallel        | *                                  |
  // | sections         | for             | +                                  |
  // | sections         | simd            | *                                  |
  // | sections         | sections        | +                                  |
  // | sections         | section         | *                                  |
  // | sections         | single          | +                                  |
  // | sections         | parallel for    | *                                  |
  // | sections         |parallel sections| *                                  |
  // +------------------+-----------------+------------------------------------+
  // | section          | parallel        | *                                  |
  // | section          | for             | +                                  |
  // | section          | simd            | *                                  |
  // | section          | sections        | +                                  |
  // | section          | section         | +                                  |
  // | section          | single          | +                                  |
  // | section          | parallel for    | *                                  |
  // | section          |parallel sections| *                                  |
  // +------------------+-----------------+------------------------------------+
  // | single           | parallel        | *                                  |
  // | single           | for             | +                                  |
  // | single           | simd            | *                                  |
  // | single           | sections        | +                                  |
  // | single           | section         | +                                  |
  // | single           | single          | +                                  |
  // | single           | parallel for    | *                                  |
  // | single           |parallel sections| *                                  |
  // +------------------+-----------------+------------------------------------+
  // | parallel for     | parallel        | *                                  |
  // | parallel for     | for             | +                                  |
  // | parallel for     | simd            | *                                  |
  // | parallel for     | sections        | +                                  |
  // | parallel for     | section         | +                                  |
  // | parallel for     | single          | +                                  |
  // | parallel for     | parallel for    | *                                  |
  // | parallel for     |parallel sections| *                                  |
  // +------------------+-----------------+------------------------------------+
  // | parallel sections| parallel        | *                                  |
  // | parallel sections| for             | +                                  |
  // | parallel sections| simd            | *                                  |
  // | parallel sections| sections        | +                                  |
  // | parallel sections| section         | *                                  |
  // | parallel sections| single          | +                                  |
  // | parallel sections| parallel for    | *                                  |
  // | parallel sections|parallel sections| *                                  |
  // +------------------+-----------------+------------------------------------+
  if (Stack->getCurScope()) {
    auto ParentRegion = Stack->getParentDirective();
    bool NestingProhibited = false;
    bool CloseNesting = true;
    bool ShouldBeInParallelRegion = false;
    if (isOpenMPSimdDirective(ParentRegion)) {
      // OpenMP [2.16, Nesting of Regions]
      // OpenMP constructs may not be nested inside a simd region.
      SemaRef.Diag(StartLoc, diag::err_omp_prohibited_region_simd);
      return true;
    }
    if (CurrentRegion == OMPD_section) {
      // OpenMP [2.7.2, sections Construct, Restrictions]
      // Orphaned section directives are prohibited. That is, the section
      // directives must appear within the sections construct and must not be
      // encountered elsewhere in the sections region.
      if (ParentRegion != OMPD_sections &&
          ParentRegion != OMPD_parallel_sections) {
        SemaRef.Diag(StartLoc, diag::err_omp_orphaned_section_directive)
            << (ParentRegion != OMPD_unknown)
            << getOpenMPDirectiveName(ParentRegion);
        return true;
      }
      return false;
    }
    if (isOpenMPWorksharingDirective(CurrentRegion) &&
        !isOpenMPParallelDirective(CurrentRegion) &&
        !isOpenMPSimdDirective(CurrentRegion)) {
      // OpenMP [2.16, Nesting of Regions]
      // A worksharing region may not be closely nested inside a worksharing,
      // explicit task, critical, ordered, atomic, or master region.
      // TODO
      NestingProhibited = isOpenMPWorksharingDirective(ParentRegion) &&
                          !isOpenMPSimdDirective(ParentRegion);
      ShouldBeInParallelRegion = true;
    }
    if (NestingProhibited) {
      SemaRef.Diag(StartLoc, diag::err_omp_prohibited_region)
          << CloseNesting << getOpenMPDirectiveName(ParentRegion)
          << ShouldBeInParallelRegion << getOpenMPDirectiveName(CurrentRegion);
      return true;
    }
  }
  return false;
}

StmtResult Sema::ActOnOpenMPExecutableDirective(OpenMPDirectiveKind Kind,
                                                ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");

  StmtResult Res = StmtError();
  if (CheckNestingOfRegions(*this, DSAStack, Kind, StartLoc))
    return StmtError();

  // Check default data sharing attributes for referenced variables.
  DSAAttrChecker DSAChecker(DSAStack, *this, cast<CapturedStmt>(AStmt));
  DSAChecker.Visit(cast<CapturedStmt>(AStmt)->getCapturedStmt());
  if (DSAChecker.isErrorFound())
    return StmtError();
  // Generate list of implicitly defined firstprivate variables.
  auto &VarsWithInheritedDSA = DSAChecker.getVarsWithInheritedDSA();
  llvm::SmallVector<OMPClause *, 8> ClausesWithImplicit;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());

  bool ErrorFound = false;
  if (!DSAChecker.getImplicitFirstprivate().empty()) {
    if (OMPClause *Implicit = ActOnOpenMPFirstprivateClause(
            DSAChecker.getImplicitFirstprivate(), SourceLocation(),
            SourceLocation(), SourceLocation())) {
      ClausesWithImplicit.push_back(Implicit);
      ErrorFound = cast<OMPFirstprivateClause>(Implicit)->varlist_size() !=
                   DSAChecker.getImplicitFirstprivate().size();
    } else
      ErrorFound = true;
  }

  switch (Kind) {
  case OMPD_parallel:
    Res = ActOnOpenMPParallelDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    break;
  case OMPD_simd:
    Res = ActOnOpenMPSimdDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc,
                                   VarsWithInheritedDSA);
    break;
  case OMPD_for:
    Res = ActOnOpenMPForDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc,
                                  VarsWithInheritedDSA);
    break;
  case OMPD_sections:
    Res = ActOnOpenMPSectionsDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    break;
  case OMPD_section:
    assert(ClausesWithImplicit.empty() &&
           "No clauses is allowed for 'omp section' directive");
    Res = ActOnOpenMPSectionDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_single:
    Res = ActOnOpenMPSingleDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_parallel_for:
    Res = ActOnOpenMPParallelForDirective(ClausesWithImplicit, AStmt, StartLoc,
                                          EndLoc, VarsWithInheritedDSA);
    break;
  case OMPD_parallel_sections:
    Res = ActOnOpenMPParallelSectionsDirective(ClausesWithImplicit, AStmt,
                                               StartLoc, EndLoc);
    break;
  case OMPD_threadprivate:
  case OMPD_task:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
    llvm_unreachable("Unknown OpenMP directive");
  }

  for (auto P : VarsWithInheritedDSA) {
    Diag(P.second->getExprLoc(), diag::err_omp_no_dsa_for_variable)
        << P.first << P.second->getSourceRange();
  }
  if (!VarsWithInheritedDSA.empty())
    return StmtError();

  if (ErrorFound)
    return StmtError();
  return Res;
}

StmtResult Sema::ActOnOpenMPParallelDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");
  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  getCurFunction()->setHasBranchProtectedScope();

  return OMPParallelDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                      AStmt);
}

namespace {
/// \brief Helper class for checking canonical form of the OpenMP loops and
/// extracting iteration space of each loop in the loop nest, that will be used
/// for IR generation.
class OpenMPIterationSpaceChecker {
  /// \brief Reference to Sema.
  Sema &SemaRef;
  /// \brief A location for diagnostics (when there is no some better location).
  SourceLocation DefaultLoc;
  /// \brief A location for diagnostics (when increment is not compatible).
  SourceLocation ConditionLoc;
  /// \brief A source location for referring to condition later.
  SourceRange ConditionSrcRange;
  /// \brief Loop variable.
  VarDecl *Var;
  /// \brief Lower bound (initializer for the var).
  Expr *LB;
  /// \brief Upper bound.
  Expr *UB;
  /// \brief Loop step (increment).
  Expr *Step;
  /// \brief This flag is true when condition is one of:
  ///   Var <  UB
  ///   Var <= UB
  ///   UB  >  Var
  ///   UB  >= Var
  bool TestIsLessOp;
  /// \brief This flag is true when condition is strict ( < or > ).
  bool TestIsStrictOp;
  /// \brief This flag is true when step is subtracted on each iteration.
  bool SubtractStep;

public:
  OpenMPIterationSpaceChecker(Sema &SemaRef, SourceLocation DefaultLoc)
      : SemaRef(SemaRef), DefaultLoc(DefaultLoc), ConditionLoc(DefaultLoc),
        ConditionSrcRange(SourceRange()), Var(nullptr), LB(nullptr),
        UB(nullptr), Step(nullptr), TestIsLessOp(false), TestIsStrictOp(false),
        SubtractStep(false) {}
  /// \brief Check init-expr for canonical loop form and save loop counter
  /// variable - #Var and its initialization value - #LB.
  bool CheckInit(Stmt *S);
  /// \brief Check test-expr for canonical form, save upper-bound (#UB), flags
  /// for less/greater and for strict/non-strict comparison.
  bool CheckCond(Expr *S);
  /// \brief Check incr-expr for canonical loop form and return true if it
  /// does not conform, otherwise save loop step (#Step).
  bool CheckInc(Expr *S);
  /// \brief Return the loop counter variable.
  VarDecl *GetLoopVar() const { return Var; }
  /// \brief Return true if any expression is dependent.
  bool Dependent() const;

private:
  /// \brief Check the right-hand side of an assignment in the increment
  /// expression.
  bool CheckIncRHS(Expr *RHS);
  /// \brief Helper to set loop counter variable and its initializer.
  bool SetVarAndLB(VarDecl *NewVar, Expr *NewLB);
  /// \brief Helper to set upper bound.
  bool SetUB(Expr *NewUB, bool LessOp, bool StrictOp, const SourceRange &SR,
             const SourceLocation &SL);
  /// \brief Helper to set loop increment.
  bool SetStep(Expr *NewStep, bool Subtract);
};

bool OpenMPIterationSpaceChecker::Dependent() const {
  if (!Var) {
    assert(!LB && !UB && !Step);
    return false;
  }
  return Var->getType()->isDependentType() || (LB && LB->isValueDependent()) ||
         (UB && UB->isValueDependent()) || (Step && Step->isValueDependent());
}

bool OpenMPIterationSpaceChecker::SetVarAndLB(VarDecl *NewVar, Expr *NewLB) {
  // State consistency checking to ensure correct usage.
  assert(Var == nullptr && LB == nullptr && UB == nullptr && Step == nullptr &&
         !TestIsLessOp && !TestIsStrictOp);
  if (!NewVar || !NewLB)
    return true;
  Var = NewVar;
  LB = NewLB;
  return false;
}

bool OpenMPIterationSpaceChecker::SetUB(Expr *NewUB, bool LessOp, bool StrictOp,
                                        const SourceRange &SR,
                                        const SourceLocation &SL) {
  // State consistency checking to ensure correct usage.
  assert(Var != nullptr && LB != nullptr && UB == nullptr && Step == nullptr &&
         !TestIsLessOp && !TestIsStrictOp);
  if (!NewUB)
    return true;
  UB = NewUB;
  TestIsLessOp = LessOp;
  TestIsStrictOp = StrictOp;
  ConditionSrcRange = SR;
  ConditionLoc = SL;
  return false;
}

bool OpenMPIterationSpaceChecker::SetStep(Expr *NewStep, bool Subtract) {
  // State consistency checking to ensure correct usage.
  assert(Var != nullptr && LB != nullptr && Step == nullptr);
  if (!NewStep)
    return true;
  if (!NewStep->isValueDependent()) {
    // Check that the step is integer expression.
    SourceLocation StepLoc = NewStep->getLocStart();
    ExprResult Val =
        SemaRef.PerformOpenMPImplicitIntegerConversion(StepLoc, NewStep);
    if (Val.isInvalid())
      return true;
    NewStep = Val.get();

    // OpenMP [2.6, Canonical Loop Form, Restrictions]
    //  If test-expr is of form var relational-op b and relational-op is < or
    //  <= then incr-expr must cause var to increase on each iteration of the
    //  loop. If test-expr is of form var relational-op b and relational-op is
    //  > or >= then incr-expr must cause var to decrease on each iteration of
    //  the loop.
    //  If test-expr is of form b relational-op var and relational-op is < or
    //  <= then incr-expr must cause var to decrease on each iteration of the
    //  loop. If test-expr is of form b relational-op var and relational-op is
    //  > or >= then incr-expr must cause var to increase on each iteration of
    //  the loop.
    llvm::APSInt Result;
    bool IsConstant = NewStep->isIntegerConstantExpr(Result, SemaRef.Context);
    bool IsUnsigned = !NewStep->getType()->hasSignedIntegerRepresentation();
    bool IsConstNeg =
        IsConstant && Result.isSigned() && (Subtract != Result.isNegative());
    bool IsConstZero = IsConstant && !Result.getBoolValue();
    if (UB && (IsConstZero ||
               (TestIsLessOp ? (IsConstNeg || (IsUnsigned && Subtract))
                             : (!IsConstNeg || (IsUnsigned && !Subtract))))) {
      SemaRef.Diag(NewStep->getExprLoc(),
                   diag::err_omp_loop_incr_not_compatible)
          << Var << TestIsLessOp << NewStep->getSourceRange();
      SemaRef.Diag(ConditionLoc,
                   diag::note_omp_loop_cond_requres_compatible_incr)
          << TestIsLessOp << ConditionSrcRange;
      return true;
    }
  }

  Step = NewStep;
  SubtractStep = Subtract;
  return false;
}

bool OpenMPIterationSpaceChecker::CheckInit(Stmt *S) {
  // Check init-expr for canonical loop form and save loop counter
  // variable - #Var and its initialization value - #LB.
  // OpenMP [2.6] Canonical loop form. init-expr may be one of the following:
  //   var = lb
  //   integer-type var = lb
  //   random-access-iterator-type var = lb
  //   pointer-type var = lb
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_init);
    return true;
  }
  if (Expr *E = dyn_cast<Expr>(S))
    S = E->IgnoreParens();
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() == BO_Assign)
      if (auto DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParens()))
        return SetVarAndLB(dyn_cast<VarDecl>(DRE->getDecl()), BO->getLHS());
  } else if (auto DS = dyn_cast<DeclStmt>(S)) {
    if (DS->isSingleDecl()) {
      if (auto Var = dyn_cast_or_null<VarDecl>(DS->getSingleDecl())) {
        if (Var->hasInit()) {
          // Accept non-canonical init form here but emit ext. warning.
          if (Var->getInitStyle() != VarDecl::CInit)
            SemaRef.Diag(S->getLocStart(),
                         diag::ext_omp_loop_not_canonical_init)
                << S->getSourceRange();
          return SetVarAndLB(Var, Var->getInit());
        }
      }
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(S))
    if (CE->getOperator() == OO_Equal)
      if (auto DRE = dyn_cast<DeclRefExpr>(CE->getArg(0)))
        return SetVarAndLB(dyn_cast<VarDecl>(DRE->getDecl()), CE->getArg(1));

  SemaRef.Diag(S->getLocStart(), diag::err_omp_loop_not_canonical_init)
      << S->getSourceRange();
  return true;
}

/// \brief Ignore parenthesizes, implicit casts, copy constructor and return the
/// variable (which may be the loop variable) if possible.
static const VarDecl *GetInitVarDecl(const Expr *E) {
  if (!E)
    return nullptr;
  E = E->IgnoreParenImpCasts();
  if (auto *CE = dyn_cast_or_null<CXXConstructExpr>(E))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if (Ctor->isCopyConstructor() && CE->getNumArgs() == 1 &&
          CE->getArg(0) != nullptr)
        E = CE->getArg(0)->IgnoreParenImpCasts();
  auto DRE = dyn_cast_or_null<DeclRefExpr>(E);
  if (!DRE)
    return nullptr;
  return dyn_cast<VarDecl>(DRE->getDecl());
}

bool OpenMPIterationSpaceChecker::CheckCond(Expr *S) {
  // Check test-expr for canonical form, save upper-bound UB, flags for
  // less/greater and for strict/non-strict comparison.
  // OpenMP [2.6] Canonical loop form. Test-expr may be one of the following:
  //   var relational-op b
  //   b relational-op var
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_cond) << Var;
    return true;
  }
  S = S->IgnoreParenImpCasts();
  SourceLocation CondLoc = S->getLocStart();
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->isRelationalOp()) {
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return SetUB(BO->getRHS(),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_LE),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT),
                     BO->getSourceRange(), BO->getOperatorLoc());
      if (GetInitVarDecl(BO->getRHS()) == Var)
        return SetUB(BO->getLHS(),
                     (BO->getOpcode() == BO_GT || BO->getOpcode() == BO_GE),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT),
                     BO->getSourceRange(), BO->getOperatorLoc());
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getNumArgs() == 2) {
      auto Op = CE->getOperator();
      switch (Op) {
      case OO_Greater:
      case OO_GreaterEqual:
      case OO_Less:
      case OO_LessEqual:
        if (GetInitVarDecl(CE->getArg(0)) == Var)
          return SetUB(CE->getArg(1), Op == OO_Less || Op == OO_LessEqual,
                       Op == OO_Less || Op == OO_Greater, CE->getSourceRange(),
                       CE->getOperatorLoc());
        if (GetInitVarDecl(CE->getArg(1)) == Var)
          return SetUB(CE->getArg(0), Op == OO_Greater || Op == OO_GreaterEqual,
                       Op == OO_Less || Op == OO_Greater, CE->getSourceRange(),
                       CE->getOperatorLoc());
        break;
      default:
        break;
      }
    }
  }
  SemaRef.Diag(CondLoc, diag::err_omp_loop_not_canonical_cond)
      << S->getSourceRange() << Var;
  return true;
}

bool OpenMPIterationSpaceChecker::CheckIncRHS(Expr *RHS) {
  // RHS of canonical loop form increment can be:
  //   var + incr
  //   incr + var
  //   var - incr
  //
  RHS = RHS->IgnoreParenImpCasts();
  if (auto BO = dyn_cast<BinaryOperator>(RHS)) {
    if (BO->isAdditiveOp()) {
      bool IsAdd = BO->getOpcode() == BO_Add;
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return SetStep(BO->getRHS(), !IsAdd);
      if (IsAdd && GetInitVarDecl(BO->getRHS()) == Var)
        return SetStep(BO->getLHS(), false);
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(RHS)) {
    bool IsAdd = CE->getOperator() == OO_Plus;
    if ((IsAdd || CE->getOperator() == OO_Minus) && CE->getNumArgs() == 2) {
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return SetStep(CE->getArg(1), !IsAdd);
      if (IsAdd && GetInitVarDecl(CE->getArg(1)) == Var)
        return SetStep(CE->getArg(0), false);
    }
  }
  SemaRef.Diag(RHS->getLocStart(), diag::err_omp_loop_not_canonical_incr)
      << RHS->getSourceRange() << Var;
  return true;
}

bool OpenMPIterationSpaceChecker::CheckInc(Expr *S) {
  // Check incr-expr for canonical loop form and return true if it
  // does not conform.
  // OpenMP [2.6] Canonical loop form. Test-expr may be one of the following:
  //   ++var
  //   var++
  //   --var
  //   var--
  //   var += incr
  //   var -= incr
  //   var = var + incr
  //   var = incr + var
  //   var = var - incr
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_incr) << Var;
    return true;
  }
  S = S->IgnoreParens();
  if (auto UO = dyn_cast<UnaryOperator>(S)) {
    if (UO->isIncrementDecrementOp() && GetInitVarDecl(UO->getSubExpr()) == Var)
      return SetStep(
          SemaRef.ActOnIntegerConstant(UO->getLocStart(),
                                       (UO->isDecrementOp() ? -1 : 1)).get(),
          false);
  } else if (auto BO = dyn_cast<BinaryOperator>(S)) {
    switch (BO->getOpcode()) {
    case BO_AddAssign:
    case BO_SubAssign:
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return SetStep(BO->getRHS(), BO->getOpcode() == BO_SubAssign);
      break;
    case BO_Assign:
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return CheckIncRHS(BO->getRHS());
      break;
    default:
      break;
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    switch (CE->getOperator()) {
    case OO_PlusPlus:
    case OO_MinusMinus:
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return SetStep(
            SemaRef.ActOnIntegerConstant(
                        CE->getLocStart(),
                        ((CE->getOperator() == OO_MinusMinus) ? -1 : 1)).get(),
            false);
      break;
    case OO_PlusEqual:
    case OO_MinusEqual:
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return SetStep(CE->getArg(1), CE->getOperator() == OO_MinusEqual);
      break;
    case OO_Equal:
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return CheckIncRHS(CE->getArg(1));
      break;
    default:
      break;
    }
  }
  SemaRef.Diag(S->getLocStart(), diag::err_omp_loop_not_canonical_incr)
      << S->getSourceRange() << Var;
  return true;
}
} // namespace

/// \brief Called on a for stmt to check and extract its iteration space
/// for further processing (such as collapsing).
static bool CheckOpenMPIterationSpace(
    OpenMPDirectiveKind DKind, Stmt *S, Sema &SemaRef, DSAStackTy &DSA,
    unsigned CurrentNestedLoopCount, unsigned NestedLoopCount,
    Expr *NestedLoopCountExpr,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  // OpenMP [2.6, Canonical Loop Form]
  //   for (init-expr; test-expr; incr-expr) structured-block
  auto For = dyn_cast_or_null<ForStmt>(S);
  if (!For) {
    SemaRef.Diag(S->getLocStart(), diag::err_omp_not_for)
        << (NestedLoopCountExpr != nullptr) << getOpenMPDirectiveName(DKind)
        << NestedLoopCount << (CurrentNestedLoopCount > 0)
        << CurrentNestedLoopCount;
    if (NestedLoopCount > 1)
      SemaRef.Diag(NestedLoopCountExpr->getExprLoc(),
                   diag::note_omp_collapse_expr)
          << NestedLoopCountExpr->getSourceRange();
    return true;
  }
  assert(For->getBody());

  OpenMPIterationSpaceChecker ISC(SemaRef, For->getForLoc());

  // Check init.
  auto Init = For->getInit();
  if (ISC.CheckInit(Init)) {
    return true;
  }

  bool HasErrors = false;

  // Check loop variable's type.
  auto Var = ISC.GetLoopVar();

  // OpenMP [2.6, Canonical Loop Form]
  // Var is one of the following:
  //   A variable of signed or unsigned integer type.
  //   For C++, a variable of a random access iterator type.
  //   For C, a variable of a pointer type.
  auto VarType = Var->getType();
  if (!VarType->isDependentType() && !VarType->isIntegerType() &&
      !VarType->isPointerType() &&
      !(SemaRef.getLangOpts().CPlusPlus && VarType->isOverloadableType())) {
    SemaRef.Diag(Init->getLocStart(), diag::err_omp_loop_variable_type)
        << SemaRef.getLangOpts().CPlusPlus;
    HasErrors = true;
  }

  // OpenMP, 2.14.1.1 Data-sharing Attribute Rules for Variables Referenced in a
  // Construct
  // The loop iteration variable(s) in the associated for-loop(s) of a for or
  // parallel for construct is (are) private.
  // The loop iteration variable in the associated for-loop of a simd construct
  // with just one associated for-loop is linear with a constant-linear-step
  // that is the increment of the associated for-loop.
  // Exclude loop var from the list of variables with implicitly defined data
  // sharing attributes.
  while (VarsWithImplicitDSA.count(Var) > 0)
    VarsWithImplicitDSA.erase(Var);

  // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables Referenced in
  // a Construct, C/C++].
  // The loop iteration variable in the associated for-loop of a simd construct
  // with just one associated for-loop may be listed in a linear clause with a
  // constant-linear-step that is the increment of the associated for-loop.
  // The loop iteration variable(s) in the associated for-loop(s) of a for or
  // parallel for construct may be listed in a private or lastprivate clause.
  DSAStackTy::DSAVarData DVar = DSA.getTopDSA(Var);
  auto PredeterminedCKind =
      isOpenMPSimdDirective(DKind)
          ? ((NestedLoopCount == 1) ? OMPC_linear : OMPC_lastprivate)
          : OMPC_private;
  if (((isOpenMPSimdDirective(DKind) && DVar.CKind != OMPC_unknown &&
        DVar.CKind != PredeterminedCKind) ||
       (isOpenMPWorksharingDirective(DKind) && DVar.CKind != OMPC_unknown &&
        DVar.CKind != OMPC_private && DVar.CKind != OMPC_lastprivate)) &&
      (DVar.CKind != OMPC_private || DVar.RefExpr != nullptr)) {
    SemaRef.Diag(Init->getLocStart(), diag::err_omp_loop_var_dsa)
        << getOpenMPClauseName(DVar.CKind) << getOpenMPDirectiveName(DKind)
        << getOpenMPClauseName(PredeterminedCKind);
    ReportOriginalDSA(SemaRef, &DSA, Var, DVar, true);
    HasErrors = true;
  } else {
    // Make the loop iteration variable private (for worksharing constructs),
    // linear (for simd directives with the only one associated loop) or
    // lastprivate (for simd directives with several collapsed loops).
    DSA.addDSA(Var, nullptr, PredeterminedCKind);
  }

  assert(isOpenMPLoopDirective(DKind) && "DSA for non-loop vars");

  // Check test-expr.
  HasErrors |= ISC.CheckCond(For->getCond());

  // Check incr-expr.
  HasErrors |= ISC.CheckInc(For->getInc());

  if (ISC.Dependent())
    return HasErrors;

  // FIXME: Build loop's iteration space representation.
  return HasErrors;
}

/// \brief A helper routine to skip no-op (attributed, compound) stmts get the
/// next nested for loop. If \a IgnoreCaptured is true, it skips captured stmt
/// to get the first for loop.
static Stmt *IgnoreContainerStmts(Stmt *S, bool IgnoreCaptured) {
  if (IgnoreCaptured)
    if (auto CapS = dyn_cast_or_null<CapturedStmt>(S))
      S = CapS->getCapturedStmt();
  // OpenMP [2.8.1, simd construct, Restrictions]
  // All loops associated with the construct must be perfectly nested; that is,
  // there must be no intervening code nor any OpenMP directive between any two
  // loops.
  while (true) {
    if (auto AS = dyn_cast_or_null<AttributedStmt>(S))
      S = AS->getSubStmt();
    else if (auto CS = dyn_cast_or_null<CompoundStmt>(S)) {
      if (CS->size() != 1)
        break;
      S = CS->body_back();
    } else
      break;
  }
  return S;
}

/// \brief Called on a for stmt to check itself and nested loops (if any).
/// \return Returns 0 if one of the collapsed stmts is not canonical for loop,
/// number of collapsed loops otherwise.
static unsigned
CheckOpenMPLoop(OpenMPDirectiveKind DKind, Expr *NestedLoopCountExpr,
                Stmt *AStmt, Sema &SemaRef, DSAStackTy &DSA,
                llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  unsigned NestedLoopCount = 1;
  if (NestedLoopCountExpr) {
    // Found 'collapse' clause - calculate collapse number.
    llvm::APSInt Result;
    if (NestedLoopCountExpr->EvaluateAsInt(Result, SemaRef.getASTContext()))
      NestedLoopCount = Result.getLimitedValue();
  }
  // This is helper routine for loop directives (e.g., 'for', 'simd',
  // 'for simd', etc.).
  Stmt *CurStmt = IgnoreContainerStmts(AStmt, true);
  for (unsigned Cnt = 0; Cnt < NestedLoopCount; ++Cnt) {
    if (CheckOpenMPIterationSpace(DKind, CurStmt, SemaRef, DSA, Cnt,
                                  NestedLoopCount, NestedLoopCountExpr,
                                  VarsWithImplicitDSA))
      return 0;
    // Move on to the next nested for loop, or to the loop body.
    CurStmt = IgnoreContainerStmts(cast<ForStmt>(CurStmt)->getBody(), false);
  }

  // FIXME: Build resulting iteration space for IR generation (collapsing
  // iteration spaces when loop count > 1 ('collapse' clause)).
  return NestedLoopCount;
}

static Expr *GetCollapseNumberExpr(ArrayRef<OMPClause *> Clauses) {
  auto CollapseFilter = [](const OMPClause *C) -> bool {
    return C->getClauseKind() == OMPC_collapse;
  };
  OMPExecutableDirective::filtered_clause_iterator<decltype(CollapseFilter)> I(
      Clauses, CollapseFilter);
  if (I)
    return cast<OMPCollapseClause>(*I)->getNumForLoops();
  return nullptr;
}

StmtResult Sema::ActOnOpenMPSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  // In presence of clause 'collapse', it will define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_simd, GetCollapseNumberExpr(Clauses), AStmt, *this,
                      *DSAStack, VarsWithImplicitDSA);
  if (NestedLoopCount == 0)
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPSimdDirective::Create(Context, StartLoc, EndLoc, NestedLoopCount,
                                  Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  // In presence of clause 'collapse', it will define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_for, GetCollapseNumberExpr(Clauses), AStmt, *this,
                      *DSAStack, VarsWithImplicitDSA);
  if (NestedLoopCount == 0)
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPForDirective::Create(Context, StartLoc, EndLoc, NestedLoopCount,
                                 Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto BaseStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  if (auto C = dyn_cast_or_null<CompoundStmt>(BaseStmt)) {
    auto S = C->children();
    if (!S)
      return StmtError();
    // All associated statements must be '#pragma omp section' except for
    // the first one.
    for (++S; S; ++S) {
      auto SectionStmt = *S;
      if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
        if (SectionStmt)
          Diag(SectionStmt->getLocStart(),
               diag::err_omp_sections_substmt_not_section);
        return StmtError();
      }
    }
  } else {
    Diag(AStmt->getLocStart(), diag::err_omp_sections_not_compound_stmt);
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPSectionsDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                      AStmt);
}

StmtResult Sema::ActOnOpenMPSectionDirective(Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");

  getCurFunction()->setHasBranchProtectedScope();

  return OMPSectionDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

StmtResult Sema::ActOnOpenMPSingleDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();
  return OMPSingleDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPParallelForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");
  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  // In presence of clause 'collapse', it will define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_parallel_for, GetCollapseNumberExpr(Clauses), AStmt,
                      *this, *DSAStack, VarsWithImplicitDSA);
  if (NestedLoopCount == 0)
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPParallelForDirective::Create(Context, StartLoc, EndLoc,
                                         NestedLoopCount, Clauses, AStmt);
}

StmtResult
Sema::ActOnOpenMPParallelSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                           Stmt *AStmt, SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto BaseStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  if (auto C = dyn_cast_or_null<CompoundStmt>(BaseStmt)) {
    auto S = C->children();
    if (!S)
      return StmtError();
    // All associated statements must be '#pragma omp section' except for
    // the first one.
    for (++S; S; ++S) {
      auto SectionStmt = *S;
      if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
        if (SectionStmt)
          Diag(SectionStmt->getLocStart(),
               diag::err_omp_parallel_sections_substmt_not_section);
        return StmtError();
      }
    }
  } else {
    Diag(AStmt->getLocStart(),
         diag::err_omp_parallel_sections_not_compound_stmt);
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPParallelSectionsDirective::Create(Context, StartLoc, EndLoc,
                                              Clauses, AStmt);
}

OMPClause *Sema::ActOnOpenMPSingleExprClause(OpenMPClauseKind Kind, Expr *Expr,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_if:
    Res = ActOnOpenMPIfClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_num_threads:
    Res = ActOnOpenMPNumThreadsClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_safelen:
    Res = ActOnOpenMPSafelenClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_collapse:
    Res = ActOnOpenMPCollapseClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_threadprivate:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPIfClause(Expr *Condition, SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = ActOnBooleanCondition(DSAStack->getCurScope(),
                                           Condition->getExprLoc(), Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = Val.get();
  }

  return new (Context) OMPIfClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::PerformOpenMPImplicitIntegerConversion(SourceLocation Loc,
                                                        Expr *Op) {
  if (!Op)
    return ExprError();

  class IntConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    IntConvertDiagnoser()
        : ICEConvertDiagnoser(/*AllowScopedEnumerations*/ false, false, true) {}
    SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                         QualType T) override {
      return S.Diag(Loc, diag::err_omp_not_integral) << T;
    }
    SemaDiagnosticBuilder diagnoseIncomplete(Sema &S, SourceLocation Loc,
                                             QualType T) override {
      return S.Diag(Loc, diag::err_omp_incomplete_type) << T;
    }
    SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S, SourceLocation Loc,
                                               QualType T,
                                               QualType ConvTy) override {
      return S.Diag(Loc, diag::err_omp_explicit_conversion) << T << ConvTy;
    }
    SemaDiagnosticBuilder noteExplicitConv(Sema &S, CXXConversionDecl *Conv,
                                           QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_omp_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                            QualType T) override {
      return S.Diag(Loc, diag::err_omp_ambiguous_conversion) << T;
    }
    SemaDiagnosticBuilder noteAmbiguous(Sema &S, CXXConversionDecl *Conv,
                                        QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_omp_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseConversion(Sema &, SourceLocation, QualType,
                                             QualType) override {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;
  return PerformContextualImplicitConversion(Loc, Op, ConvertDiagnoser);
}

OMPClause *Sema::ActOnOpenMPNumThreadsClause(Expr *NumThreads,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  Expr *ValExpr = NumThreads;
  if (!NumThreads->isValueDependent() && !NumThreads->isTypeDependent() &&
      !NumThreads->isInstantiationDependent() &&
      !NumThreads->containsUnexpandedParameterPack()) {
    SourceLocation NumThreadsLoc = NumThreads->getLocStart();
    ExprResult Val =
        PerformOpenMPImplicitIntegerConversion(NumThreadsLoc, NumThreads);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = Val.get();

    // OpenMP [2.5, Restrictions]
    //  The num_threads expression must evaluate to a positive integer value.
    llvm::APSInt Result;
    if (ValExpr->isIntegerConstantExpr(Result, Context) && Result.isSigned() &&
        !Result.isStrictlyPositive()) {
      Diag(NumThreadsLoc, diag::err_omp_negative_expression_in_clause)
          << "num_threads" << NumThreads->getSourceRange();
      return nullptr;
    }
  }

  return new (Context)
      OMPNumThreadsClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::VerifyPositiveIntegerConstantInClause(Expr *E,
                                                       OpenMPClauseKind CKind) {
  if (!E)
    return ExprError();
  if (E->isValueDependent() || E->isTypeDependent() ||
      E->isInstantiationDependent() || E->containsUnexpandedParameterPack())
    return E;
  llvm::APSInt Result;
  ExprResult ICE = VerifyIntegerConstantExpression(E, &Result);
  if (ICE.isInvalid())
    return ExprError();
  if (!Result.isStrictlyPositive()) {
    Diag(E->getExprLoc(), diag::err_omp_negative_expression_in_clause)
        << getOpenMPClauseName(CKind) << E->getSourceRange();
    return ExprError();
  }
  return ICE;
}

OMPClause *Sema::ActOnOpenMPSafelenClause(Expr *Len, SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  // OpenMP [2.8.1, simd construct, Description]
  // The parameter of the safelen clause must be a constant
  // positive integer expression.
  ExprResult Safelen = VerifyPositiveIntegerConstantInClause(Len, OMPC_safelen);
  if (Safelen.isInvalid())
    return nullptr;
  return new (Context)
      OMPSafelenClause(Safelen.get(), StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPCollapseClause(Expr *NumForLoops,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  // OpenMP [2.7.1, loop construct, Description]
  // OpenMP [2.8.1, simd construct, Description]
  // OpenMP [2.9.6, distribute construct, Description]
  // The parameter of the collapse clause must be a constant
  // positive integer expression.
  ExprResult NumForLoopsResult =
      VerifyPositiveIntegerConstantInClause(NumForLoops, OMPC_collapse);
  if (NumForLoopsResult.isInvalid())
    return nullptr;
  return new (Context)
      OMPCollapseClause(NumForLoopsResult.get(), StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSimpleClause(
    OpenMPClauseKind Kind, unsigned Argument, SourceLocation ArgumentLoc,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_default:
    Res =
        ActOnOpenMPDefaultClause(static_cast<OpenMPDefaultClauseKind>(Argument),
                                 ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_proc_bind:
    Res = ActOnOpenMPProcBindClause(
        static_cast<OpenMPProcBindClauseKind>(Argument), ArgumentLoc, StartLoc,
        LParenLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_collapse:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_threadprivate:
  case OMPC_unknown:
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
    static_assert(OMPC_DEFAULT_unknown > 0,
                  "OMPC_DEFAULT_unknown not greater than 0");
    std::string Sep(", ");
    for (unsigned i = 0; i < OMPC_DEFAULT_unknown; ++i) {
      Values += "'";
      Values += getOpenMPSimpleClauseTypeName(OMPC_default, i);
      Values += "'";
      switch (i) {
      case OMPC_DEFAULT_unknown - 2:
        Values += " or ";
        break;
      case OMPC_DEFAULT_unknown - 1:
        break;
      default:
        Values += Sep;
        break;
      }
    }
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_default);
    return nullptr;
  }
  switch (Kind) {
  case OMPC_DEFAULT_none:
    DSAStack->setDefaultDSANone(KindKwLoc);
    break;
  case OMPC_DEFAULT_shared:
    DSAStack->setDefaultDSAShared(KindKwLoc);
    break;
  case OMPC_DEFAULT_unknown:
    llvm_unreachable("Clause kind is not allowed.");
    break;
  }
  return new (Context)
      OMPDefaultClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPProcBindClause(OpenMPProcBindClauseKind Kind,
                                           SourceLocation KindKwLoc,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  if (Kind == OMPC_PROC_BIND_unknown) {
    std::string Values;
    std::string Sep(", ");
    for (unsigned i = 0; i < OMPC_PROC_BIND_unknown; ++i) {
      Values += "'";
      Values += getOpenMPSimpleClauseTypeName(OMPC_proc_bind, i);
      Values += "'";
      switch (i) {
      case OMPC_PROC_BIND_unknown - 2:
        Values += " or ";
        break;
      case OMPC_PROC_BIND_unknown - 1:
        break;
      default:
        Values += Sep;
        break;
      }
    }
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_proc_bind);
    return nullptr;
  }
  return new (Context)
      OMPProcBindClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSingleExprWithArgClause(
    OpenMPClauseKind Kind, unsigned Argument, Expr *Expr,
    SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ArgumentLoc, SourceLocation CommaLoc,
    SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_schedule:
    Res = ActOnOpenMPScheduleClause(
        static_cast<OpenMPScheduleClauseKind>(Argument), Expr, StartLoc,
        LParenLoc, ArgumentLoc, CommaLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_threadprivate:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPScheduleClause(
    OpenMPScheduleClauseKind Kind, Expr *ChunkSize, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation KindLoc, SourceLocation CommaLoc,
    SourceLocation EndLoc) {
  if (Kind == OMPC_SCHEDULE_unknown) {
    std::string Values;
    std::string Sep(", ");
    for (unsigned i = 0; i < OMPC_SCHEDULE_unknown; ++i) {
      Values += "'";
      Values += getOpenMPSimpleClauseTypeName(OMPC_schedule, i);
      Values += "'";
      switch (i) {
      case OMPC_SCHEDULE_unknown - 2:
        Values += " or ";
        break;
      case OMPC_SCHEDULE_unknown - 1:
        break;
      default:
        Values += Sep;
        break;
      }
    }
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_schedule);
    return nullptr;
  }
  Expr *ValExpr = ChunkSize;
  if (ChunkSize) {
    if (!ChunkSize->isValueDependent() && !ChunkSize->isTypeDependent() &&
        !ChunkSize->isInstantiationDependent() &&
        !ChunkSize->containsUnexpandedParameterPack()) {
      SourceLocation ChunkSizeLoc = ChunkSize->getLocStart();
      ExprResult Val =
          PerformOpenMPImplicitIntegerConversion(ChunkSizeLoc, ChunkSize);
      if (Val.isInvalid())
        return nullptr;

      ValExpr = Val.get();

      // OpenMP [2.7.1, Restrictions]
      //  chunk_size must be a loop invariant integer expression with a positive
      //  value.
      llvm::APSInt Result;
      if (ValExpr->isIntegerConstantExpr(Result, Context) &&
          Result.isSigned() && !Result.isStrictlyPositive()) {
        Diag(ChunkSizeLoc, diag::err_omp_negative_expression_in_clause)
            << "schedule" << ChunkSize->getSourceRange();
        return nullptr;
      }
    }
  }

  return new (Context) OMPScheduleClause(StartLoc, LParenLoc, KindLoc, CommaLoc,
                                         EndLoc, Kind, ValExpr);
}

OMPClause *Sema::ActOnOpenMPClause(OpenMPClauseKind Kind,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_ordered:
    Res = ActOnOpenMPOrderedClause(StartLoc, EndLoc);
    break;
  case OMPC_nowait:
    Res = ActOnOpenMPNowaitClause(StartLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_collapse:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_threadprivate:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPOrderedClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPOrderedClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNowaitClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPNowaitClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPVarListClause(
    OpenMPClauseKind Kind, ArrayRef<Expr *> VarList, Expr *TailExpr,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation EndLoc, CXXScopeSpec &ReductionIdScopeSpec,
    const DeclarationNameInfo &ReductionId) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_private:
    Res = ActOnOpenMPPrivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_firstprivate:
    Res = ActOnOpenMPFirstprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_lastprivate:
    Res = ActOnOpenMPLastprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_shared:
    Res = ActOnOpenMPSharedClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_reduction:
    Res = ActOnOpenMPReductionClause(VarList, StartLoc, LParenLoc, ColonLoc,
                                     EndLoc, ReductionIdScopeSpec, ReductionId);
    break;
  case OMPC_linear:
    Res = ActOnOpenMPLinearClause(VarList, TailExpr, StartLoc, LParenLoc,
                                  ColonLoc, EndLoc);
    break;
  case OMPC_aligned:
    Res = ActOnOpenMPAlignedClause(VarList, TailExpr, StartLoc, LParenLoc,
                                   ColonLoc, EndLoc);
    break;
  case OMPC_copyin:
    Res = ActOnOpenMPCopyinClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_copyprivate:
    Res = ActOnOpenMPCopyprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_threadprivate:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPPrivateClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP private clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
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
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accessible, unambiguous default constructor for the
    //  class type.
    while (Type.getNonReferenceType()->isArrayType()) {
      Type = cast<ArrayType>(Type.getNonReferenceType().getTypePtr())
                 ->getElementType();
    }
    CXXRecordDecl *RD = getLangOpts().CPlusPlus
                            ? Type.getNonReferenceType()->getAsCXXRecordDecl()
                            : nullptr;
    // FIXME This code must be replaced by actual constructing/destructing of
    // the private variable.
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
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
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
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
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
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_private);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_private);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPFirstprivateClause(ArrayRef<Expr *> VarList,
                                               SourceLocation StartLoc,
                                               SourceLocation LParenLoc,
                                               SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP firstprivate clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
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
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.3.4, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accessible, unambiguous copy constructor for the
    //  class type.
    Type = Context.getBaseElementType(Type);
    CXXRecordDecl *RD = getLangOpts().CPlusPlus
                            ? Type.getNonReferenceType()->getAsCXXRecordDecl()
                            : nullptr;
    // FIXME This code must be replaced by actual constructing/destructing of
    // the firstprivate variable.
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
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
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
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
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
      if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_firstprivate &&
          DVar.CKind != OMPC_lastprivate && DVar.RefExpr) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_firstprivate);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
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
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }

      OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
      // OpenMP [2.9.3.4, Restrictions, p.2]
      //  A list item that is private within a parallel region must not appear
      //  in a firstprivate clause on a worksharing construct if any of the
      //  worksharing regions arising from the worksharing construct ever bind
      //  to any of the parallel regions arising from the parallel construct.
      if (isOpenMPWorksharingDirective(CurrDir) &&
          !isOpenMPParallelDirective(CurrDir)) {
        DVar = DSAStack->getImplicitDSA(VD);
        if (DVar.CKind != OMPC_shared) {
          Diag(ELoc, diag::err_omp_required_access)
              << getOpenMPClauseName(OMPC_firstprivate)
              << getOpenMPClauseName(OMPC_shared);
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
      }
      // OpenMP [2.9.3.4, Restrictions, p.3]
      //  A list item that appears in a reduction clause of a parallel construct
      //  must not appear in a firstprivate clause on a worksharing or task
      //  construct if any of the worksharing or task regions arising from the
      //  worksharing or task construct ever bind to any of the parallel regions
      //  arising from the parallel construct.
      // TODO
      // OpenMP [2.9.3.4, Restrictions, p.4]
      //  A list item that appears in a reduction clause in worksharing
      //  construct must not appear in a firstprivate clause in a task construct
      //  encountered during execution of any of the worksharing regions arising
      //  from the worksharing construct.
      // TODO
    }

    DSAStack->addDSA(VD, DE, OMPC_firstprivate);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPFirstprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                       Vars);
}

OMPClause *Sema::ActOnOpenMPLastprivateClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP lastprivate clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.3.5, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or structure
    //  element) cannot appear in a lastprivate clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
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

    // OpenMP [2.14.3.5, Restrictions, C/C++, p.2]
    //  A variable that appears in a lastprivate clause must not have an
    //  incomplete type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_lastprivate_incomplete_type)) {
      continue;
    }
    if (Type->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
          << getOpenMPClauseName(OMPC_lastprivate) << Type;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_lastprivate &&
        DVar.CKind != OMPC_firstprivate &&
        (DVar.CKind != OMPC_private || DVar.RefExpr != nullptr)) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(DVar.CKind)
          << getOpenMPClauseName(OMPC_lastprivate);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    // OpenMP [2.14.3.5, Restrictions, p.2]
    // A list item that is private within a parallel region, or that appears in
    // the reduction clause of a parallel construct, must not appear in a
    // lastprivate clause on a worksharing construct if any of the corresponding
    // worksharing regions ever binds to any of the corresponding parallel
    // regions.
    if (isOpenMPWorksharingDirective(CurrDir) &&
        !isOpenMPParallelDirective(CurrDir)) {
      DVar = DSAStack->getImplicitDSA(VD);
      if (DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_lastprivate)
            << getOpenMPClauseName(OMPC_shared);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }
    }
    // OpenMP [2.14.3.5, Restrictions, C++, p.1,2]
    //  A variable of class type (or array thereof) that appears in a
    //  lastprivate clause requires an accessible, unambiguous default
    //  constructor for the class type, unless the list item is also specified
    //  in a firstprivate clause.
    //  A variable of class type (or array thereof) that appears in a
    //  lastprivate clause requires an accessible, unambiguous copy assignment
    //  operator for the class type.
    while (Type.getNonReferenceType()->isArrayType())
      Type = cast<ArrayType>(Type.getNonReferenceType().getTypePtr())
                 ->getElementType();
    CXXRecordDecl *RD = getLangOpts().CPlusPlus
                            ? Type.getNonReferenceType()->getAsCXXRecordDecl()
                            : nullptr;
    // FIXME This code must be replaced by actual copying and destructing of the
    // lastprivate variable.
    if (RD) {
      CXXMethodDecl *MD = LookupCopyingAssignment(RD, 0, false, 0);
      DeclAccessPair FoundDecl = DeclAccessPair::make(MD, MD->getAccess());
      if (MD) {
        if (CheckMemberAccess(ELoc, RD, FoundDecl) == AR_inaccessible ||
            MD->isDeleted()) {
          Diag(ELoc, diag::err_omp_required_method)
              << getOpenMPClauseName(OMPC_lastprivate) << 2;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        }
        MarkFunctionReferenced(ELoc, MD);
        DiagnoseUseOfDecl(MD, ELoc);
      }

      CXXDestructorDecl *DD = RD->getDestructor();
      if (DD) {
        PartialDiagnostic PD =
            PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
        if (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
            DD->isDeleted()) {
          Diag(ELoc, diag::err_omp_required_method)
              << getOpenMPClauseName(OMPC_lastprivate) << 4;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        }
        MarkFunctionReferenced(ELoc, DD);
        DiagnoseUseOfDecl(DD, ELoc);
      }
    }

    if (DVar.CKind != OMPC_firstprivate)
      DSAStack->addDSA(VD, DE, OMPC_lastprivate);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPLastprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                      Vars);
}

OMPClause *Sema::ActOnOpenMPSharedClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP shared clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.3.2, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or structure
    //  element) cannot appear in a shared unless it is a static data member
    //  of a C++ class.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
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
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_shared &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_shared);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_shared);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPSharedClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

namespace {
class DSARefChecker : public StmtVisitor<DSARefChecker, bool> {
  DSAStackTy *Stack;

public:
  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(VD);
      if (DVar.CKind == OMPC_shared && !DVar.RefExpr)
        return false;
      if (DVar.CKind != OMPC_unknown)
        return true;
      DSAStackTy::DSAVarData DVarPrivate =
          Stack->hasDSA(VD, isOpenMPPrivate, MatchesAlways());
      if (DVarPrivate.CKind != OMPC_unknown)
        return true;
      return false;
    }
    return false;
  }
  bool VisitStmt(Stmt *S) {
    for (auto Child : S->children()) {
      if (Child && Visit(Child))
        return true;
    }
    return false;
  }
  explicit DSARefChecker(DSAStackTy *S) : Stack(S) {}
};
} // namespace

OMPClause *Sema::ActOnOpenMPReductionClause(
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec,
    const DeclarationNameInfo &ReductionId) {
  // TODO: Allow scope specification search when 'declare reduction' is
  // supported.
  assert(ReductionIdScopeSpec.isEmpty() &&
         "No support for scoped reduction identifiers yet.");

  auto DN = ReductionId.getName();
  auto OOK = DN.getCXXOverloadedOperator();
  BinaryOperatorKind BOK = BO_Comma;

  // OpenMP [2.14.3.6, reduction clause]
  // C
  // reduction-identifier is either an identifier or one of the following
  // operators: +, -, *,  &, |, ^, && and ||
  // C++
  // reduction-identifier is either an id-expression or one of the following
  // operators: +, -, *, &, |, ^, && and ||
  // FIXME: Only 'min' and 'max' identifiers are supported for now.
  switch (OOK) {
  case OO_Plus:
  case OO_Minus:
    BOK = BO_AddAssign;
    break;
  case OO_Star:
    BOK = BO_MulAssign;
    break;
  case OO_Amp:
    BOK = BO_AndAssign;
    break;
  case OO_Pipe:
    BOK = BO_OrAssign;
    break;
  case OO_Caret:
    BOK = BO_XorAssign;
    break;
  case OO_AmpAmp:
    BOK = BO_LAnd;
    break;
  case OO_PipePipe:
    BOK = BO_LOr;
    break;
  default:
    if (auto II = DN.getAsIdentifierInfo()) {
      if (II->isStr("max"))
        BOK = BO_GT;
      else if (II->isStr("min"))
        BOK = BO_LT;
    }
    break;
  }
  SourceRange ReductionIdRange;
  if (ReductionIdScopeSpec.isValid()) {
    ReductionIdRange.setBegin(ReductionIdScopeSpec.getBeginLoc());
  }
  ReductionIdRange.setEnd(ReductionId.getEndLoc());
  if (BOK == BO_Comma) {
    // Not allowed reduction identifier is found.
    Diag(ReductionId.getLocStart(), diag::err_omp_unknown_reduction_identifier)
        << ReductionIdRange;
    return nullptr;
  }

  SmallVector<Expr *, 8> Vars;
  for (auto RefExpr : VarList) {
    assert(RefExpr && "nullptr expr in OpenMP reduction clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    if (RefExpr->isTypeDependent() || RefExpr->isValueDependent() ||
        RefExpr->isInstantiationDependent() ||
        RefExpr->containsUnexpandedParameterPack()) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    auto ELoc = RefExpr->getExprLoc();
    auto ERange = RefExpr->getSourceRange();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable or array section, subject to the restrictions
    //  specified in Section 2.4 on page 42 and in each of the sections
    // describing clauses and directives for which a list appears.
    // OpenMP  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    auto DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << ERange;
      continue;
    }
    auto D = DE->getDecl();
    auto VD = cast<VarDecl>(D);
    auto Type = VD->getType();
    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_reduction_incomplete_type))
      continue;
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // Arrays may not appear in a reduction clause.
    if (Type.getNonReferenceType()->isArrayType()) {
      Diag(ELoc, diag::err_omp_reduction_type_array) << Type << ERange;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // A list item that appears in a reduction clause must not be
    // const-qualified.
    if (Type.getNonReferenceType().isConstant(Context)) {
      Diag(ELoc, diag::err_omp_const_variable)
          << getOpenMPClauseName(OMPC_reduction) << Type << ERange;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }
    // OpenMP [2.9.3.6, Restrictions, C/C++, p.4]
    //  If a list-item is a reference type then it must bind to the same object
    //  for all threads of the team.
    VarDecl *VDDef = VD->getDefinition();
    if (Type->isReferenceType() && VDDef) {
      DSARefChecker Check(DSAStack);
      if (Check.Visit(VDDef->getInit())) {
        Diag(ELoc, diag::err_omp_reduction_ref_type_arg) << ERange;
        Diag(VDDef->getLocation(), diag::note_defined_here) << VDDef;
        continue;
      }
    }
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // The type of a list item that appears in a reduction clause must be valid
    // for the reduction-identifier. For a max or min reduction in C, the type
    // of the list item must be an allowed arithmetic data type: char, int,
    // float, double, or _Bool, possibly modified with long, short, signed, or
    // unsigned. For a max or min reduction in C++, the type of the list item
    // must be an allowed arithmetic data type: char, wchar_t, int, float,
    // double, or bool, possibly modified with long, short, signed, or unsigned.
    if ((BOK == BO_GT || BOK == BO_LT) &&
        !(Type->isScalarType() ||
          (getLangOpts().CPlusPlus && Type->isArithmeticType()))) {
      Diag(ELoc, diag::err_omp_clause_not_arithmetic_type_arg)
          << getLangOpts().CPlusPlus;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }
    if ((BOK == BO_OrAssign || BOK == BO_AndAssign || BOK == BO_XorAssign) &&
        !getLangOpts().CPlusPlus && Type->isFloatingType()) {
      Diag(ELoc, diag::err_omp_clause_floating_type_arg);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }
    bool Suppress = getDiagnostics().getSuppressAllDiagnostics();
    getDiagnostics().setSuppressAllDiagnostics(true);
    ExprResult ReductionOp =
        BuildBinOp(DSAStack->getCurScope(), ReductionId.getLocStart(), BOK,
                   RefExpr, RefExpr);
    getDiagnostics().setSuppressAllDiagnostics(Suppress);
    if (ReductionOp.isInvalid()) {
      Diag(ELoc, diag::err_omp_reduction_id_not_compatible) << Type
                                                            << ReductionIdRange;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    // OpenMP [2.14.3.6, Restrictions, p.3]
    //  Any number of reduction clauses can be specified on the directive,
    //  but a list item can appear only once in the reduction clauses for that
    //  directive.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD);
    if (DVar.CKind == OMPC_reduction) {
      Diag(ELoc, diag::err_omp_once_referenced)
          << getOpenMPClauseName(OMPC_reduction);
      if (DVar.RefExpr) {
        Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_referenced);
      }
    } else if (DVar.CKind != OMPC_unknown) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(DVar.CKind)
          << getOpenMPClauseName(OMPC_reduction);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    // OpenMP [2.14.3.6, Restrictions, p.1]
    //  A list item that appears in a reduction clause of a worksharing
    //  construct must be shared in the parallel regions to which any of the
    //  worksharing regions arising from the worksharing construct bind.
    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    if (isOpenMPWorksharingDirective(CurrDir) &&
        !isOpenMPParallelDirective(CurrDir)) {
      DVar = DSAStack->getImplicitDSA(VD);
      if (DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_reduction)
            << getOpenMPClauseName(OMPC_shared);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }
    }

    CXXRecordDecl *RD = getLangOpts().CPlusPlus
                            ? Type.getNonReferenceType()->getAsCXXRecordDecl()
                            : nullptr;
    // FIXME This code must be replaced by actual constructing/destructing of
    // the reduction variable.
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
            << getOpenMPClauseName(OMPC_reduction) << 0;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
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
              << getOpenMPClauseName(OMPC_reduction) << 4;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        }
        MarkFunctionReferenced(ELoc, DD);
        DiagnoseUseOfDecl(DD, ELoc);
      }
    }

    DSAStack->addDSA(VD, DE, OMPC_reduction);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPReductionClause::Create(
      Context, StartLoc, LParenLoc, ColonLoc, EndLoc, Vars,
      ReductionIdScopeSpec.getWithLocInContext(Context), ReductionId);
}

OMPClause *Sema::ActOnOpenMPLinearClause(ArrayRef<Expr *> VarList, Expr *Step,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation ColonLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP linear clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    // OpenMP [2.14.3.7, linear clause]
    // A list item that appears in a linear clause is subject to the private
    // clause semantics described in Section 2.14.3.3 on page 159 except as
    // noted. In addition, the value of the new list item on each iteration
    // of the associated loop(s) corresponds to the value of the original
    // list item before entering the construct plus the logical number of
    // the iteration times linear-step.

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }

    VarDecl *VD = cast<VarDecl>(DE->getDecl());

    // OpenMP [2.14.3.7, linear clause]
    //  A list-item cannot appear in more than one linear clause.
    //  A list-item that appears in a linear clause cannot appear in any
    //  other data-sharing attribute clause.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD);
    if (DVar.RefExpr) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_linear);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    QualType QType = VD->getType();
    if (QType->isDependentType() || QType->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      continue;
    }

    // A variable must not have an incomplete type or a reference type.
    if (RequireCompleteType(ELoc, QType,
                            diag::err_omp_linear_incomplete_type)) {
      continue;
    }
    if (QType->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
          << getOpenMPClauseName(OMPC_linear) << QType;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // A list item must not be const-qualified.
    if (QType.isConstant(Context)) {
      Diag(ELoc, diag::err_omp_const_variable)
          << getOpenMPClauseName(OMPC_linear);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // A list item must be of integral or pointer type.
    QType = QType.getUnqualifiedType().getCanonicalType();
    const Type *Ty = QType.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isIntegralType(Context) &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_linear_expected_int_or_ptr) << QType;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_linear);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  Expr *StepExpr = Step;
  if (Step && !Step->isValueDependent() && !Step->isTypeDependent() &&
      !Step->isInstantiationDependent() &&
      !Step->containsUnexpandedParameterPack()) {
    SourceLocation StepLoc = Step->getLocStart();
    ExprResult Val = PerformOpenMPImplicitIntegerConversion(StepLoc, Step);
    if (Val.isInvalid())
      return nullptr;
    StepExpr = Val.get();

    // Warn about zero linear step (it would be probably better specified as
    // making corresponding variables 'const').
    llvm::APSInt Result;
    if (StepExpr->isIntegerConstantExpr(Result, Context) &&
        !Result.isNegative() && !Result.isStrictlyPositive())
      Diag(StepLoc, diag::warn_omp_linear_step_zero) << Vars[0]
                                                     << (Vars.size() > 1);
  }

  return OMPLinearClause::Create(Context, StartLoc, LParenLoc, ColonLoc, EndLoc,
                                 Vars, StepExpr);
}

OMPClause *Sema::ActOnOpenMPAlignedClause(
    ArrayRef<Expr *> VarList, Expr *Alignment, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc) {

  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP aligned clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }

    VarDecl *VD = cast<VarDecl>(DE->getDecl());

    // OpenMP  [2.8.1, simd construct, Restrictions]
    // The type of list items appearing in the aligned clause must be
    // array, pointer, reference to array, or reference to pointer.
    QualType QType = DE->getType()
                         .getNonReferenceType()
                         .getUnqualifiedType()
                         .getCanonicalType();
    const Type *Ty = QType.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isArrayType() &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_aligned_expected_array_or_ptr)
          << QType << getLangOpts().CPlusPlus << RefExpr->getSourceRange();
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP  [2.8.1, simd construct, Restrictions]
    // A list-item cannot appear in more than one aligned clause.
    if (DeclRefExpr *PrevRef = DSAStack->addUniqueAligned(VD, DE)) {
      Diag(ELoc, diag::err_omp_aligned_twice) << RefExpr->getSourceRange();
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(OMPC_aligned);
      continue;
    }

    Vars.push_back(DE);
  }

  // OpenMP [2.8.1, simd construct, Description]
  // The parameter of the aligned clause, alignment, must be a constant
  // positive integer expression.
  // If no optional parameter is specified, implementation-defined default
  // alignments for SIMD instructions on the target platforms are assumed.
  if (Alignment != nullptr) {
    ExprResult AlignResult =
        VerifyPositiveIntegerConstantInClause(Alignment, OMPC_aligned);
    if (AlignResult.isInvalid())
      return nullptr;
    Alignment = AlignResult.get();
  }
  if (Vars.empty())
    return nullptr;

  return OMPAlignedClause::Create(Context, StartLoc, LParenLoc, ColonLoc,
                                  EndLoc, Vars, Alignment);
}

OMPClause *Sema::ActOnOpenMPCopyinClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP copyin clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.4.1, Restrictions, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
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

    // OpenMP [2.14.4.1, Restrictions, C/C++, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    if (!DSAStack->isThreadPrivate(VD)) {
      Diag(ELoc, diag::err_omp_required_access)
          << getOpenMPClauseName(OMPC_copyin)
          << getOpenMPDirectiveName(OMPD_threadprivate);
      continue;
    }

    // OpenMP [2.14.4.1, Restrictions, C/C++, p.2]
    //  A variable of class type (or array thereof) that appears in a
    //  copyin clause requires an accessible, unambiguous copy assignment
    //  operator for the class type.
    Type = Context.getBaseElementType(Type);
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : nullptr;
    // FIXME This code must be replaced by actual assignment of the
    // threadprivate variable.
    if (RD) {
      CXXMethodDecl *MD = LookupCopyingAssignment(RD, 0, false, 0);
      DeclAccessPair FoundDecl = DeclAccessPair::make(MD, MD->getAccess());
      if (MD) {
        if (CheckMemberAccess(ELoc, RD, FoundDecl) == AR_inaccessible ||
            MD->isDeleted()) {
          Diag(ELoc, diag::err_omp_required_method)
              << getOpenMPClauseName(OMPC_copyin) << 2;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        }
        MarkFunctionReferenced(ELoc, MD);
        DiagnoseUseOfDecl(MD, ELoc);
      }
    }

    DSAStack->addDSA(VD, DE, OMPC_copyin);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPCopyinClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPCopyprivateClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP copyprivate clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.4.1, Restrictions, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
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

    // OpenMP [2.14.4.2, Restrictions, p.2]
    //  A list item that appears in a copyprivate clause may not appear in a
    //  private or firstprivate clause on the single construct.
    if (!DSAStack->isThreadPrivate(VD)) {
      auto DVar = DSAStack->getTopDSA(VD);
      if (DVar.CKind != OMPC_copyprivate && DVar.CKind != OMPC_unknown &&
          !(DVar.CKind == OMPC_private && !DVar.RefExpr)) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_copyprivate);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }

      // OpenMP [2.11.4.2, Restrictions, p.1]
      //  All list items that appear in a copyprivate clause must be either
      //  threadprivate or private in the enclosing context.
      if (DVar.CKind == OMPC_unknown) {
        DVar = DSAStack->getImplicitDSA(VD);
        if (DVar.CKind == OMPC_shared) {
          Diag(ELoc, diag::err_omp_required_access)
              << getOpenMPClauseName(OMPC_copyprivate)
              << "threadprivate or private in the enclosing context";
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
      }
    }

    // OpenMP [2.14.4.1, Restrictions, C/C++, p.2]
    //  A variable of class type (or array thereof) that appears in a
    //  copyin clause requires an accessible, unambiguous copy assignment
    //  operator for the class type.
    Type = Context.getBaseElementType(Type);
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : nullptr;
    // FIXME This code must be replaced by actual assignment of the
    // threadprivate variable.
    if (RD) {
      CXXMethodDecl *MD = LookupCopyingAssignment(RD, 0, false, 0);
      DeclAccessPair FoundDecl = DeclAccessPair::make(MD, MD->getAccess());
      if (MD) {
        if (CheckMemberAccess(ELoc, RD, FoundDecl) == AR_inaccessible ||
            MD->isDeleted()) {
          Diag(ELoc, diag::err_omp_required_method)
              << getOpenMPClauseName(OMPC_copyprivate) << 2;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        }
        MarkFunctionReferenced(ELoc, MD);
        DiagnoseUseOfDecl(MD, ELoc);
      }
    }

    // No need to mark vars as copyprivate, they are already threadprivate or
    // implicitly private.
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPCopyprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

#undef DSAStack
