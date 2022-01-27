//===--- SemaOpenMP.cpp - Semantic Analysis for OpenMP constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OpenMP directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Frontend/OpenMP/OMPAssume.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <set>

using namespace clang;
using namespace llvm::omp;

//===----------------------------------------------------------------------===//
// Stack of data-sharing attributes for variables
//===----------------------------------------------------------------------===//

static const Expr *checkMapClauseExpressionBase(
    Sema &SemaRef, Expr *E,
    OMPClauseMappableExprCommon::MappableExprComponentList &CurComponents,
    OpenMPClauseKind CKind, OpenMPDirectiveKind DKind, bool NoDiagnose);

namespace {
/// Default data sharing attributes, which can be applied to directive.
enum DefaultDataSharingAttributes {
  DSA_unspecified = 0,       /// Data sharing attribute not specified.
  DSA_none = 1 << 0,         /// Default data sharing attribute 'none'.
  DSA_shared = 1 << 1,       /// Default data sharing attribute 'shared'.
  DSA_firstprivate = 1 << 2, /// Default data sharing attribute 'firstprivate'.
};

/// Stack for tracking declarations used in OpenMP directives and
/// clauses and their data-sharing attributes.
class DSAStackTy {
public:
  struct DSAVarData {
    OpenMPDirectiveKind DKind = OMPD_unknown;
    OpenMPClauseKind CKind = OMPC_unknown;
    unsigned Modifier = 0;
    const Expr *RefExpr = nullptr;
    DeclRefExpr *PrivateCopy = nullptr;
    SourceLocation ImplicitDSALoc;
    bool AppliedToPointee = false;
    DSAVarData() = default;
    DSAVarData(OpenMPDirectiveKind DKind, OpenMPClauseKind CKind,
               const Expr *RefExpr, DeclRefExpr *PrivateCopy,
               SourceLocation ImplicitDSALoc, unsigned Modifier,
               bool AppliedToPointee)
        : DKind(DKind), CKind(CKind), Modifier(Modifier), RefExpr(RefExpr),
          PrivateCopy(PrivateCopy), ImplicitDSALoc(ImplicitDSALoc),
          AppliedToPointee(AppliedToPointee) {}
  };
  using OperatorOffsetTy =
      llvm::SmallVector<std::pair<Expr *, OverloadedOperatorKind>, 4>;
  using DoacrossDependMapTy =
      llvm::DenseMap<OMPDependClause *, OperatorOffsetTy>;
  /// Kind of the declaration used in the uses_allocators clauses.
  enum class UsesAllocatorsDeclKind {
    /// Predefined allocator
    PredefinedAllocator,
    /// User-defined allocator
    UserDefinedAllocator,
    /// The declaration that represent allocator trait
    AllocatorTrait,
  };

private:
  struct DSAInfo {
    OpenMPClauseKind Attributes = OMPC_unknown;
    unsigned Modifier = 0;
    /// Pointer to a reference expression and a flag which shows that the
    /// variable is marked as lastprivate(true) or not (false).
    llvm::PointerIntPair<const Expr *, 1, bool> RefExpr;
    DeclRefExpr *PrivateCopy = nullptr;
    /// true if the attribute is applied to the pointee, not the variable
    /// itself.
    bool AppliedToPointee = false;
  };
  using DeclSAMapTy = llvm::SmallDenseMap<const ValueDecl *, DSAInfo, 8>;
  using UsedRefMapTy = llvm::SmallDenseMap<const ValueDecl *, const Expr *, 8>;
  using LCDeclInfo = std::pair<unsigned, VarDecl *>;
  using LoopControlVariablesMapTy =
      llvm::SmallDenseMap<const ValueDecl *, LCDeclInfo, 8>;
  /// Struct that associates a component with the clause kind where they are
  /// found.
  struct MappedExprComponentTy {
    OMPClauseMappableExprCommon::MappableExprComponentLists Components;
    OpenMPClauseKind Kind = OMPC_unknown;
  };
  using MappedExprComponentsTy =
      llvm::DenseMap<const ValueDecl *, MappedExprComponentTy>;
  using CriticalsWithHintsTy =
      llvm::StringMap<std::pair<const OMPCriticalDirective *, llvm::APSInt>>;
  struct ReductionData {
    using BOKPtrType = llvm::PointerEmbeddedInt<BinaryOperatorKind, 16>;
    SourceRange ReductionRange;
    llvm::PointerUnion<const Expr *, BOKPtrType> ReductionOp;
    ReductionData() = default;
    void set(BinaryOperatorKind BO, SourceRange RR) {
      ReductionRange = RR;
      ReductionOp = BO;
    }
    void set(const Expr *RefExpr, SourceRange RR) {
      ReductionRange = RR;
      ReductionOp = RefExpr;
    }
  };
  using DeclReductionMapTy =
      llvm::SmallDenseMap<const ValueDecl *, ReductionData, 4>;
  struct DefaultmapInfo {
    OpenMPDefaultmapClauseModifier ImplicitBehavior =
        OMPC_DEFAULTMAP_MODIFIER_unknown;
    SourceLocation SLoc;
    DefaultmapInfo() = default;
    DefaultmapInfo(OpenMPDefaultmapClauseModifier M, SourceLocation Loc)
        : ImplicitBehavior(M), SLoc(Loc) {}
  };

  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    DeclReductionMapTy ReductionMap;
    UsedRefMapTy AlignedMap;
    UsedRefMapTy NontemporalMap;
    MappedExprComponentsTy MappedExprComponents;
    LoopControlVariablesMapTy LCVMap;
    DefaultDataSharingAttributes DefaultAttr = DSA_unspecified;
    SourceLocation DefaultAttrLoc;
    DefaultmapInfo DefaultmapMap[OMPC_DEFAULTMAP_unknown];
    OpenMPDirectiveKind Directive = OMPD_unknown;
    DeclarationNameInfo DirectiveName;
    Scope *CurScope = nullptr;
    DeclContext *Context = nullptr;
    SourceLocation ConstructLoc;
    /// Set of 'depend' clauses with 'sink|source' dependence kind. Required to
    /// get the data (loop counters etc.) about enclosing loop-based construct.
    /// This data is required during codegen.
    DoacrossDependMapTy DoacrossDepends;
    /// First argument (Expr *) contains optional argument of the
    /// 'ordered' clause, the second one is true if the regions has 'ordered'
    /// clause, false otherwise.
    llvm::Optional<std::pair<const Expr *, OMPOrderedClause *>> OrderedRegion;
    unsigned AssociatedLoops = 1;
    bool HasMutipleLoops = false;
    const Decl *PossiblyLoopCounter = nullptr;
    bool NowaitRegion = false;
    bool CancelRegion = false;
    bool LoopStart = false;
    bool BodyComplete = false;
    SourceLocation PrevScanLocation;
    SourceLocation PrevOrderedLocation;
    SourceLocation InnerTeamsRegionLoc;
    /// Reference to the taskgroup task_reduction reference expression.
    Expr *TaskgroupReductionRef = nullptr;
    llvm::DenseSet<QualType> MappedClassesQualTypes;
    SmallVector<Expr *, 4> InnerUsedAllocators;
    llvm::DenseSet<CanonicalDeclPtr<Decl>> ImplicitTaskFirstprivates;
    /// List of globals marked as declare target link in this target region
    /// (isOpenMPTargetExecutionDirective(Directive) == true).
    llvm::SmallVector<DeclRefExpr *, 4> DeclareTargetLinkVarDecls;
    /// List of decls used in inclusive/exclusive clauses of the scan directive.
    llvm::DenseSet<CanonicalDeclPtr<Decl>> UsedInScanDirective;
    llvm::DenseMap<CanonicalDeclPtr<const Decl>, UsesAllocatorsDeclKind>
        UsesAllocatorsDecls;
    Expr *DeclareMapperVar = nullptr;
    SharingMapTy(OpenMPDirectiveKind DKind, DeclarationNameInfo Name,
                 Scope *CurScope, SourceLocation Loc)
        : Directive(DKind), DirectiveName(Name), CurScope(CurScope),
          ConstructLoc(Loc) {}
    SharingMapTy() = default;
  };

  using StackTy = SmallVector<SharingMapTy, 4>;

  /// Stack of used declaration and their data-sharing attributes.
  DeclSAMapTy Threadprivates;
  const FunctionScopeInfo *CurrentNonCapturingFunctionScope = nullptr;
  SmallVector<std::pair<StackTy, const FunctionScopeInfo *>, 4> Stack;
  /// true, if check for DSA must be from parent directive, false, if
  /// from current directive.
  OpenMPClauseKind ClauseKindMode = OMPC_unknown;
  Sema &SemaRef;
  bool ForceCapturing = false;
  /// true if all the variables in the target executable directives must be
  /// captured by reference.
  bool ForceCaptureByReferenceInTargetExecutable = false;
  CriticalsWithHintsTy Criticals;
  unsigned IgnoredStackElements = 0;

  /// Iterators over the stack iterate in order from innermost to outermost
  /// directive.
  using const_iterator = StackTy::const_reverse_iterator;
  const_iterator begin() const {
    return Stack.empty() ? const_iterator()
                         : Stack.back().first.rbegin() + IgnoredStackElements;
  }
  const_iterator end() const {
    return Stack.empty() ? const_iterator() : Stack.back().first.rend();
  }
  using iterator = StackTy::reverse_iterator;
  iterator begin() {
    return Stack.empty() ? iterator()
                         : Stack.back().first.rbegin() + IgnoredStackElements;
  }
  iterator end() {
    return Stack.empty() ? iterator() : Stack.back().first.rend();
  }

  // Convenience operations to get at the elements of the stack.

  bool isStackEmpty() const {
    return Stack.empty() ||
           Stack.back().second != CurrentNonCapturingFunctionScope ||
           Stack.back().first.size() <= IgnoredStackElements;
  }
  size_t getStackSize() const {
    return isStackEmpty() ? 0
                          : Stack.back().first.size() - IgnoredStackElements;
  }

  SharingMapTy *getTopOfStackOrNull() {
    size_t Size = getStackSize();
    if (Size == 0)
      return nullptr;
    return &Stack.back().first[Size - 1];
  }
  const SharingMapTy *getTopOfStackOrNull() const {
    return const_cast<DSAStackTy &>(*this).getTopOfStackOrNull();
  }
  SharingMapTy &getTopOfStack() {
    assert(!isStackEmpty() && "no current directive");
    return *getTopOfStackOrNull();
  }
  const SharingMapTy &getTopOfStack() const {
    return const_cast<DSAStackTy &>(*this).getTopOfStack();
  }

  SharingMapTy *getSecondOnStackOrNull() {
    size_t Size = getStackSize();
    if (Size <= 1)
      return nullptr;
    return &Stack.back().first[Size - 2];
  }
  const SharingMapTy *getSecondOnStackOrNull() const {
    return const_cast<DSAStackTy &>(*this).getSecondOnStackOrNull();
  }

  /// Get the stack element at a certain level (previously returned by
  /// \c getNestingLevel).
  ///
  /// Note that nesting levels count from outermost to innermost, and this is
  /// the reverse of our iteration order where new inner levels are pushed at
  /// the front of the stack.
  SharingMapTy &getStackElemAtLevel(unsigned Level) {
    assert(Level < getStackSize() && "no such stack element");
    return Stack.back().first[Level];
  }
  const SharingMapTy &getStackElemAtLevel(unsigned Level) const {
    return const_cast<DSAStackTy &>(*this).getStackElemAtLevel(Level);
  }

  DSAVarData getDSA(const_iterator &Iter, ValueDecl *D) const;

  /// Checks if the variable is a local for OpenMP region.
  bool isOpenMPLocal(VarDecl *D, const_iterator Iter) const;

  /// Vector of previously declared requires directives
  SmallVector<const OMPRequiresDecl *, 2> RequiresDecls;
  /// omp_allocator_handle_t type.
  QualType OMPAllocatorHandleT;
  /// omp_depend_t type.
  QualType OMPDependT;
  /// omp_event_handle_t type.
  QualType OMPEventHandleT;
  /// omp_alloctrait_t type.
  QualType OMPAlloctraitT;
  /// Expression for the predefined allocators.
  Expr *OMPPredefinedAllocators[OMPAllocateDeclAttr::OMPUserDefinedMemAlloc] = {
      nullptr};
  /// Vector of previously encountered target directives
  SmallVector<SourceLocation, 2> TargetLocations;
  SourceLocation AtomicLocation;
  /// Vector of declare variant construct traits.
  SmallVector<llvm::omp::TraitProperty, 8> ConstructTraits;

public:
  explicit DSAStackTy(Sema &S) : SemaRef(S) {}

  /// Sets omp_allocator_handle_t type.
  void setOMPAllocatorHandleT(QualType Ty) { OMPAllocatorHandleT = Ty; }
  /// Gets omp_allocator_handle_t type.
  QualType getOMPAllocatorHandleT() const { return OMPAllocatorHandleT; }
  /// Sets omp_alloctrait_t type.
  void setOMPAlloctraitT(QualType Ty) { OMPAlloctraitT = Ty; }
  /// Gets omp_alloctrait_t type.
  QualType getOMPAlloctraitT() const { return OMPAlloctraitT; }
  /// Sets the given default allocator.
  void setAllocator(OMPAllocateDeclAttr::AllocatorTypeTy AllocatorKind,
                    Expr *Allocator) {
    OMPPredefinedAllocators[AllocatorKind] = Allocator;
  }
  /// Returns the specified default allocator.
  Expr *getAllocator(OMPAllocateDeclAttr::AllocatorTypeTy AllocatorKind) const {
    return OMPPredefinedAllocators[AllocatorKind];
  }
  /// Sets omp_depend_t type.
  void setOMPDependT(QualType Ty) { OMPDependT = Ty; }
  /// Gets omp_depend_t type.
  QualType getOMPDependT() const { return OMPDependT; }

  /// Sets omp_event_handle_t type.
  void setOMPEventHandleT(QualType Ty) { OMPEventHandleT = Ty; }
  /// Gets omp_event_handle_t type.
  QualType getOMPEventHandleT() const { return OMPEventHandleT; }

  bool isClauseParsingMode() const { return ClauseKindMode != OMPC_unknown; }
  OpenMPClauseKind getClauseParsingMode() const {
    assert(isClauseParsingMode() && "Must be in clause parsing mode.");
    return ClauseKindMode;
  }
  void setClauseParsingMode(OpenMPClauseKind K) { ClauseKindMode = K; }

  bool isBodyComplete() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top && Top->BodyComplete;
  }
  void setBodyComplete() { getTopOfStack().BodyComplete = true; }

  bool isForceVarCapturing() const { return ForceCapturing; }
  void setForceVarCapturing(bool V) { ForceCapturing = V; }

  void setForceCaptureByReferenceInTargetExecutable(bool V) {
    ForceCaptureByReferenceInTargetExecutable = V;
  }
  bool isForceCaptureByReferenceInTargetExecutable() const {
    return ForceCaptureByReferenceInTargetExecutable;
  }

  void push(OpenMPDirectiveKind DKind, const DeclarationNameInfo &DirName,
            Scope *CurScope, SourceLocation Loc) {
    assert(!IgnoredStackElements &&
           "cannot change stack while ignoring elements");
    if (Stack.empty() ||
        Stack.back().second != CurrentNonCapturingFunctionScope)
      Stack.emplace_back(StackTy(), CurrentNonCapturingFunctionScope);
    Stack.back().first.emplace_back(DKind, DirName, CurScope, Loc);
    Stack.back().first.back().DefaultAttrLoc = Loc;
  }

  void pop() {
    assert(!IgnoredStackElements &&
           "cannot change stack while ignoring elements");
    assert(!Stack.back().first.empty() &&
           "Data-sharing attributes stack is empty!");
    Stack.back().first.pop_back();
  }

  /// RAII object to temporarily leave the scope of a directive when we want to
  /// logically operate in its parent.
  class ParentDirectiveScope {
    DSAStackTy &Self;
    bool Active;

  public:
    ParentDirectiveScope(DSAStackTy &Self, bool Activate)
        : Self(Self), Active(false) {
      if (Activate)
        enable();
    }
    ~ParentDirectiveScope() { disable(); }
    void disable() {
      if (Active) {
        --Self.IgnoredStackElements;
        Active = false;
      }
    }
    void enable() {
      if (!Active) {
        ++Self.IgnoredStackElements;
        Active = true;
      }
    }
  };

  /// Marks that we're started loop parsing.
  void loopInit() {
    assert(isOpenMPLoopDirective(getCurrentDirective()) &&
           "Expected loop-based directive.");
    getTopOfStack().LoopStart = true;
  }
  /// Start capturing of the variables in the loop context.
  void loopStart() {
    assert(isOpenMPLoopDirective(getCurrentDirective()) &&
           "Expected loop-based directive.");
    getTopOfStack().LoopStart = false;
  }
  /// true, if variables are captured, false otherwise.
  bool isLoopStarted() const {
    assert(isOpenMPLoopDirective(getCurrentDirective()) &&
           "Expected loop-based directive.");
    return !getTopOfStack().LoopStart;
  }
  /// Marks (or clears) declaration as possibly loop counter.
  void resetPossibleLoopCounter(const Decl *D = nullptr) {
    getTopOfStack().PossiblyLoopCounter = D ? D->getCanonicalDecl() : D;
  }
  /// Gets the possible loop counter decl.
  const Decl *getPossiblyLoopCunter() const {
    return getTopOfStack().PossiblyLoopCounter;
  }
  /// Start new OpenMP region stack in new non-capturing function.
  void pushFunction() {
    assert(!IgnoredStackElements &&
           "cannot change stack while ignoring elements");
    const FunctionScopeInfo *CurFnScope = SemaRef.getCurFunction();
    assert(!isa<CapturingScopeInfo>(CurFnScope));
    CurrentNonCapturingFunctionScope = CurFnScope;
  }
  /// Pop region stack for non-capturing function.
  void popFunction(const FunctionScopeInfo *OldFSI) {
    assert(!IgnoredStackElements &&
           "cannot change stack while ignoring elements");
    if (!Stack.empty() && Stack.back().second == OldFSI) {
      assert(Stack.back().first.empty());
      Stack.pop_back();
    }
    CurrentNonCapturingFunctionScope = nullptr;
    for (const FunctionScopeInfo *FSI : llvm::reverse(SemaRef.FunctionScopes)) {
      if (!isa<CapturingScopeInfo>(FSI)) {
        CurrentNonCapturingFunctionScope = FSI;
        break;
      }
    }
  }

  void addCriticalWithHint(const OMPCriticalDirective *D, llvm::APSInt Hint) {
    Criticals.try_emplace(D->getDirectiveName().getAsString(), D, Hint);
  }
  const std::pair<const OMPCriticalDirective *, llvm::APSInt>
  getCriticalWithHint(const DeclarationNameInfo &Name) const {
    auto I = Criticals.find(Name.getAsString());
    if (I != Criticals.end())
      return I->second;
    return std::make_pair(nullptr, llvm::APSInt());
  }
  /// If 'aligned' declaration for given variable \a D was not seen yet,
  /// add it and return NULL; otherwise return previous occurrence's expression
  /// for diagnostics.
  const Expr *addUniqueAligned(const ValueDecl *D, const Expr *NewDE);
  /// If 'nontemporal' declaration for given variable \a D was not seen yet,
  /// add it and return NULL; otherwise return previous occurrence's expression
  /// for diagnostics.
  const Expr *addUniqueNontemporal(const ValueDecl *D, const Expr *NewDE);

  /// Register specified variable as loop control variable.
  void addLoopControlVariable(const ValueDecl *D, VarDecl *Capture);
  /// Check if the specified variable is a loop control variable for
  /// current region.
  /// \return The index of the loop control variable in the list of associated
  /// for-loops (from outer to inner).
  const LCDeclInfo isLoopControlVariable(const ValueDecl *D) const;
  /// Check if the specified variable is a loop control variable for
  /// parent region.
  /// \return The index of the loop control variable in the list of associated
  /// for-loops (from outer to inner).
  const LCDeclInfo isParentLoopControlVariable(const ValueDecl *D) const;
  /// Check if the specified variable is a loop control variable for
  /// current region.
  /// \return The index of the loop control variable in the list of associated
  /// for-loops (from outer to inner).
  const LCDeclInfo isLoopControlVariable(const ValueDecl *D,
                                         unsigned Level) const;
  /// Get the loop control variable for the I-th loop (or nullptr) in
  /// parent directive.
  const ValueDecl *getParentLoopControlVariable(unsigned I) const;

  /// Marks the specified decl \p D as used in scan directive.
  void markDeclAsUsedInScanDirective(ValueDecl *D) {
    if (SharingMapTy *Stack = getSecondOnStackOrNull())
      Stack->UsedInScanDirective.insert(D);
  }

  /// Checks if the specified declaration was used in the inner scan directive.
  bool isUsedInScanDirective(ValueDecl *D) const {
    if (const SharingMapTy *Stack = getTopOfStackOrNull())
      return Stack->UsedInScanDirective.contains(D);
    return false;
  }

  /// Adds explicit data sharing attribute to the specified declaration.
  void addDSA(const ValueDecl *D, const Expr *E, OpenMPClauseKind A,
              DeclRefExpr *PrivateCopy = nullptr, unsigned Modifier = 0,
              bool AppliedToPointee = false);

  /// Adds additional information for the reduction items with the reduction id
  /// represented as an operator.
  void addTaskgroupReductionData(const ValueDecl *D, SourceRange SR,
                                 BinaryOperatorKind BOK);
  /// Adds additional information for the reduction items with the reduction id
  /// represented as reduction identifier.
  void addTaskgroupReductionData(const ValueDecl *D, SourceRange SR,
                                 const Expr *ReductionRef);
  /// Returns the location and reduction operation from the innermost parent
  /// region for the given \p D.
  const DSAVarData
  getTopMostTaskgroupReductionData(const ValueDecl *D, SourceRange &SR,
                                   BinaryOperatorKind &BOK,
                                   Expr *&TaskgroupDescriptor) const;
  /// Returns the location and reduction operation from the innermost parent
  /// region for the given \p D.
  const DSAVarData
  getTopMostTaskgroupReductionData(const ValueDecl *D, SourceRange &SR,
                                   const Expr *&ReductionRef,
                                   Expr *&TaskgroupDescriptor) const;
  /// Return reduction reference expression for the current taskgroup or
  /// parallel/worksharing directives with task reductions.
  Expr *getTaskgroupReductionRef() const {
    assert((getTopOfStack().Directive == OMPD_taskgroup ||
            ((isOpenMPParallelDirective(getTopOfStack().Directive) ||
              isOpenMPWorksharingDirective(getTopOfStack().Directive)) &&
             !isOpenMPSimdDirective(getTopOfStack().Directive))) &&
           "taskgroup reference expression requested for non taskgroup or "
           "parallel/worksharing directive.");
    return getTopOfStack().TaskgroupReductionRef;
  }
  /// Checks if the given \p VD declaration is actually a taskgroup reduction
  /// descriptor variable at the \p Level of OpenMP regions.
  bool isTaskgroupReductionRef(const ValueDecl *VD, unsigned Level) const {
    return getStackElemAtLevel(Level).TaskgroupReductionRef &&
           cast<DeclRefExpr>(getStackElemAtLevel(Level).TaskgroupReductionRef)
                   ->getDecl() == VD;
  }

  /// Returns data sharing attributes from top of the stack for the
  /// specified declaration.
  const DSAVarData getTopDSA(ValueDecl *D, bool FromParent);
  /// Returns data-sharing attributes for the specified declaration.
  const DSAVarData getImplicitDSA(ValueDecl *D, bool FromParent) const;
  /// Returns data-sharing attributes for the specified declaration.
  const DSAVarData getImplicitDSA(ValueDecl *D, unsigned Level) const;
  /// Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any directive which matches \a DPred
  /// predicate.
  const DSAVarData
  hasDSA(ValueDecl *D,
         const llvm::function_ref<bool(OpenMPClauseKind, bool)> CPred,
         const llvm::function_ref<bool(OpenMPDirectiveKind)> DPred,
         bool FromParent) const;
  /// Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any innermost directive which
  /// matches \a DPred predicate.
  const DSAVarData
  hasInnermostDSA(ValueDecl *D,
                  const llvm::function_ref<bool(OpenMPClauseKind, bool)> CPred,
                  const llvm::function_ref<bool(OpenMPDirectiveKind)> DPred,
                  bool FromParent) const;
  /// Checks if the specified variables has explicit data-sharing
  /// attributes which match specified \a CPred predicate at the specified
  /// OpenMP region.
  bool
  hasExplicitDSA(const ValueDecl *D,
                 const llvm::function_ref<bool(OpenMPClauseKind, bool)> CPred,
                 unsigned Level, bool NotLastprivate = false) const;

  /// Returns true if the directive at level \Level matches in the
  /// specified \a DPred predicate.
  bool hasExplicitDirective(
      const llvm::function_ref<bool(OpenMPDirectiveKind)> DPred,
      unsigned Level) const;

  /// Finds a directive which matches specified \a DPred predicate.
  bool hasDirective(
      const llvm::function_ref<bool(
          OpenMPDirectiveKind, const DeclarationNameInfo &, SourceLocation)>
          DPred,
      bool FromParent) const;

  /// Returns currently analyzed directive.
  OpenMPDirectiveKind getCurrentDirective() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->Directive : OMPD_unknown;
  }
  /// Returns directive kind at specified level.
  OpenMPDirectiveKind getDirective(unsigned Level) const {
    assert(!isStackEmpty() && "No directive at specified level.");
    return getStackElemAtLevel(Level).Directive;
  }
  /// Returns the capture region at the specified level.
  OpenMPDirectiveKind getCaptureRegion(unsigned Level,
                                       unsigned OpenMPCaptureLevel) const {
    SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
    getOpenMPCaptureRegions(CaptureRegions, getDirective(Level));
    return CaptureRegions[OpenMPCaptureLevel];
  }
  /// Returns parent directive.
  OpenMPDirectiveKind getParentDirective() const {
    const SharingMapTy *Parent = getSecondOnStackOrNull();
    return Parent ? Parent->Directive : OMPD_unknown;
  }

  /// Add requires decl to internal vector
  void addRequiresDecl(OMPRequiresDecl *RD) { RequiresDecls.push_back(RD); }

  /// Checks if the defined 'requires' directive has specified type of clause.
  template <typename ClauseType> bool hasRequiresDeclWithClause() const {
    return llvm::any_of(RequiresDecls, [](const OMPRequiresDecl *D) {
      return llvm::any_of(D->clauselists(), [](const OMPClause *C) {
        return isa<ClauseType>(C);
      });
    });
  }

  /// Checks for a duplicate clause amongst previously declared requires
  /// directives
  bool hasDuplicateRequiresClause(ArrayRef<OMPClause *> ClauseList) const {
    bool IsDuplicate = false;
    for (OMPClause *CNew : ClauseList) {
      for (const OMPRequiresDecl *D : RequiresDecls) {
        for (const OMPClause *CPrev : D->clauselists()) {
          if (CNew->getClauseKind() == CPrev->getClauseKind()) {
            SemaRef.Diag(CNew->getBeginLoc(),
                         diag::err_omp_requires_clause_redeclaration)
                << getOpenMPClauseName(CNew->getClauseKind());
            SemaRef.Diag(CPrev->getBeginLoc(),
                         diag::note_omp_requires_previous_clause)
                << getOpenMPClauseName(CPrev->getClauseKind());
            IsDuplicate = true;
          }
        }
      }
    }
    return IsDuplicate;
  }

  /// Add location of previously encountered target to internal vector
  void addTargetDirLocation(SourceLocation LocStart) {
    TargetLocations.push_back(LocStart);
  }

  /// Add location for the first encountered atomicc directive.
  void addAtomicDirectiveLoc(SourceLocation Loc) {
    if (AtomicLocation.isInvalid())
      AtomicLocation = Loc;
  }

  /// Returns the location of the first encountered atomic directive in the
  /// module.
  SourceLocation getAtomicDirectiveLoc() const { return AtomicLocation; }

  // Return previously encountered target region locations.
  ArrayRef<SourceLocation> getEncounteredTargetLocs() const {
    return TargetLocations;
  }

  /// Set default data sharing attribute to none.
  void setDefaultDSANone(SourceLocation Loc) {
    getTopOfStack().DefaultAttr = DSA_none;
    getTopOfStack().DefaultAttrLoc = Loc;
  }
  /// Set default data sharing attribute to shared.
  void setDefaultDSAShared(SourceLocation Loc) {
    getTopOfStack().DefaultAttr = DSA_shared;
    getTopOfStack().DefaultAttrLoc = Loc;
  }
  /// Set default data sharing attribute to firstprivate.
  void setDefaultDSAFirstPrivate(SourceLocation Loc) {
    getTopOfStack().DefaultAttr = DSA_firstprivate;
    getTopOfStack().DefaultAttrLoc = Loc;
  }
  /// Set default data mapping attribute to Modifier:Kind
  void setDefaultDMAAttr(OpenMPDefaultmapClauseModifier M,
                         OpenMPDefaultmapClauseKind Kind, SourceLocation Loc) {
    DefaultmapInfo &DMI = getTopOfStack().DefaultmapMap[Kind];
    DMI.ImplicitBehavior = M;
    DMI.SLoc = Loc;
  }
  /// Check whether the implicit-behavior has been set in defaultmap
  bool checkDefaultmapCategory(OpenMPDefaultmapClauseKind VariableCategory) {
    if (VariableCategory == OMPC_DEFAULTMAP_unknown)
      return getTopOfStack()
                     .DefaultmapMap[OMPC_DEFAULTMAP_aggregate]
                     .ImplicitBehavior != OMPC_DEFAULTMAP_MODIFIER_unknown ||
             getTopOfStack()
                     .DefaultmapMap[OMPC_DEFAULTMAP_scalar]
                     .ImplicitBehavior != OMPC_DEFAULTMAP_MODIFIER_unknown ||
             getTopOfStack()
                     .DefaultmapMap[OMPC_DEFAULTMAP_pointer]
                     .ImplicitBehavior != OMPC_DEFAULTMAP_MODIFIER_unknown;
    return getTopOfStack().DefaultmapMap[VariableCategory].ImplicitBehavior !=
           OMPC_DEFAULTMAP_MODIFIER_unknown;
  }

  ArrayRef<llvm::omp::TraitProperty> getConstructTraits() {
    return ConstructTraits;
  }
  void handleConstructTrait(ArrayRef<llvm::omp::TraitProperty> Traits,
                            bool ScopeEntry) {
    if (ScopeEntry)
      ConstructTraits.append(Traits.begin(), Traits.end());
    else
      for (llvm::omp::TraitProperty Trait : llvm::reverse(Traits)) {
        llvm::omp::TraitProperty Top = ConstructTraits.pop_back_val();
        assert(Top == Trait && "Something left a trait on the stack!");
        (void)Trait;
        (void)Top;
      }
  }

  DefaultDataSharingAttributes getDefaultDSA(unsigned Level) const {
    return getStackSize() <= Level ? DSA_unspecified
                                   : getStackElemAtLevel(Level).DefaultAttr;
  }
  DefaultDataSharingAttributes getDefaultDSA() const {
    return isStackEmpty() ? DSA_unspecified : getTopOfStack().DefaultAttr;
  }
  SourceLocation getDefaultDSALocation() const {
    return isStackEmpty() ? SourceLocation() : getTopOfStack().DefaultAttrLoc;
  }
  OpenMPDefaultmapClauseModifier
  getDefaultmapModifier(OpenMPDefaultmapClauseKind Kind) const {
    return isStackEmpty()
               ? OMPC_DEFAULTMAP_MODIFIER_unknown
               : getTopOfStack().DefaultmapMap[Kind].ImplicitBehavior;
  }
  OpenMPDefaultmapClauseModifier
  getDefaultmapModifierAtLevel(unsigned Level,
                               OpenMPDefaultmapClauseKind Kind) const {
    return getStackElemAtLevel(Level).DefaultmapMap[Kind].ImplicitBehavior;
  }
  bool isDefaultmapCapturedByRef(unsigned Level,
                                 OpenMPDefaultmapClauseKind Kind) const {
    OpenMPDefaultmapClauseModifier M =
        getDefaultmapModifierAtLevel(Level, Kind);
    if (Kind == OMPC_DEFAULTMAP_scalar || Kind == OMPC_DEFAULTMAP_pointer) {
      return (M == OMPC_DEFAULTMAP_MODIFIER_alloc) ||
             (M == OMPC_DEFAULTMAP_MODIFIER_to) ||
             (M == OMPC_DEFAULTMAP_MODIFIER_from) ||
             (M == OMPC_DEFAULTMAP_MODIFIER_tofrom);
    }
    return true;
  }
  static bool mustBeFirstprivateBase(OpenMPDefaultmapClauseModifier M,
                                     OpenMPDefaultmapClauseKind Kind) {
    switch (Kind) {
    case OMPC_DEFAULTMAP_scalar:
    case OMPC_DEFAULTMAP_pointer:
      return (M == OMPC_DEFAULTMAP_MODIFIER_unknown) ||
             (M == OMPC_DEFAULTMAP_MODIFIER_firstprivate) ||
             (M == OMPC_DEFAULTMAP_MODIFIER_default);
    case OMPC_DEFAULTMAP_aggregate:
      return M == OMPC_DEFAULTMAP_MODIFIER_firstprivate;
    default:
      break;
    }
    llvm_unreachable("Unexpected OpenMPDefaultmapClauseKind enum");
  }
  bool mustBeFirstprivateAtLevel(unsigned Level,
                                 OpenMPDefaultmapClauseKind Kind) const {
    OpenMPDefaultmapClauseModifier M =
        getDefaultmapModifierAtLevel(Level, Kind);
    return mustBeFirstprivateBase(M, Kind);
  }
  bool mustBeFirstprivate(OpenMPDefaultmapClauseKind Kind) const {
    OpenMPDefaultmapClauseModifier M = getDefaultmapModifier(Kind);
    return mustBeFirstprivateBase(M, Kind);
  }

  /// Checks if the specified variable is a threadprivate.
  bool isThreadPrivate(VarDecl *D) {
    const DSAVarData DVar = getTopDSA(D, false);
    return isOpenMPThreadPrivate(DVar.CKind);
  }

  /// Marks current region as ordered (it has an 'ordered' clause).
  void setOrderedRegion(bool IsOrdered, const Expr *Param,
                        OMPOrderedClause *Clause) {
    if (IsOrdered)
      getTopOfStack().OrderedRegion.emplace(Param, Clause);
    else
      getTopOfStack().OrderedRegion.reset();
  }
  /// Returns true, if region is ordered (has associated 'ordered' clause),
  /// false - otherwise.
  bool isOrderedRegion() const {
    if (const SharingMapTy *Top = getTopOfStackOrNull())
      return Top->OrderedRegion.hasValue();
    return false;
  }
  /// Returns optional parameter for the ordered region.
  std::pair<const Expr *, OMPOrderedClause *> getOrderedRegionParam() const {
    if (const SharingMapTy *Top = getTopOfStackOrNull())
      if (Top->OrderedRegion.hasValue())
        return Top->OrderedRegion.getValue();
    return std::make_pair(nullptr, nullptr);
  }
  /// Returns true, if parent region is ordered (has associated
  /// 'ordered' clause), false - otherwise.
  bool isParentOrderedRegion() const {
    if (const SharingMapTy *Parent = getSecondOnStackOrNull())
      return Parent->OrderedRegion.hasValue();
    return false;
  }
  /// Returns optional parameter for the ordered region.
  std::pair<const Expr *, OMPOrderedClause *>
  getParentOrderedRegionParam() const {
    if (const SharingMapTy *Parent = getSecondOnStackOrNull())
      if (Parent->OrderedRegion.hasValue())
        return Parent->OrderedRegion.getValue();
    return std::make_pair(nullptr, nullptr);
  }
  /// Marks current region as nowait (it has a 'nowait' clause).
  void setNowaitRegion(bool IsNowait = true) {
    getTopOfStack().NowaitRegion = IsNowait;
  }
  /// Returns true, if parent region is nowait (has associated
  /// 'nowait' clause), false - otherwise.
  bool isParentNowaitRegion() const {
    if (const SharingMapTy *Parent = getSecondOnStackOrNull())
      return Parent->NowaitRegion;
    return false;
  }
  /// Marks parent region as cancel region.
  void setParentCancelRegion(bool Cancel = true) {
    if (SharingMapTy *Parent = getSecondOnStackOrNull())
      Parent->CancelRegion |= Cancel;
  }
  /// Return true if current region has inner cancel construct.
  bool isCancelRegion() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->CancelRegion : false;
  }

  /// Mark that parent region already has scan directive.
  void setParentHasScanDirective(SourceLocation Loc) {
    if (SharingMapTy *Parent = getSecondOnStackOrNull())
      Parent->PrevScanLocation = Loc;
  }
  /// Return true if current region has inner cancel construct.
  bool doesParentHasScanDirective() const {
    const SharingMapTy *Top = getSecondOnStackOrNull();
    return Top ? Top->PrevScanLocation.isValid() : false;
  }
  /// Return true if current region has inner cancel construct.
  SourceLocation getParentScanDirectiveLoc() const {
    const SharingMapTy *Top = getSecondOnStackOrNull();
    return Top ? Top->PrevScanLocation : SourceLocation();
  }
  /// Mark that parent region already has ordered directive.
  void setParentHasOrderedDirective(SourceLocation Loc) {
    if (SharingMapTy *Parent = getSecondOnStackOrNull())
      Parent->PrevOrderedLocation = Loc;
  }
  /// Return true if current region has inner ordered construct.
  bool doesParentHasOrderedDirective() const {
    const SharingMapTy *Top = getSecondOnStackOrNull();
    return Top ? Top->PrevOrderedLocation.isValid() : false;
  }
  /// Returns the location of the previously specified ordered directive.
  SourceLocation getParentOrderedDirectiveLoc() const {
    const SharingMapTy *Top = getSecondOnStackOrNull();
    return Top ? Top->PrevOrderedLocation : SourceLocation();
  }

  /// Set collapse value for the region.
  void setAssociatedLoops(unsigned Val) {
    getTopOfStack().AssociatedLoops = Val;
    if (Val > 1)
      getTopOfStack().HasMutipleLoops = true;
  }
  /// Return collapse value for region.
  unsigned getAssociatedLoops() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->AssociatedLoops : 0;
  }
  /// Returns true if the construct is associated with multiple loops.
  bool hasMutipleLoops() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->HasMutipleLoops : false;
  }

  /// Marks current target region as one with closely nested teams
  /// region.
  void setParentTeamsRegionLoc(SourceLocation TeamsRegionLoc) {
    if (SharingMapTy *Parent = getSecondOnStackOrNull())
      Parent->InnerTeamsRegionLoc = TeamsRegionLoc;
  }
  /// Returns true, if current region has closely nested teams region.
  bool hasInnerTeamsRegion() const {
    return getInnerTeamsRegionLoc().isValid();
  }
  /// Returns location of the nested teams region (if any).
  SourceLocation getInnerTeamsRegionLoc() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->InnerTeamsRegionLoc : SourceLocation();
  }

  Scope *getCurScope() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->CurScope : nullptr;
  }
  void setContext(DeclContext *DC) { getTopOfStack().Context = DC; }
  SourceLocation getConstructLoc() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->ConstructLoc : SourceLocation();
  }

  /// Do the check specified in \a Check to all component lists and return true
  /// if any issue is found.
  bool checkMappableExprComponentListsForDecl(
      const ValueDecl *VD, bool CurrentRegionOnly,
      const llvm::function_ref<
          bool(OMPClauseMappableExprCommon::MappableExprComponentListRef,
               OpenMPClauseKind)>
          Check) const {
    if (isStackEmpty())
      return false;
    auto SI = begin();
    auto SE = end();

    if (SI == SE)
      return false;

    if (CurrentRegionOnly)
      SE = std::next(SI);
    else
      std::advance(SI, 1);

    for (; SI != SE; ++SI) {
      auto MI = SI->MappedExprComponents.find(VD);
      if (MI != SI->MappedExprComponents.end())
        for (OMPClauseMappableExprCommon::MappableExprComponentListRef L :
             MI->second.Components)
          if (Check(L, MI->second.Kind))
            return true;
    }
    return false;
  }

  /// Do the check specified in \a Check to all component lists at a given level
  /// and return true if any issue is found.
  bool checkMappableExprComponentListsForDeclAtLevel(
      const ValueDecl *VD, unsigned Level,
      const llvm::function_ref<
          bool(OMPClauseMappableExprCommon::MappableExprComponentListRef,
               OpenMPClauseKind)>
          Check) const {
    if (getStackSize() <= Level)
      return false;

    const SharingMapTy &StackElem = getStackElemAtLevel(Level);
    auto MI = StackElem.MappedExprComponents.find(VD);
    if (MI != StackElem.MappedExprComponents.end())
      for (OMPClauseMappableExprCommon::MappableExprComponentListRef L :
           MI->second.Components)
        if (Check(L, MI->second.Kind))
          return true;
    return false;
  }

  /// Create a new mappable expression component list associated with a given
  /// declaration and initialize it with the provided list of components.
  void addMappableExpressionComponents(
      const ValueDecl *VD,
      OMPClauseMappableExprCommon::MappableExprComponentListRef Components,
      OpenMPClauseKind WhereFoundClauseKind) {
    MappedExprComponentTy &MEC = getTopOfStack().MappedExprComponents[VD];
    // Create new entry and append the new components there.
    MEC.Components.resize(MEC.Components.size() + 1);
    MEC.Components.back().append(Components.begin(), Components.end());
    MEC.Kind = WhereFoundClauseKind;
  }

  unsigned getNestingLevel() const {
    assert(!isStackEmpty());
    return getStackSize() - 1;
  }
  void addDoacrossDependClause(OMPDependClause *C,
                               const OperatorOffsetTy &OpsOffs) {
    SharingMapTy *Parent = getSecondOnStackOrNull();
    assert(Parent && isOpenMPWorksharingDirective(Parent->Directive));
    Parent->DoacrossDepends.try_emplace(C, OpsOffs);
  }
  llvm::iterator_range<DoacrossDependMapTy::const_iterator>
  getDoacrossDependClauses() const {
    const SharingMapTy &StackElem = getTopOfStack();
    if (isOpenMPWorksharingDirective(StackElem.Directive)) {
      const DoacrossDependMapTy &Ref = StackElem.DoacrossDepends;
      return llvm::make_range(Ref.begin(), Ref.end());
    }
    return llvm::make_range(StackElem.DoacrossDepends.end(),
                            StackElem.DoacrossDepends.end());
  }

  // Store types of classes which have been explicitly mapped
  void addMappedClassesQualTypes(QualType QT) {
    SharingMapTy &StackElem = getTopOfStack();
    StackElem.MappedClassesQualTypes.insert(QT);
  }

  // Return set of mapped classes types
  bool isClassPreviouslyMapped(QualType QT) const {
    const SharingMapTy &StackElem = getTopOfStack();
    return StackElem.MappedClassesQualTypes.contains(QT);
  }

  /// Adds global declare target to the parent target region.
  void addToParentTargetRegionLinkGlobals(DeclRefExpr *E) {
    assert(*OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(
               E->getDecl()) == OMPDeclareTargetDeclAttr::MT_Link &&
           "Expected declare target link global.");
    for (auto &Elem : *this) {
      if (isOpenMPTargetExecutionDirective(Elem.Directive)) {
        Elem.DeclareTargetLinkVarDecls.push_back(E);
        return;
      }
    }
  }

  /// Returns the list of globals with declare target link if current directive
  /// is target.
  ArrayRef<DeclRefExpr *> getLinkGlobals() const {
    assert(isOpenMPTargetExecutionDirective(getCurrentDirective()) &&
           "Expected target executable directive.");
    return getTopOfStack().DeclareTargetLinkVarDecls;
  }

  /// Adds list of allocators expressions.
  void addInnerAllocatorExpr(Expr *E) {
    getTopOfStack().InnerUsedAllocators.push_back(E);
  }
  /// Return list of used allocators.
  ArrayRef<Expr *> getInnerAllocators() const {
    return getTopOfStack().InnerUsedAllocators;
  }
  /// Marks the declaration as implicitly firstprivate nin the task-based
  /// regions.
  void addImplicitTaskFirstprivate(unsigned Level, Decl *D) {
    getStackElemAtLevel(Level).ImplicitTaskFirstprivates.insert(D);
  }
  /// Checks if the decl is implicitly firstprivate in the task-based region.
  bool isImplicitTaskFirstprivate(Decl *D) const {
    return getTopOfStack().ImplicitTaskFirstprivates.contains(D);
  }

  /// Marks decl as used in uses_allocators clause as the allocator.
  void addUsesAllocatorsDecl(const Decl *D, UsesAllocatorsDeclKind Kind) {
    getTopOfStack().UsesAllocatorsDecls.try_emplace(D, Kind);
  }
  /// Checks if specified decl is used in uses allocator clause as the
  /// allocator.
  Optional<UsesAllocatorsDeclKind> isUsesAllocatorsDecl(unsigned Level,
                                                        const Decl *D) const {
    const SharingMapTy &StackElem = getTopOfStack();
    auto I = StackElem.UsesAllocatorsDecls.find(D);
    if (I == StackElem.UsesAllocatorsDecls.end())
      return None;
    return I->getSecond();
  }
  Optional<UsesAllocatorsDeclKind> isUsesAllocatorsDecl(const Decl *D) const {
    const SharingMapTy &StackElem = getTopOfStack();
    auto I = StackElem.UsesAllocatorsDecls.find(D);
    if (I == StackElem.UsesAllocatorsDecls.end())
      return None;
    return I->getSecond();
  }

  void addDeclareMapperVarRef(Expr *Ref) {
    SharingMapTy &StackElem = getTopOfStack();
    StackElem.DeclareMapperVar = Ref;
  }
  const Expr *getDeclareMapperVarRef() const {
    const SharingMapTy *Top = getTopOfStackOrNull();
    return Top ? Top->DeclareMapperVar : nullptr;
  }
};

bool isImplicitTaskingRegion(OpenMPDirectiveKind DKind) {
  return isOpenMPParallelDirective(DKind) || isOpenMPTeamsDirective(DKind);
}

bool isImplicitOrExplicitTaskingRegion(OpenMPDirectiveKind DKind) {
  return isImplicitTaskingRegion(DKind) || isOpenMPTaskingDirective(DKind) ||
         DKind == OMPD_unknown;
}

} // namespace

static const Expr *getExprAsWritten(const Expr *E) {
  if (const auto *FE = dyn_cast<FullExpr>(E))
    E = FE->getSubExpr();

  if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    E = MTE->getSubExpr();

  while (const auto *Binder = dyn_cast<CXXBindTemporaryExpr>(E))
    E = Binder->getSubExpr();

  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    E = ICE->getSubExprAsWritten();
  return E->IgnoreParens();
}

static Expr *getExprAsWritten(Expr *E) {
  return const_cast<Expr *>(getExprAsWritten(const_cast<const Expr *>(E)));
}

static const ValueDecl *getCanonicalDecl(const ValueDecl *D) {
  if (const auto *CED = dyn_cast<OMPCapturedExprDecl>(D))
    if (const auto *ME = dyn_cast<MemberExpr>(getExprAsWritten(CED->getInit())))
      D = ME->getMemberDecl();
  const auto *VD = dyn_cast<VarDecl>(D);
  const auto *FD = dyn_cast<FieldDecl>(D);
  if (VD != nullptr) {
    VD = VD->getCanonicalDecl();
    D = VD;
  } else {
    assert(FD);
    FD = FD->getCanonicalDecl();
    D = FD;
  }
  return D;
}

static ValueDecl *getCanonicalDecl(ValueDecl *D) {
  return const_cast<ValueDecl *>(
      getCanonicalDecl(const_cast<const ValueDecl *>(D)));
}

DSAStackTy::DSAVarData DSAStackTy::getDSA(const_iterator &Iter,
                                          ValueDecl *D) const {
  D = getCanonicalDecl(D);
  auto *VD = dyn_cast<VarDecl>(D);
  const auto *FD = dyn_cast<FieldDecl>(D);
  DSAVarData DVar;
  if (Iter == end()) {
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  File-scope or namespace-scope variables referenced in called routines
    //  in the region are shared unless they appear in a threadprivate
    //  directive.
    if (VD && !VD->isFunctionOrMethodVarDecl() && !isa<ParmVarDecl>(VD))
      DVar.CKind = OMPC_shared;

    // OpenMP [2.9.1.2, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  Variables with static storage duration that are declared in called
    //  routines in the region are shared.
    if (VD && VD->hasGlobalStorage())
      DVar.CKind = OMPC_shared;

    // Non-static data members are shared by default.
    if (FD)
      DVar.CKind = OMPC_shared;

    return DVar;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  // Variables with automatic storage duration that are declared in a scope
  // inside the construct are private.
  if (VD && isOpenMPLocal(VD, Iter) && VD->isLocalVarDecl() &&
      (VD->getStorageClass() == SC_Auto || VD->getStorageClass() == SC_None)) {
    DVar.CKind = OMPC_private;
    return DVar;
  }

  DVar.DKind = Iter->Directive;
  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  if (Iter->SharingMap.count(D)) {
    const DSAInfo &Data = Iter->SharingMap.lookup(D);
    DVar.RefExpr = Data.RefExpr.getPointer();
    DVar.PrivateCopy = Data.PrivateCopy;
    DVar.CKind = Data.Attributes;
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    DVar.Modifier = Data.Modifier;
    DVar.AppliedToPointee = Data.AppliedToPointee;
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
  case DSA_firstprivate:
    if (VD->getStorageDuration() == SD_Static &&
        VD->getDeclContext()->isFileContext()) {
      DVar.CKind = OMPC_unknown;
    } else {
      DVar.CKind = OMPC_firstprivate;
    }
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    return DVar;
  case DSA_unspecified:
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.2]
    //  In a parallel construct, if no default clause is present, these
    //  variables are shared.
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    if ((isOpenMPParallelDirective(DVar.DKind) &&
         !isOpenMPTaskLoopDirective(DVar.DKind)) ||
        isOpenMPTeamsDirective(DVar.DKind)) {
      DVar.CKind = OMPC_shared;
      return DVar;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.4]
    //  In a task construct, if no default clause is present, a variable that in
    //  the enclosing context is determined to be shared by all implicit tasks
    //  bound to the current team is shared.
    if (isOpenMPTaskingDirective(DVar.DKind)) {
      DSAVarData DVarTemp;
      const_iterator I = Iter, E = end();
      do {
        ++I;
        // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables
        // Referenced in a Construct, implicitly determined, p.6]
        //  In a task construct, if no default clause is present, a variable
        //  whose data-sharing attribute is not determined by the rules above is
        //  firstprivate.
        DVarTemp = getDSA(I, D);
        if (DVarTemp.CKind != OMPC_shared) {
          DVar.RefExpr = nullptr;
          DVar.CKind = OMPC_firstprivate;
          return DVar;
        }
      } while (I != E && !isImplicitTaskingRegion(I->Directive));
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
  return getDSA(++Iter, D);
}

const Expr *DSAStackTy::addUniqueAligned(const ValueDecl *D,
                                         const Expr *NewDE) {
  assert(!isStackEmpty() && "Data sharing attributes stack is empty");
  D = getCanonicalDecl(D);
  SharingMapTy &StackElem = getTopOfStack();
  auto It = StackElem.AlignedMap.find(D);
  if (It == StackElem.AlignedMap.end()) {
    assert(NewDE && "Unexpected nullptr expr to be added into aligned map");
    StackElem.AlignedMap[D] = NewDE;
    return nullptr;
  }
  assert(It->second && "Unexpected nullptr expr in the aligned map");
  return It->second;
}

const Expr *DSAStackTy::addUniqueNontemporal(const ValueDecl *D,
                                             const Expr *NewDE) {
  assert(!isStackEmpty() && "Data sharing attributes stack is empty");
  D = getCanonicalDecl(D);
  SharingMapTy &StackElem = getTopOfStack();
  auto It = StackElem.NontemporalMap.find(D);
  if (It == StackElem.NontemporalMap.end()) {
    assert(NewDE && "Unexpected nullptr expr to be added into aligned map");
    StackElem.NontemporalMap[D] = NewDE;
    return nullptr;
  }
  assert(It->second && "Unexpected nullptr expr in the aligned map");
  return It->second;
}

void DSAStackTy::addLoopControlVariable(const ValueDecl *D, VarDecl *Capture) {
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  D = getCanonicalDecl(D);
  SharingMapTy &StackElem = getTopOfStack();
  StackElem.LCVMap.try_emplace(
      D, LCDeclInfo(StackElem.LCVMap.size() + 1, Capture));
}

const DSAStackTy::LCDeclInfo
DSAStackTy::isLoopControlVariable(const ValueDecl *D) const {
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  D = getCanonicalDecl(D);
  const SharingMapTy &StackElem = getTopOfStack();
  auto It = StackElem.LCVMap.find(D);
  if (It != StackElem.LCVMap.end())
    return It->second;
  return {0, nullptr};
}

const DSAStackTy::LCDeclInfo
DSAStackTy::isLoopControlVariable(const ValueDecl *D, unsigned Level) const {
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  D = getCanonicalDecl(D);
  for (unsigned I = Level + 1; I > 0; --I) {
    const SharingMapTy &StackElem = getStackElemAtLevel(I - 1);
    auto It = StackElem.LCVMap.find(D);
    if (It != StackElem.LCVMap.end())
      return It->second;
  }
  return {0, nullptr};
}

const DSAStackTy::LCDeclInfo
DSAStackTy::isParentLoopControlVariable(const ValueDecl *D) const {
  const SharingMapTy *Parent = getSecondOnStackOrNull();
  assert(Parent && "Data-sharing attributes stack is empty");
  D = getCanonicalDecl(D);
  auto It = Parent->LCVMap.find(D);
  if (It != Parent->LCVMap.end())
    return It->second;
  return {0, nullptr};
}

const ValueDecl *DSAStackTy::getParentLoopControlVariable(unsigned I) const {
  const SharingMapTy *Parent = getSecondOnStackOrNull();
  assert(Parent && "Data-sharing attributes stack is empty");
  if (Parent->LCVMap.size() < I)
    return nullptr;
  for (const auto &Pair : Parent->LCVMap)
    if (Pair.second.first == I)
      return Pair.first;
  return nullptr;
}

void DSAStackTy::addDSA(const ValueDecl *D, const Expr *E, OpenMPClauseKind A,
                        DeclRefExpr *PrivateCopy, unsigned Modifier,
                        bool AppliedToPointee) {
  D = getCanonicalDecl(D);
  if (A == OMPC_threadprivate) {
    DSAInfo &Data = Threadprivates[D];
    Data.Attributes = A;
    Data.RefExpr.setPointer(E);
    Data.PrivateCopy = nullptr;
    Data.Modifier = Modifier;
  } else {
    DSAInfo &Data = getTopOfStack().SharingMap[D];
    assert(Data.Attributes == OMPC_unknown || (A == Data.Attributes) ||
           (A == OMPC_firstprivate && Data.Attributes == OMPC_lastprivate) ||
           (A == OMPC_lastprivate && Data.Attributes == OMPC_firstprivate) ||
           (isLoopControlVariable(D).first && A == OMPC_private));
    Data.Modifier = Modifier;
    if (A == OMPC_lastprivate && Data.Attributes == OMPC_firstprivate) {
      Data.RefExpr.setInt(/*IntVal=*/true);
      return;
    }
    const bool IsLastprivate =
        A == OMPC_lastprivate || Data.Attributes == OMPC_lastprivate;
    Data.Attributes = A;
    Data.RefExpr.setPointerAndInt(E, IsLastprivate);
    Data.PrivateCopy = PrivateCopy;
    Data.AppliedToPointee = AppliedToPointee;
    if (PrivateCopy) {
      DSAInfo &Data = getTopOfStack().SharingMap[PrivateCopy->getDecl()];
      Data.Modifier = Modifier;
      Data.Attributes = A;
      Data.RefExpr.setPointerAndInt(PrivateCopy, IsLastprivate);
      Data.PrivateCopy = nullptr;
      Data.AppliedToPointee = AppliedToPointee;
    }
  }
}

/// Build a variable declaration for OpenMP loop iteration variable.
static VarDecl *buildVarDecl(Sema &SemaRef, SourceLocation Loc, QualType Type,
                             StringRef Name, const AttrVec *Attrs = nullptr,
                             DeclRefExpr *OrigRef = nullptr) {
  DeclContext *DC = SemaRef.CurContext;
  IdentifierInfo *II = &SemaRef.PP.getIdentifierTable().get(Name);
  TypeSourceInfo *TInfo = SemaRef.Context.getTrivialTypeSourceInfo(Type, Loc);
  auto *Decl =
      VarDecl::Create(SemaRef.Context, DC, Loc, Loc, II, Type, TInfo, SC_None);
  if (Attrs) {
    for (specific_attr_iterator<AlignedAttr> I(Attrs->begin()), E(Attrs->end());
         I != E; ++I)
      Decl->addAttr(*I);
  }
  Decl->setImplicit();
  if (OrigRef) {
    Decl->addAttr(
        OMPReferencedVarAttr::CreateImplicit(SemaRef.Context, OrigRef));
  }
  return Decl;
}

static DeclRefExpr *buildDeclRefExpr(Sema &S, VarDecl *D, QualType Ty,
                                     SourceLocation Loc,
                                     bool RefersToCapture = false) {
  D->setReferenced();
  D->markUsed(S.Context);
  return DeclRefExpr::Create(S.getASTContext(), NestedNameSpecifierLoc(),
                             SourceLocation(), D, RefersToCapture, Loc, Ty,
                             VK_LValue);
}

void DSAStackTy::addTaskgroupReductionData(const ValueDecl *D, SourceRange SR,
                                           BinaryOperatorKind BOK) {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  assert(
      getTopOfStack().SharingMap[D].Attributes == OMPC_reduction &&
      "Additional reduction info may be specified only for reduction items.");
  ReductionData &ReductionData = getTopOfStack().ReductionMap[D];
  assert(ReductionData.ReductionRange.isInvalid() &&
         (getTopOfStack().Directive == OMPD_taskgroup ||
          ((isOpenMPParallelDirective(getTopOfStack().Directive) ||
            isOpenMPWorksharingDirective(getTopOfStack().Directive)) &&
           !isOpenMPSimdDirective(getTopOfStack().Directive))) &&
         "Additional reduction info may be specified only once for reduction "
         "items.");
  ReductionData.set(BOK, SR);
  Expr *&TaskgroupReductionRef = getTopOfStack().TaskgroupReductionRef;
  if (!TaskgroupReductionRef) {
    VarDecl *VD = buildVarDecl(SemaRef, SR.getBegin(),
                               SemaRef.Context.VoidPtrTy, ".task_red.");
    TaskgroupReductionRef =
        buildDeclRefExpr(SemaRef, VD, SemaRef.Context.VoidPtrTy, SR.getBegin());
  }
}

void DSAStackTy::addTaskgroupReductionData(const ValueDecl *D, SourceRange SR,
                                           const Expr *ReductionRef) {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  assert(
      getTopOfStack().SharingMap[D].Attributes == OMPC_reduction &&
      "Additional reduction info may be specified only for reduction items.");
  ReductionData &ReductionData = getTopOfStack().ReductionMap[D];
  assert(ReductionData.ReductionRange.isInvalid() &&
         (getTopOfStack().Directive == OMPD_taskgroup ||
          ((isOpenMPParallelDirective(getTopOfStack().Directive) ||
            isOpenMPWorksharingDirective(getTopOfStack().Directive)) &&
           !isOpenMPSimdDirective(getTopOfStack().Directive))) &&
         "Additional reduction info may be specified only once for reduction "
         "items.");
  ReductionData.set(ReductionRef, SR);
  Expr *&TaskgroupReductionRef = getTopOfStack().TaskgroupReductionRef;
  if (!TaskgroupReductionRef) {
    VarDecl *VD = buildVarDecl(SemaRef, SR.getBegin(),
                               SemaRef.Context.VoidPtrTy, ".task_red.");
    TaskgroupReductionRef =
        buildDeclRefExpr(SemaRef, VD, SemaRef.Context.VoidPtrTy, SR.getBegin());
  }
}

const DSAStackTy::DSAVarData DSAStackTy::getTopMostTaskgroupReductionData(
    const ValueDecl *D, SourceRange &SR, BinaryOperatorKind &BOK,
    Expr *&TaskgroupDescriptor) const {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty.");
  for (const_iterator I = begin() + 1, E = end(); I != E; ++I) {
    const DSAInfo &Data = I->SharingMap.lookup(D);
    if (Data.Attributes != OMPC_reduction ||
        Data.Modifier != OMPC_REDUCTION_task)
      continue;
    const ReductionData &ReductionData = I->ReductionMap.lookup(D);
    if (!ReductionData.ReductionOp ||
        ReductionData.ReductionOp.is<const Expr *>())
      return DSAVarData();
    SR = ReductionData.ReductionRange;
    BOK = ReductionData.ReductionOp.get<ReductionData::BOKPtrType>();
    assert(I->TaskgroupReductionRef && "taskgroup reduction reference "
                                       "expression for the descriptor is not "
                                       "set.");
    TaskgroupDescriptor = I->TaskgroupReductionRef;
    return DSAVarData(I->Directive, OMPC_reduction, Data.RefExpr.getPointer(),
                      Data.PrivateCopy, I->DefaultAttrLoc, OMPC_REDUCTION_task,
                      /*AppliedToPointee=*/false);
  }
  return DSAVarData();
}

const DSAStackTy::DSAVarData DSAStackTy::getTopMostTaskgroupReductionData(
    const ValueDecl *D, SourceRange &SR, const Expr *&ReductionRef,
    Expr *&TaskgroupDescriptor) const {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty.");
  for (const_iterator I = begin() + 1, E = end(); I != E; ++I) {
    const DSAInfo &Data = I->SharingMap.lookup(D);
    if (Data.Attributes != OMPC_reduction ||
        Data.Modifier != OMPC_REDUCTION_task)
      continue;
    const ReductionData &ReductionData = I->ReductionMap.lookup(D);
    if (!ReductionData.ReductionOp ||
        !ReductionData.ReductionOp.is<const Expr *>())
      return DSAVarData();
    SR = ReductionData.ReductionRange;
    ReductionRef = ReductionData.ReductionOp.get<const Expr *>();
    assert(I->TaskgroupReductionRef && "taskgroup reduction reference "
                                       "expression for the descriptor is not "
                                       "set.");
    TaskgroupDescriptor = I->TaskgroupReductionRef;
    return DSAVarData(I->Directive, OMPC_reduction, Data.RefExpr.getPointer(),
                      Data.PrivateCopy, I->DefaultAttrLoc, OMPC_REDUCTION_task,
                      /*AppliedToPointee=*/false);
  }
  return DSAVarData();
}

bool DSAStackTy::isOpenMPLocal(VarDecl *D, const_iterator I) const {
  D = D->getCanonicalDecl();
  for (const_iterator E = end(); I != E; ++I) {
    if (isImplicitOrExplicitTaskingRegion(I->Directive) ||
        isOpenMPTargetExecutionDirective(I->Directive)) {
      if (I->CurScope) {
        Scope *TopScope = I->CurScope->getParent();
        Scope *CurScope = getCurScope();
        while (CurScope && CurScope != TopScope && !CurScope->isDeclScope(D))
          CurScope = CurScope->getParent();
        return CurScope != TopScope;
      }
      for (DeclContext *DC = D->getDeclContext(); DC; DC = DC->getParent())
        if (I->Context == DC)
          return true;
      return false;
    }
  }
  return false;
}

static bool isConstNotMutableType(Sema &SemaRef, QualType Type,
                                  bool AcceptIfMutable = true,
                                  bool *IsClassType = nullptr) {
  ASTContext &Context = SemaRef.getASTContext();
  Type = Type.getNonReferenceType().getCanonicalType();
  bool IsConstant = Type.isConstant(Context);
  Type = Context.getBaseElementType(Type);
  const CXXRecordDecl *RD = AcceptIfMutable && SemaRef.getLangOpts().CPlusPlus
                                ? Type->getAsCXXRecordDecl()
                                : nullptr;
  if (const auto *CTSD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(RD))
    if (const ClassTemplateDecl *CTD = CTSD->getSpecializedTemplate())
      RD = CTD->getTemplatedDecl();
  if (IsClassType)
    *IsClassType = RD;
  return IsConstant && !(SemaRef.getLangOpts().CPlusPlus && RD &&
                         RD->hasDefinition() && RD->hasMutableFields());
}

static bool rejectConstNotMutableType(Sema &SemaRef, const ValueDecl *D,
                                      QualType Type, OpenMPClauseKind CKind,
                                      SourceLocation ELoc,
                                      bool AcceptIfMutable = true,
                                      bool ListItemNotVar = false) {
  ASTContext &Context = SemaRef.getASTContext();
  bool IsClassType;
  if (isConstNotMutableType(SemaRef, Type, AcceptIfMutable, &IsClassType)) {
    unsigned Diag = ListItemNotVar ? diag::err_omp_const_list_item
                    : IsClassType  ? diag::err_omp_const_not_mutable_variable
                                   : diag::err_omp_const_variable;
    SemaRef.Diag(ELoc, Diag) << getOpenMPClauseName(CKind);
    if (!ListItemNotVar && D) {
      const VarDecl *VD = dyn_cast<VarDecl>(D);
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      SemaRef.Diag(D->getLocation(),
                   IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
    }
    return true;
  }
  return false;
}

const DSAStackTy::DSAVarData DSAStackTy::getTopDSA(ValueDecl *D,
                                                   bool FromParent) {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  auto *VD = dyn_cast<VarDecl>(D);
  auto TI = Threadprivates.find(D);
  if (TI != Threadprivates.end()) {
    DVar.RefExpr = TI->getSecond().RefExpr.getPointer();
    DVar.CKind = OMPC_threadprivate;
    DVar.Modifier = TI->getSecond().Modifier;
    return DVar;
  }
  if (VD && VD->hasAttr<OMPThreadPrivateDeclAttr>()) {
    DVar.RefExpr = buildDeclRefExpr(
        SemaRef, VD, D->getType().getNonReferenceType(),
        VD->getAttr<OMPThreadPrivateDeclAttr>()->getLocation());
    DVar.CKind = OMPC_threadprivate;
    addDSA(D, DVar.RefExpr, OMPC_threadprivate);
    return DVar;
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  //  Variables appearing in threadprivate directives are threadprivate.
  if ((VD && VD->getTLSKind() != VarDecl::TLS_None &&
       !(VD->hasAttr<OMPThreadPrivateDeclAttr>() &&
         SemaRef.getLangOpts().OpenMPUseTLS &&
         SemaRef.getASTContext().getTargetInfo().isTLSSupported())) ||
      (VD && VD->getStorageClass() == SC_Register &&
       VD->hasAttr<AsmLabelAttr>() && !VD->isLocalVarDecl())) {
    DVar.RefExpr = buildDeclRefExpr(
        SemaRef, VD, D->getType().getNonReferenceType(), D->getLocation());
    DVar.CKind = OMPC_threadprivate;
    addDSA(D, DVar.RefExpr, OMPC_threadprivate);
    return DVar;
  }
  if (SemaRef.getLangOpts().OpenMPCUDAMode && VD &&
      VD->isLocalVarDeclOrParm() && !isStackEmpty() &&
      !isLoopControlVariable(D).first) {
    const_iterator IterTarget =
        std::find_if(begin(), end(), [](const SharingMapTy &Data) {
          return isOpenMPTargetExecutionDirective(Data.Directive);
        });
    if (IterTarget != end()) {
      const_iterator ParentIterTarget = IterTarget + 1;
      for (const_iterator Iter = begin(); Iter != ParentIterTarget; ++Iter) {
        if (isOpenMPLocal(VD, Iter)) {
          DVar.RefExpr =
              buildDeclRefExpr(SemaRef, VD, D->getType().getNonReferenceType(),
                               D->getLocation());
          DVar.CKind = OMPC_threadprivate;
          return DVar;
        }
      }
      if (!isClauseParsingMode() || IterTarget != begin()) {
        auto DSAIter = IterTarget->SharingMap.find(D);
        if (DSAIter != IterTarget->SharingMap.end() &&
            isOpenMPPrivate(DSAIter->getSecond().Attributes)) {
          DVar.RefExpr = DSAIter->getSecond().RefExpr.getPointer();
          DVar.CKind = OMPC_threadprivate;
          return DVar;
        }
        const_iterator End = end();
        if (!SemaRef.isOpenMPCapturedByRef(D,
                                           std::distance(ParentIterTarget, End),
                                           /*OpenMPCaptureLevel=*/0)) {
          DVar.RefExpr =
              buildDeclRefExpr(SemaRef, VD, D->getType().getNonReferenceType(),
                               IterTarget->ConstructLoc);
          DVar.CKind = OMPC_threadprivate;
          return DVar;
        }
      }
    }
  }

  if (isStackEmpty())
    // Not in OpenMP execution region and top scope was already checked.
    return DVar;

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.4]
  //  Static data members are shared.
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.7]
  //  Variables with static storage duration that are declared in a scope
  //  inside the construct are shared.
  if (VD && VD->isStaticDataMember()) {
    // Check for explicitly specified attributes.
    const_iterator I = begin();
    const_iterator EndI = end();
    if (FromParent && I != EndI)
      ++I;
    if (I != EndI) {
      auto It = I->SharingMap.find(D);
      if (It != I->SharingMap.end()) {
        const DSAInfo &Data = It->getSecond();
        DVar.RefExpr = Data.RefExpr.getPointer();
        DVar.PrivateCopy = Data.PrivateCopy;
        DVar.CKind = Data.Attributes;
        DVar.ImplicitDSALoc = I->DefaultAttrLoc;
        DVar.DKind = I->Directive;
        DVar.Modifier = Data.Modifier;
        DVar.AppliedToPointee = Data.AppliedToPointee;
        return DVar;
      }
    }

    DVar.CKind = OMPC_shared;
    return DVar;
  }

  auto &&MatchesAlways = [](OpenMPDirectiveKind) { return true; };
  // The predetermined shared attribute for const-qualified types having no
  // mutable members was removed after OpenMP 3.1.
  if (SemaRef.LangOpts.OpenMP <= 31) {
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, C/C++, predetermined, p.6]
    //  Variables with const qualified type having no mutable member are
    //  shared.
    if (isConstNotMutableType(SemaRef, D->getType())) {
      // Variables with const-qualified type having no mutable member may be
      // listed in a firstprivate clause, even if they are static data members.
      DSAVarData DVarTemp = hasInnermostDSA(
          D,
          [](OpenMPClauseKind C, bool) {
            return C == OMPC_firstprivate || C == OMPC_shared;
          },
          MatchesAlways, FromParent);
      if (DVarTemp.CKind != OMPC_unknown && DVarTemp.RefExpr)
        return DVarTemp;

      DVar.CKind = OMPC_shared;
      return DVar;
    }
  }

  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  const_iterator I = begin();
  const_iterator EndI = end();
  if (FromParent && I != EndI)
    ++I;
  if (I == EndI)
    return DVar;
  auto It = I->SharingMap.find(D);
  if (It != I->SharingMap.end()) {
    const DSAInfo &Data = It->getSecond();
    DVar.RefExpr = Data.RefExpr.getPointer();
    DVar.PrivateCopy = Data.PrivateCopy;
    DVar.CKind = Data.Attributes;
    DVar.ImplicitDSALoc = I->DefaultAttrLoc;
    DVar.DKind = I->Directive;
    DVar.Modifier = Data.Modifier;
    DVar.AppliedToPointee = Data.AppliedToPointee;
  }

  return DVar;
}

const DSAStackTy::DSAVarData DSAStackTy::getImplicitDSA(ValueDecl *D,
                                                        bool FromParent) const {
  if (isStackEmpty()) {
    const_iterator I;
    return getDSA(I, D);
  }
  D = getCanonicalDecl(D);
  const_iterator StartI = begin();
  const_iterator EndI = end();
  if (FromParent && StartI != EndI)
    ++StartI;
  return getDSA(StartI, D);
}

const DSAStackTy::DSAVarData DSAStackTy::getImplicitDSA(ValueDecl *D,
                                                        unsigned Level) const {
  if (getStackSize() <= Level)
    return DSAVarData();
  D = getCanonicalDecl(D);
  const_iterator StartI = std::next(begin(), getStackSize() - 1 - Level);
  return getDSA(StartI, D);
}

const DSAStackTy::DSAVarData
DSAStackTy::hasDSA(ValueDecl *D,
                   const llvm::function_ref<bool(OpenMPClauseKind, bool)> CPred,
                   const llvm::function_ref<bool(OpenMPDirectiveKind)> DPred,
                   bool FromParent) const {
  if (isStackEmpty())
    return {};
  D = getCanonicalDecl(D);
  const_iterator I = begin();
  const_iterator EndI = end();
  if (FromParent && I != EndI)
    ++I;
  for (; I != EndI; ++I) {
    if (!DPred(I->Directive) &&
        !isImplicitOrExplicitTaskingRegion(I->Directive))
      continue;
    const_iterator NewI = I;
    DSAVarData DVar = getDSA(NewI, D);
    if (I == NewI && CPred(DVar.CKind, DVar.AppliedToPointee))
      return DVar;
  }
  return {};
}

const DSAStackTy::DSAVarData DSAStackTy::hasInnermostDSA(
    ValueDecl *D, const llvm::function_ref<bool(OpenMPClauseKind, bool)> CPred,
    const llvm::function_ref<bool(OpenMPDirectiveKind)> DPred,
    bool FromParent) const {
  if (isStackEmpty())
    return {};
  D = getCanonicalDecl(D);
  const_iterator StartI = begin();
  const_iterator EndI = end();
  if (FromParent && StartI != EndI)
    ++StartI;
  if (StartI == EndI || !DPred(StartI->Directive))
    return {};
  const_iterator NewI = StartI;
  DSAVarData DVar = getDSA(NewI, D);
  return (NewI == StartI && CPred(DVar.CKind, DVar.AppliedToPointee))
             ? DVar
             : DSAVarData();
}

bool DSAStackTy::hasExplicitDSA(
    const ValueDecl *D,
    const llvm::function_ref<bool(OpenMPClauseKind, bool)> CPred,
    unsigned Level, bool NotLastprivate) const {
  if (getStackSize() <= Level)
    return false;
  D = getCanonicalDecl(D);
  const SharingMapTy &StackElem = getStackElemAtLevel(Level);
  auto I = StackElem.SharingMap.find(D);
  if (I != StackElem.SharingMap.end() && I->getSecond().RefExpr.getPointer() &&
      CPred(I->getSecond().Attributes, I->getSecond().AppliedToPointee) &&
      (!NotLastprivate || !I->getSecond().RefExpr.getInt()))
    return true;
  // Check predetermined rules for the loop control variables.
  auto LI = StackElem.LCVMap.find(D);
  if (LI != StackElem.LCVMap.end())
    return CPred(OMPC_private, /*AppliedToPointee=*/false);
  return false;
}

bool DSAStackTy::hasExplicitDirective(
    const llvm::function_ref<bool(OpenMPDirectiveKind)> DPred,
    unsigned Level) const {
  if (getStackSize() <= Level)
    return false;
  const SharingMapTy &StackElem = getStackElemAtLevel(Level);
  return DPred(StackElem.Directive);
}

bool DSAStackTy::hasDirective(
    const llvm::function_ref<bool(OpenMPDirectiveKind,
                                  const DeclarationNameInfo &, SourceLocation)>
        DPred,
    bool FromParent) const {
  // We look only in the enclosing region.
  size_t Skip = FromParent ? 2 : 1;
  for (const_iterator I = begin() + std::min(Skip, getStackSize()), E = end();
       I != E; ++I) {
    if (DPred(I->Directive, I->DirectiveName, I->ConstructLoc))
      return true;
  }
  return false;
}

void Sema::InitDataSharingAttributesStack() {
  VarDataSharingAttributesStack = new DSAStackTy(*this);
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStack)

void Sema::pushOpenMPFunctionRegion() { DSAStack->pushFunction(); }

void Sema::popOpenMPFunctionRegion(const FunctionScopeInfo *OldFSI) {
  DSAStack->popFunction(OldFSI);
}

static bool isOpenMPDeviceDelayedContext(Sema &S) {
  assert(S.LangOpts.OpenMP && S.LangOpts.OpenMPIsDevice &&
         "Expected OpenMP device compilation.");
  return !S.isInOpenMPTargetExecutionDirective();
}

namespace {
/// Status of the function emission on the host/device.
enum class FunctionEmissionStatus {
  Emitted,
  Discarded,
  Unknown,
};
} // anonymous namespace

Sema::SemaDiagnosticBuilder Sema::diagIfOpenMPDeviceCode(SourceLocation Loc,
                                                         unsigned DiagID,
                                                         FunctionDecl *FD) {
  assert(LangOpts.OpenMP && LangOpts.OpenMPIsDevice &&
         "Expected OpenMP device compilation.");

  SemaDiagnosticBuilder::Kind Kind = SemaDiagnosticBuilder::K_Nop;
  if (FD) {
    FunctionEmissionStatus FES = getEmissionStatus(FD);
    switch (FES) {
    case FunctionEmissionStatus::Emitted:
      Kind = SemaDiagnosticBuilder::K_Immediate;
      break;
    case FunctionEmissionStatus::Unknown:
      // TODO: We should always delay diagnostics here in case a target
      //       region is in a function we do not emit. However, as the
      //       current diagnostics are associated with the function containing
      //       the target region and we do not emit that one, we would miss out
      //       on diagnostics for the target region itself. We need to anchor
      //       the diagnostics with the new generated function *or* ensure we
      //       emit diagnostics associated with the surrounding function.
      Kind = isOpenMPDeviceDelayedContext(*this)
                 ? SemaDiagnosticBuilder::K_Deferred
                 : SemaDiagnosticBuilder::K_Immediate;
      break;
    case FunctionEmissionStatus::TemplateDiscarded:
    case FunctionEmissionStatus::OMPDiscarded:
      Kind = SemaDiagnosticBuilder::K_Nop;
      break;
    case FunctionEmissionStatus::CUDADiscarded:
      llvm_unreachable("CUDADiscarded unexpected in OpenMP device compilation");
      break;
    }
  }

  return SemaDiagnosticBuilder(Kind, Loc, DiagID, FD, *this);
}

Sema::SemaDiagnosticBuilder Sema::diagIfOpenMPHostCode(SourceLocation Loc,
                                                       unsigned DiagID,
                                                       FunctionDecl *FD) {
  assert(LangOpts.OpenMP && !LangOpts.OpenMPIsDevice &&
         "Expected OpenMP host compilation.");

  SemaDiagnosticBuilder::Kind Kind = SemaDiagnosticBuilder::K_Nop;
  if (FD) {
    FunctionEmissionStatus FES = getEmissionStatus(FD);
    switch (FES) {
    case FunctionEmissionStatus::Emitted:
      Kind = SemaDiagnosticBuilder::K_Immediate;
      break;
    case FunctionEmissionStatus::Unknown:
      Kind = SemaDiagnosticBuilder::K_Deferred;
      break;
    case FunctionEmissionStatus::TemplateDiscarded:
    case FunctionEmissionStatus::OMPDiscarded:
    case FunctionEmissionStatus::CUDADiscarded:
      Kind = SemaDiagnosticBuilder::K_Nop;
      break;
    }
  }

  return SemaDiagnosticBuilder(Kind, Loc, DiagID, FD, *this);
}

static OpenMPDefaultmapClauseKind
getVariableCategoryFromDecl(const LangOptions &LO, const ValueDecl *VD) {
  if (LO.OpenMP <= 45) {
    if (VD->getType().getNonReferenceType()->isScalarType())
      return OMPC_DEFAULTMAP_scalar;
    return OMPC_DEFAULTMAP_aggregate;
  }
  if (VD->getType().getNonReferenceType()->isAnyPointerType())
    return OMPC_DEFAULTMAP_pointer;
  if (VD->getType().getNonReferenceType()->isScalarType())
    return OMPC_DEFAULTMAP_scalar;
  return OMPC_DEFAULTMAP_aggregate;
}

bool Sema::isOpenMPCapturedByRef(const ValueDecl *D, unsigned Level,
                                 unsigned OpenMPCaptureLevel) const {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");

  ASTContext &Ctx = getASTContext();
  bool IsByRef = true;

  // Find the directive that is associated with the provided scope.
  D = cast<ValueDecl>(D->getCanonicalDecl());
  QualType Ty = D->getType();

  bool IsVariableUsedInMapClause = false;
  if (DSAStack->hasExplicitDirective(isOpenMPTargetExecutionDirective, Level)) {
    // This table summarizes how a given variable should be passed to the device
    // given its type and the clauses where it appears. This table is based on
    // the description in OpenMP 4.5 [2.10.4, target Construct] and
    // OpenMP 4.5 [2.15.5, Data-mapping Attribute Rules and Clauses].
    //
    // =========================================================================
    // | type |  defaultmap   | pvt | first | is_device_ptr |    map   | res.  |
    // |      |(tofrom:scalar)|     |  pvt  |               |          |       |
    // =========================================================================
    // | scl  |               |     |       |       -       |          | bycopy|
    // | scl  |               |  -  |   x   |       -       |     -    | bycopy|
    // | scl  |               |  x  |   -   |       -       |     -    | null  |
    // | scl  |       x       |     |       |       -       |          | byref |
    // | scl  |       x       |  -  |   x   |       -       |     -    | bycopy|
    // | scl  |       x       |  x  |   -   |       -       |     -    | null  |
    // | scl  |               |  -  |   -   |       -       |     x    | byref |
    // | scl  |       x       |  -  |   -   |       -       |     x    | byref |
    //
    // | agg  |      n.a.     |     |       |       -       |          | byref |
    // | agg  |      n.a.     |  -  |   x   |       -       |     -    | byref |
    // | agg  |      n.a.     |  x  |   -   |       -       |     -    | null  |
    // | agg  |      n.a.     |  -  |   -   |       -       |     x    | byref |
    // | agg  |      n.a.     |  -  |   -   |       -       |    x[]   | byref |
    //
    // | ptr  |      n.a.     |     |       |       -       |          | bycopy|
    // | ptr  |      n.a.     |  -  |   x   |       -       |     -    | bycopy|
    // | ptr  |      n.a.     |  x  |   -   |       -       |     -    | null  |
    // | ptr  |      n.a.     |  -  |   -   |       -       |     x    | byref |
    // | ptr  |      n.a.     |  -  |   -   |       -       |    x[]   | bycopy|
    // | ptr  |      n.a.     |  -  |   -   |       x       |          | bycopy|
    // | ptr  |      n.a.     |  -  |   -   |       x       |     x    | bycopy|
    // | ptr  |      n.a.     |  -  |   -   |       x       |    x[]   | bycopy|
    // =========================================================================
    // Legend:
    //  scl - scalar
    //  ptr - pointer
    //  agg - aggregate
    //  x - applies
    //  - - invalid in this combination
    //  [] - mapped with an array section
    //  byref - should be mapped by reference
    //  byval - should be mapped by value
    //  null - initialize a local variable to null on the device
    //
    // Observations:
    //  - All scalar declarations that show up in a map clause have to be passed
    //    by reference, because they may have been mapped in the enclosing data
    //    environment.
    //  - If the scalar value does not fit the size of uintptr, it has to be
    //    passed by reference, regardless the result in the table above.
    //  - For pointers mapped by value that have either an implicit map or an
    //    array section, the runtime library may pass the NULL value to the
    //    device instead of the value passed to it by the compiler.

    if (Ty->isReferenceType())
      Ty = Ty->castAs<ReferenceType>()->getPointeeType();

    // Locate map clauses and see if the variable being captured is referred to
    // in any of those clauses. Here we only care about variables, not fields,
    // because fields are part of aggregates.
    bool IsVariableAssociatedWithSection = false;

    DSAStack->checkMappableExprComponentListsForDeclAtLevel(
        D, Level,
        [&IsVariableUsedInMapClause, &IsVariableAssociatedWithSection,
         D](OMPClauseMappableExprCommon::MappableExprComponentListRef
                MapExprComponents,
            OpenMPClauseKind WhereFoundClauseKind) {
          // Only the map clause information influences how a variable is
          // captured. E.g. is_device_ptr does not require changing the default
          // behavior.
          if (WhereFoundClauseKind != OMPC_map)
            return false;

          auto EI = MapExprComponents.rbegin();
          auto EE = MapExprComponents.rend();

          assert(EI != EE && "Invalid map expression!");

          if (isa<DeclRefExpr>(EI->getAssociatedExpression()))
            IsVariableUsedInMapClause |= EI->getAssociatedDeclaration() == D;

          ++EI;
          if (EI == EE)
            return false;

          if (isa<ArraySubscriptExpr>(EI->getAssociatedExpression()) ||
              isa<OMPArraySectionExpr>(EI->getAssociatedExpression()) ||
              isa<MemberExpr>(EI->getAssociatedExpression()) ||
              isa<OMPArrayShapingExpr>(EI->getAssociatedExpression())) {
            IsVariableAssociatedWithSection = true;
            // There is nothing more we need to know about this variable.
            return true;
          }

          // Keep looking for more map info.
          return false;
        });

    if (IsVariableUsedInMapClause) {
      // If variable is identified in a map clause it is always captured by
      // reference except if it is a pointer that is dereferenced somehow.
      IsByRef = !(Ty->isPointerType() && IsVariableAssociatedWithSection);
    } else {
      // By default, all the data that has a scalar type is mapped by copy
      // (except for reduction variables).
      // Defaultmap scalar is mutual exclusive to defaultmap pointer
      IsByRef = (DSAStack->isForceCaptureByReferenceInTargetExecutable() &&
                 !Ty->isAnyPointerType()) ||
                !Ty->isScalarType() ||
                DSAStack->isDefaultmapCapturedByRef(
                    Level, getVariableCategoryFromDecl(LangOpts, D)) ||
                DSAStack->hasExplicitDSA(
                    D,
                    [](OpenMPClauseKind K, bool AppliedToPointee) {
                      return K == OMPC_reduction && !AppliedToPointee;
                    },
                    Level);
    }
  }

  if (IsByRef && Ty.getNonReferenceType()->isScalarType()) {
    IsByRef =
        ((IsVariableUsedInMapClause &&
          DSAStack->getCaptureRegion(Level, OpenMPCaptureLevel) ==
              OMPD_target) ||
         !(DSAStack->hasExplicitDSA(
               D,
               [](OpenMPClauseKind K, bool AppliedToPointee) -> bool {
                 return K == OMPC_firstprivate ||
                        (K == OMPC_reduction && AppliedToPointee);
               },
               Level, /*NotLastprivate=*/true) ||
           DSAStack->isUsesAllocatorsDecl(Level, D))) &&
        // If the variable is artificial and must be captured by value - try to
        // capture by value.
        !(isa<OMPCapturedExprDecl>(D) && !D->hasAttr<OMPCaptureNoInitAttr>() &&
          !cast<OMPCapturedExprDecl>(D)->getInit()->isGLValue()) &&
        // If the variable is implicitly firstprivate and scalar - capture by
        // copy
        !(DSAStack->getDefaultDSA() == DSA_firstprivate &&
          !DSAStack->hasExplicitDSA(
              D, [](OpenMPClauseKind K, bool) { return K != OMPC_unknown; },
              Level) &&
          !DSAStack->isLoopControlVariable(D, Level).first);
  }

  // When passing data by copy, we need to make sure it fits the uintptr size
  // and alignment, because the runtime library only deals with uintptr types.
  // If it does not fit the uintptr size, we need to pass the data by reference
  // instead.
  if (!IsByRef &&
      (Ctx.getTypeSizeInChars(Ty) >
           Ctx.getTypeSizeInChars(Ctx.getUIntPtrType()) ||
       Ctx.getDeclAlign(D) > Ctx.getTypeAlignInChars(Ctx.getUIntPtrType()))) {
    IsByRef = true;
  }

  return IsByRef;
}

unsigned Sema::getOpenMPNestingLevel() const {
  assert(getLangOpts().OpenMP);
  return DSAStack->getNestingLevel();
}

bool Sema::isInOpenMPTargetExecutionDirective() const {
  return (isOpenMPTargetExecutionDirective(DSAStack->getCurrentDirective()) &&
          !DSAStack->isClauseParsingMode()) ||
         DSAStack->hasDirective(
             [](OpenMPDirectiveKind K, const DeclarationNameInfo &,
                SourceLocation) -> bool {
               return isOpenMPTargetExecutionDirective(K);
             },
             false);
}

VarDecl *Sema::isOpenMPCapturedDecl(ValueDecl *D, bool CheckScopeInfo,
                                    unsigned StopAt) {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  D = getCanonicalDecl(D);

  auto *VD = dyn_cast<VarDecl>(D);
  // Do not capture constexpr variables.
  if (VD && VD->isConstexpr())
    return nullptr;

  // If we want to determine whether the variable should be captured from the
  // perspective of the current capturing scope, and we've already left all the
  // capturing scopes of the top directive on the stack, check from the
  // perspective of its parent directive (if any) instead.
  DSAStackTy::ParentDirectiveScope InParentDirectiveRAII(
      *DSAStack, CheckScopeInfo && DSAStack->isBodyComplete());

  // If we are attempting to capture a global variable in a directive with
  // 'target' we return true so that this global is also mapped to the device.
  //
  if (VD && !VD->hasLocalStorage() &&
      (getCurCapturedRegion() || getCurBlock() || getCurLambda())) {
    if (isInOpenMPTargetExecutionDirective()) {
      DSAStackTy::DSAVarData DVarTop =
          DSAStack->getTopDSA(D, DSAStack->isClauseParsingMode());
      if (DVarTop.CKind != OMPC_unknown && DVarTop.RefExpr)
        return VD;
      // If the declaration is enclosed in a 'declare target' directive,
      // then it should not be captured.
      //
      if (OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD))
        return nullptr;
      CapturedRegionScopeInfo *CSI = nullptr;
      for (FunctionScopeInfo *FSI : llvm::drop_begin(
               llvm::reverse(FunctionScopes),
               CheckScopeInfo ? (FunctionScopes.size() - (StopAt + 1)) : 0)) {
        if (!isa<CapturingScopeInfo>(FSI))
          return nullptr;
        if (auto *RSI = dyn_cast<CapturedRegionScopeInfo>(FSI))
          if (RSI->CapRegionKind == CR_OpenMP) {
            CSI = RSI;
            break;
          }
      }
      assert(CSI && "Failed to find CapturedRegionScopeInfo");
      SmallVector<OpenMPDirectiveKind, 4> Regions;
      getOpenMPCaptureRegions(Regions,
                              DSAStack->getDirective(CSI->OpenMPLevel));
      if (Regions[CSI->OpenMPCaptureLevel] != OMPD_task)
        return VD;
    }
    if (isInOpenMPDeclareTargetContext()) {
      // Try to mark variable as declare target if it is used in capturing
      // regions.
      if (LangOpts.OpenMP <= 45 &&
          !OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD))
        checkDeclIsAllowedInOpenMPTarget(nullptr, VD);
      return nullptr;
    }
  }

  if (CheckScopeInfo) {
    bool OpenMPFound = false;
    for (unsigned I = StopAt + 1; I > 0; --I) {
      FunctionScopeInfo *FSI = FunctionScopes[I - 1];
      if (!isa<CapturingScopeInfo>(FSI))
        return nullptr;
      if (auto *RSI = dyn_cast<CapturedRegionScopeInfo>(FSI))
        if (RSI->CapRegionKind == CR_OpenMP) {
          OpenMPFound = true;
          break;
        }
    }
    if (!OpenMPFound)
      return nullptr;
  }

  if (DSAStack->getCurrentDirective() != OMPD_unknown &&
      (!DSAStack->isClauseParsingMode() ||
       DSAStack->getParentDirective() != OMPD_unknown)) {
    auto &&Info = DSAStack->isLoopControlVariable(D);
    if (Info.first ||
        (VD && VD->hasLocalStorage() &&
         isImplicitOrExplicitTaskingRegion(DSAStack->getCurrentDirective())) ||
        (VD && DSAStack->isForceVarCapturing()))
      return VD ? VD : Info.second;
    DSAStackTy::DSAVarData DVarTop =
        DSAStack->getTopDSA(D, DSAStack->isClauseParsingMode());
    if (DVarTop.CKind != OMPC_unknown && isOpenMPPrivate(DVarTop.CKind) &&
        (!VD || VD->hasLocalStorage() || !DVarTop.AppliedToPointee))
      return VD ? VD : cast<VarDecl>(DVarTop.PrivateCopy->getDecl());
    // Threadprivate variables must not be captured.
    if (isOpenMPThreadPrivate(DVarTop.CKind))
      return nullptr;
    // The variable is not private or it is the variable in the directive with
    // default(none) clause and not used in any clause.
    DSAStackTy::DSAVarData DVarPrivate = DSAStack->hasDSA(
        D,
        [](OpenMPClauseKind C, bool AppliedToPointee) {
          return isOpenMPPrivate(C) && !AppliedToPointee;
        },
        [](OpenMPDirectiveKind) { return true; },
        DSAStack->isClauseParsingMode());
    // Global shared must not be captured.
    if (VD && !VD->hasLocalStorage() && DVarPrivate.CKind == OMPC_unknown &&
        ((DSAStack->getDefaultDSA() != DSA_none &&
          DSAStack->getDefaultDSA() != DSA_firstprivate) ||
         DVarTop.CKind == OMPC_shared))
      return nullptr;
    if (DVarPrivate.CKind != OMPC_unknown ||
        (VD && (DSAStack->getDefaultDSA() == DSA_none ||
                DSAStack->getDefaultDSA() == DSA_firstprivate)))
      return VD ? VD : cast<VarDecl>(DVarPrivate.PrivateCopy->getDecl());
  }
  return nullptr;
}

void Sema::adjustOpenMPTargetScopeIndex(unsigned &FunctionScopesIndex,
                                        unsigned Level) const {
  FunctionScopesIndex -= getOpenMPCaptureLevels(DSAStack->getDirective(Level));
}

void Sema::startOpenMPLoop() {
  assert(LangOpts.OpenMP && "OpenMP must be enabled.");
  if (isOpenMPLoopDirective(DSAStack->getCurrentDirective()))
    DSAStack->loopInit();
}

void Sema::startOpenMPCXXRangeFor() {
  assert(LangOpts.OpenMP && "OpenMP must be enabled.");
  if (isOpenMPLoopDirective(DSAStack->getCurrentDirective())) {
    DSAStack->resetPossibleLoopCounter();
    DSAStack->loopStart();
  }
}

OpenMPClauseKind Sema::isOpenMPPrivateDecl(ValueDecl *D, unsigned Level,
                                           unsigned CapLevel) const {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  if (DSAStack->hasExplicitDirective(isOpenMPTaskingDirective, Level)) {
    bool IsTriviallyCopyable =
        D->getType().getNonReferenceType().isTriviallyCopyableType(Context) &&
        !D->getType()
             .getNonReferenceType()
             .getCanonicalType()
             ->getAsCXXRecordDecl();
    OpenMPDirectiveKind DKind = DSAStack->getDirective(Level);
    SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
    getOpenMPCaptureRegions(CaptureRegions, DKind);
    if (isOpenMPTaskingDirective(CaptureRegions[CapLevel]) &&
        (IsTriviallyCopyable ||
         !isOpenMPTaskLoopDirective(CaptureRegions[CapLevel]))) {
      if (DSAStack->hasExplicitDSA(
              D,
              [](OpenMPClauseKind K, bool) { return K == OMPC_firstprivate; },
              Level, /*NotLastprivate=*/true))
        return OMPC_firstprivate;
      DSAStackTy::DSAVarData DVar = DSAStack->getImplicitDSA(D, Level);
      if (DVar.CKind != OMPC_shared &&
          !DSAStack->isLoopControlVariable(D, Level).first && !DVar.RefExpr) {
        DSAStack->addImplicitTaskFirstprivate(Level, D);
        return OMPC_firstprivate;
      }
    }
  }
  if (isOpenMPLoopDirective(DSAStack->getCurrentDirective())) {
    if (DSAStack->getAssociatedLoops() > 0 && !DSAStack->isLoopStarted()) {
      DSAStack->resetPossibleLoopCounter(D);
      DSAStack->loopStart();
      return OMPC_private;
    }
    if ((DSAStack->getPossiblyLoopCunter() == D->getCanonicalDecl() ||
         DSAStack->isLoopControlVariable(D).first) &&
        !DSAStack->hasExplicitDSA(
            D, [](OpenMPClauseKind K, bool) { return K != OMPC_private; },
            Level) &&
        !isOpenMPSimdDirective(DSAStack->getCurrentDirective()))
      return OMPC_private;
  }
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (DSAStack->isThreadPrivate(const_cast<VarDecl *>(VD)) &&
        DSAStack->isForceVarCapturing() &&
        !DSAStack->hasExplicitDSA(
            D, [](OpenMPClauseKind K, bool) { return K == OMPC_copyin; },
            Level))
      return OMPC_private;
  }
  // User-defined allocators are private since they must be defined in the
  // context of target region.
  if (DSAStack->hasExplicitDirective(isOpenMPTargetExecutionDirective, Level) &&
      DSAStack->isUsesAllocatorsDecl(Level, D).getValueOr(
          DSAStackTy::UsesAllocatorsDeclKind::AllocatorTrait) ==
          DSAStackTy::UsesAllocatorsDeclKind::UserDefinedAllocator)
    return OMPC_private;
  return (DSAStack->hasExplicitDSA(
              D, [](OpenMPClauseKind K, bool) { return K == OMPC_private; },
              Level) ||
          (DSAStack->isClauseParsingMode() &&
           DSAStack->getClauseParsingMode() == OMPC_private) ||
          // Consider taskgroup reduction descriptor variable a private
          // to avoid possible capture in the region.
          (DSAStack->hasExplicitDirective(
               [](OpenMPDirectiveKind K) {
                 return K == OMPD_taskgroup ||
                        ((isOpenMPParallelDirective(K) ||
                          isOpenMPWorksharingDirective(K)) &&
                         !isOpenMPSimdDirective(K));
               },
               Level) &&
           DSAStack->isTaskgroupReductionRef(D, Level)))
             ? OMPC_private
             : OMPC_unknown;
}

void Sema::setOpenMPCaptureKind(FieldDecl *FD, const ValueDecl *D,
                                unsigned Level) {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  D = getCanonicalDecl(D);
  OpenMPClauseKind OMPC = OMPC_unknown;
  for (unsigned I = DSAStack->getNestingLevel() + 1; I > Level; --I) {
    const unsigned NewLevel = I - 1;
    if (DSAStack->hasExplicitDSA(
            D,
            [&OMPC](const OpenMPClauseKind K, bool AppliedToPointee) {
              if (isOpenMPPrivate(K) && !AppliedToPointee) {
                OMPC = K;
                return true;
              }
              return false;
            },
            NewLevel))
      break;
    if (DSAStack->checkMappableExprComponentListsForDeclAtLevel(
            D, NewLevel,
            [](OMPClauseMappableExprCommon::MappableExprComponentListRef,
               OpenMPClauseKind) { return true; })) {
      OMPC = OMPC_map;
      break;
    }
    if (DSAStack->hasExplicitDirective(isOpenMPTargetExecutionDirective,
                                       NewLevel)) {
      OMPC = OMPC_map;
      if (DSAStack->mustBeFirstprivateAtLevel(
              NewLevel, getVariableCategoryFromDecl(LangOpts, D)))
        OMPC = OMPC_firstprivate;
      break;
    }
  }
  if (OMPC != OMPC_unknown)
    FD->addAttr(OMPCaptureKindAttr::CreateImplicit(Context, unsigned(OMPC)));
}

bool Sema::isOpenMPTargetCapturedDecl(const ValueDecl *D, unsigned Level,
                                      unsigned CaptureLevel) const {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  // Return true if the current level is no longer enclosed in a target region.

  SmallVector<OpenMPDirectiveKind, 4> Regions;
  getOpenMPCaptureRegions(Regions, DSAStack->getDirective(Level));
  const auto *VD = dyn_cast<VarDecl>(D);
  return VD && !VD->hasLocalStorage() &&
         DSAStack->hasExplicitDirective(isOpenMPTargetExecutionDirective,
                                        Level) &&
         Regions[CaptureLevel] != OMPD_task;
}

bool Sema::isOpenMPGlobalCapturedDecl(ValueDecl *D, unsigned Level,
                                      unsigned CaptureLevel) const {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  // Return true if the current level is no longer enclosed in a target region.

  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (!VD->hasLocalStorage()) {
      if (isInOpenMPTargetExecutionDirective())
        return true;
      DSAStackTy::DSAVarData TopDVar =
          DSAStack->getTopDSA(D, /*FromParent=*/false);
      unsigned NumLevels =
          getOpenMPCaptureLevels(DSAStack->getDirective(Level));
      if (Level == 0)
        return (NumLevels == CaptureLevel + 1) && TopDVar.CKind != OMPC_shared;
      do {
        --Level;
        DSAStackTy::DSAVarData DVar = DSAStack->getImplicitDSA(D, Level);
        if (DVar.CKind != OMPC_shared)
          return true;
      } while (Level > 0);
    }
  }
  return true;
}

void Sema::DestroyDataSharingAttributesStack() { delete DSAStack; }

void Sema::ActOnOpenMPBeginDeclareVariant(SourceLocation Loc,
                                          OMPTraitInfo &TI) {
  OMPDeclareVariantScopes.push_back(OMPDeclareVariantScope(TI));
}

void Sema::ActOnOpenMPEndDeclareVariant() {
  assert(isInOpenMPDeclareVariantScope() &&
         "Not in OpenMP declare variant scope!");

  OMPDeclareVariantScopes.pop_back();
}

void Sema::finalizeOpenMPDelayedAnalysis(const FunctionDecl *Caller,
                                         const FunctionDecl *Callee,
                                         SourceLocation Loc) {
  assert(LangOpts.OpenMP && "Expected OpenMP compilation mode.");
  Optional<OMPDeclareTargetDeclAttr::DevTypeTy> DevTy =
      OMPDeclareTargetDeclAttr::getDeviceType(Caller->getMostRecentDecl());
  // Ignore host functions during device analyzis.
  if (LangOpts.OpenMPIsDevice &&
      (!DevTy || *DevTy == OMPDeclareTargetDeclAttr::DT_Host))
    return;
  // Ignore nohost functions during host analyzis.
  if (!LangOpts.OpenMPIsDevice && DevTy &&
      *DevTy == OMPDeclareTargetDeclAttr::DT_NoHost)
    return;
  const FunctionDecl *FD = Callee->getMostRecentDecl();
  DevTy = OMPDeclareTargetDeclAttr::getDeviceType(FD);
  if (LangOpts.OpenMPIsDevice && DevTy &&
      *DevTy == OMPDeclareTargetDeclAttr::DT_Host) {
    // Diagnose host function called during device codegen.
    StringRef HostDevTy =
        getOpenMPSimpleClauseTypeName(OMPC_device_type, OMPC_DEVICE_TYPE_host);
    Diag(Loc, diag::err_omp_wrong_device_function_call) << HostDevTy << 0;
    Diag(*OMPDeclareTargetDeclAttr::getLocation(FD),
         diag::note_omp_marked_device_type_here)
        << HostDevTy;
    return;
  }
  if (!LangOpts.OpenMPIsDevice && DevTy &&
      *DevTy == OMPDeclareTargetDeclAttr::DT_NoHost) {
    // Diagnose nohost function called during host codegen.
    StringRef NoHostDevTy = getOpenMPSimpleClauseTypeName(
        OMPC_device_type, OMPC_DEVICE_TYPE_nohost);
    Diag(Loc, diag::err_omp_wrong_device_function_call) << NoHostDevTy << 1;
    Diag(*OMPDeclareTargetDeclAttr::getLocation(FD),
         diag::note_omp_marked_device_type_here)
        << NoHostDevTy;
  }
}

void Sema::StartOpenMPDSABlock(OpenMPDirectiveKind DKind,
                               const DeclarationNameInfo &DirName,
                               Scope *CurScope, SourceLocation Loc) {
  DSAStack->push(DKind, DirName, CurScope, Loc);
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);
}

void Sema::StartOpenMPClause(OpenMPClauseKind K) {
  DSAStack->setClauseParsingMode(K);
}

void Sema::EndOpenMPClause() {
  DSAStack->setClauseParsingMode(/*K=*/OMPC_unknown);
  CleanupVarDeclMarking();
}

static std::pair<ValueDecl *, bool>
getPrivateItem(Sema &S, Expr *&RefExpr, SourceLocation &ELoc,
               SourceRange &ERange, bool AllowArraySection = false);

/// Check consistency of the reduction clauses.
static void checkReductionClauses(Sema &S, DSAStackTy *Stack,
                                  ArrayRef<OMPClause *> Clauses) {
  bool InscanFound = false;
  SourceLocation InscanLoc;
  // OpenMP 5.0, 2.19.5.4 reduction Clause, Restrictions.
  // A reduction clause without the inscan reduction-modifier may not appear on
  // a construct on which a reduction clause with the inscan reduction-modifier
  // appears.
  for (OMPClause *C : Clauses) {
    if (C->getClauseKind() != OMPC_reduction)
      continue;
    auto *RC = cast<OMPReductionClause>(C);
    if (RC->getModifier() == OMPC_REDUCTION_inscan) {
      InscanFound = true;
      InscanLoc = RC->getModifierLoc();
      continue;
    }
    if (RC->getModifier() == OMPC_REDUCTION_task) {
      // OpenMP 5.0, 2.19.5.4 reduction Clause.
      // A reduction clause with the task reduction-modifier may only appear on
      // a parallel construct, a worksharing construct or a combined or
      // composite construct for which any of the aforementioned constructs is a
      // constituent construct and simd or loop are not constituent constructs.
      OpenMPDirectiveKind CurDir = Stack->getCurrentDirective();
      if (!(isOpenMPParallelDirective(CurDir) ||
            isOpenMPWorksharingDirective(CurDir)) ||
          isOpenMPSimdDirective(CurDir))
        S.Diag(RC->getModifierLoc(),
               diag::err_omp_reduction_task_not_parallel_or_worksharing);
      continue;
    }
  }
  if (InscanFound) {
    for (OMPClause *C : Clauses) {
      if (C->getClauseKind() != OMPC_reduction)
        continue;
      auto *RC = cast<OMPReductionClause>(C);
      if (RC->getModifier() != OMPC_REDUCTION_inscan) {
        S.Diag(RC->getModifier() == OMPC_REDUCTION_unknown
                   ? RC->getBeginLoc()
                   : RC->getModifierLoc(),
               diag::err_omp_inscan_reduction_expected);
        S.Diag(InscanLoc, diag::note_omp_previous_inscan_reduction);
        continue;
      }
      for (Expr *Ref : RC->varlists()) {
        assert(Ref && "NULL expr in OpenMP nontemporal clause.");
        SourceLocation ELoc;
        SourceRange ERange;
        Expr *SimpleRefExpr = Ref;
        auto Res = getPrivateItem(S, SimpleRefExpr, ELoc, ERange,
                                  /*AllowArraySection=*/true);
        ValueDecl *D = Res.first;
        if (!D)
          continue;
        if (!Stack->isUsedInScanDirective(getCanonicalDecl(D))) {
          S.Diag(Ref->getExprLoc(),
                 diag::err_omp_reduction_not_inclusive_exclusive)
              << Ref->getSourceRange();
        }
      }
    }
  }
}

static void checkAllocateClauses(Sema &S, DSAStackTy *Stack,
                                 ArrayRef<OMPClause *> Clauses);
static DeclRefExpr *buildCapture(Sema &S, ValueDecl *D, Expr *CaptureExpr,
                                 bool WithInit);

static void reportOriginalDsa(Sema &SemaRef, const DSAStackTy *Stack,
                              const ValueDecl *D,
                              const DSAStackTy::DSAVarData &DVar,
                              bool IsLoopIterVar = false);

void Sema::EndOpenMPDSABlock(Stmt *CurDirective) {
  // OpenMP [2.14.3.5, Restrictions, C/C++, p.1]
  //  A variable of class type (or array thereof) that appears in a lastprivate
  //  clause requires an accessible, unambiguous default constructor for the
  //  class type, unless the list item is also specified in a firstprivate
  //  clause.
  if (const auto *D = dyn_cast_or_null<OMPExecutableDirective>(CurDirective)) {
    for (OMPClause *C : D->clauses()) {
      if (auto *Clause = dyn_cast<OMPLastprivateClause>(C)) {
        SmallVector<Expr *, 8> PrivateCopies;
        for (Expr *DE : Clause->varlists()) {
          if (DE->isValueDependent() || DE->isTypeDependent()) {
            PrivateCopies.push_back(nullptr);
            continue;
          }
          auto *DRE = cast<DeclRefExpr>(DE->IgnoreParens());
          auto *VD = cast<VarDecl>(DRE->getDecl());
          QualType Type = VD->getType().getNonReferenceType();
          const DSAStackTy::DSAVarData DVar =
              DSAStack->getTopDSA(VD, /*FromParent=*/false);
          if (DVar.CKind == OMPC_lastprivate) {
            // Generate helper private variable and initialize it with the
            // default value. The address of the original variable is replaced
            // by the address of the new private variable in CodeGen. This new
            // variable is not added to IdResolver, so the code in the OpenMP
            // region uses original variable for proper diagnostics.
            VarDecl *VDPrivate = buildVarDecl(
                *this, DE->getExprLoc(), Type.getUnqualifiedType(),
                VD->getName(), VD->hasAttrs() ? &VD->getAttrs() : nullptr, DRE);
            ActOnUninitializedDecl(VDPrivate);
            if (VDPrivate->isInvalidDecl()) {
              PrivateCopies.push_back(nullptr);
              continue;
            }
            PrivateCopies.push_back(buildDeclRefExpr(
                *this, VDPrivate, DE->getType(), DE->getExprLoc()));
          } else {
            // The variable is also a firstprivate, so initialization sequence
            // for private copy is generated already.
            PrivateCopies.push_back(nullptr);
          }
        }
        Clause->setPrivateCopies(PrivateCopies);
        continue;
      }
      // Finalize nontemporal clause by handling private copies, if any.
      if (auto *Clause = dyn_cast<OMPNontemporalClause>(C)) {
        SmallVector<Expr *, 8> PrivateRefs;
        for (Expr *RefExpr : Clause->varlists()) {
          assert(RefExpr && "NULL expr in OpenMP nontemporal clause.");
          SourceLocation ELoc;
          SourceRange ERange;
          Expr *SimpleRefExpr = RefExpr;
          auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
          if (Res.second)
            // It will be analyzed later.
            PrivateRefs.push_back(RefExpr);
          ValueDecl *D = Res.first;
          if (!D)
            continue;

          const DSAStackTy::DSAVarData DVar =
              DSAStack->getTopDSA(D, /*FromParent=*/false);
          PrivateRefs.push_back(DVar.PrivateCopy ? DVar.PrivateCopy
                                                 : SimpleRefExpr);
        }
        Clause->setPrivateRefs(PrivateRefs);
        continue;
      }
      if (auto *Clause = dyn_cast<OMPUsesAllocatorsClause>(C)) {
        for (unsigned I = 0, E = Clause->getNumberOfAllocators(); I < E; ++I) {
          OMPUsesAllocatorsClause::Data D = Clause->getAllocatorData(I);
          auto *DRE = dyn_cast<DeclRefExpr>(D.Allocator->IgnoreParenImpCasts());
          if (!DRE)
            continue;
          ValueDecl *VD = DRE->getDecl();
          if (!VD || !isa<VarDecl>(VD))
            continue;
          DSAStackTy::DSAVarData DVar =
              DSAStack->getTopDSA(VD, /*FromParent=*/false);
          // OpenMP [2.12.5, target Construct]
          // Memory allocators that appear in a uses_allocators clause cannot
          // appear in other data-sharing attribute clauses or data-mapping
          // attribute clauses in the same construct.
          Expr *MapExpr = nullptr;
          if (DVar.RefExpr ||
              DSAStack->checkMappableExprComponentListsForDecl(
                  VD, /*CurrentRegionOnly=*/true,
                  [VD, &MapExpr](
                      OMPClauseMappableExprCommon::MappableExprComponentListRef
                          MapExprComponents,
                      OpenMPClauseKind C) {
                    auto MI = MapExprComponents.rbegin();
                    auto ME = MapExprComponents.rend();
                    if (MI != ME &&
                        MI->getAssociatedDeclaration()->getCanonicalDecl() ==
                            VD->getCanonicalDecl()) {
                      MapExpr = MI->getAssociatedExpression();
                      return true;
                    }
                    return false;
                  })) {
            Diag(D.Allocator->getExprLoc(),
                 diag::err_omp_allocator_used_in_clauses)
                << D.Allocator->getSourceRange();
            if (DVar.RefExpr)
              reportOriginalDsa(*this, DSAStack, VD, DVar);
            else
              Diag(MapExpr->getExprLoc(), diag::note_used_here)
                  << MapExpr->getSourceRange();
          }
        }
        continue;
      }
    }
    // Check allocate clauses.
    if (!CurContext->isDependentContext())
      checkAllocateClauses(*this, DSAStack, D->clauses());
    checkReductionClauses(*this, DSAStack, D->clauses());
  }

  DSAStack->pop();
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

static bool FinishOpenMPLinearClause(OMPLinearClause &Clause, DeclRefExpr *IV,
                                     Expr *NumIterations, Sema &SemaRef,
                                     Scope *S, DSAStackTy *Stack);

namespace {

class VarDeclFilterCCC final : public CorrectionCandidateCallback {
private:
  Sema &SemaRef;

public:
  explicit VarDeclFilterCCC(Sema &S) : SemaRef(S) {}
  bool ValidateCandidate(const TypoCorrection &Candidate) override {
    NamedDecl *ND = Candidate.getCorrectionDecl();
    if (const auto *VD = dyn_cast_or_null<VarDecl>(ND)) {
      return VD->hasGlobalStorage() &&
             SemaRef.isDeclInScope(ND, SemaRef.getCurLexicalContext(),
                                   SemaRef.getCurScope());
    }
    return false;
  }

  std::unique_ptr<CorrectionCandidateCallback> clone() override {
    return std::make_unique<VarDeclFilterCCC>(*this);
  }
};

class VarOrFuncDeclFilterCCC final : public CorrectionCandidateCallback {
private:
  Sema &SemaRef;

public:
  explicit VarOrFuncDeclFilterCCC(Sema &S) : SemaRef(S) {}
  bool ValidateCandidate(const TypoCorrection &Candidate) override {
    NamedDecl *ND = Candidate.getCorrectionDecl();
    if (ND && ((isa<VarDecl>(ND) && ND->getKind() == Decl::Var) ||
               isa<FunctionDecl>(ND))) {
      return SemaRef.isDeclInScope(ND, SemaRef.getCurLexicalContext(),
                                   SemaRef.getCurScope());
    }
    return false;
  }

  std::unique_ptr<CorrectionCandidateCallback> clone() override {
    return std::make_unique<VarOrFuncDeclFilterCCC>(*this);
  }
};

} // namespace

ExprResult Sema::ActOnOpenMPIdExpression(Scope *CurScope,
                                         CXXScopeSpec &ScopeSpec,
                                         const DeclarationNameInfo &Id,
                                         OpenMPDirectiveKind Kind) {
  LookupResult Lookup(*this, Id, LookupOrdinaryName);
  LookupParsedName(Lookup, CurScope, &ScopeSpec, true);

  if (Lookup.isAmbiguous())
    return ExprError();

  VarDecl *VD;
  if (!Lookup.isSingleResult()) {
    VarDeclFilterCCC CCC(*this);
    if (TypoCorrection Corrected =
            CorrectTypo(Id, LookupOrdinaryName, CurScope, nullptr, CCC,
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
  } else if (!(VD = Lookup.getAsSingle<VarDecl>())) {
    Diag(Id.getLoc(), diag::err_omp_expected_var_arg) << Id.getName();
    Diag(Lookup.getFoundDecl()->getLocation(), diag::note_declared_at);
    return ExprError();
  }
  Lookup.suppressDiagnostics();

  // OpenMP [2.9.2, Syntax, C/C++]
  //   Variables must be file-scope, namespace-scope, or static block-scope.
  if (Kind == OMPD_threadprivate && !VD->hasGlobalStorage()) {
    Diag(Id.getLoc(), diag::err_omp_global_var_arg)
        << getOpenMPDirectiveName(Kind) << !VD->isStaticLocal();
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }

  VarDecl *CanonicalVD = VD->getCanonicalDecl();
  NamedDecl *ND = CanonicalVD;
  // OpenMP [2.9.2, Restrictions, C/C++, p.2]
  //   A threadprivate directive for file-scope variables must appear outside
  //   any definition or declaration.
  if (CanonicalVD->getDeclContext()->isTranslationUnit() &&
      !getCurLexicalContext()->isTranslationUnit()) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(Kind) << VD;
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
        << getOpenMPDirectiveName(Kind) << VD;
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
        << getOpenMPDirectiveName(Kind) << VD;
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
  if (CanonicalVD->isLocalVarDecl() && CurScope &&
      !isDeclInScope(ND, getCurLexicalContext(), CurScope)) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(Kind) << VD;
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
  if (Kind == OMPD_threadprivate && VD->isUsed() &&
      !DSAStack->isThreadPrivate(VD)) {
    Diag(Id.getLoc(), diag::err_omp_var_used)
        << getOpenMPDirectiveName(Kind) << VD;
    return ExprError();
  }

  QualType ExprType = VD->getType().getNonReferenceType();
  return DeclRefExpr::Create(Context, NestedNameSpecifierLoc(),
                             SourceLocation(), VD,
                             /*RefersToEnclosingVariableOrCapture=*/false,
                             Id.getLoc(), ExprType, VK_LValue);
}

Sema::DeclGroupPtrTy
Sema::ActOnOpenMPThreadprivateDirective(SourceLocation Loc,
                                        ArrayRef<Expr *> VarList) {
  if (OMPThreadPrivateDecl *D = CheckOMPThreadPrivateDecl(Loc, VarList)) {
    CurContext->addDecl(D);
    return DeclGroupPtrTy::make(DeclGroupRef(D));
  }
  return nullptr;
}

namespace {
class LocalVarRefChecker final
    : public ConstStmtVisitor<LocalVarRefChecker, bool> {
  Sema &SemaRef;

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (const auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      if (VD->hasLocalStorage()) {
        SemaRef.Diag(E->getBeginLoc(),
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
    for (const Stmt *Child : S->children()) {
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
  for (Expr *RefExpr : VarList) {
    auto *DE = cast<DeclRefExpr>(RefExpr);
    auto *VD = cast<VarDecl>(DE->getDecl());
    SourceLocation ILoc = DE->getExprLoc();

    // Mark variable as used.
    VD->setReferenced();
    VD->markUsed(Context);

    QualType QType = VD->getType();
    if (QType->isDependentType() || QType->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      continue;
    }

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

    // Check if this is a TLS variable. If TLS is not being supported, produce
    // the corresponding diagnostic.
    if ((VD->getTLSKind() != VarDecl::TLS_None &&
         !(VD->hasAttr<OMPThreadPrivateDeclAttr>() &&
           getLangOpts().OpenMPUseTLS &&
           getASTContext().getTargetInfo().isTLSSupported())) ||
        (VD->getStorageClass() == SC_Register && VD->hasAttr<AsmLabelAttr>() &&
         !VD->isLocalVarDecl())) {
      Diag(ILoc, diag::err_omp_var_thread_local)
          << VD << ((VD->getTLSKind() != VarDecl::TLS_None) ? 0 : 1);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // Check if initial value of threadprivate variable reference variable with
    // local storage (it is not supported by runtime).
    if (const Expr *Init = VD->getAnyInitializer()) {
      LocalVarRefChecker Checker(*this);
      if (Checker.Visit(Init))
        continue;
    }

    Vars.push_back(RefExpr);
    DSAStack->addDSA(VD, DE, OMPC_threadprivate);
    VD->addAttr(OMPThreadPrivateDeclAttr::CreateImplicit(
        Context, SourceRange(Loc, Loc)));
    if (ASTMutationListener *ML = Context.getASTMutationListener())
      ML->DeclarationMarkedOpenMPThreadPrivate(VD);
  }
  OMPThreadPrivateDecl *D = nullptr;
  if (!Vars.empty()) {
    D = OMPThreadPrivateDecl::Create(Context, getCurLexicalContext(), Loc,
                                     Vars);
    D->setAccess(AS_public);
  }
  return D;
}

static OMPAllocateDeclAttr::AllocatorTypeTy
getAllocatorKind(Sema &S, DSAStackTy *Stack, Expr *Allocator) {
  if (!Allocator)
    return OMPAllocateDeclAttr::OMPNullMemAlloc;
  if (Allocator->isTypeDependent() || Allocator->isValueDependent() ||
      Allocator->isInstantiationDependent() ||
      Allocator->containsUnexpandedParameterPack())
    return OMPAllocateDeclAttr::OMPUserDefinedMemAlloc;
  auto AllocatorKindRes = OMPAllocateDeclAttr::OMPUserDefinedMemAlloc;
  const Expr *AE = Allocator->IgnoreParenImpCasts();
  for (int I = 0; I < OMPAllocateDeclAttr::OMPUserDefinedMemAlloc; ++I) {
    auto AllocatorKind = static_cast<OMPAllocateDeclAttr::AllocatorTypeTy>(I);
    const Expr *DefAllocator = Stack->getAllocator(AllocatorKind);
    llvm::FoldingSetNodeID AEId, DAEId;
    AE->Profile(AEId, S.getASTContext(), /*Canonical=*/true);
    DefAllocator->Profile(DAEId, S.getASTContext(), /*Canonical=*/true);
    if (AEId == DAEId) {
      AllocatorKindRes = AllocatorKind;
      break;
    }
  }
  return AllocatorKindRes;
}

static bool checkPreviousOMPAllocateAttribute(
    Sema &S, DSAStackTy *Stack, Expr *RefExpr, VarDecl *VD,
    OMPAllocateDeclAttr::AllocatorTypeTy AllocatorKind, Expr *Allocator) {
  if (!VD->hasAttr<OMPAllocateDeclAttr>())
    return false;
  const auto *A = VD->getAttr<OMPAllocateDeclAttr>();
  Expr *PrevAllocator = A->getAllocator();
  OMPAllocateDeclAttr::AllocatorTypeTy PrevAllocatorKind =
      getAllocatorKind(S, Stack, PrevAllocator);
  bool AllocatorsMatch = AllocatorKind == PrevAllocatorKind;
  if (AllocatorsMatch &&
      AllocatorKind == OMPAllocateDeclAttr::OMPUserDefinedMemAlloc &&
      Allocator && PrevAllocator) {
    const Expr *AE = Allocator->IgnoreParenImpCasts();
    const Expr *PAE = PrevAllocator->IgnoreParenImpCasts();
    llvm::FoldingSetNodeID AEId, PAEId;
    AE->Profile(AEId, S.Context, /*Canonical=*/true);
    PAE->Profile(PAEId, S.Context, /*Canonical=*/true);
    AllocatorsMatch = AEId == PAEId;
  }
  if (!AllocatorsMatch) {
    SmallString<256> AllocatorBuffer;
    llvm::raw_svector_ostream AllocatorStream(AllocatorBuffer);
    if (Allocator)
      Allocator->printPretty(AllocatorStream, nullptr, S.getPrintingPolicy());
    SmallString<256> PrevAllocatorBuffer;
    llvm::raw_svector_ostream PrevAllocatorStream(PrevAllocatorBuffer);
    if (PrevAllocator)
      PrevAllocator->printPretty(PrevAllocatorStream, nullptr,
                                 S.getPrintingPolicy());

    SourceLocation AllocatorLoc =
        Allocator ? Allocator->getExprLoc() : RefExpr->getExprLoc();
    SourceRange AllocatorRange =
        Allocator ? Allocator->getSourceRange() : RefExpr->getSourceRange();
    SourceLocation PrevAllocatorLoc =
        PrevAllocator ? PrevAllocator->getExprLoc() : A->getLocation();
    SourceRange PrevAllocatorRange =
        PrevAllocator ? PrevAllocator->getSourceRange() : A->getRange();
    S.Diag(AllocatorLoc, diag::warn_omp_used_different_allocator)
        << (Allocator ? 1 : 0) << AllocatorStream.str()
        << (PrevAllocator ? 1 : 0) << PrevAllocatorStream.str()
        << AllocatorRange;
    S.Diag(PrevAllocatorLoc, diag::note_omp_previous_allocator)
        << PrevAllocatorRange;
    return true;
  }
  return false;
}

static void
applyOMPAllocateAttribute(Sema &S, VarDecl *VD,
                          OMPAllocateDeclAttr::AllocatorTypeTy AllocatorKind,
                          Expr *Allocator, Expr *Alignment, SourceRange SR) {
  if (VD->hasAttr<OMPAllocateDeclAttr>())
    return;
  if (Alignment &&
      (Alignment->isTypeDependent() || Alignment->isValueDependent() ||
       Alignment->isInstantiationDependent() ||
       Alignment->containsUnexpandedParameterPack()))
    // Apply later when we have a usable value.
    return;
  if (Allocator &&
      (Allocator->isTypeDependent() || Allocator->isValueDependent() ||
       Allocator->isInstantiationDependent() ||
       Allocator->containsUnexpandedParameterPack()))
    return;
  auto *A = OMPAllocateDeclAttr::CreateImplicit(S.Context, AllocatorKind,
                                                Allocator, Alignment, SR);
  VD->addAttr(A);
  if (ASTMutationListener *ML = S.Context.getASTMutationListener())
    ML->DeclarationMarkedOpenMPAllocate(VD, A);
}

Sema::DeclGroupPtrTy
Sema::ActOnOpenMPAllocateDirective(SourceLocation Loc, ArrayRef<Expr *> VarList,
                                   ArrayRef<OMPClause *> Clauses,
                                   DeclContext *Owner) {
  assert(Clauses.size() <= 2 && "Expected at most two clauses.");
  Expr *Alignment = nullptr;
  Expr *Allocator = nullptr;
  if (Clauses.empty()) {
    // OpenMP 5.0, 2.11.3 allocate Directive, Restrictions.
    // allocate directives that appear in a target region must specify an
    // allocator clause unless a requires directive with the dynamic_allocators
    // clause is present in the same compilation unit.
    if (LangOpts.OpenMPIsDevice &&
        !DSAStack->hasRequiresDeclWithClause<OMPDynamicAllocatorsClause>())
      targetDiag(Loc, diag::err_expected_allocator_clause);
  } else {
    for (const OMPClause *C : Clauses)
      if (const auto *AC = dyn_cast<OMPAllocatorClause>(C))
        Allocator = AC->getAllocator();
      else if (const auto *AC = dyn_cast<OMPAlignClause>(C))
        Alignment = AC->getAlignment();
      else
        llvm_unreachable("Unexpected clause on allocate directive");
  }
  OMPAllocateDeclAttr::AllocatorTypeTy AllocatorKind =
      getAllocatorKind(*this, DSAStack, Allocator);
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : VarList) {
    auto *DE = cast<DeclRefExpr>(RefExpr);
    auto *VD = cast<VarDecl>(DE->getDecl());

    // Check if this is a TLS variable or global register.
    if (VD->getTLSKind() != VarDecl::TLS_None ||
        VD->hasAttr<OMPThreadPrivateDeclAttr>() ||
        (VD->getStorageClass() == SC_Register && VD->hasAttr<AsmLabelAttr>() &&
         !VD->isLocalVarDecl()))
      continue;

    // If the used several times in the allocate directive, the same allocator
    // must be used.
    if (checkPreviousOMPAllocateAttribute(*this, DSAStack, RefExpr, VD,
                                          AllocatorKind, Allocator))
      continue;

    // OpenMP, 2.11.3 allocate Directive, Restrictions, C / C++
    // If a list item has a static storage type, the allocator expression in the
    // allocator clause must be a constant expression that evaluates to one of
    // the predefined memory allocator values.
    if (Allocator && VD->hasGlobalStorage()) {
      if (AllocatorKind == OMPAllocateDeclAttr::OMPUserDefinedMemAlloc) {
        Diag(Allocator->getExprLoc(),
             diag::err_omp_expected_predefined_allocator)
            << Allocator->getSourceRange();
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        continue;
      }
    }

    Vars.push_back(RefExpr);
    applyOMPAllocateAttribute(*this, VD, AllocatorKind, Allocator, Alignment,
                              DE->getSourceRange());
  }
  if (Vars.empty())
    return nullptr;
  if (!Owner)
    Owner = getCurLexicalContext();
  auto *D = OMPAllocateDecl::Create(Context, Owner, Loc, Vars, Clauses);
  D->setAccess(AS_public);
  Owner->addDecl(D);
  return DeclGroupPtrTy::make(DeclGroupRef(D));
}

Sema::DeclGroupPtrTy
Sema::ActOnOpenMPRequiresDirective(SourceLocation Loc,
                                   ArrayRef<OMPClause *> ClauseList) {
  OMPRequiresDecl *D = nullptr;
  if (!CurContext->isFileContext()) {
    Diag(Loc, diag::err_omp_invalid_scope) << "requires";
  } else {
    D = CheckOMPRequiresDecl(Loc, ClauseList);
    if (D) {
      CurContext->addDecl(D);
      DSAStack->addRequiresDecl(D);
    }
  }
  return DeclGroupPtrTy::make(DeclGroupRef(D));
}

void Sema::ActOnOpenMPAssumesDirective(SourceLocation Loc,
                                       OpenMPDirectiveKind DKind,
                                       ArrayRef<std::string> Assumptions,
                                       bool SkippedClauses) {
  if (!SkippedClauses && Assumptions.empty())
    Diag(Loc, diag::err_omp_no_clause_for_directive)
        << llvm::omp::getAllAssumeClauseOptions()
        << llvm::omp::getOpenMPDirectiveName(DKind);

  auto *AA = AssumptionAttr::Create(Context, llvm::join(Assumptions, ","), Loc);
  if (DKind == llvm::omp::Directive::OMPD_begin_assumes) {
    OMPAssumeScoped.push_back(AA);
    return;
  }

  // Global assumes without assumption clauses are ignored.
  if (Assumptions.empty())
    return;

  assert(DKind == llvm::omp::Directive::OMPD_assumes &&
         "Unexpected omp assumption directive!");
  OMPAssumeGlobal.push_back(AA);

  // The OMPAssumeGlobal scope above will take care of new declarations but
  // we also want to apply the assumption to existing ones, e.g., to
  // declarations in included headers. To this end, we traverse all existing
  // declaration contexts and annotate function declarations here.
  SmallVector<DeclContext *, 8> DeclContexts;
  auto *Ctx = CurContext;
  while (Ctx->getLexicalParent())
    Ctx = Ctx->getLexicalParent();
  DeclContexts.push_back(Ctx);
  while (!DeclContexts.empty()) {
    DeclContext *DC = DeclContexts.pop_back_val();
    for (auto *SubDC : DC->decls()) {
      if (SubDC->isInvalidDecl())
        continue;
      if (auto *CTD = dyn_cast<ClassTemplateDecl>(SubDC)) {
        DeclContexts.push_back(CTD->getTemplatedDecl());
        for (auto *S : CTD->specializations())
          DeclContexts.push_back(S);
        continue;
      }
      if (auto *DC = dyn_cast<DeclContext>(SubDC))
        DeclContexts.push_back(DC);
      if (auto *F = dyn_cast<FunctionDecl>(SubDC)) {
        F->addAttr(AA);
        continue;
      }
    }
  }
}

void Sema::ActOnOpenMPEndAssumesDirective() {
  assert(isInOpenMPAssumeScope() && "Not in OpenMP assumes scope!");
  OMPAssumeScoped.pop_back();
}

OMPRequiresDecl *Sema::CheckOMPRequiresDecl(SourceLocation Loc,
                                            ArrayRef<OMPClause *> ClauseList) {
  /// For target specific clauses, the requires directive cannot be
  /// specified after the handling of any of the target regions in the
  /// current compilation unit.
  ArrayRef<SourceLocation> TargetLocations =
      DSAStack->getEncounteredTargetLocs();
  SourceLocation AtomicLoc = DSAStack->getAtomicDirectiveLoc();
  if (!TargetLocations.empty() || !AtomicLoc.isInvalid()) {
    for (const OMPClause *CNew : ClauseList) {
      // Check if any of the requires clauses affect target regions.
      if (isa<OMPUnifiedSharedMemoryClause>(CNew) ||
          isa<OMPUnifiedAddressClause>(CNew) ||
          isa<OMPReverseOffloadClause>(CNew) ||
          isa<OMPDynamicAllocatorsClause>(CNew)) {
        Diag(Loc, diag::err_omp_directive_before_requires)
            << "target" << getOpenMPClauseName(CNew->getClauseKind());
        for (SourceLocation TargetLoc : TargetLocations) {
          Diag(TargetLoc, diag::note_omp_requires_encountered_directive)
              << "target";
        }
      } else if (!AtomicLoc.isInvalid() &&
                 isa<OMPAtomicDefaultMemOrderClause>(CNew)) {
        Diag(Loc, diag::err_omp_directive_before_requires)
            << "atomic" << getOpenMPClauseName(CNew->getClauseKind());
        Diag(AtomicLoc, diag::note_omp_requires_encountered_directive)
            << "atomic";
      }
    }
  }

  if (!DSAStack->hasDuplicateRequiresClause(ClauseList))
    return OMPRequiresDecl::Create(Context, getCurLexicalContext(), Loc,
                                   ClauseList);
  return nullptr;
}

static void reportOriginalDsa(Sema &SemaRef, const DSAStackTy *Stack,
                              const ValueDecl *D,
                              const DSAStackTy::DSAVarData &DVar,
                              bool IsLoopIterVar) {
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
    PDSA_TaskVarFirstprivate,
    PDSA_LocalVarPrivate,
    PDSA_Implicit
  } Reason = PDSA_Implicit;
  bool ReportHint = false;
  auto ReportLoc = D->getLocation();
  auto *VD = dyn_cast<VarDecl>(D);
  if (IsLoopIterVar) {
    if (DVar.CKind == OMPC_private)
      Reason = PDSA_LoopIterVarPrivate;
    else if (DVar.CKind == OMPC_lastprivate)
      Reason = PDSA_LoopIterVarLastprivate;
    else
      Reason = PDSA_LoopIterVarLinear;
  } else if (isOpenMPTaskingDirective(DVar.DKind) &&
             DVar.CKind == OMPC_firstprivate) {
    Reason = PDSA_TaskVarFirstprivate;
    ReportLoc = DVar.ImplicitDSALoc;
  } else if (VD && VD->isStaticLocal())
    Reason = PDSA_StaticLocalVarShared;
  else if (VD && VD->isStaticDataMember())
    Reason = PDSA_StaticMemberShared;
  else if (VD && VD->isFileVarDecl())
    Reason = PDSA_GlobalVarShared;
  else if (D->getType().isConstant(SemaRef.getASTContext()))
    Reason = PDSA_ConstVarShared;
  else if (VD && VD->isLocalVarDecl() && DVar.CKind == OMPC_private) {
    ReportHint = true;
    Reason = PDSA_LocalVarPrivate;
  }
  if (Reason != PDSA_Implicit) {
    SemaRef.Diag(ReportLoc, diag::note_omp_predetermined_dsa)
        << Reason << ReportHint
        << getOpenMPDirectiveName(Stack->getCurrentDirective());
  } else if (DVar.ImplicitDSALoc.isValid()) {
    SemaRef.Diag(DVar.ImplicitDSALoc, diag::note_omp_implicit_dsa)
        << getOpenMPClauseName(DVar.CKind);
  }
}

static OpenMPMapClauseKind
getMapClauseKindFromModifier(OpenMPDefaultmapClauseModifier M,
                             bool IsAggregateOrDeclareTarget) {
  OpenMPMapClauseKind Kind = OMPC_MAP_unknown;
  switch (M) {
  case OMPC_DEFAULTMAP_MODIFIER_alloc:
    Kind = OMPC_MAP_alloc;
    break;
  case OMPC_DEFAULTMAP_MODIFIER_to:
    Kind = OMPC_MAP_to;
    break;
  case OMPC_DEFAULTMAP_MODIFIER_from:
    Kind = OMPC_MAP_from;
    break;
  case OMPC_DEFAULTMAP_MODIFIER_tofrom:
    Kind = OMPC_MAP_tofrom;
    break;
  case OMPC_DEFAULTMAP_MODIFIER_present:
    // OpenMP 5.1 [2.21.7.3] defaultmap clause, Description]
    // If implicit-behavior is present, each variable referenced in the
    // construct in the category specified by variable-category is treated as if
    // it had been listed in a map clause with the map-type of alloc and
    // map-type-modifier of present.
    Kind = OMPC_MAP_alloc;
    break;
  case OMPC_DEFAULTMAP_MODIFIER_firstprivate:
  case OMPC_DEFAULTMAP_MODIFIER_last:
    llvm_unreachable("Unexpected defaultmap implicit behavior");
  case OMPC_DEFAULTMAP_MODIFIER_none:
  case OMPC_DEFAULTMAP_MODIFIER_default:
  case OMPC_DEFAULTMAP_MODIFIER_unknown:
    // IsAggregateOrDeclareTarget could be true if:
    // 1. the implicit behavior for aggregate is tofrom
    // 2. it's a declare target link
    if (IsAggregateOrDeclareTarget) {
      Kind = OMPC_MAP_tofrom;
      break;
    }
    llvm_unreachable("Unexpected defaultmap implicit behavior");
  }
  assert(Kind != OMPC_MAP_unknown && "Expect map kind to be known");
  return Kind;
}

namespace {
class DSAAttrChecker final : public StmtVisitor<DSAAttrChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  bool ErrorFound = false;
  bool TryCaptureCXXThisMembers = false;
  CapturedStmt *CS = nullptr;
  const static unsigned DefaultmapKindNum = OMPC_DEFAULTMAP_pointer + 1;
  llvm::SmallVector<Expr *, 4> ImplicitFirstprivate;
  llvm::SmallVector<Expr *, 4> ImplicitMap[DefaultmapKindNum][OMPC_MAP_delete];
  llvm::SmallVector<OpenMPMapModifierKind, NumberOfOMPMapClauseModifiers>
      ImplicitMapModifier[DefaultmapKindNum];
  Sema::VarsWithInheritedDSAType VarsWithInheritedDSA;
  llvm::SmallDenseSet<const ValueDecl *, 4> ImplicitDeclarations;

  void VisitSubCaptures(OMPExecutableDirective *S) {
    // Check implicitly captured variables.
    if (!S->hasAssociatedStmt() || !S->getAssociatedStmt())
      return;
    if (S->getDirectiveKind() == OMPD_atomic ||
        S->getDirectiveKind() == OMPD_critical ||
        S->getDirectiveKind() == OMPD_section ||
        S->getDirectiveKind() == OMPD_master ||
        S->getDirectiveKind() == OMPD_masked ||
        isOpenMPLoopTransformationDirective(S->getDirectiveKind())) {
      Visit(S->getAssociatedStmt());
      return;
    }
    visitSubCaptures(S->getInnermostCapturedStmt());
    // Try to capture inner this->member references to generate correct mappings
    // and diagnostics.
    if (TryCaptureCXXThisMembers ||
        (isOpenMPTargetExecutionDirective(Stack->getCurrentDirective()) &&
         llvm::any_of(S->getInnermostCapturedStmt()->captures(),
                      [](const CapturedStmt::Capture &C) {
                        return C.capturesThis();
                      }))) {
      bool SavedTryCaptureCXXThisMembers = TryCaptureCXXThisMembers;
      TryCaptureCXXThisMembers = true;
      Visit(S->getInnermostCapturedStmt()->getCapturedStmt());
      TryCaptureCXXThisMembers = SavedTryCaptureCXXThisMembers;
    }
    // In tasks firstprivates are not captured anymore, need to analyze them
    // explicitly.
    if (isOpenMPTaskingDirective(S->getDirectiveKind()) &&
        !isOpenMPTaskLoopDirective(S->getDirectiveKind())) {
      for (OMPClause *C : S->clauses())
        if (auto *FC = dyn_cast<OMPFirstprivateClause>(C)) {
          for (Expr *Ref : FC->varlists())
            Visit(Ref);
        }
    }
  }

public:
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (TryCaptureCXXThisMembers || E->isTypeDependent() ||
        E->isValueDependent() || E->containsUnexpandedParameterPack() ||
        E->isInstantiationDependent())
      return;
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      // Check the datasharing rules for the expressions in the clauses.
      if (!CS || (isa<OMPCapturedExprDecl>(VD) && !CS->capturesVariable(VD) &&
                  !Stack->getTopDSA(VD, /*FromParent=*/false).RefExpr)) {
        if (auto *CED = dyn_cast<OMPCapturedExprDecl>(VD))
          if (!CED->hasAttr<OMPCaptureNoInitAttr>()) {
            Visit(CED->getInit());
            return;
          }
      } else if (VD->isImplicit() || isa<OMPCapturedExprDecl>(VD))
        // Do not analyze internal variables and do not enclose them into
        // implicit clauses.
        return;
      VD = VD->getCanonicalDecl();
      // Skip internally declared variables.
      if (VD->hasLocalStorage() && CS && !CS->capturesVariable(VD) &&
          !Stack->isImplicitTaskFirstprivate(VD))
        return;
      // Skip allocators in uses_allocators clauses.
      if (Stack->isUsesAllocatorsDecl(VD).hasValue())
        return;

      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(VD, /*FromParent=*/false);
      // Check if the variable has explicit DSA set and stop analysis if it so.
      if (DVar.RefExpr || !ImplicitDeclarations.insert(VD).second)
        return;

      // Skip internally declared static variables.
      llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
          OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD);
      if (VD->hasGlobalStorage() && CS && !CS->capturesVariable(VD) &&
          (Stack->hasRequiresDeclWithClause<OMPUnifiedSharedMemoryClause>() ||
           !Res || *Res != OMPDeclareTargetDeclAttr::MT_Link) &&
          !Stack->isImplicitTaskFirstprivate(VD))
        return;

      SourceLocation ELoc = E->getExprLoc();
      OpenMPDirectiveKind DKind = Stack->getCurrentDirective();
      // The default(none) clause requires that each variable that is referenced
      // in the construct, and does not have a predetermined data-sharing
      // attribute, must have its data-sharing attribute explicitly determined
      // by being listed in a data-sharing attribute clause.
      if (DVar.CKind == OMPC_unknown &&
          (Stack->getDefaultDSA() == DSA_none ||
           Stack->getDefaultDSA() == DSA_firstprivate) &&
          isImplicitOrExplicitTaskingRegion(DKind) &&
          VarsWithInheritedDSA.count(VD) == 0) {
        bool InheritedDSA = Stack->getDefaultDSA() == DSA_none;
        if (!InheritedDSA && Stack->getDefaultDSA() == DSA_firstprivate) {
          DSAStackTy::DSAVarData DVar =
              Stack->getImplicitDSA(VD, /*FromParent=*/false);
          InheritedDSA = DVar.CKind == OMPC_unknown;
        }
        if (InheritedDSA)
          VarsWithInheritedDSA[VD] = E;
        return;
      }

      // OpenMP 5.0 [2.19.7.2, defaultmap clause, Description]
      // If implicit-behavior is none, each variable referenced in the
      // construct that does not have a predetermined data-sharing attribute
      // and does not appear in a to or link clause on a declare target
      // directive must be listed in a data-mapping attribute clause, a
      // data-haring attribute clause (including a data-sharing attribute
      // clause on a combined construct where target. is one of the
      // constituent constructs), or an is_device_ptr clause.
      OpenMPDefaultmapClauseKind ClauseKind =
          getVariableCategoryFromDecl(SemaRef.getLangOpts(), VD);
      if (SemaRef.getLangOpts().OpenMP >= 50) {
        bool IsModifierNone = Stack->getDefaultmapModifier(ClauseKind) ==
                              OMPC_DEFAULTMAP_MODIFIER_none;
        if (DVar.CKind == OMPC_unknown && IsModifierNone &&
            VarsWithInheritedDSA.count(VD) == 0 && !Res) {
          // Only check for data-mapping attribute and is_device_ptr here
          // since we have already make sure that the declaration does not
          // have a data-sharing attribute above
          if (!Stack->checkMappableExprComponentListsForDecl(
                  VD, /*CurrentRegionOnly=*/true,
                  [VD](OMPClauseMappableExprCommon::MappableExprComponentListRef
                           MapExprComponents,
                       OpenMPClauseKind) {
                    auto MI = MapExprComponents.rbegin();
                    auto ME = MapExprComponents.rend();
                    return MI != ME && MI->getAssociatedDeclaration() == VD;
                  })) {
            VarsWithInheritedDSA[VD] = E;
            return;
          }
        }
      }
      if (SemaRef.getLangOpts().OpenMP > 50) {
        bool IsModifierPresent = Stack->getDefaultmapModifier(ClauseKind) ==
                                 OMPC_DEFAULTMAP_MODIFIER_present;
        if (IsModifierPresent) {
          if (llvm::find(ImplicitMapModifier[ClauseKind],
                         OMPC_MAP_MODIFIER_present) ==
              std::end(ImplicitMapModifier[ClauseKind])) {
            ImplicitMapModifier[ClauseKind].push_back(
                OMPC_MAP_MODIFIER_present);
          }
        }
      }

      if (isOpenMPTargetExecutionDirective(DKind) &&
          !Stack->isLoopControlVariable(VD).first) {
        if (!Stack->checkMappableExprComponentListsForDecl(
                VD, /*CurrentRegionOnly=*/true,
                [this](OMPClauseMappableExprCommon::MappableExprComponentListRef
                           StackComponents,
                       OpenMPClauseKind) {
                  if (SemaRef.LangOpts.OpenMP >= 50)
                    return !StackComponents.empty();
                  // Variable is used if it has been marked as an array, array
                  // section, array shaping or the variable iself.
                  return StackComponents.size() == 1 ||
                         std::all_of(
                             std::next(StackComponents.rbegin()),
                             StackComponents.rend(),
                             [](const OMPClauseMappableExprCommon::
                                    MappableComponent &MC) {
                               return MC.getAssociatedDeclaration() ==
                                          nullptr &&
                                      (isa<OMPArraySectionExpr>(
                                           MC.getAssociatedExpression()) ||
                                       isa<OMPArrayShapingExpr>(
                                           MC.getAssociatedExpression()) ||
                                       isa<ArraySubscriptExpr>(
                                           MC.getAssociatedExpression()));
                             });
                })) {
          bool IsFirstprivate = false;
          // By default lambdas are captured as firstprivates.
          if (const auto *RD =
                  VD->getType().getNonReferenceType()->getAsCXXRecordDecl())
            IsFirstprivate = RD->isLambda();
          IsFirstprivate =
              IsFirstprivate || (Stack->mustBeFirstprivate(ClauseKind) && !Res);
          if (IsFirstprivate) {
            ImplicitFirstprivate.emplace_back(E);
          } else {
            OpenMPDefaultmapClauseModifier M =
                Stack->getDefaultmapModifier(ClauseKind);
            OpenMPMapClauseKind Kind = getMapClauseKindFromModifier(
                M, ClauseKind == OMPC_DEFAULTMAP_aggregate || Res);
            ImplicitMap[ClauseKind][Kind].emplace_back(E);
          }
          return;
        }
      }

      // OpenMP [2.9.3.6, Restrictions, p.2]
      //  A list item that appears in a reduction clause of the innermost
      //  enclosing worksharing or parallel construct may not be accessed in an
      //  explicit task.
      DVar = Stack->hasInnermostDSA(
          VD,
          [](OpenMPClauseKind C, bool AppliedToPointee) {
            return C == OMPC_reduction && !AppliedToPointee;
          },
          [](OpenMPDirectiveKind K) {
            return isOpenMPParallelDirective(K) ||
                   isOpenMPWorksharingDirective(K) || isOpenMPTeamsDirective(K);
          },
          /*FromParent=*/true);
      if (isOpenMPTaskingDirective(DKind) && DVar.CKind == OMPC_reduction) {
        ErrorFound = true;
        SemaRef.Diag(ELoc, diag::err_omp_reduction_in_task);
        reportOriginalDsa(SemaRef, Stack, VD, DVar);
        return;
      }

      // Define implicit data-sharing attributes for task.
      DVar = Stack->getImplicitDSA(VD, /*FromParent=*/false);
      if (((isOpenMPTaskingDirective(DKind) && DVar.CKind != OMPC_shared) ||
           (Stack->getDefaultDSA() == DSA_firstprivate &&
            DVar.CKind == OMPC_firstprivate && !DVar.RefExpr)) &&
          !Stack->isLoopControlVariable(VD).first) {
        ImplicitFirstprivate.push_back(E);
        return;
      }

      // Store implicitly used globals with declare target link for parent
      // target.
      if (!isOpenMPTargetExecutionDirective(DKind) && Res &&
          *Res == OMPDeclareTargetDeclAttr::MT_Link) {
        Stack->addToParentTargetRegionLinkGlobals(E);
        return;
      }
    }
  }
  void VisitMemberExpr(MemberExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;
    auto *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
    OpenMPDirectiveKind DKind = Stack->getCurrentDirective();
    if (auto *TE = dyn_cast<CXXThisExpr>(E->getBase()->IgnoreParenCasts())) {
      if (!FD)
        return;
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(FD, /*FromParent=*/false);
      // Check if the variable has explicit DSA set and stop analysis if it
      // so.
      if (DVar.RefExpr || !ImplicitDeclarations.insert(FD).second)
        return;

      if (isOpenMPTargetExecutionDirective(DKind) &&
          !Stack->isLoopControlVariable(FD).first &&
          !Stack->checkMappableExprComponentListsForDecl(
              FD, /*CurrentRegionOnly=*/true,
              [](OMPClauseMappableExprCommon::MappableExprComponentListRef
                     StackComponents,
                 OpenMPClauseKind) {
                return isa<CXXThisExpr>(
                    cast<MemberExpr>(
                        StackComponents.back().getAssociatedExpression())
                        ->getBase()
                        ->IgnoreParens());
              })) {
        // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C/C++, p.3]
        //  A bit-field cannot appear in a map clause.
        //
        if (FD->isBitField())
          return;

        // Check to see if the member expression is referencing a class that
        // has already been explicitly mapped
        if (Stack->isClassPreviouslyMapped(TE->getType()))
          return;

        OpenMPDefaultmapClauseModifier Modifier =
            Stack->getDefaultmapModifier(OMPC_DEFAULTMAP_aggregate);
        OpenMPDefaultmapClauseKind ClauseKind =
            getVariableCategoryFromDecl(SemaRef.getLangOpts(), FD);
        OpenMPMapClauseKind Kind = getMapClauseKindFromModifier(
            Modifier, /*IsAggregateOrDeclareTarget*/ true);
        ImplicitMap[ClauseKind][Kind].emplace_back(E);
        return;
      }

      SourceLocation ELoc = E->getExprLoc();
      // OpenMP [2.9.3.6, Restrictions, p.2]
      //  A list item that appears in a reduction clause of the innermost
      //  enclosing worksharing or parallel construct may not be accessed in
      //  an  explicit task.
      DVar = Stack->hasInnermostDSA(
          FD,
          [](OpenMPClauseKind C, bool AppliedToPointee) {
            return C == OMPC_reduction && !AppliedToPointee;
          },
          [](OpenMPDirectiveKind K) {
            return isOpenMPParallelDirective(K) ||
                   isOpenMPWorksharingDirective(K) || isOpenMPTeamsDirective(K);
          },
          /*FromParent=*/true);
      if (isOpenMPTaskingDirective(DKind) && DVar.CKind == OMPC_reduction) {
        ErrorFound = true;
        SemaRef.Diag(ELoc, diag::err_omp_reduction_in_task);
        reportOriginalDsa(SemaRef, Stack, FD, DVar);
        return;
      }

      // Define implicit data-sharing attributes for task.
      DVar = Stack->getImplicitDSA(FD, /*FromParent=*/false);
      if (isOpenMPTaskingDirective(DKind) && DVar.CKind != OMPC_shared &&
          !Stack->isLoopControlVariable(FD).first) {
        // Check if there is a captured expression for the current field in the
        // region. Do not mark it as firstprivate unless there is no captured
        // expression.
        // TODO: try to make it firstprivate.
        if (DVar.CKind != OMPC_unknown)
          ImplicitFirstprivate.push_back(E);
      }
      return;
    }
    if (isOpenMPTargetExecutionDirective(DKind)) {
      OMPClauseMappableExprCommon::MappableExprComponentList CurComponents;
      if (!checkMapClauseExpressionBase(SemaRef, E, CurComponents, OMPC_map,
                                        Stack->getCurrentDirective(),
                                        /*NoDiagnose=*/true))
        return;
      const auto *VD = cast<ValueDecl>(
          CurComponents.back().getAssociatedDeclaration()->getCanonicalDecl());
      if (!Stack->checkMappableExprComponentListsForDecl(
              VD, /*CurrentRegionOnly=*/true,
              [&CurComponents](
                  OMPClauseMappableExprCommon::MappableExprComponentListRef
                      StackComponents,
                  OpenMPClauseKind) {
                auto CCI = CurComponents.rbegin();
                auto CCE = CurComponents.rend();
                for (const auto &SC : llvm::reverse(StackComponents)) {
                  // Do both expressions have the same kind?
                  if (CCI->getAssociatedExpression()->getStmtClass() !=
                      SC.getAssociatedExpression()->getStmtClass())
                    if (!((isa<OMPArraySectionExpr>(
                               SC.getAssociatedExpression()) ||
                           isa<OMPArrayShapingExpr>(
                               SC.getAssociatedExpression())) &&
                          isa<ArraySubscriptExpr>(
                              CCI->getAssociatedExpression())))
                      return false;

                  const Decl *CCD = CCI->getAssociatedDeclaration();
                  const Decl *SCD = SC.getAssociatedDeclaration();
                  CCD = CCD ? CCD->getCanonicalDecl() : nullptr;
                  SCD = SCD ? SCD->getCanonicalDecl() : nullptr;
                  if (SCD != CCD)
                    return false;
                  std::advance(CCI, 1);
                  if (CCI == CCE)
                    break;
                }
                return true;
              })) {
        Visit(E->getBase());
      }
    } else if (!TryCaptureCXXThisMembers) {
      Visit(E->getBase());
    }
  }
  void VisitOMPExecutableDirective(OMPExecutableDirective *S) {
    for (OMPClause *C : S->clauses()) {
      // Skip analysis of arguments of private clauses for task|target
      // directives.
      if (isa_and_nonnull<OMPPrivateClause>(C))
        continue;
      // Skip analysis of arguments of implicitly defined firstprivate clause
      // for task|target directives.
      // Skip analysis of arguments of implicitly defined map clause for target
      // directives.
      if (C && !((isa<OMPFirstprivateClause>(C) || isa<OMPMapClause>(C)) &&
                 C->isImplicit() &&
                 !isOpenMPTaskingDirective(Stack->getCurrentDirective()))) {
        for (Stmt *CC : C->children()) {
          if (CC)
            Visit(CC);
        }
      }
    }
    // Check implicitly captured variables.
    VisitSubCaptures(S);
  }

  void VisitOMPLoopTransformationDirective(OMPLoopTransformationDirective *S) {
    // Loop transformation directives do not introduce data sharing
    VisitStmt(S);
  }

  void VisitCallExpr(CallExpr *S) {
    for (Stmt *C : S->arguments()) {
      if (C) {
        // Check implicitly captured variables in the task-based directives to
        // check if they must be firstprivatized.
        Visit(C);
      }
    }
    if (Expr *Callee = S->getCallee())
      if (auto *CE = dyn_cast<MemberExpr>(Callee->IgnoreParenImpCasts()))
        Visit(CE->getBase());
  }
  void VisitStmt(Stmt *S) {
    for (Stmt *C : S->children()) {
      if (C) {
        // Check implicitly captured variables in the task-based directives to
        // check if they must be firstprivatized.
        Visit(C);
      }
    }
  }

  void visitSubCaptures(CapturedStmt *S) {
    for (const CapturedStmt::Capture &Cap : S->captures()) {
      if (!Cap.capturesVariable() && !Cap.capturesVariableByCopy())
        continue;
      VarDecl *VD = Cap.getCapturedVar();
      // Do not try to map the variable if it or its sub-component was mapped
      // already.
      if (isOpenMPTargetExecutionDirective(Stack->getCurrentDirective()) &&
          Stack->checkMappableExprComponentListsForDecl(
              VD, /*CurrentRegionOnly=*/true,
              [](OMPClauseMappableExprCommon::MappableExprComponentListRef,
                 OpenMPClauseKind) { return true; }))
        continue;
      DeclRefExpr *DRE = buildDeclRefExpr(
          SemaRef, VD, VD->getType().getNonLValueExprType(SemaRef.Context),
          Cap.getLocation(), /*RefersToCapture=*/true);
      Visit(DRE);
    }
  }
  bool isErrorFound() const { return ErrorFound; }
  ArrayRef<Expr *> getImplicitFirstprivate() const {
    return ImplicitFirstprivate;
  }
  ArrayRef<Expr *> getImplicitMap(OpenMPDefaultmapClauseKind DK,
                                  OpenMPMapClauseKind MK) const {
    return ImplicitMap[DK][MK];
  }
  ArrayRef<OpenMPMapModifierKind>
  getImplicitMapModifier(OpenMPDefaultmapClauseKind Kind) const {
    return ImplicitMapModifier[Kind];
  }
  const Sema::VarsWithInheritedDSAType &getVarsWithInheritedDSA() const {
    return VarsWithInheritedDSA;
  }

  DSAAttrChecker(DSAStackTy *S, Sema &SemaRef, CapturedStmt *CS)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false), CS(CS) {
    // Process declare target link variables for the target directives.
    if (isOpenMPTargetExecutionDirective(S->getCurrentDirective())) {
      for (DeclRefExpr *E : Stack->getLinkGlobals())
        Visit(E);
    }
  }
};
} // namespace

static void handleDeclareVariantConstructTrait(DSAStackTy *Stack,
                                               OpenMPDirectiveKind DKind,
                                               bool ScopeEntry) {
  SmallVector<llvm::omp::TraitProperty, 8> Traits;
  if (isOpenMPTargetExecutionDirective(DKind))
    Traits.emplace_back(llvm::omp::TraitProperty::construct_target_target);
  if (isOpenMPTeamsDirective(DKind))
    Traits.emplace_back(llvm::omp::TraitProperty::construct_teams_teams);
  if (isOpenMPParallelDirective(DKind))
    Traits.emplace_back(llvm::omp::TraitProperty::construct_parallel_parallel);
  if (isOpenMPWorksharingDirective(DKind))
    Traits.emplace_back(llvm::omp::TraitProperty::construct_for_for);
  if (isOpenMPSimdDirective(DKind))
    Traits.emplace_back(llvm::omp::TraitProperty::construct_simd_simd);
  Stack->handleConstructTrait(Traits, ScopeEntry);
}

void Sema::ActOnOpenMPRegionStart(OpenMPDirectiveKind DKind, Scope *CurScope) {
  switch (DKind) {
  case OMPD_parallel:
  case OMPD_parallel_for:
  case OMPD_parallel_for_simd:
  case OMPD_parallel_sections:
  case OMPD_parallel_master:
  case OMPD_teams:
  case OMPD_teams_distribute:
  case OMPD_teams_distribute_simd: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_target_teams:
  case OMPD_target_parallel:
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
  case OMPD_target_teams_distribute:
  case OMPD_target_teams_distribute_simd: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType VoidPtrTy = Context.VoidPtrTy.withConst().withRestrict();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    QualType Args[] = {VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32PtrTy),
        std::make_pair(".privates.", VoidPtrTy),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(".task_t.", Context.VoidPtrTy.withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params, /*OpenMPCaptureLevel=*/0);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, {}, AttributeCommonInfo::AS_Keyword,
            AlwaysInlineAttr::Keyword_forceinline));
    Sema::CapturedParamNameType ParamsTarget[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'target' with no implicit parameters.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsTarget, /*OpenMPCaptureLevel=*/1);
    Sema::CapturedParamNameType ParamsTeamsOrParallel[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'teams' or 'parallel'.  Both regions have
    // the same implicit parameters.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsTeamsOrParallel, /*OpenMPCaptureLevel=*/2);
    break;
  }
  case OMPD_target:
  case OMPD_target_simd: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType VoidPtrTy = Context.VoidPtrTy.withConst().withRestrict();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    QualType Args[] = {VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32PtrTy),
        std::make_pair(".privates.", VoidPtrTy),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(".task_t.", Context.VoidPtrTy.withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params, /*OpenMPCaptureLevel=*/0);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, {}, AttributeCommonInfo::AS_Keyword,
            AlwaysInlineAttr::Keyword_forceinline));
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             std::make_pair(StringRef(), QualType()),
                             /*OpenMPCaptureLevel=*/1);
    break;
  }
  case OMPD_atomic:
  case OMPD_critical:
  case OMPD_section:
  case OMPD_master:
  case OMPD_masked:
  case OMPD_tile:
  case OMPD_unroll:
    break;
  case OMPD_loop:
    // TODO: 'loop' may require additional parameters depending on the binding.
    // Treat similar to OMPD_simd/OMPD_for for now.
  case OMPD_simd:
  case OMPD_for:
  case OMPD_for_simd:
  case OMPD_sections:
  case OMPD_single:
  case OMPD_taskgroup:
  case OMPD_distribute:
  case OMPD_distribute_simd:
  case OMPD_ordered:
  case OMPD_target_data:
  case OMPD_dispatch: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_task: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType VoidPtrTy = Context.VoidPtrTy.withConst().withRestrict();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    QualType Args[] = {VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32PtrTy),
        std::make_pair(".privates.", VoidPtrTy),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(".task_t.", Context.VoidPtrTy.withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, {}, AttributeCommonInfo::AS_Keyword,
            AlwaysInlineAttr::Keyword_forceinline));
    break;
  }
  case OMPD_taskloop:
  case OMPD_taskloop_simd:
  case OMPD_master_taskloop:
  case OMPD_master_taskloop_simd: {
    QualType KmpInt32Ty =
        Context.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1)
            .withConst();
    QualType KmpUInt64Ty =
        Context.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/0)
            .withConst();
    QualType KmpInt64Ty =
        Context.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1)
            .withConst();
    QualType VoidPtrTy = Context.VoidPtrTy.withConst().withRestrict();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    QualType Args[] = {VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32PtrTy),
        std::make_pair(".privates.", VoidPtrTy),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(".task_t.", Context.VoidPtrTy.withConst()),
        std::make_pair(".lb.", KmpUInt64Ty),
        std::make_pair(".ub.", KmpUInt64Ty),
        std::make_pair(".st.", KmpInt64Ty),
        std::make_pair(".liter.", KmpInt32Ty),
        std::make_pair(".reductions.", VoidPtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, {}, AttributeCommonInfo::AS_Keyword,
            AlwaysInlineAttr::Keyword_forceinline));
    break;
  }
  case OMPD_parallel_master_taskloop:
  case OMPD_parallel_master_taskloop_simd: {
    QualType KmpInt32Ty =
        Context.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1)
            .withConst();
    QualType KmpUInt64Ty =
        Context.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/0)
            .withConst();
    QualType KmpInt64Ty =
        Context.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1)
            .withConst();
    QualType VoidPtrTy = Context.VoidPtrTy.withConst().withRestrict();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType ParamsParallel[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'parallel'.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsParallel, /*OpenMPCaptureLevel=*/0);
    QualType Args[] = {VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32PtrTy),
        std::make_pair(".privates.", VoidPtrTy),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(".task_t.", Context.VoidPtrTy.withConst()),
        std::make_pair(".lb.", KmpUInt64Ty),
        std::make_pair(".ub.", KmpUInt64Ty),
        std::make_pair(".st.", KmpInt64Ty),
        std::make_pair(".liter.", KmpInt32Ty),
        std::make_pair(".reductions.", VoidPtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params, /*OpenMPCaptureLevel=*/1);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, {}, AttributeCommonInfo::AS_Keyword,
            AlwaysInlineAttr::Keyword_forceinline));
    break;
  }
  case OMPD_distribute_parallel_for_simd:
  case OMPD_distribute_parallel_for: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(".previous.lb.", Context.getSizeType().withConst()),
        std::make_pair(".previous.ub.", Context.getSizeType().withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_target_teams_distribute_parallel_for:
  case OMPD_target_teams_distribute_parallel_for_simd: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    QualType VoidPtrTy = Context.VoidPtrTy.withConst().withRestrict();

    QualType Args[] = {VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32PtrTy),
        std::make_pair(".privates.", VoidPtrTy),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(".task_t.", Context.VoidPtrTy.withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params, /*OpenMPCaptureLevel=*/0);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, {}, AttributeCommonInfo::AS_Keyword,
            AlwaysInlineAttr::Keyword_forceinline));
    Sema::CapturedParamNameType ParamsTarget[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'target' with no implicit parameters.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsTarget, /*OpenMPCaptureLevel=*/1);

    Sema::CapturedParamNameType ParamsTeams[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'target' with no implicit parameters.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsTeams, /*OpenMPCaptureLevel=*/2);

    Sema::CapturedParamNameType ParamsParallel[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(".previous.lb.", Context.getSizeType().withConst()),
        std::make_pair(".previous.ub.", Context.getSizeType().withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'teams' or 'parallel'.  Both regions have
    // the same implicit parameters.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsParallel, /*OpenMPCaptureLevel=*/3);
    break;
  }

  case OMPD_teams_distribute_parallel_for:
  case OMPD_teams_distribute_parallel_for_simd: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();

    Sema::CapturedParamNameType ParamsTeams[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'target' with no implicit parameters.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsTeams, /*OpenMPCaptureLevel=*/0);

    Sema::CapturedParamNameType ParamsParallel[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(".previous.lb.", Context.getSizeType().withConst()),
        std::make_pair(".previous.ub.", Context.getSizeType().withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    // Start a captured region for 'teams' or 'parallel'.  Both regions have
    // the same implicit parameters.
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             ParamsParallel, /*OpenMPCaptureLevel=*/1);
    break;
  }
  case OMPD_target_update:
  case OMPD_target_enter_data:
  case OMPD_target_exit_data: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1).withConst();
    QualType VoidPtrTy = Context.VoidPtrTy.withConst().withRestrict();
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    QualType Args[] = {VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32PtrTy),
        std::make_pair(".privates.", VoidPtrTy),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(".task_t.", Context.VoidPtrTy.withConst()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, {}, AttributeCommonInfo::AS_Keyword,
            AlwaysInlineAttr::Keyword_forceinline));
    break;
  }
  case OMPD_threadprivate:
  case OMPD_allocate:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_cancellation_point:
  case OMPD_cancel:
  case OMPD_flush:
  case OMPD_depobj:
  case OMPD_scan:
  case OMPD_declare_reduction:
  case OMPD_declare_mapper:
  case OMPD_declare_simd:
  case OMPD_declare_target:
  case OMPD_end_declare_target:
  case OMPD_requires:
  case OMPD_declare_variant:
  case OMPD_begin_declare_variant:
  case OMPD_end_declare_variant:
  case OMPD_metadirective:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
  default:
    llvm_unreachable("Unknown OpenMP directive");
  }
  DSAStack->setContext(CurContext);
  handleDeclareVariantConstructTrait(DSAStack, DKind, /* ScopeEntry */ true);
}

int Sema::getNumberOfConstructScopes(unsigned Level) const {
  return getOpenMPCaptureLevels(DSAStack->getDirective(Level));
}

int Sema::getOpenMPCaptureLevels(OpenMPDirectiveKind DKind) {
  SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
  getOpenMPCaptureRegions(CaptureRegions, DKind);
  return CaptureRegions.size();
}

static OMPCapturedExprDecl *buildCaptureDecl(Sema &S, IdentifierInfo *Id,
                                             Expr *CaptureExpr, bool WithInit,
                                             bool AsExpression) {
  assert(CaptureExpr);
  ASTContext &C = S.getASTContext();
  Expr *Init = AsExpression ? CaptureExpr : CaptureExpr->IgnoreImpCasts();
  QualType Ty = Init->getType();
  if (CaptureExpr->getObjectKind() == OK_Ordinary && CaptureExpr->isGLValue()) {
    if (S.getLangOpts().CPlusPlus) {
      Ty = C.getLValueReferenceType(Ty);
    } else {
      Ty = C.getPointerType(Ty);
      ExprResult Res =
          S.CreateBuiltinUnaryOp(CaptureExpr->getExprLoc(), UO_AddrOf, Init);
      if (!Res.isUsable())
        return nullptr;
      Init = Res.get();
    }
    WithInit = true;
  }
  auto *CED = OMPCapturedExprDecl::Create(C, S.CurContext, Id, Ty,
                                          CaptureExpr->getBeginLoc());
  if (!WithInit)
    CED->addAttr(OMPCaptureNoInitAttr::CreateImplicit(C));
  S.CurContext->addHiddenDecl(CED);
  Sema::TentativeAnalysisScope Trap(S);
  S.AddInitializerToDecl(CED, Init, /*DirectInit=*/false);
  return CED;
}

static DeclRefExpr *buildCapture(Sema &S, ValueDecl *D, Expr *CaptureExpr,
                                 bool WithInit) {
  OMPCapturedExprDecl *CD;
  if (VarDecl *VD = S.isOpenMPCapturedDecl(D))
    CD = cast<OMPCapturedExprDecl>(VD);
  else
    CD = buildCaptureDecl(S, D->getIdentifier(), CaptureExpr, WithInit,
                          /*AsExpression=*/false);
  return buildDeclRefExpr(S, CD, CD->getType().getNonReferenceType(),
                          CaptureExpr->getExprLoc());
}

static ExprResult buildCapture(Sema &S, Expr *CaptureExpr, DeclRefExpr *&Ref) {
  CaptureExpr = S.DefaultLvalueConversion(CaptureExpr).get();
  if (!Ref) {
    OMPCapturedExprDecl *CD = buildCaptureDecl(
        S, &S.getASTContext().Idents.get(".capture_expr."), CaptureExpr,
        /*WithInit=*/true, /*AsExpression=*/true);
    Ref = buildDeclRefExpr(S, CD, CD->getType().getNonReferenceType(),
                           CaptureExpr->getExprLoc());
  }
  ExprResult Res = Ref;
  if (!S.getLangOpts().CPlusPlus &&
      CaptureExpr->getObjectKind() == OK_Ordinary && CaptureExpr->isGLValue() &&
      Ref->getType()->isPointerType()) {
    Res = S.CreateBuiltinUnaryOp(CaptureExpr->getExprLoc(), UO_Deref, Ref);
    if (!Res.isUsable())
      return ExprError();
  }
  return S.DefaultLvalueConversion(Res.get());
}

namespace {
// OpenMP directives parsed in this section are represented as a
// CapturedStatement with an associated statement.  If a syntax error
// is detected during the parsing of the associated statement, the
// compiler must abort processing and close the CapturedStatement.
//
// Combined directives such as 'target parallel' have more than one
// nested CapturedStatements.  This RAII ensures that we unwind out
// of all the nested CapturedStatements when an error is found.
class CaptureRegionUnwinderRAII {
private:
  Sema &S;
  bool &ErrorFound;
  OpenMPDirectiveKind DKind = OMPD_unknown;

public:
  CaptureRegionUnwinderRAII(Sema &S, bool &ErrorFound,
                            OpenMPDirectiveKind DKind)
      : S(S), ErrorFound(ErrorFound), DKind(DKind) {}
  ~CaptureRegionUnwinderRAII() {
    if (ErrorFound) {
      int ThisCaptureLevel = S.getOpenMPCaptureLevels(DKind);
      while (--ThisCaptureLevel >= 0)
        S.ActOnCapturedRegionError();
    }
  }
};
} // namespace

void Sema::tryCaptureOpenMPLambdas(ValueDecl *V) {
  // Capture variables captured by reference in lambdas for target-based
  // directives.
  if (!CurContext->isDependentContext() &&
      (isOpenMPTargetExecutionDirective(DSAStack->getCurrentDirective()) ||
       isOpenMPTargetDataManagementDirective(
           DSAStack->getCurrentDirective()))) {
    QualType Type = V->getType();
    if (const auto *RD = Type.getCanonicalType()
                             .getNonReferenceType()
                             ->getAsCXXRecordDecl()) {
      bool SavedForceCaptureByReferenceInTargetExecutable =
          DSAStack->isForceCaptureByReferenceInTargetExecutable();
      DSAStack->setForceCaptureByReferenceInTargetExecutable(
          /*V=*/true);
      if (RD->isLambda()) {
        llvm::DenseMap<const VarDecl *, FieldDecl *> Captures;
        FieldDecl *ThisCapture;
        RD->getCaptureFields(Captures, ThisCapture);
        for (const LambdaCapture &LC : RD->captures()) {
          if (LC.getCaptureKind() == LCK_ByRef) {
            VarDecl *VD = LC.getCapturedVar();
            DeclContext *VDC = VD->getDeclContext();
            if (!VDC->Encloses(CurContext))
              continue;
            MarkVariableReferenced(LC.getLocation(), VD);
          } else if (LC.getCaptureKind() == LCK_This) {
            QualType ThisTy = getCurrentThisType();
            if (!ThisTy.isNull() &&
                Context.typesAreCompatible(ThisTy, ThisCapture->getType()))
              CheckCXXThisCapture(LC.getLocation());
          }
        }
      }
      DSAStack->setForceCaptureByReferenceInTargetExecutable(
          SavedForceCaptureByReferenceInTargetExecutable);
    }
  }
}

static bool checkOrderedOrderSpecified(Sema &S,
                                       const ArrayRef<OMPClause *> Clauses) {
  const OMPOrderedClause *Ordered = nullptr;
  const OMPOrderClause *Order = nullptr;

  for (const OMPClause *Clause : Clauses) {
    if (Clause->getClauseKind() == OMPC_ordered)
      Ordered = cast<OMPOrderedClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_order) {
      Order = cast<OMPOrderClause>(Clause);
      if (Order->getKind() != OMPC_ORDER_concurrent)
        Order = nullptr;
    }
    if (Ordered && Order)
      break;
  }

  if (Ordered && Order) {
    S.Diag(Order->getKindKwLoc(),
           diag::err_omp_simple_clause_incompatible_with_ordered)
        << getOpenMPClauseName(OMPC_order)
        << getOpenMPSimpleClauseTypeName(OMPC_order, OMPC_ORDER_concurrent)
        << SourceRange(Order->getBeginLoc(), Order->getEndLoc());
    S.Diag(Ordered->getBeginLoc(), diag::note_omp_ordered_param)
        << 0 << SourceRange(Ordered->getBeginLoc(), Ordered->getEndLoc());
    return true;
  }
  return false;
}

StmtResult Sema::ActOnOpenMPRegionEnd(StmtResult S,
                                      ArrayRef<OMPClause *> Clauses) {
  handleDeclareVariantConstructTrait(DSAStack, DSAStack->getCurrentDirective(),
                                     /* ScopeEntry */ false);
  if (DSAStack->getCurrentDirective() == OMPD_atomic ||
      DSAStack->getCurrentDirective() == OMPD_critical ||
      DSAStack->getCurrentDirective() == OMPD_section ||
      DSAStack->getCurrentDirective() == OMPD_master ||
      DSAStack->getCurrentDirective() == OMPD_masked)
    return S;

  bool ErrorFound = false;
  CaptureRegionUnwinderRAII CaptureRegionUnwinder(
      *this, ErrorFound, DSAStack->getCurrentDirective());
  if (!S.isUsable()) {
    ErrorFound = true;
    return StmtError();
  }

  SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
  getOpenMPCaptureRegions(CaptureRegions, DSAStack->getCurrentDirective());
  OMPOrderedClause *OC = nullptr;
  OMPScheduleClause *SC = nullptr;
  SmallVector<const OMPLinearClause *, 4> LCs;
  SmallVector<const OMPClauseWithPreInit *, 4> PICs;
  // This is required for proper codegen.
  for (OMPClause *Clause : Clauses) {
    if (!LangOpts.OpenMPSimd &&
        isOpenMPTaskingDirective(DSAStack->getCurrentDirective()) &&
        Clause->getClauseKind() == OMPC_in_reduction) {
      // Capture taskgroup task_reduction descriptors inside the tasking regions
      // with the corresponding in_reduction items.
      auto *IRC = cast<OMPInReductionClause>(Clause);
      for (Expr *E : IRC->taskgroup_descriptors())
        if (E)
          MarkDeclarationsReferencedInExpr(E);
    }
    if (isOpenMPPrivate(Clause->getClauseKind()) ||
        Clause->getClauseKind() == OMPC_copyprivate ||
        (getLangOpts().OpenMPUseTLS &&
         getASTContext().getTargetInfo().isTLSSupported() &&
         Clause->getClauseKind() == OMPC_copyin)) {
      DSAStack->setForceVarCapturing(Clause->getClauseKind() == OMPC_copyin);
      // Mark all variables in private list clauses as used in inner region.
      for (Stmt *VarRef : Clause->children()) {
        if (auto *E = cast_or_null<Expr>(VarRef)) {
          MarkDeclarationsReferencedInExpr(E);
        }
      }
      DSAStack->setForceVarCapturing(/*V=*/false);
    } else if (isOpenMPLoopTransformationDirective(
                   DSAStack->getCurrentDirective())) {
      assert(CaptureRegions.empty() &&
             "No captured regions in loop transformation directives.");
    } else if (CaptureRegions.size() > 1 ||
               CaptureRegions.back() != OMPD_unknown) {
      if (auto *C = OMPClauseWithPreInit::get(Clause))
        PICs.push_back(C);
      if (auto *C = OMPClauseWithPostUpdate::get(Clause)) {
        if (Expr *E = C->getPostUpdateExpr())
          MarkDeclarationsReferencedInExpr(E);
      }
    }
    if (Clause->getClauseKind() == OMPC_schedule)
      SC = cast<OMPScheduleClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_ordered)
      OC = cast<OMPOrderedClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_linear)
      LCs.push_back(cast<OMPLinearClause>(Clause));
  }
  // Capture allocator expressions if used.
  for (Expr *E : DSAStack->getInnerAllocators())
    MarkDeclarationsReferencedInExpr(E);
  // OpenMP, 2.7.1 Loop Construct, Restrictions
  // The nonmonotonic modifier cannot be specified if an ordered clause is
  // specified.
  if (SC &&
      (SC->getFirstScheduleModifier() == OMPC_SCHEDULE_MODIFIER_nonmonotonic ||
       SC->getSecondScheduleModifier() ==
           OMPC_SCHEDULE_MODIFIER_nonmonotonic) &&
      OC) {
    Diag(SC->getFirstScheduleModifier() == OMPC_SCHEDULE_MODIFIER_nonmonotonic
             ? SC->getFirstScheduleModifierLoc()
             : SC->getSecondScheduleModifierLoc(),
         diag::err_omp_simple_clause_incompatible_with_ordered)
        << getOpenMPClauseName(OMPC_schedule)
        << getOpenMPSimpleClauseTypeName(OMPC_schedule,
                                         OMPC_SCHEDULE_MODIFIER_nonmonotonic)
        << SourceRange(OC->getBeginLoc(), OC->getEndLoc());
    ErrorFound = true;
  }
  // OpenMP 5.0, 2.9.2 Worksharing-Loop Construct, Restrictions.
  // If an order(concurrent) clause is present, an ordered clause may not appear
  // on the same directive.
  if (checkOrderedOrderSpecified(*this, Clauses))
    ErrorFound = true;
  if (!LCs.empty() && OC && OC->getNumForLoops()) {
    for (const OMPLinearClause *C : LCs) {
      Diag(C->getBeginLoc(), diag::err_omp_linear_ordered)
          << SourceRange(OC->getBeginLoc(), OC->getEndLoc());
    }
    ErrorFound = true;
  }
  if (isOpenMPWorksharingDirective(DSAStack->getCurrentDirective()) &&
      isOpenMPSimdDirective(DSAStack->getCurrentDirective()) && OC &&
      OC->getNumForLoops()) {
    Diag(OC->getBeginLoc(), diag::err_omp_ordered_simd)
        << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
    ErrorFound = true;
  }
  if (ErrorFound) {
    return StmtError();
  }
  StmtResult SR = S;
  unsigned CompletedRegions = 0;
  for (OpenMPDirectiveKind ThisCaptureRegion : llvm::reverse(CaptureRegions)) {
    // Mark all variables in private list clauses as used in inner region.
    // Required for proper codegen of combined directives.
    // TODO: add processing for other clauses.
    if (ThisCaptureRegion != OMPD_unknown) {
      for (const clang::OMPClauseWithPreInit *C : PICs) {
        OpenMPDirectiveKind CaptureRegion = C->getCaptureRegion();
        // Find the particular capture region for the clause if the
        // directive is a combined one with multiple capture regions.
        // If the directive is not a combined one, the capture region
        // associated with the clause is OMPD_unknown and is generated
        // only once.
        if (CaptureRegion == ThisCaptureRegion ||
            CaptureRegion == OMPD_unknown) {
          if (auto *DS = cast_or_null<DeclStmt>(C->getPreInitStmt())) {
            for (Decl *D : DS->decls())
              MarkVariableReferenced(D->getLocation(), cast<VarDecl>(D));
          }
        }
      }
    }
    if (ThisCaptureRegion == OMPD_target) {
      // Capture allocator traits in the target region. They are used implicitly
      // and, thus, are not captured by default.
      for (OMPClause *C : Clauses) {
        if (const auto *UAC = dyn_cast<OMPUsesAllocatorsClause>(C)) {
          for (unsigned I = 0, End = UAC->getNumberOfAllocators(); I < End;
               ++I) {
            OMPUsesAllocatorsClause::Data D = UAC->getAllocatorData(I);
            if (Expr *E = D.AllocatorTraits)
              MarkDeclarationsReferencedInExpr(E);
          }
          continue;
        }
      }
    }
    if (ThisCaptureRegion == OMPD_parallel) {
      // Capture temp arrays for inscan reductions and locals in aligned
      // clauses.
      for (OMPClause *C : Clauses) {
        if (auto *RC = dyn_cast<OMPReductionClause>(C)) {
          if (RC->getModifier() != OMPC_REDUCTION_inscan)
            continue;
          for (Expr *E : RC->copy_array_temps())
            MarkDeclarationsReferencedInExpr(E);
        }
        if (auto *AC = dyn_cast<OMPAlignedClause>(C)) {
          for (Expr *E : AC->varlists())
            MarkDeclarationsReferencedInExpr(E);
        }
      }
    }
    if (++CompletedRegions == CaptureRegions.size())
      DSAStack->setBodyComplete();
    SR = ActOnCapturedRegionEnd(SR.get());
  }
  return SR;
}

static bool checkCancelRegion(Sema &SemaRef, OpenMPDirectiveKind CurrentRegion,
                              OpenMPDirectiveKind CancelRegion,
                              SourceLocation StartLoc) {
  // CancelRegion is only needed for cancel and cancellation_point.
  if (CurrentRegion != OMPD_cancel && CurrentRegion != OMPD_cancellation_point)
    return false;

  if (CancelRegion == OMPD_parallel || CancelRegion == OMPD_for ||
      CancelRegion == OMPD_sections || CancelRegion == OMPD_taskgroup)
    return false;

  SemaRef.Diag(StartLoc, diag::err_omp_wrong_cancel_region)
      << getOpenMPDirectiveName(CancelRegion);
  return true;
}

static bool checkNestingOfRegions(Sema &SemaRef, const DSAStackTy *Stack,
                                  OpenMPDirectiveKind CurrentRegion,
                                  const DeclarationNameInfo &CurrentName,
                                  OpenMPDirectiveKind CancelRegion,
                                  OpenMPBindClauseKind BindKind,
                                  SourceLocation StartLoc) {
  if (Stack->getCurScope()) {
    OpenMPDirectiveKind ParentRegion = Stack->getParentDirective();
    OpenMPDirectiveKind OffendingRegion = ParentRegion;
    bool NestingProhibited = false;
    bool CloseNesting = true;
    bool OrphanSeen = false;
    enum {
      NoRecommend,
      ShouldBeInParallelRegion,
      ShouldBeInOrderedRegion,
      ShouldBeInTargetRegion,
      ShouldBeInTeamsRegion,
      ShouldBeInLoopSimdRegion,
    } Recommend = NoRecommend;
    if (isOpenMPSimdDirective(ParentRegion) &&
        ((SemaRef.LangOpts.OpenMP <= 45 && CurrentRegion != OMPD_ordered) ||
         (SemaRef.LangOpts.OpenMP >= 50 && CurrentRegion != OMPD_ordered &&
          CurrentRegion != OMPD_simd && CurrentRegion != OMPD_atomic &&
          CurrentRegion != OMPD_scan))) {
      // OpenMP [2.16, Nesting of Regions]
      // OpenMP constructs may not be nested inside a simd region.
      // OpenMP [2.8.1,simd Construct, Restrictions]
      // An ordered construct with the simd clause is the only OpenMP
      // construct that can appear in the simd region.
      // Allowing a SIMD construct nested in another SIMD construct is an
      // extension. The OpenMP 4.5 spec does not allow it. Issue a warning
      // message.
      // OpenMP 5.0 [2.9.3.1, simd Construct, Restrictions]
      // The only OpenMP constructs that can be encountered during execution of
      // a simd region are the atomic construct, the loop construct, the simd
      // construct and the ordered construct with the simd clause.
      SemaRef.Diag(StartLoc, (CurrentRegion != OMPD_simd)
                                 ? diag::err_omp_prohibited_region_simd
                                 : diag::warn_omp_nesting_simd)
          << (SemaRef.LangOpts.OpenMP >= 50 ? 1 : 0);
      return CurrentRegion != OMPD_simd;
    }
    if (ParentRegion == OMPD_atomic) {
      // OpenMP [2.16, Nesting of Regions]
      // OpenMP constructs may not be nested inside an atomic region.
      SemaRef.Diag(StartLoc, diag::err_omp_prohibited_region_atomic);
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
    // Allow some constructs (except teams and cancellation constructs) to be
    // orphaned (they could be used in functions, called from OpenMP regions
    // with the required preconditions).
    if (ParentRegion == OMPD_unknown &&
        !isOpenMPNestingTeamsDirective(CurrentRegion) &&
        CurrentRegion != OMPD_cancellation_point &&
        CurrentRegion != OMPD_cancel && CurrentRegion != OMPD_scan)
      return false;
    if (CurrentRegion == OMPD_cancellation_point ||
        CurrentRegion == OMPD_cancel) {
      // OpenMP [2.16, Nesting of Regions]
      // A cancellation point construct for which construct-type-clause is
      // taskgroup must be nested inside a task construct. A cancellation
      // point construct for which construct-type-clause is not taskgroup must
      // be closely nested inside an OpenMP construct that matches the type
      // specified in construct-type-clause.
      // A cancel construct for which construct-type-clause is taskgroup must be
      // nested inside a task construct. A cancel construct for which
      // construct-type-clause is not taskgroup must be closely nested inside an
      // OpenMP construct that matches the type specified in
      // construct-type-clause.
      NestingProhibited =
          !((CancelRegion == OMPD_parallel &&
             (ParentRegion == OMPD_parallel ||
              ParentRegion == OMPD_target_parallel)) ||
            (CancelRegion == OMPD_for &&
             (ParentRegion == OMPD_for || ParentRegion == OMPD_parallel_for ||
              ParentRegion == OMPD_target_parallel_for ||
              ParentRegion == OMPD_distribute_parallel_for ||
              ParentRegion == OMPD_teams_distribute_parallel_for ||
              ParentRegion == OMPD_target_teams_distribute_parallel_for)) ||
            (CancelRegion == OMPD_taskgroup &&
             (ParentRegion == OMPD_task ||
              (SemaRef.getLangOpts().OpenMP >= 50 &&
               (ParentRegion == OMPD_taskloop ||
                ParentRegion == OMPD_master_taskloop ||
                ParentRegion == OMPD_parallel_master_taskloop)))) ||
            (CancelRegion == OMPD_sections &&
             (ParentRegion == OMPD_section || ParentRegion == OMPD_sections ||
              ParentRegion == OMPD_parallel_sections)));
      OrphanSeen = ParentRegion == OMPD_unknown;
    } else if (CurrentRegion == OMPD_master || CurrentRegion == OMPD_masked) {
      // OpenMP 5.1 [2.22, Nesting of Regions]
      // A masked region may not be closely nested inside a worksharing, loop,
      // atomic, task, or taskloop region.
      NestingProhibited = isOpenMPWorksharingDirective(ParentRegion) ||
                          isOpenMPGenericLoopDirective(ParentRegion) ||
                          isOpenMPTaskingDirective(ParentRegion);
    } else if (CurrentRegion == OMPD_critical && CurrentName.getName()) {
      // OpenMP [2.16, Nesting of Regions]
      // A critical region may not be nested (closely or otherwise) inside a
      // critical region with the same name. Note that this restriction is not
      // sufficient to prevent deadlock.
      SourceLocation PreviousCriticalLoc;
      bool DeadLock = Stack->hasDirective(
          [CurrentName, &PreviousCriticalLoc](OpenMPDirectiveKind K,
                                              const DeclarationNameInfo &DNI,
                                              SourceLocation Loc) {
            if (K == OMPD_critical && DNI.getName() == CurrentName.getName()) {
              PreviousCriticalLoc = Loc;
              return true;
            }
            return false;
          },
          false /* skip top directive */);
      if (DeadLock) {
        SemaRef.Diag(StartLoc,
                     diag::err_omp_prohibited_region_critical_same_name)
            << CurrentName.getName();
        if (PreviousCriticalLoc.isValid())
          SemaRef.Diag(PreviousCriticalLoc,
                       diag::note_omp_previous_critical_region);
        return true;
      }
    } else if (CurrentRegion == OMPD_barrier) {
      // OpenMP 5.1 [2.22, Nesting of Regions]
      // A barrier region may not be closely nested inside a worksharing, loop,
      // task, taskloop, critical, ordered, atomic, or masked region.
      NestingProhibited =
          isOpenMPWorksharingDirective(ParentRegion) ||
          isOpenMPGenericLoopDirective(ParentRegion) ||
          isOpenMPTaskingDirective(ParentRegion) ||
          ParentRegion == OMPD_master || ParentRegion == OMPD_masked ||
          ParentRegion == OMPD_parallel_master ||
          ParentRegion == OMPD_critical || ParentRegion == OMPD_ordered;
    } else if (isOpenMPWorksharingDirective(CurrentRegion) &&
               !isOpenMPParallelDirective(CurrentRegion) &&
               !isOpenMPTeamsDirective(CurrentRegion)) {
      // OpenMP 5.1 [2.22, Nesting of Regions]
      // A loop region that binds to a parallel region or a worksharing region
      // may not be closely nested inside a worksharing, loop, task, taskloop,
      // critical, ordered, atomic, or masked region.
      NestingProhibited =
          isOpenMPWorksharingDirective(ParentRegion) ||
          isOpenMPGenericLoopDirective(ParentRegion) ||
          isOpenMPTaskingDirective(ParentRegion) ||
          ParentRegion == OMPD_master || ParentRegion == OMPD_masked ||
          ParentRegion == OMPD_parallel_master ||
          ParentRegion == OMPD_critical || ParentRegion == OMPD_ordered;
      Recommend = ShouldBeInParallelRegion;
    } else if (CurrentRegion == OMPD_ordered) {
      // OpenMP [2.16, Nesting of Regions]
      // An ordered region may not be closely nested inside a critical,
      // atomic, or explicit task region.
      // An ordered region must be closely nested inside a loop region (or
      // parallel loop region) with an ordered clause.
      // OpenMP [2.8.1,simd Construct, Restrictions]
      // An ordered construct with the simd clause is the only OpenMP construct
      // that can appear in the simd region.
      NestingProhibited = ParentRegion == OMPD_critical ||
                          isOpenMPTaskingDirective(ParentRegion) ||
                          !(isOpenMPSimdDirective(ParentRegion) ||
                            Stack->isParentOrderedRegion());
      Recommend = ShouldBeInOrderedRegion;
    } else if (isOpenMPNestingTeamsDirective(CurrentRegion)) {
      // OpenMP [2.16, Nesting of Regions]
      // If specified, a teams construct must be contained within a target
      // construct.
      NestingProhibited =
          (SemaRef.LangOpts.OpenMP <= 45 && ParentRegion != OMPD_target) ||
          (SemaRef.LangOpts.OpenMP >= 50 && ParentRegion != OMPD_unknown &&
           ParentRegion != OMPD_target);
      OrphanSeen = ParentRegion == OMPD_unknown;
      Recommend = ShouldBeInTargetRegion;
    } else if (CurrentRegion == OMPD_scan) {
      // OpenMP [2.16, Nesting of Regions]
      // If specified, a teams construct must be contained within a target
      // construct.
      NestingProhibited =
          SemaRef.LangOpts.OpenMP < 50 ||
          (ParentRegion != OMPD_simd && ParentRegion != OMPD_for &&
           ParentRegion != OMPD_for_simd && ParentRegion != OMPD_parallel_for &&
           ParentRegion != OMPD_parallel_for_simd);
      OrphanSeen = ParentRegion == OMPD_unknown;
      Recommend = ShouldBeInLoopSimdRegion;
    }
    if (!NestingProhibited &&
        !isOpenMPTargetExecutionDirective(CurrentRegion) &&
        !isOpenMPTargetDataManagementDirective(CurrentRegion) &&
        (ParentRegion == OMPD_teams || ParentRegion == OMPD_target_teams)) {
      // OpenMP [5.1, 2.22, Nesting of Regions]
      // distribute, distribute simd, distribute parallel worksharing-loop,
      // distribute parallel worksharing-loop SIMD, loop, parallel regions,
      // including any parallel regions arising from combined constructs,
      // omp_get_num_teams() regions, and omp_get_team_num() regions are the
      // only OpenMP regions that may be strictly nested inside the teams
      // region.
      NestingProhibited = !isOpenMPParallelDirective(CurrentRegion) &&
                          !isOpenMPDistributeDirective(CurrentRegion) &&
                          CurrentRegion != OMPD_loop;
      Recommend = ShouldBeInParallelRegion;
    }
    if (!NestingProhibited && CurrentRegion == OMPD_loop) {
      // OpenMP [5.1, 2.11.7, loop Construct, Restrictions]
      // If the bind clause is present on the loop construct and binding is
      // teams then the corresponding loop region must be strictly nested inside
      // a teams region.
      NestingProhibited = BindKind == OMPC_BIND_teams &&
                          ParentRegion != OMPD_teams &&
                          ParentRegion != OMPD_target_teams;
      Recommend = ShouldBeInTeamsRegion;
    }
    if (!NestingProhibited &&
        isOpenMPNestingDistributeDirective(CurrentRegion)) {
      // OpenMP 4.5 [2.17 Nesting of Regions]
      // The region associated with the distribute construct must be strictly
      // nested inside a teams region
      NestingProhibited =
          (ParentRegion != OMPD_teams && ParentRegion != OMPD_target_teams);
      Recommend = ShouldBeInTeamsRegion;
    }
    if (!NestingProhibited &&
        (isOpenMPTargetExecutionDirective(CurrentRegion) ||
         isOpenMPTargetDataManagementDirective(CurrentRegion))) {
      // OpenMP 4.5 [2.17 Nesting of Regions]
      // If a target, target update, target data, target enter data, or
      // target exit data construct is encountered during execution of a
      // target region, the behavior is unspecified.
      NestingProhibited = Stack->hasDirective(
          [&OffendingRegion](OpenMPDirectiveKind K, const DeclarationNameInfo &,
                             SourceLocation) {
            if (isOpenMPTargetExecutionDirective(K)) {
              OffendingRegion = K;
              return true;
            }
            return false;
          },
          false /* don't skip top directive */);
      CloseNesting = false;
    }
    if (NestingProhibited) {
      if (OrphanSeen) {
        SemaRef.Diag(StartLoc, diag::err_omp_orphaned_device_directive)
            << getOpenMPDirectiveName(CurrentRegion) << Recommend;
      } else {
        SemaRef.Diag(StartLoc, diag::err_omp_prohibited_region)
            << CloseNesting << getOpenMPDirectiveName(OffendingRegion)
            << Recommend << getOpenMPDirectiveName(CurrentRegion);
      }
      return true;
    }
  }
  return false;
}

struct Kind2Unsigned {
  using argument_type = OpenMPDirectiveKind;
  unsigned operator()(argument_type DK) { return unsigned(DK); }
};
static bool checkIfClauses(Sema &S, OpenMPDirectiveKind Kind,
                           ArrayRef<OMPClause *> Clauses,
                           ArrayRef<OpenMPDirectiveKind> AllowedNameModifiers) {
  bool ErrorFound = false;
  unsigned NamedModifiersNumber = 0;
  llvm::IndexedMap<const OMPIfClause *, Kind2Unsigned> FoundNameModifiers;
  FoundNameModifiers.resize(llvm::omp::Directive_enumSize + 1);
  SmallVector<SourceLocation, 4> NameModifierLoc;
  for (const OMPClause *C : Clauses) {
    if (const auto *IC = dyn_cast_or_null<OMPIfClause>(C)) {
      // At most one if clause without a directive-name-modifier can appear on
      // the directive.
      OpenMPDirectiveKind CurNM = IC->getNameModifier();
      if (FoundNameModifiers[CurNM]) {
        S.Diag(C->getBeginLoc(), diag::err_omp_more_one_clause)
            << getOpenMPDirectiveName(Kind) << getOpenMPClauseName(OMPC_if)
            << (CurNM != OMPD_unknown) << getOpenMPDirectiveName(CurNM);
        ErrorFound = true;
      } else if (CurNM != OMPD_unknown) {
        NameModifierLoc.push_back(IC->getNameModifierLoc());
        ++NamedModifiersNumber;
      }
      FoundNameModifiers[CurNM] = IC;
      if (CurNM == OMPD_unknown)
        continue;
      // Check if the specified name modifier is allowed for the current
      // directive.
      // At most one if clause with the particular directive-name-modifier can
      // appear on the directive.
      if (!llvm::is_contained(AllowedNameModifiers, CurNM)) {
        S.Diag(IC->getNameModifierLoc(),
               diag::err_omp_wrong_if_directive_name_modifier)
            << getOpenMPDirectiveName(CurNM) << getOpenMPDirectiveName(Kind);
        ErrorFound = true;
      }
    }
  }
  // If any if clause on the directive includes a directive-name-modifier then
  // all if clauses on the directive must include a directive-name-modifier.
  if (FoundNameModifiers[OMPD_unknown] && NamedModifiersNumber > 0) {
    if (NamedModifiersNumber == AllowedNameModifiers.size()) {
      S.Diag(FoundNameModifiers[OMPD_unknown]->getBeginLoc(),
             diag::err_omp_no_more_if_clause);
    } else {
      std::string Values;
      std::string Sep(", ");
      unsigned AllowedCnt = 0;
      unsigned TotalAllowedNum =
          AllowedNameModifiers.size() - NamedModifiersNumber;
      for (unsigned Cnt = 0, End = AllowedNameModifiers.size(); Cnt < End;
           ++Cnt) {
        OpenMPDirectiveKind NM = AllowedNameModifiers[Cnt];
        if (!FoundNameModifiers[NM]) {
          Values += "'";
          Values += getOpenMPDirectiveName(NM);
          Values += "'";
          if (AllowedCnt + 2 == TotalAllowedNum)
            Values += " or ";
          else if (AllowedCnt + 1 != TotalAllowedNum)
            Values += Sep;
          ++AllowedCnt;
        }
      }
      S.Diag(FoundNameModifiers[OMPD_unknown]->getCondition()->getBeginLoc(),
             diag::err_omp_unnamed_if_clause)
          << (TotalAllowedNum > 1) << Values;
    }
    for (SourceLocation Loc : NameModifierLoc) {
      S.Diag(Loc, diag::note_omp_previous_named_if_clause);
    }
    ErrorFound = true;
  }
  return ErrorFound;
}

static std::pair<ValueDecl *, bool> getPrivateItem(Sema &S, Expr *&RefExpr,
                                                   SourceLocation &ELoc,
                                                   SourceRange &ERange,
                                                   bool AllowArraySection) {
  if (RefExpr->isTypeDependent() || RefExpr->isValueDependent() ||
      RefExpr->containsUnexpandedParameterPack())
    return std::make_pair(nullptr, true);

  // OpenMP [3.1, C/C++]
  //  A list item is a variable name.
  // OpenMP  [2.9.3.3, Restrictions, p.1]
  //  A variable that is part of another variable (as an array or
  //  structure element) cannot appear in a private clause.
  RefExpr = RefExpr->IgnoreParens();
  enum {
    NoArrayExpr = -1,
    ArraySubscript = 0,
    OMPArraySection = 1
  } IsArrayExpr = NoArrayExpr;
  if (AllowArraySection) {
    if (auto *ASE = dyn_cast_or_null<ArraySubscriptExpr>(RefExpr)) {
      Expr *Base = ASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
        Base = TempASE->getBase()->IgnoreParenImpCasts();
      RefExpr = Base;
      IsArrayExpr = ArraySubscript;
    } else if (auto *OASE = dyn_cast_or_null<OMPArraySectionExpr>(RefExpr)) {
      Expr *Base = OASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempOASE = dyn_cast<OMPArraySectionExpr>(Base))
        Base = TempOASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
        Base = TempASE->getBase()->IgnoreParenImpCasts();
      RefExpr = Base;
      IsArrayExpr = OMPArraySection;
    }
  }
  ELoc = RefExpr->getExprLoc();
  ERange = RefExpr->getSourceRange();
  RefExpr = RefExpr->IgnoreParenImpCasts();
  auto *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
  auto *ME = dyn_cast_or_null<MemberExpr>(RefExpr);
  if ((!DE || !isa<VarDecl>(DE->getDecl())) &&
      (S.getCurrentThisType().isNull() || !ME ||
       !isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()) ||
       !isa<FieldDecl>(ME->getMemberDecl()))) {
    if (IsArrayExpr != NoArrayExpr) {
      S.Diag(ELoc, diag::err_omp_expected_base_var_name)
          << IsArrayExpr << ERange;
    } else {
      S.Diag(ELoc,
             AllowArraySection
                 ? diag::err_omp_expected_var_name_member_expr_or_array_item
                 : diag::err_omp_expected_var_name_member_expr)
          << (S.getCurrentThisType().isNull() ? 0 : 1) << ERange;
    }
    return std::make_pair(nullptr, false);
  }
  return std::make_pair(
      getCanonicalDecl(DE ? DE->getDecl() : ME->getMemberDecl()), false);
}

namespace {
/// Checks if the allocator is used in uses_allocators clause to be allowed in
/// target regions.
class AllocatorChecker final : public ConstStmtVisitor<AllocatorChecker, bool> {
  DSAStackTy *S = nullptr;

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    return S->isUsesAllocatorsDecl(E->getDecl())
               .getValueOr(
                   DSAStackTy::UsesAllocatorsDeclKind::AllocatorTrait) ==
           DSAStackTy::UsesAllocatorsDeclKind::AllocatorTrait;
  }
  bool VisitStmt(const Stmt *S) {
    for (const Stmt *Child : S->children()) {
      if (Child && Visit(Child))
        return true;
    }
    return false;
  }
  explicit AllocatorChecker(DSAStackTy *S) : S(S) {}
};
} // namespace

static void checkAllocateClauses(Sema &S, DSAStackTy *Stack,
                                 ArrayRef<OMPClause *> Clauses) {
  assert(!S.CurContext->isDependentContext() &&
         "Expected non-dependent context.");
  auto AllocateRange =
      llvm::make_filter_range(Clauses, OMPAllocateClause::classof);
  llvm::DenseMap<CanonicalDeclPtr<Decl>, CanonicalDeclPtr<VarDecl>> DeclToCopy;
  auto PrivateRange = llvm::make_filter_range(Clauses, [](const OMPClause *C) {
    return isOpenMPPrivate(C->getClauseKind());
  });
  for (OMPClause *Cl : PrivateRange) {
    MutableArrayRef<Expr *>::iterator I, It, Et;
    if (Cl->getClauseKind() == OMPC_private) {
      auto *PC = cast<OMPPrivateClause>(Cl);
      I = PC->private_copies().begin();
      It = PC->varlist_begin();
      Et = PC->varlist_end();
    } else if (Cl->getClauseKind() == OMPC_firstprivate) {
      auto *PC = cast<OMPFirstprivateClause>(Cl);
      I = PC->private_copies().begin();
      It = PC->varlist_begin();
      Et = PC->varlist_end();
    } else if (Cl->getClauseKind() == OMPC_lastprivate) {
      auto *PC = cast<OMPLastprivateClause>(Cl);
      I = PC->private_copies().begin();
      It = PC->varlist_begin();
      Et = PC->varlist_end();
    } else if (Cl->getClauseKind() == OMPC_linear) {
      auto *PC = cast<OMPLinearClause>(Cl);
      I = PC->privates().begin();
      It = PC->varlist_begin();
      Et = PC->varlist_end();
    } else if (Cl->getClauseKind() == OMPC_reduction) {
      auto *PC = cast<OMPReductionClause>(Cl);
      I = PC->privates().begin();
      It = PC->varlist_begin();
      Et = PC->varlist_end();
    } else if (Cl->getClauseKind() == OMPC_task_reduction) {
      auto *PC = cast<OMPTaskReductionClause>(Cl);
      I = PC->privates().begin();
      It = PC->varlist_begin();
      Et = PC->varlist_end();
    } else if (Cl->getClauseKind() == OMPC_in_reduction) {
      auto *PC = cast<OMPInReductionClause>(Cl);
      I = PC->privates().begin();
      It = PC->varlist_begin();
      Et = PC->varlist_end();
    } else {
      llvm_unreachable("Expected private clause.");
    }
    for (Expr *E : llvm::make_range(It, Et)) {
      if (!*I) {
        ++I;
        continue;
      }
      SourceLocation ELoc;
      SourceRange ERange;
      Expr *SimpleRefExpr = E;
      auto Res = getPrivateItem(S, SimpleRefExpr, ELoc, ERange,
                                /*AllowArraySection=*/true);
      DeclToCopy.try_emplace(Res.first,
                             cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()));
      ++I;
    }
  }
  for (OMPClause *C : AllocateRange) {
    auto *AC = cast<OMPAllocateClause>(C);
    if (S.getLangOpts().OpenMP >= 50 &&
        !Stack->hasRequiresDeclWithClause<OMPDynamicAllocatorsClause>() &&
        isOpenMPTargetExecutionDirective(Stack->getCurrentDirective()) &&
        AC->getAllocator()) {
      Expr *Allocator = AC->getAllocator();
      // OpenMP, 2.12.5 target Construct
      // Memory allocators that do not appear in a uses_allocators clause cannot
      // appear as an allocator in an allocate clause or be used in the target
      // region unless a requires directive with the dynamic_allocators clause
      // is present in the same compilation unit.
      AllocatorChecker Checker(Stack);
      if (Checker.Visit(Allocator))
        S.Diag(Allocator->getExprLoc(),
               diag::err_omp_allocator_not_in_uses_allocators)
            << Allocator->getSourceRange();
    }
    OMPAllocateDeclAttr::AllocatorTypeTy AllocatorKind =
        getAllocatorKind(S, Stack, AC->getAllocator());
    // OpenMP, 2.11.4 allocate Clause, Restrictions.
    // For task, taskloop or target directives, allocation requests to memory
    // allocators with the trait access set to thread result in unspecified
    // behavior.
    if (AllocatorKind == OMPAllocateDeclAttr::OMPThreadMemAlloc &&
        (isOpenMPTaskingDirective(Stack->getCurrentDirective()) ||
         isOpenMPTargetExecutionDirective(Stack->getCurrentDirective()))) {
      S.Diag(AC->getAllocator()->getExprLoc(),
             diag::warn_omp_allocate_thread_on_task_target_directive)
          << getOpenMPDirectiveName(Stack->getCurrentDirective());
    }
    for (Expr *E : AC->varlists()) {
      SourceLocation ELoc;
      SourceRange ERange;
      Expr *SimpleRefExpr = E;
      auto Res = getPrivateItem(S, SimpleRefExpr, ELoc, ERange);
      ValueDecl *VD = Res.first;
      DSAStackTy::DSAVarData Data = Stack->getTopDSA(VD, /*FromParent=*/false);
      if (!isOpenMPPrivate(Data.CKind)) {
        S.Diag(E->getExprLoc(),
               diag::err_omp_expected_private_copy_for_allocate);
        continue;
      }
      VarDecl *PrivateVD = DeclToCopy[VD];
      if (checkPreviousOMPAllocateAttribute(S, Stack, E, PrivateVD,
                                            AllocatorKind, AC->getAllocator()))
        continue;
      // Placeholder until allocate clause supports align modifier.
      Expr *Alignment = nullptr;
      applyOMPAllocateAttribute(S, PrivateVD, AllocatorKind, AC->getAllocator(),
                                Alignment, E->getSourceRange());
    }
  }
}

namespace {
/// Rewrite statements and expressions for Sema \p Actions CurContext.
///
/// Used to wrap already parsed statements/expressions into a new CapturedStmt
/// context. DeclRefExpr used inside the new context are changed to refer to the
/// captured variable instead.
class CaptureVars : public TreeTransform<CaptureVars> {
  using BaseTransform = TreeTransform<CaptureVars>;

public:
  CaptureVars(Sema &Actions) : BaseTransform(Actions) {}

  bool AlwaysRebuild() { return true; }
};
} // namespace

static VarDecl *precomputeExpr(Sema &Actions,
                               SmallVectorImpl<Stmt *> &BodyStmts, Expr *E,
                               StringRef Name) {
  Expr *NewE = AssertSuccess(CaptureVars(Actions).TransformExpr(E));
  VarDecl *NewVar = buildVarDecl(Actions, {}, NewE->getType(), Name, nullptr,
                                 dyn_cast<DeclRefExpr>(E->IgnoreImplicit()));
  auto *NewDeclStmt = cast<DeclStmt>(AssertSuccess(
      Actions.ActOnDeclStmt(Actions.ConvertDeclToDeclGroup(NewVar), {}, {})));
  Actions.AddInitializerToDecl(NewDeclStmt->getSingleDecl(), NewE, false);
  BodyStmts.push_back(NewDeclStmt);
  return NewVar;
}

/// Create a closure that computes the number of iterations of a loop.
///
/// \param Actions   The Sema object.
/// \param LogicalTy Type for the logical iteration number.
/// \param Rel       Comparison operator of the loop condition.
/// \param StartExpr Value of the loop counter at the first iteration.
/// \param StopExpr  Expression the loop counter is compared against in the loop
/// condition. \param StepExpr      Amount of increment after each iteration.
///
/// \return Closure (CapturedStmt) of the distance calculation.
static CapturedStmt *buildDistanceFunc(Sema &Actions, QualType LogicalTy,
                                       BinaryOperator::Opcode Rel,
                                       Expr *StartExpr, Expr *StopExpr,
                                       Expr *StepExpr) {
  ASTContext &Ctx = Actions.getASTContext();
  TypeSourceInfo *LogicalTSI = Ctx.getTrivialTypeSourceInfo(LogicalTy);

  // Captured regions currently don't support return values, we use an
  // out-parameter instead. All inputs are implicit captures.
  // TODO: Instead of capturing each DeclRefExpr occurring in
  // StartExpr/StopExpr/Step, these could also be passed as a value capture.
  QualType ResultTy = Ctx.getLValueReferenceType(LogicalTy);
  Sema::CapturedParamNameType Params[] = {{"Distance", ResultTy},
                                          {StringRef(), QualType()}};
  Actions.ActOnCapturedRegionStart({}, nullptr, CR_Default, Params);

  Stmt *Body;
  {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    CapturedDecl *CS = cast<CapturedDecl>(Actions.CurContext);

    // Get the LValue expression for the result.
    ImplicitParamDecl *DistParam = CS->getParam(0);
    DeclRefExpr *DistRef = Actions.BuildDeclRefExpr(
        DistParam, LogicalTy, VK_LValue, {}, nullptr, nullptr, {}, nullptr);

    SmallVector<Stmt *, 4> BodyStmts;

    // Capture all referenced variable references.
    // TODO: Instead of computing NewStart/NewStop/NewStep inside the
    // CapturedStmt, we could compute them before and capture the result, to be
    // used jointly with the LoopVar function.
    VarDecl *NewStart = precomputeExpr(Actions, BodyStmts, StartExpr, ".start");
    VarDecl *NewStop = precomputeExpr(Actions, BodyStmts, StopExpr, ".stop");
    VarDecl *NewStep = precomputeExpr(Actions, BodyStmts, StepExpr, ".step");
    auto BuildVarRef = [&](VarDecl *VD) {
      return buildDeclRefExpr(Actions, VD, VD->getType(), {});
    };

    IntegerLiteral *Zero = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(LogicalTy), 0), LogicalTy, {});
    Expr *Dist;
    if (Rel == BO_NE) {
      // When using a != comparison, the increment can be +1 or -1. This can be
      // dynamic at runtime, so we need to check for the direction.
      Expr *IsNegStep = AssertSuccess(
          Actions.BuildBinOp(nullptr, {}, BO_LT, BuildVarRef(NewStep), Zero));

      // Positive increment.
      Expr *ForwardRange = AssertSuccess(Actions.BuildBinOp(
          nullptr, {}, BO_Sub, BuildVarRef(NewStop), BuildVarRef(NewStart)));
      ForwardRange = AssertSuccess(
          Actions.BuildCStyleCastExpr({}, LogicalTSI, {}, ForwardRange));
      Expr *ForwardDist = AssertSuccess(Actions.BuildBinOp(
          nullptr, {}, BO_Div, ForwardRange, BuildVarRef(NewStep)));

      // Negative increment.
      Expr *BackwardRange = AssertSuccess(Actions.BuildBinOp(
          nullptr, {}, BO_Sub, BuildVarRef(NewStart), BuildVarRef(NewStop)));
      BackwardRange = AssertSuccess(
          Actions.BuildCStyleCastExpr({}, LogicalTSI, {}, BackwardRange));
      Expr *NegIncAmount = AssertSuccess(
          Actions.BuildUnaryOp(nullptr, {}, UO_Minus, BuildVarRef(NewStep)));
      Expr *BackwardDist = AssertSuccess(
          Actions.BuildBinOp(nullptr, {}, BO_Div, BackwardRange, NegIncAmount));

      // Use the appropriate case.
      Dist = AssertSuccess(Actions.ActOnConditionalOp(
          {}, {}, IsNegStep, BackwardDist, ForwardDist));
    } else {
      assert((Rel == BO_LT || Rel == BO_LE || Rel == BO_GE || Rel == BO_GT) &&
             "Expected one of these relational operators");

      // We can derive the direction from any other comparison operator. It is
      // non well-formed OpenMP if Step increments/decrements in the other
      // directions. Whether at least the first iteration passes the loop
      // condition.
      Expr *HasAnyIteration = AssertSuccess(Actions.BuildBinOp(
          nullptr, {}, Rel, BuildVarRef(NewStart), BuildVarRef(NewStop)));

      // Compute the range between first and last counter value.
      Expr *Range;
      if (Rel == BO_GE || Rel == BO_GT)
        Range = AssertSuccess(Actions.BuildBinOp(
            nullptr, {}, BO_Sub, BuildVarRef(NewStart), BuildVarRef(NewStop)));
      else
        Range = AssertSuccess(Actions.BuildBinOp(
            nullptr, {}, BO_Sub, BuildVarRef(NewStop), BuildVarRef(NewStart)));

      // Ensure unsigned range space.
      Range =
          AssertSuccess(Actions.BuildCStyleCastExpr({}, LogicalTSI, {}, Range));

      if (Rel == BO_LE || Rel == BO_GE) {
        // Add one to the range if the relational operator is inclusive.
        Range = AssertSuccess(Actions.BuildBinOp(
            nullptr, {}, BO_Add, Range,
            Actions.ActOnIntegerConstant(SourceLocation(), 1).get()));
      }

      // Divide by the absolute step amount.
      Expr *Divisor = BuildVarRef(NewStep);
      if (Rel == BO_GE || Rel == BO_GT)
        Divisor =
            AssertSuccess(Actions.BuildUnaryOp(nullptr, {}, UO_Minus, Divisor));
      Dist = AssertSuccess(
          Actions.BuildBinOp(nullptr, {}, BO_Div, Range, Divisor));

      // If there is not at least one iteration, the range contains garbage. Fix
      // to zero in this case.
      Dist = AssertSuccess(
          Actions.ActOnConditionalOp({}, {}, HasAnyIteration, Dist, Zero));
    }

    // Assign the result to the out-parameter.
    Stmt *ResultAssign = AssertSuccess(Actions.BuildBinOp(
        Actions.getCurScope(), {}, BO_Assign, DistRef, Dist));
    BodyStmts.push_back(ResultAssign);

    Body = AssertSuccess(Actions.ActOnCompoundStmt({}, {}, BodyStmts, false));
  }

  return cast<CapturedStmt>(
      AssertSuccess(Actions.ActOnCapturedRegionEnd(Body)));
}

/// Create a closure that computes the loop variable from the logical iteration
/// number.
///
/// \param Actions   The Sema object.
/// \param LoopVarTy Type for the loop variable used for result value.
/// \param LogicalTy Type for the logical iteration number.
/// \param StartExpr Value of the loop counter at the first iteration.
/// \param Step      Amount of increment after each iteration.
/// \param Deref     Whether the loop variable is a dereference of the loop
/// counter variable.
///
/// \return Closure (CapturedStmt) of the loop value calculation.
static CapturedStmt *buildLoopVarFunc(Sema &Actions, QualType LoopVarTy,
                                      QualType LogicalTy,
                                      DeclRefExpr *StartExpr, Expr *Step,
                                      bool Deref) {
  ASTContext &Ctx = Actions.getASTContext();

  // Pass the result as an out-parameter. Passing as return value would require
  // the OpenMPIRBuilder to know additional C/C++ semantics, such as how to
  // invoke a copy constructor.
  QualType TargetParamTy = Ctx.getLValueReferenceType(LoopVarTy);
  Sema::CapturedParamNameType Params[] = {{"LoopVar", TargetParamTy},
                                          {"Logical", LogicalTy},
                                          {StringRef(), QualType()}};
  Actions.ActOnCapturedRegionStart({}, nullptr, CR_Default, Params);

  // Capture the initial iterator which represents the LoopVar value at the
  // zero's logical iteration. Since the original ForStmt/CXXForRangeStmt update
  // it in every iteration, capture it by value before it is modified.
  VarDecl *StartVar = cast<VarDecl>(StartExpr->getDecl());
  bool Invalid = Actions.tryCaptureVariable(StartVar, {},
                                            Sema::TryCapture_ExplicitByVal, {});
  (void)Invalid;
  assert(!Invalid && "Expecting capture-by-value to work.");

  Expr *Body;
  {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    auto *CS = cast<CapturedDecl>(Actions.CurContext);

    ImplicitParamDecl *TargetParam = CS->getParam(0);
    DeclRefExpr *TargetRef = Actions.BuildDeclRefExpr(
        TargetParam, LoopVarTy, VK_LValue, {}, nullptr, nullptr, {}, nullptr);
    ImplicitParamDecl *IndvarParam = CS->getParam(1);
    DeclRefExpr *LogicalRef = Actions.BuildDeclRefExpr(
        IndvarParam, LogicalTy, VK_LValue, {}, nullptr, nullptr, {}, nullptr);

    // Capture the Start expression.
    CaptureVars Recap(Actions);
    Expr *NewStart = AssertSuccess(Recap.TransformExpr(StartExpr));
    Expr *NewStep = AssertSuccess(Recap.TransformExpr(Step));

    Expr *Skip = AssertSuccess(
        Actions.BuildBinOp(nullptr, {}, BO_Mul, NewStep, LogicalRef));
    // TODO: Explicitly cast to the iterator's difference_type instead of
    // relying on implicit conversion.
    Expr *Advanced =
        AssertSuccess(Actions.BuildBinOp(nullptr, {}, BO_Add, NewStart, Skip));

    if (Deref) {
      // For range-based for-loops convert the loop counter value to a concrete
      // loop variable value by dereferencing the iterator.
      Advanced =
          AssertSuccess(Actions.BuildUnaryOp(nullptr, {}, UO_Deref, Advanced));
    }

    // Assign the result to the output parameter.
    Body = AssertSuccess(Actions.BuildBinOp(Actions.getCurScope(), {},
                                            BO_Assign, TargetRef, Advanced));
  }
  return cast<CapturedStmt>(
      AssertSuccess(Actions.ActOnCapturedRegionEnd(Body)));
}

StmtResult Sema::ActOnOpenMPCanonicalLoop(Stmt *AStmt) {
  ASTContext &Ctx = getASTContext();

  // Extract the common elements of ForStmt and CXXForRangeStmt:
  // Loop variable, repeat condition, increment
  Expr *Cond, *Inc;
  VarDecl *LIVDecl, *LUVDecl;
  if (auto *For = dyn_cast<ForStmt>(AStmt)) {
    Stmt *Init = For->getInit();
    if (auto *LCVarDeclStmt = dyn_cast<DeclStmt>(Init)) {
      // For statement declares loop variable.
      LIVDecl = cast<VarDecl>(LCVarDeclStmt->getSingleDecl());
    } else if (auto *LCAssign = dyn_cast<BinaryOperator>(Init)) {
      // For statement reuses variable.
      assert(LCAssign->getOpcode() == BO_Assign &&
             "init part must be a loop variable assignment");
      auto *CounterRef = cast<DeclRefExpr>(LCAssign->getLHS());
      LIVDecl = cast<VarDecl>(CounterRef->getDecl());
    } else
      llvm_unreachable("Cannot determine loop variable");
    LUVDecl = LIVDecl;

    Cond = For->getCond();
    Inc = For->getInc();
  } else if (auto *RangeFor = dyn_cast<CXXForRangeStmt>(AStmt)) {
    DeclStmt *BeginStmt = RangeFor->getBeginStmt();
    LIVDecl = cast<VarDecl>(BeginStmt->getSingleDecl());
    LUVDecl = RangeFor->getLoopVariable();

    Cond = RangeFor->getCond();
    Inc = RangeFor->getInc();
  } else
    llvm_unreachable("unhandled kind of loop");

  QualType CounterTy = LIVDecl->getType();
  QualType LVTy = LUVDecl->getType();

  // Analyze the loop condition.
  Expr *LHS, *RHS;
  BinaryOperator::Opcode CondRel;
  Cond = Cond->IgnoreImplicit();
  if (auto *CondBinExpr = dyn_cast<BinaryOperator>(Cond)) {
    LHS = CondBinExpr->getLHS();
    RHS = CondBinExpr->getRHS();
    CondRel = CondBinExpr->getOpcode();
  } else if (auto *CondCXXOp = dyn_cast<CXXOperatorCallExpr>(Cond)) {
    assert(CondCXXOp->getNumArgs() == 2 && "Comparison should have 2 operands");
    LHS = CondCXXOp->getArg(0);
    RHS = CondCXXOp->getArg(1);
    switch (CondCXXOp->getOperator()) {
    case OO_ExclaimEqual:
      CondRel = BO_NE;
      break;
    case OO_Less:
      CondRel = BO_LT;
      break;
    case OO_LessEqual:
      CondRel = BO_LE;
      break;
    case OO_Greater:
      CondRel = BO_GT;
      break;
    case OO_GreaterEqual:
      CondRel = BO_GE;
      break;
    default:
      llvm_unreachable("unexpected iterator operator");
    }
  } else
    llvm_unreachable("unexpected loop condition");

  // Normalize such that the loop counter is on the LHS.
  if (!isa<DeclRefExpr>(LHS->IgnoreImplicit()) ||
      cast<DeclRefExpr>(LHS->IgnoreImplicit())->getDecl() != LIVDecl) {
    std::swap(LHS, RHS);
    CondRel = BinaryOperator::reverseComparisonOp(CondRel);
  }
  auto *CounterRef = cast<DeclRefExpr>(LHS->IgnoreImplicit());

  // Decide the bit width for the logical iteration counter. By default use the
  // unsigned ptrdiff_t integer size (for iterators and pointers).
  // TODO: For iterators, use iterator::difference_type,
  // std::iterator_traits<>::difference_type or decltype(it - end).
  QualType LogicalTy = Ctx.getUnsignedPointerDiffType();
  if (CounterTy->isIntegerType()) {
    unsigned BitWidth = Ctx.getIntWidth(CounterTy);
    LogicalTy = Ctx.getIntTypeForBitwidth(BitWidth, false);
  }

  // Analyze the loop increment.
  Expr *Step;
  if (auto *IncUn = dyn_cast<UnaryOperator>(Inc)) {
    int Direction;
    switch (IncUn->getOpcode()) {
    case UO_PreInc:
    case UO_PostInc:
      Direction = 1;
      break;
    case UO_PreDec:
    case UO_PostDec:
      Direction = -1;
      break;
    default:
      llvm_unreachable("unhandled unary increment operator");
    }
    Step = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(LogicalTy), Direction), LogicalTy, {});
  } else if (auto *IncBin = dyn_cast<BinaryOperator>(Inc)) {
    if (IncBin->getOpcode() == BO_AddAssign) {
      Step = IncBin->getRHS();
    } else if (IncBin->getOpcode() == BO_SubAssign) {
      Step =
          AssertSuccess(BuildUnaryOp(nullptr, {}, UO_Minus, IncBin->getRHS()));
    } else
      llvm_unreachable("unhandled binary increment operator");
  } else if (auto *CondCXXOp = dyn_cast<CXXOperatorCallExpr>(Inc)) {
    switch (CondCXXOp->getOperator()) {
    case OO_PlusPlus:
      Step = IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(LogicalTy), 1), LogicalTy, {});
      break;
    case OO_MinusMinus:
      Step = IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(LogicalTy), -1), LogicalTy, {});
      break;
    case OO_PlusEqual:
      Step = CondCXXOp->getArg(1);
      break;
    case OO_MinusEqual:
      Step = AssertSuccess(
          BuildUnaryOp(nullptr, {}, UO_Minus, CondCXXOp->getArg(1)));
      break;
    default:
      llvm_unreachable("unhandled overloaded increment operator");
    }
  } else
    llvm_unreachable("unknown increment expression");

  CapturedStmt *DistanceFunc =
      buildDistanceFunc(*this, LogicalTy, CondRel, LHS, RHS, Step);
  CapturedStmt *LoopVarFunc = buildLoopVarFunc(
      *this, LVTy, LogicalTy, CounterRef, Step, isa<CXXForRangeStmt>(AStmt));
  DeclRefExpr *LVRef = BuildDeclRefExpr(LUVDecl, LUVDecl->getType(), VK_LValue,
                                        {}, nullptr, nullptr, {}, nullptr);
  return OMPCanonicalLoop::create(getASTContext(), AStmt, DistanceFunc,
                                  LoopVarFunc, LVRef);
}

StmtResult Sema::ActOnOpenMPLoopnest(Stmt *AStmt) {
  // Handle a literal loop.
  if (isa<ForStmt>(AStmt) || isa<CXXForRangeStmt>(AStmt))
    return ActOnOpenMPCanonicalLoop(AStmt);

  // If not a literal loop, it must be the result of a loop transformation.
  OMPExecutableDirective *LoopTransform = cast<OMPExecutableDirective>(AStmt);
  assert(
      isOpenMPLoopTransformationDirective(LoopTransform->getDirectiveKind()) &&
      "Loop transformation directive expected");
  return LoopTransform;
}

static ExprResult buildUserDefinedMapperRef(Sema &SemaRef, Scope *S,
                                            CXXScopeSpec &MapperIdScopeSpec,
                                            const DeclarationNameInfo &MapperId,
                                            QualType Type,
                                            Expr *UnresolvedMapper);

/// Perform DFS through the structure/class data members trying to find
/// member(s) with user-defined 'default' mapper and generate implicit map
/// clauses for such members with the found 'default' mapper.
static void
processImplicitMapsWithDefaultMappers(Sema &S, DSAStackTy *Stack,
                                      SmallVectorImpl<OMPClause *> &Clauses) {
  // Check for the deault mapper for data members.
  if (S.getLangOpts().OpenMP < 50)
    return;
  SmallVector<OMPClause *, 4> ImplicitMaps;
  for (int Cnt = 0, EndCnt = Clauses.size(); Cnt < EndCnt; ++Cnt) {
    auto *C = dyn_cast<OMPMapClause>(Clauses[Cnt]);
    if (!C)
      continue;
    SmallVector<Expr *, 4> SubExprs;
    auto *MI = C->mapperlist_begin();
    for (auto I = C->varlist_begin(), End = C->varlist_end(); I != End;
         ++I, ++MI) {
      // Expression is mapped using mapper - skip it.
      if (*MI)
        continue;
      Expr *E = *I;
      // Expression is dependent - skip it, build the mapper when it gets
      // instantiated.
      if (E->isTypeDependent() || E->isValueDependent() ||
          E->containsUnexpandedParameterPack())
        continue;
      // Array section - need to check for the mapping of the array section
      // element.
      QualType CanonType = E->getType().getCanonicalType();
      if (CanonType->isSpecificBuiltinType(BuiltinType::OMPArraySection)) {
        const auto *OASE = cast<OMPArraySectionExpr>(E->IgnoreParenImpCasts());
        QualType BaseType =
            OMPArraySectionExpr::getBaseOriginalType(OASE->getBase());
        QualType ElemType;
        if (const auto *ATy = BaseType->getAsArrayTypeUnsafe())
          ElemType = ATy->getElementType();
        else
          ElemType = BaseType->getPointeeType();
        CanonType = ElemType;
      }

      // DFS over data members in structures/classes.
      SmallVector<std::pair<QualType, FieldDecl *>, 4> Types(
          1, {CanonType, nullptr});
      llvm::DenseMap<const Type *, Expr *> Visited;
      SmallVector<std::pair<FieldDecl *, unsigned>, 4> ParentChain(
          1, {nullptr, 1});
      while (!Types.empty()) {
        QualType BaseType;
        FieldDecl *CurFD;
        std::tie(BaseType, CurFD) = Types.pop_back_val();
        while (ParentChain.back().second == 0)
          ParentChain.pop_back();
        --ParentChain.back().second;
        if (BaseType.isNull())
          continue;
        // Only structs/classes are allowed to have mappers.
        const RecordDecl *RD = BaseType.getCanonicalType()->getAsRecordDecl();
        if (!RD)
          continue;
        auto It = Visited.find(BaseType.getTypePtr());
        if (It == Visited.end()) {
          // Try to find the associated user-defined mapper.
          CXXScopeSpec MapperIdScopeSpec;
          DeclarationNameInfo DefaultMapperId;
          DefaultMapperId.setName(S.Context.DeclarationNames.getIdentifier(
              &S.Context.Idents.get("default")));
          DefaultMapperId.setLoc(E->getExprLoc());
          ExprResult ER = buildUserDefinedMapperRef(
              S, Stack->getCurScope(), MapperIdScopeSpec, DefaultMapperId,
              BaseType, /*UnresolvedMapper=*/nullptr);
          if (ER.isInvalid())
            continue;
          It = Visited.try_emplace(BaseType.getTypePtr(), ER.get()).first;
        }
        // Found default mapper.
        if (It->second) {
          auto *OE = new (S.Context) OpaqueValueExpr(E->getExprLoc(), CanonType,
                                                     VK_LValue, OK_Ordinary, E);
          OE->setIsUnique(/*V=*/true);
          Expr *BaseExpr = OE;
          for (const auto &P : ParentChain) {
            if (P.first) {
              BaseExpr = S.BuildMemberExpr(
                  BaseExpr, /*IsArrow=*/false, E->getExprLoc(),
                  NestedNameSpecifierLoc(), SourceLocation(), P.first,
                  DeclAccessPair::make(P.first, P.first->getAccess()),
                  /*HadMultipleCandidates=*/false, DeclarationNameInfo(),
                  P.first->getType(), VK_LValue, OK_Ordinary);
              BaseExpr = S.DefaultLvalueConversion(BaseExpr).get();
            }
          }
          if (CurFD)
            BaseExpr = S.BuildMemberExpr(
                BaseExpr, /*IsArrow=*/false, E->getExprLoc(),
                NestedNameSpecifierLoc(), SourceLocation(), CurFD,
                DeclAccessPair::make(CurFD, CurFD->getAccess()),
                /*HadMultipleCandidates=*/false, DeclarationNameInfo(),
                CurFD->getType(), VK_LValue, OK_Ordinary);
          SubExprs.push_back(BaseExpr);
          continue;
        }
        // Check for the "default" mapper for data members.
        bool FirstIter = true;
        for (FieldDecl *FD : RD->fields()) {
          if (!FD)
            continue;
          QualType FieldTy = FD->getType();
          if (FieldTy.isNull() ||
              !(FieldTy->isStructureOrClassType() || FieldTy->isUnionType()))
            continue;
          if (FirstIter) {
            FirstIter = false;
            ParentChain.emplace_back(CurFD, 1);
          } else {
            ++ParentChain.back().second;
          }
          Types.emplace_back(FieldTy, FD);
        }
      }
    }
    if (SubExprs.empty())
      continue;
    CXXScopeSpec MapperIdScopeSpec;
    DeclarationNameInfo MapperId;
    if (OMPClause *NewClause = S.ActOnOpenMPMapClause(
            C->getMapTypeModifiers(), C->getMapTypeModifiersLoc(),
            MapperIdScopeSpec, MapperId, C->getMapType(),
            /*IsMapTypeImplicit=*/true, SourceLocation(), SourceLocation(),
            SubExprs, OMPVarListLocTy()))
      Clauses.push_back(NewClause);
  }
}

StmtResult Sema::ActOnOpenMPExecutableDirective(
    OpenMPDirectiveKind Kind, const DeclarationNameInfo &DirName,
    OpenMPDirectiveKind CancelRegion, ArrayRef<OMPClause *> Clauses,
    Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {
  StmtResult Res = StmtError();
  OpenMPBindClauseKind BindKind = OMPC_BIND_unknown;
  if (const OMPBindClause *BC =
          OMPExecutableDirective::getSingleClause<OMPBindClause>(Clauses))
    BindKind = BC->getBindKind();
  // First check CancelRegion which is then used in checkNestingOfRegions.
  if (checkCancelRegion(*this, Kind, CancelRegion, StartLoc) ||
      checkNestingOfRegions(*this, DSAStack, Kind, DirName, CancelRegion,
                            BindKind, StartLoc))
    return StmtError();

  llvm::SmallVector<OMPClause *, 8> ClausesWithImplicit;
  VarsWithInheritedDSAType VarsWithInheritedDSA;
  bool ErrorFound = false;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());
  if (AStmt && !CurContext->isDependentContext() && Kind != OMPD_atomic &&
      Kind != OMPD_critical && Kind != OMPD_section && Kind != OMPD_master &&
      Kind != OMPD_masked && !isOpenMPLoopTransformationDirective(Kind)) {
    assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

    // Check default data sharing attributes for referenced variables.
    DSAAttrChecker DSAChecker(DSAStack, *this, cast<CapturedStmt>(AStmt));
    int ThisCaptureLevel = getOpenMPCaptureLevels(Kind);
    Stmt *S = AStmt;
    while (--ThisCaptureLevel >= 0)
      S = cast<CapturedStmt>(S)->getCapturedStmt();
    DSAChecker.Visit(S);
    if (!isOpenMPTargetDataManagementDirective(Kind) &&
        !isOpenMPTaskingDirective(Kind)) {
      // Visit subcaptures to generate implicit clauses for captured vars.
      auto *CS = cast<CapturedStmt>(AStmt);
      SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
      getOpenMPCaptureRegions(CaptureRegions, Kind);
      // Ignore outer tasking regions for target directives.
      if (CaptureRegions.size() > 1 && CaptureRegions.front() == OMPD_task)
        CS = cast<CapturedStmt>(CS->getCapturedStmt());
      DSAChecker.visitSubCaptures(CS);
    }
    if (DSAChecker.isErrorFound())
      return StmtError();
    // Generate list of implicitly defined firstprivate variables.
    VarsWithInheritedDSA = DSAChecker.getVarsWithInheritedDSA();

    SmallVector<Expr *, 4> ImplicitFirstprivates(
        DSAChecker.getImplicitFirstprivate().begin(),
        DSAChecker.getImplicitFirstprivate().end());
    const unsigned DefaultmapKindNum = OMPC_DEFAULTMAP_pointer + 1;
    SmallVector<Expr *, 4> ImplicitMaps[DefaultmapKindNum][OMPC_MAP_delete];
    SmallVector<OpenMPMapModifierKind, NumberOfOMPMapClauseModifiers>
        ImplicitMapModifiers[DefaultmapKindNum];
    SmallVector<SourceLocation, NumberOfOMPMapClauseModifiers>
        ImplicitMapModifiersLoc[DefaultmapKindNum];
    // Get the original location of present modifier from Defaultmap clause.
    SourceLocation PresentModifierLocs[DefaultmapKindNum];
    for (OMPClause *C : Clauses) {
      if (auto *DMC = dyn_cast<OMPDefaultmapClause>(C))
        if (DMC->getDefaultmapModifier() == OMPC_DEFAULTMAP_MODIFIER_present)
          PresentModifierLocs[DMC->getDefaultmapKind()] =
              DMC->getDefaultmapModifierLoc();
    }
    for (unsigned VC = 0; VC < DefaultmapKindNum; ++VC) {
      auto Kind = static_cast<OpenMPDefaultmapClauseKind>(VC);
      for (unsigned I = 0; I < OMPC_MAP_delete; ++I) {
        ArrayRef<Expr *> ImplicitMap = DSAChecker.getImplicitMap(
            Kind, static_cast<OpenMPMapClauseKind>(I));
        ImplicitMaps[VC][I].append(ImplicitMap.begin(), ImplicitMap.end());
      }
      ArrayRef<OpenMPMapModifierKind> ImplicitModifier =
          DSAChecker.getImplicitMapModifier(Kind);
      ImplicitMapModifiers[VC].append(ImplicitModifier.begin(),
                                      ImplicitModifier.end());
      std::fill_n(std::back_inserter(ImplicitMapModifiersLoc[VC]),
                  ImplicitModifier.size(), PresentModifierLocs[VC]);
    }
    // Mark taskgroup task_reduction descriptors as implicitly firstprivate.
    for (OMPClause *C : Clauses) {
      if (auto *IRC = dyn_cast<OMPInReductionClause>(C)) {
        for (Expr *E : IRC->taskgroup_descriptors())
          if (E)
            ImplicitFirstprivates.emplace_back(E);
      }
      // OpenMP 5.0, 2.10.1 task Construct
      // [detach clause]... The event-handle will be considered as if it was
      // specified on a firstprivate clause.
      if (auto *DC = dyn_cast<OMPDetachClause>(C))
        ImplicitFirstprivates.push_back(DC->getEventHandler());
    }
    if (!ImplicitFirstprivates.empty()) {
      if (OMPClause *Implicit = ActOnOpenMPFirstprivateClause(
              ImplicitFirstprivates, SourceLocation(), SourceLocation(),
              SourceLocation())) {
        ClausesWithImplicit.push_back(Implicit);
        ErrorFound = cast<OMPFirstprivateClause>(Implicit)->varlist_size() !=
                     ImplicitFirstprivates.size();
      } else {
        ErrorFound = true;
      }
    }
    // OpenMP 5.0 [2.19.7]
    // If a list item appears in a reduction, lastprivate or linear
    // clause on a combined target construct then it is treated as
    // if it also appears in a map clause with a map-type of tofrom
    if (getLangOpts().OpenMP >= 50 && Kind != OMPD_target &&
        isOpenMPTargetExecutionDirective(Kind)) {
      SmallVector<Expr *, 4> ImplicitExprs;
      for (OMPClause *C : Clauses) {
        if (auto *RC = dyn_cast<OMPReductionClause>(C))
          for (Expr *E : RC->varlists())
            if (!isa<DeclRefExpr>(E->IgnoreParenImpCasts()))
              ImplicitExprs.emplace_back(E);
      }
      if (!ImplicitExprs.empty()) {
        ArrayRef<Expr *> Exprs = ImplicitExprs;
        CXXScopeSpec MapperIdScopeSpec;
        DeclarationNameInfo MapperId;
        if (OMPClause *Implicit = ActOnOpenMPMapClause(
                OMPC_MAP_MODIFIER_unknown, SourceLocation(), MapperIdScopeSpec,
                MapperId, OMPC_MAP_tofrom,
                /*IsMapTypeImplicit=*/true, SourceLocation(), SourceLocation(),
                Exprs, OMPVarListLocTy(), /*NoDiagnose=*/true))
          ClausesWithImplicit.emplace_back(Implicit);
      }
    }
    for (unsigned I = 0, E = DefaultmapKindNum; I < E; ++I) {
      int ClauseKindCnt = -1;
      for (ArrayRef<Expr *> ImplicitMap : ImplicitMaps[I]) {
        ++ClauseKindCnt;
        if (ImplicitMap.empty())
          continue;
        CXXScopeSpec MapperIdScopeSpec;
        DeclarationNameInfo MapperId;
        auto Kind = static_cast<OpenMPMapClauseKind>(ClauseKindCnt);
        if (OMPClause *Implicit = ActOnOpenMPMapClause(
                ImplicitMapModifiers[I], ImplicitMapModifiersLoc[I],
                MapperIdScopeSpec, MapperId, Kind, /*IsMapTypeImplicit=*/true,
                SourceLocation(), SourceLocation(), ImplicitMap,
                OMPVarListLocTy())) {
          ClausesWithImplicit.emplace_back(Implicit);
          ErrorFound |= cast<OMPMapClause>(Implicit)->varlist_size() !=
                        ImplicitMap.size();
        } else {
          ErrorFound = true;
        }
      }
    }
    // Build expressions for implicit maps of data members with 'default'
    // mappers.
    if (LangOpts.OpenMP >= 50)
      processImplicitMapsWithDefaultMappers(*this, DSAStack,
                                            ClausesWithImplicit);
  }

  llvm::SmallVector<OpenMPDirectiveKind, 4> AllowedNameModifiers;
  switch (Kind) {
  case OMPD_parallel:
    Res = ActOnOpenMPParallelDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_simd:
    Res = ActOnOpenMPSimdDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc,
                                   VarsWithInheritedDSA);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_tile:
    Res =
        ActOnOpenMPTileDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_unroll:
    Res = ActOnOpenMPUnrollDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_for:
    Res = ActOnOpenMPForDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc,
                                  VarsWithInheritedDSA);
    break;
  case OMPD_for_simd:
    Res = ActOnOpenMPForSimdDirective(ClausesWithImplicit, AStmt, StartLoc,
                                      EndLoc, VarsWithInheritedDSA);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_sections:
    Res = ActOnOpenMPSectionsDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    break;
  case OMPD_section:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp section' directive");
    Res = ActOnOpenMPSectionDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_single:
    Res = ActOnOpenMPSingleDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_master:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp master' directive");
    Res = ActOnOpenMPMasterDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_masked:
    Res = ActOnOpenMPMaskedDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_critical:
    Res = ActOnOpenMPCriticalDirective(DirName, ClausesWithImplicit, AStmt,
                                       StartLoc, EndLoc);
    break;
  case OMPD_parallel_for:
    Res = ActOnOpenMPParallelForDirective(ClausesWithImplicit, AStmt, StartLoc,
                                          EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_parallel_for_simd:
    Res = ActOnOpenMPParallelForSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_parallel_master:
    Res = ActOnOpenMPParallelMasterDirective(ClausesWithImplicit, AStmt,
                                             StartLoc, EndLoc);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_parallel_sections:
    Res = ActOnOpenMPParallelSectionsDirective(ClausesWithImplicit, AStmt,
                                               StartLoc, EndLoc);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_task:
    Res =
        ActOnOpenMPTaskDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    AllowedNameModifiers.push_back(OMPD_task);
    break;
  case OMPD_taskyield:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp taskyield' directive");
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp taskyield' directive");
    Res = ActOnOpenMPTaskyieldDirective(StartLoc, EndLoc);
    break;
  case OMPD_barrier:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp barrier' directive");
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp barrier' directive");
    Res = ActOnOpenMPBarrierDirective(StartLoc, EndLoc);
    break;
  case OMPD_taskwait:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp taskwait' directive");
    Res = ActOnOpenMPTaskwaitDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_taskgroup:
    Res = ActOnOpenMPTaskgroupDirective(ClausesWithImplicit, AStmt, StartLoc,
                                        EndLoc);
    break;
  case OMPD_flush:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp flush' directive");
    Res = ActOnOpenMPFlushDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_depobj:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp depobj' directive");
    Res = ActOnOpenMPDepobjDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_scan:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp scan' directive");
    Res = ActOnOpenMPScanDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_ordered:
    Res = ActOnOpenMPOrderedDirective(ClausesWithImplicit, AStmt, StartLoc,
                                      EndLoc);
    break;
  case OMPD_atomic:
    Res = ActOnOpenMPAtomicDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_teams:
    Res =
        ActOnOpenMPTeamsDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_target:
    Res = ActOnOpenMPTargetDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    AllowedNameModifiers.push_back(OMPD_target);
    break;
  case OMPD_target_parallel:
    Res = ActOnOpenMPTargetParallelDirective(ClausesWithImplicit, AStmt,
                                             StartLoc, EndLoc);
    AllowedNameModifiers.push_back(OMPD_target);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_target_parallel_for:
    Res = ActOnOpenMPTargetParallelForDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_target);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_cancellation_point:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp cancellation point' directive");
    assert(AStmt == nullptr && "No associated statement allowed for 'omp "
                               "cancellation point' directive");
    Res = ActOnOpenMPCancellationPointDirective(StartLoc, EndLoc, CancelRegion);
    break;
  case OMPD_cancel:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp cancel' directive");
    Res = ActOnOpenMPCancelDirective(ClausesWithImplicit, StartLoc, EndLoc,
                                     CancelRegion);
    AllowedNameModifiers.push_back(OMPD_cancel);
    break;
  case OMPD_target_data:
    Res = ActOnOpenMPTargetDataDirective(ClausesWithImplicit, AStmt, StartLoc,
                                         EndLoc);
    AllowedNameModifiers.push_back(OMPD_target_data);
    break;
  case OMPD_target_enter_data:
    Res = ActOnOpenMPTargetEnterDataDirective(ClausesWithImplicit, StartLoc,
                                              EndLoc, AStmt);
    AllowedNameModifiers.push_back(OMPD_target_enter_data);
    break;
  case OMPD_target_exit_data:
    Res = ActOnOpenMPTargetExitDataDirective(ClausesWithImplicit, StartLoc,
                                             EndLoc, AStmt);
    AllowedNameModifiers.push_back(OMPD_target_exit_data);
    break;
  case OMPD_taskloop:
    Res = ActOnOpenMPTaskLoopDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    break;
  case OMPD_taskloop_simd:
    Res = ActOnOpenMPTaskLoopSimdDirective(ClausesWithImplicit, AStmt, StartLoc,
                                           EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_master_taskloop:
    Res = ActOnOpenMPMasterTaskLoopDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    break;
  case OMPD_master_taskloop_simd:
    Res = ActOnOpenMPMasterTaskLoopSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_parallel_master_taskloop:
    Res = ActOnOpenMPParallelMasterTaskLoopDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_parallel_master_taskloop_simd:
    Res = ActOnOpenMPParallelMasterTaskLoopSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    AllowedNameModifiers.push_back(OMPD_parallel);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_distribute:
    Res = ActOnOpenMPDistributeDirective(ClausesWithImplicit, AStmt, StartLoc,
                                         EndLoc, VarsWithInheritedDSA);
    break;
  case OMPD_target_update:
    Res = ActOnOpenMPTargetUpdateDirective(ClausesWithImplicit, StartLoc,
                                           EndLoc, AStmt);
    AllowedNameModifiers.push_back(OMPD_target_update);
    break;
  case OMPD_distribute_parallel_for:
    Res = ActOnOpenMPDistributeParallelForDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_distribute_parallel_for_simd:
    Res = ActOnOpenMPDistributeParallelForSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_distribute_simd:
    Res = ActOnOpenMPDistributeSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_target_parallel_for_simd:
    Res = ActOnOpenMPTargetParallelForSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_target);
    AllowedNameModifiers.push_back(OMPD_parallel);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_target_simd:
    Res = ActOnOpenMPTargetSimdDirective(ClausesWithImplicit, AStmt, StartLoc,
                                         EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_target);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_teams_distribute:
    Res = ActOnOpenMPTeamsDistributeDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    break;
  case OMPD_teams_distribute_simd:
    Res = ActOnOpenMPTeamsDistributeSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_teams_distribute_parallel_for_simd:
    Res = ActOnOpenMPTeamsDistributeParallelForSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_teams_distribute_parallel_for:
    Res = ActOnOpenMPTeamsDistributeParallelForDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_target_teams:
    Res = ActOnOpenMPTargetTeamsDirective(ClausesWithImplicit, AStmt, StartLoc,
                                          EndLoc);
    AllowedNameModifiers.push_back(OMPD_target);
    break;
  case OMPD_target_teams_distribute:
    Res = ActOnOpenMPTargetTeamsDistributeDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_target);
    break;
  case OMPD_target_teams_distribute_parallel_for:
    Res = ActOnOpenMPTargetTeamsDistributeParallelForDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_target);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_target_teams_distribute_parallel_for_simd:
    Res = ActOnOpenMPTargetTeamsDistributeParallelForSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_target);
    AllowedNameModifiers.push_back(OMPD_parallel);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_target_teams_distribute_simd:
    Res = ActOnOpenMPTargetTeamsDistributeSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_target);
    if (LangOpts.OpenMP >= 50)
      AllowedNameModifiers.push_back(OMPD_simd);
    break;
  case OMPD_interop:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp interop' directive");
    Res = ActOnOpenMPInteropDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_dispatch:
    Res = ActOnOpenMPDispatchDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    break;
  case OMPD_loop:
    Res = ActOnOpenMPGenericLoopDirective(ClausesWithImplicit, AStmt, StartLoc,
                                          EndLoc, VarsWithInheritedDSA);
    break;
  case OMPD_declare_target:
  case OMPD_end_declare_target:
  case OMPD_threadprivate:
  case OMPD_allocate:
  case OMPD_declare_reduction:
  case OMPD_declare_mapper:
  case OMPD_declare_simd:
  case OMPD_requires:
  case OMPD_declare_variant:
  case OMPD_begin_declare_variant:
  case OMPD_end_declare_variant:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
  default:
    llvm_unreachable("Unknown OpenMP directive");
  }

  ErrorFound = Res.isInvalid() || ErrorFound;

  // Check variables in the clauses if default(none) or
  // default(firstprivate) was specified.
  if (DSAStack->getDefaultDSA() == DSA_none ||
      DSAStack->getDefaultDSA() == DSA_firstprivate) {
    DSAAttrChecker DSAChecker(DSAStack, *this, nullptr);
    for (OMPClause *C : Clauses) {
      switch (C->getClauseKind()) {
      case OMPC_num_threads:
      case OMPC_dist_schedule:
        // Do not analyse if no parent teams directive.
        if (isOpenMPTeamsDirective(Kind))
          break;
        continue;
      case OMPC_if:
        if (isOpenMPTeamsDirective(Kind) &&
            cast<OMPIfClause>(C)->getNameModifier() != OMPD_target)
          break;
        if (isOpenMPParallelDirective(Kind) &&
            isOpenMPTaskLoopDirective(Kind) &&
            cast<OMPIfClause>(C)->getNameModifier() != OMPD_parallel)
          break;
        continue;
      case OMPC_schedule:
      case OMPC_detach:
        break;
      case OMPC_grainsize:
      case OMPC_num_tasks:
      case OMPC_final:
      case OMPC_priority:
      case OMPC_novariants:
      case OMPC_nocontext:
        // Do not analyze if no parent parallel directive.
        if (isOpenMPParallelDirective(Kind))
          break;
        continue;
      case OMPC_ordered:
      case OMPC_device:
      case OMPC_num_teams:
      case OMPC_thread_limit:
      case OMPC_hint:
      case OMPC_collapse:
      case OMPC_safelen:
      case OMPC_simdlen:
      case OMPC_sizes:
      case OMPC_default:
      case OMPC_proc_bind:
      case OMPC_private:
      case OMPC_firstprivate:
      case OMPC_lastprivate:
      case OMPC_shared:
      case OMPC_reduction:
      case OMPC_task_reduction:
      case OMPC_in_reduction:
      case OMPC_linear:
      case OMPC_aligned:
      case OMPC_copyin:
      case OMPC_copyprivate:
      case OMPC_nowait:
      case OMPC_untied:
      case OMPC_mergeable:
      case OMPC_allocate:
      case OMPC_read:
      case OMPC_write:
      case OMPC_update:
      case OMPC_capture:
      case OMPC_compare:
      case OMPC_seq_cst:
      case OMPC_acq_rel:
      case OMPC_acquire:
      case OMPC_release:
      case OMPC_relaxed:
      case OMPC_depend:
      case OMPC_threads:
      case OMPC_simd:
      case OMPC_map:
      case OMPC_nogroup:
      case OMPC_defaultmap:
      case OMPC_to:
      case OMPC_from:
      case OMPC_use_device_ptr:
      case OMPC_use_device_addr:
      case OMPC_is_device_ptr:
      case OMPC_nontemporal:
      case OMPC_order:
      case OMPC_destroy:
      case OMPC_inclusive:
      case OMPC_exclusive:
      case OMPC_uses_allocators:
      case OMPC_affinity:
      case OMPC_bind:
        continue;
      case OMPC_allocator:
      case OMPC_flush:
      case OMPC_depobj:
      case OMPC_threadprivate:
      case OMPC_uniform:
      case OMPC_unknown:
      case OMPC_unified_address:
      case OMPC_unified_shared_memory:
      case OMPC_reverse_offload:
      case OMPC_dynamic_allocators:
      case OMPC_atomic_default_mem_order:
      case OMPC_device_type:
      case OMPC_match:
      case OMPC_when:
      default:
        llvm_unreachable("Unexpected clause");
      }
      for (Stmt *CC : C->children()) {
        if (CC)
          DSAChecker.Visit(CC);
      }
    }
    for (const auto &P : DSAChecker.getVarsWithInheritedDSA())
      VarsWithInheritedDSA[P.getFirst()] = P.getSecond();
  }
  for (const auto &P : VarsWithInheritedDSA) {
    if (P.getFirst()->isImplicit() || isa<OMPCapturedExprDecl>(P.getFirst()))
      continue;
    ErrorFound = true;
    if (DSAStack->getDefaultDSA() == DSA_none ||
        DSAStack->getDefaultDSA() == DSA_firstprivate) {
      Diag(P.second->getExprLoc(), diag::err_omp_no_dsa_for_variable)
          << P.first << P.second->getSourceRange();
      Diag(DSAStack->getDefaultDSALocation(), diag::note_omp_default_dsa_none);
    } else if (getLangOpts().OpenMP >= 50) {
      Diag(P.second->getExprLoc(),
           diag::err_omp_defaultmap_no_attr_for_variable)
          << P.first << P.second->getSourceRange();
      Diag(DSAStack->getDefaultDSALocation(),
           diag::note_omp_defaultmap_attr_none);
    }
  }

  if (!AllowedNameModifiers.empty())
    ErrorFound = checkIfClauses(*this, Kind, Clauses, AllowedNameModifiers) ||
                 ErrorFound;

  if (ErrorFound)
    return StmtError();

  if (!CurContext->isDependentContext() &&
      isOpenMPTargetExecutionDirective(Kind) &&
      !(DSAStack->hasRequiresDeclWithClause<OMPUnifiedSharedMemoryClause>() ||
        DSAStack->hasRequiresDeclWithClause<OMPUnifiedAddressClause>() ||
        DSAStack->hasRequiresDeclWithClause<OMPReverseOffloadClause>() ||
        DSAStack->hasRequiresDeclWithClause<OMPDynamicAllocatorsClause>())) {
    // Register target to DSA Stack.
    DSAStack->addTargetDirLocation(StartLoc);
  }

  return Res;
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPDeclareSimdDirective(
    DeclGroupPtrTy DG, OMPDeclareSimdDeclAttr::BranchStateTy BS, Expr *Simdlen,
    ArrayRef<Expr *> Uniforms, ArrayRef<Expr *> Aligneds,
    ArrayRef<Expr *> Alignments, ArrayRef<Expr *> Linears,
    ArrayRef<unsigned> LinModifiers, ArrayRef<Expr *> Steps, SourceRange SR) {
  assert(Aligneds.size() == Alignments.size());
  assert(Linears.size() == LinModifiers.size());
  assert(Linears.size() == Steps.size());
  if (!DG || DG.get().isNull())
    return DeclGroupPtrTy();

  const int SimdId = 0;
  if (!DG.get().isSingleDecl()) {
    Diag(SR.getBegin(), diag::err_omp_single_decl_in_declare_simd_variant)
        << SimdId;
    return DG;
  }
  Decl *ADecl = DG.get().getSingleDecl();
  if (auto *FTD = dyn_cast<FunctionTemplateDecl>(ADecl))
    ADecl = FTD->getTemplatedDecl();

  auto *FD = dyn_cast<FunctionDecl>(ADecl);
  if (!FD) {
    Diag(ADecl->getLocation(), diag::err_omp_function_expected) << SimdId;
    return DeclGroupPtrTy();
  }

  // OpenMP [2.8.2, declare simd construct, Description]
  // The parameter of the simdlen clause must be a constant positive integer
  // expression.
  ExprResult SL;
  if (Simdlen)
    SL = VerifyPositiveIntegerConstantInClause(Simdlen, OMPC_simdlen);
  // OpenMP [2.8.2, declare simd construct, Description]
  // The special this pointer can be used as if was one of the arguments to the
  // function in any of the linear, aligned, or uniform clauses.
  // The uniform clause declares one or more arguments to have an invariant
  // value for all concurrent invocations of the function in the execution of a
  // single SIMD loop.
  llvm::DenseMap<const Decl *, const Expr *> UniformedArgs;
  const Expr *UniformedLinearThis = nullptr;
  for (const Expr *E : Uniforms) {
    E = E->IgnoreParenImpCasts();
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
      if (const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl()))
        if (FD->getNumParams() > PVD->getFunctionScopeIndex() &&
            FD->getParamDecl(PVD->getFunctionScopeIndex())
                    ->getCanonicalDecl() == PVD->getCanonicalDecl()) {
          UniformedArgs.try_emplace(PVD->getCanonicalDecl(), E);
          continue;
        }
    if (isa<CXXThisExpr>(E)) {
      UniformedLinearThis = E;
      continue;
    }
    Diag(E->getExprLoc(), diag::err_omp_param_or_this_in_clause)
        << FD->getDeclName() << (isa<CXXMethodDecl>(ADecl) ? 1 : 0);
  }
  // OpenMP [2.8.2, declare simd construct, Description]
  // The aligned clause declares that the object to which each list item points
  // is aligned to the number of bytes expressed in the optional parameter of
  // the aligned clause.
  // The special this pointer can be used as if was one of the arguments to the
  // function in any of the linear, aligned, or uniform clauses.
  // The type of list items appearing in the aligned clause must be array,
  // pointer, reference to array, or reference to pointer.
  llvm::DenseMap<const Decl *, const Expr *> AlignedArgs;
  const Expr *AlignedThis = nullptr;
  for (const Expr *E : Aligneds) {
    E = E->IgnoreParenImpCasts();
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
      if (const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
        const VarDecl *CanonPVD = PVD->getCanonicalDecl();
        if (FD->getNumParams() > PVD->getFunctionScopeIndex() &&
            FD->getParamDecl(PVD->getFunctionScopeIndex())
                    ->getCanonicalDecl() == CanonPVD) {
          // OpenMP  [2.8.1, simd construct, Restrictions]
          // A list-item cannot appear in more than one aligned clause.
          if (AlignedArgs.count(CanonPVD) > 0) {
            Diag(E->getExprLoc(), diag::err_omp_used_in_clause_twice)
                << 1 << getOpenMPClauseName(OMPC_aligned)
                << E->getSourceRange();
            Diag(AlignedArgs[CanonPVD]->getExprLoc(),
                 diag::note_omp_explicit_dsa)
                << getOpenMPClauseName(OMPC_aligned);
            continue;
          }
          AlignedArgs[CanonPVD] = E;
          QualType QTy = PVD->getType()
                             .getNonReferenceType()
                             .getUnqualifiedType()
                             .getCanonicalType();
          const Type *Ty = QTy.getTypePtrOrNull();
          if (!Ty || (!Ty->isArrayType() && !Ty->isPointerType())) {
            Diag(E->getExprLoc(), diag::err_omp_aligned_expected_array_or_ptr)
                << QTy << getLangOpts().CPlusPlus << E->getSourceRange();
            Diag(PVD->getLocation(), diag::note_previous_decl) << PVD;
          }
          continue;
        }
      }
    if (isa<CXXThisExpr>(E)) {
      if (AlignedThis) {
        Diag(E->getExprLoc(), diag::err_omp_used_in_clause_twice)
            << 2 << getOpenMPClauseName(OMPC_aligned) << E->getSourceRange();
        Diag(AlignedThis->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(OMPC_aligned);
      }
      AlignedThis = E;
      continue;
    }
    Diag(E->getExprLoc(), diag::err_omp_param_or_this_in_clause)
        << FD->getDeclName() << (isa<CXXMethodDecl>(ADecl) ? 1 : 0);
  }
  // The optional parameter of the aligned clause, alignment, must be a constant
  // positive integer expression. If no optional parameter is specified,
  // implementation-defined default alignments for SIMD instructions on the
  // target platforms are assumed.
  SmallVector<const Expr *, 4> NewAligns;
  for (Expr *E : Alignments) {
    ExprResult Align;
    if (E)
      Align = VerifyPositiveIntegerConstantInClause(E, OMPC_aligned);
    NewAligns.push_back(Align.get());
  }
  // OpenMP [2.8.2, declare simd construct, Description]
  // The linear clause declares one or more list items to be private to a SIMD
  // lane and to have a linear relationship with respect to the iteration space
  // of a loop.
  // The special this pointer can be used as if was one of the arguments to the
  // function in any of the linear, aligned, or uniform clauses.
  // When a linear-step expression is specified in a linear clause it must be
  // either a constant integer expression or an integer-typed parameter that is
  // specified in a uniform clause on the directive.
  llvm::DenseMap<const Decl *, const Expr *> LinearArgs;
  const bool IsUniformedThis = UniformedLinearThis != nullptr;
  auto MI = LinModifiers.begin();
  for (const Expr *E : Linears) {
    auto LinKind = static_cast<OpenMPLinearClauseKind>(*MI);
    ++MI;
    E = E->IgnoreParenImpCasts();
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
      if (const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
        const VarDecl *CanonPVD = PVD->getCanonicalDecl();
        if (FD->getNumParams() > PVD->getFunctionScopeIndex() &&
            FD->getParamDecl(PVD->getFunctionScopeIndex())
                    ->getCanonicalDecl() == CanonPVD) {
          // OpenMP  [2.15.3.7, linear Clause, Restrictions]
          // A list-item cannot appear in more than one linear clause.
          if (LinearArgs.count(CanonPVD) > 0) {
            Diag(E->getExprLoc(), diag::err_omp_wrong_dsa)
                << getOpenMPClauseName(OMPC_linear)
                << getOpenMPClauseName(OMPC_linear) << E->getSourceRange();
            Diag(LinearArgs[CanonPVD]->getExprLoc(),
                 diag::note_omp_explicit_dsa)
                << getOpenMPClauseName(OMPC_linear);
            continue;
          }
          // Each argument can appear in at most one uniform or linear clause.
          if (UniformedArgs.count(CanonPVD) > 0) {
            Diag(E->getExprLoc(), diag::err_omp_wrong_dsa)
                << getOpenMPClauseName(OMPC_linear)
                << getOpenMPClauseName(OMPC_uniform) << E->getSourceRange();
            Diag(UniformedArgs[CanonPVD]->getExprLoc(),
                 diag::note_omp_explicit_dsa)
                << getOpenMPClauseName(OMPC_uniform);
            continue;
          }
          LinearArgs[CanonPVD] = E;
          if (E->isValueDependent() || E->isTypeDependent() ||
              E->isInstantiationDependent() ||
              E->containsUnexpandedParameterPack())
            continue;
          (void)CheckOpenMPLinearDecl(CanonPVD, E->getExprLoc(), LinKind,
                                      PVD->getOriginalType(),
                                      /*IsDeclareSimd=*/true);
          continue;
        }
      }
    if (isa<CXXThisExpr>(E)) {
      if (UniformedLinearThis) {
        Diag(E->getExprLoc(), diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(OMPC_linear)
            << getOpenMPClauseName(IsUniformedThis ? OMPC_uniform : OMPC_linear)
            << E->getSourceRange();
        Diag(UniformedLinearThis->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(IsUniformedThis ? OMPC_uniform
                                                   : OMPC_linear);
        continue;
      }
      UniformedLinearThis = E;
      if (E->isValueDependent() || E->isTypeDependent() ||
          E->isInstantiationDependent() || E->containsUnexpandedParameterPack())
        continue;
      (void)CheckOpenMPLinearDecl(/*D=*/nullptr, E->getExprLoc(), LinKind,
                                  E->getType(), /*IsDeclareSimd=*/true);
      continue;
    }
    Diag(E->getExprLoc(), diag::err_omp_param_or_this_in_clause)
        << FD->getDeclName() << (isa<CXXMethodDecl>(ADecl) ? 1 : 0);
  }
  Expr *Step = nullptr;
  Expr *NewStep = nullptr;
  SmallVector<Expr *, 4> NewSteps;
  for (Expr *E : Steps) {
    // Skip the same step expression, it was checked already.
    if (Step == E || !E) {
      NewSteps.push_back(E ? NewStep : nullptr);
      continue;
    }
    Step = E;
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Step))
      if (const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
        const VarDecl *CanonPVD = PVD->getCanonicalDecl();
        if (UniformedArgs.count(CanonPVD) == 0) {
          Diag(Step->getExprLoc(), diag::err_omp_expected_uniform_param)
              << Step->getSourceRange();
        } else if (E->isValueDependent() || E->isTypeDependent() ||
                   E->isInstantiationDependent() ||
                   E->containsUnexpandedParameterPack() ||
                   CanonPVD->getType()->hasIntegerRepresentation()) {
          NewSteps.push_back(Step);
        } else {
          Diag(Step->getExprLoc(), diag::err_omp_expected_int_param)
              << Step->getSourceRange();
        }
        continue;
      }
    NewStep = Step;
    if (Step && !Step->isValueDependent() && !Step->isTypeDependent() &&
        !Step->isInstantiationDependent() &&
        !Step->containsUnexpandedParameterPack()) {
      NewStep = PerformOpenMPImplicitIntegerConversion(Step->getExprLoc(), Step)
                    .get();
      if (NewStep)
        NewStep =
            VerifyIntegerConstantExpression(NewStep, /*FIXME*/ AllowFold).get();
    }
    NewSteps.push_back(NewStep);
  }
  auto *NewAttr = OMPDeclareSimdDeclAttr::CreateImplicit(
      Context, BS, SL.get(), const_cast<Expr **>(Uniforms.data()),
      Uniforms.size(), const_cast<Expr **>(Aligneds.data()), Aligneds.size(),
      const_cast<Expr **>(NewAligns.data()), NewAligns.size(),
      const_cast<Expr **>(Linears.data()), Linears.size(),
      const_cast<unsigned *>(LinModifiers.data()), LinModifiers.size(),
      NewSteps.data(), NewSteps.size(), SR);
  ADecl->addAttr(NewAttr);
  return DG;
}

static void setPrototype(Sema &S, FunctionDecl *FD, FunctionDecl *FDWithProto,
                         QualType NewType) {
  assert(NewType->isFunctionProtoType() &&
         "Expected function type with prototype.");
  assert(FD->getType()->isFunctionNoProtoType() &&
         "Expected function with type with no prototype.");
  assert(FDWithProto->getType()->isFunctionProtoType() &&
         "Expected function with prototype.");
  // Synthesize parameters with the same types.
  FD->setType(NewType);
  SmallVector<ParmVarDecl *, 16> Params;
  for (const ParmVarDecl *P : FDWithProto->parameters()) {
    auto *Param = ParmVarDecl::Create(S.getASTContext(), FD, SourceLocation(),
                                      SourceLocation(), nullptr, P->getType(),
                                      /*TInfo=*/nullptr, SC_None, nullptr);
    Param->setScopeInfo(0, Params.size());
    Param->setImplicit();
    Params.push_back(Param);
  }

  FD->setParams(Params);
}

void Sema::ActOnFinishedFunctionDefinitionInOpenMPAssumeScope(Decl *D) {
  if (D->isInvalidDecl())
    return;
  FunctionDecl *FD = nullptr;
  if (auto *UTemplDecl = dyn_cast<FunctionTemplateDecl>(D))
    FD = UTemplDecl->getTemplatedDecl();
  else
    FD = cast<FunctionDecl>(D);
  assert(FD && "Expected a function declaration!");

  // If we are instantiating templates we do *not* apply scoped assumptions but
  // only global ones. We apply scoped assumption to the template definition
  // though.
  if (!inTemplateInstantiation()) {
    for (AssumptionAttr *AA : OMPAssumeScoped)
      FD->addAttr(AA);
  }
  for (AssumptionAttr *AA : OMPAssumeGlobal)
    FD->addAttr(AA);
}

Sema::OMPDeclareVariantScope::OMPDeclareVariantScope(OMPTraitInfo &TI)
    : TI(&TI), NameSuffix(TI.getMangledName()) {}

void Sema::ActOnStartOfFunctionDefinitionInOpenMPDeclareVariantScope(
    Scope *S, Declarator &D, MultiTemplateParamsArg TemplateParamLists,
    SmallVectorImpl<FunctionDecl *> &Bases) {
  if (!D.getIdentifier())
    return;

  OMPDeclareVariantScope &DVScope = OMPDeclareVariantScopes.back();

  // Template specialization is an extension, check if we do it.
  bool IsTemplated = !TemplateParamLists.empty();
  if (IsTemplated &
      !DVScope.TI->isExtensionActive(
          llvm::omp::TraitProperty::implementation_extension_allow_templates))
    return;

  IdentifierInfo *BaseII = D.getIdentifier();
  LookupResult Lookup(*this, DeclarationName(BaseII), D.getIdentifierLoc(),
                      LookupOrdinaryName);
  LookupParsedName(Lookup, S, &D.getCXXScopeSpec());

  TypeSourceInfo *TInfo = GetTypeForDeclarator(D, S);
  QualType FType = TInfo->getType();

  bool IsConstexpr =
      D.getDeclSpec().getConstexprSpecifier() == ConstexprSpecKind::Constexpr;
  bool IsConsteval =
      D.getDeclSpec().getConstexprSpecifier() == ConstexprSpecKind::Consteval;

  for (auto *Candidate : Lookup) {
    auto *CandidateDecl = Candidate->getUnderlyingDecl();
    FunctionDecl *UDecl = nullptr;
    if (IsTemplated && isa<FunctionTemplateDecl>(CandidateDecl)) {
      auto *FTD = cast<FunctionTemplateDecl>(CandidateDecl);
      if (FTD->getTemplateParameters()->size() == TemplateParamLists.size())
        UDecl = FTD->getTemplatedDecl();
    } else if (!IsTemplated)
      UDecl = dyn_cast<FunctionDecl>(CandidateDecl);
    if (!UDecl)
      continue;

    // Don't specialize constexpr/consteval functions with
    // non-constexpr/consteval functions.
    if (UDecl->isConstexpr() && !IsConstexpr)
      continue;
    if (UDecl->isConsteval() && !IsConsteval)
      continue;

    QualType UDeclTy = UDecl->getType();
    if (!UDeclTy->isDependentType()) {
      QualType NewType = Context.mergeFunctionTypes(
          FType, UDeclTy, /* OfBlockPointer */ false,
          /* Unqualified */ false, /* AllowCXX */ true);
      if (NewType.isNull())
        continue;
    }

    // Found a base!
    Bases.push_back(UDecl);
  }

  bool UseImplicitBase = !DVScope.TI->isExtensionActive(
      llvm::omp::TraitProperty::implementation_extension_disable_implicit_base);
  // If no base was found we create a declaration that we use as base.
  if (Bases.empty() && UseImplicitBase) {
    D.setFunctionDefinitionKind(FunctionDefinitionKind::Declaration);
    Decl *BaseD = HandleDeclarator(S, D, TemplateParamLists);
    BaseD->setImplicit(true);
    if (auto *BaseTemplD = dyn_cast<FunctionTemplateDecl>(BaseD))
      Bases.push_back(BaseTemplD->getTemplatedDecl());
    else
      Bases.push_back(cast<FunctionDecl>(BaseD));
  }

  std::string MangledName;
  MangledName += D.getIdentifier()->getName();
  MangledName += getOpenMPVariantManglingSeparatorStr();
  MangledName += DVScope.NameSuffix;
  IdentifierInfo &VariantII = Context.Idents.get(MangledName);

  VariantII.setMangledOpenMPVariantName(true);
  D.SetIdentifier(&VariantII, D.getBeginLoc());
}

void Sema::ActOnFinishedFunctionDefinitionInOpenMPDeclareVariantScope(
    Decl *D, SmallVectorImpl<FunctionDecl *> &Bases) {
  // Do not mark function as is used to prevent its emission if this is the
  // only place where it is used.
  EnterExpressionEvaluationContext Unevaluated(
      *this, Sema::ExpressionEvaluationContext::Unevaluated);

  FunctionDecl *FD = nullptr;
  if (auto *UTemplDecl = dyn_cast<FunctionTemplateDecl>(D))
    FD = UTemplDecl->getTemplatedDecl();
  else
    FD = cast<FunctionDecl>(D);
  auto *VariantFuncRef = DeclRefExpr::Create(
      Context, NestedNameSpecifierLoc(), SourceLocation(), FD,
      /* RefersToEnclosingVariableOrCapture */ false,
      /* NameLoc */ FD->getLocation(), FD->getType(),
      ExprValueKind::VK_PRValue);

  OMPDeclareVariantScope &DVScope = OMPDeclareVariantScopes.back();
  auto *OMPDeclareVariantA = OMPDeclareVariantAttr::CreateImplicit(
      Context, VariantFuncRef, DVScope.TI,
      /*NothingArgs=*/nullptr, /*NothingArgsSize=*/0,
      /*NeedDevicePtrArgs=*/nullptr, /*NeedDevicePtrArgsSize=*/0,
      /*AppendArgs=*/nullptr, /*AppendArgsSize=*/0);
  for (FunctionDecl *BaseFD : Bases)
    BaseFD->addAttr(OMPDeclareVariantA);
}

ExprResult Sema::ActOnOpenMPCall(ExprResult Call, Scope *Scope,
                                 SourceLocation LParenLoc,
                                 MultiExprArg ArgExprs,
                                 SourceLocation RParenLoc, Expr *ExecConfig) {
  // The common case is a regular call we do not want to specialize at all. Try
  // to make that case fast by bailing early.
  CallExpr *CE = dyn_cast<CallExpr>(Call.get());
  if (!CE)
    return Call;

  FunctionDecl *CalleeFnDecl = CE->getDirectCallee();
  if (!CalleeFnDecl)
    return Call;

  if (!CalleeFnDecl->hasAttr<OMPDeclareVariantAttr>())
    return Call;

  ASTContext &Context = getASTContext();
  std::function<void(StringRef)> DiagUnknownTrait = [this,
                                                     CE](StringRef ISATrait) {
    // TODO Track the selector locations in a way that is accessible here to
    // improve the diagnostic location.
    Diag(CE->getBeginLoc(), diag::warn_unknown_declare_variant_isa_trait)
        << ISATrait;
  };
  TargetOMPContext OMPCtx(Context, std::move(DiagUnknownTrait),
                          getCurFunctionDecl(), DSAStack->getConstructTraits());

  QualType CalleeFnType = CalleeFnDecl->getType();

  SmallVector<Expr *, 4> Exprs;
  SmallVector<VariantMatchInfo, 4> VMIs;
  while (CalleeFnDecl) {
    for (OMPDeclareVariantAttr *A :
         CalleeFnDecl->specific_attrs<OMPDeclareVariantAttr>()) {
      Expr *VariantRef = A->getVariantFuncRef();

      VariantMatchInfo VMI;
      OMPTraitInfo &TI = A->getTraitInfo();
      TI.getAsVariantMatchInfo(Context, VMI);
      if (!isVariantApplicableInContext(VMI, OMPCtx,
                                        /* DeviceSetOnly */ false))
        continue;

      VMIs.push_back(VMI);
      Exprs.push_back(VariantRef);
    }

    CalleeFnDecl = CalleeFnDecl->getPreviousDecl();
  }

  ExprResult NewCall;
  do {
    int BestIdx = getBestVariantMatchForContext(VMIs, OMPCtx);
    if (BestIdx < 0)
      return Call;
    Expr *BestExpr = cast<DeclRefExpr>(Exprs[BestIdx]);
    Decl *BestDecl = cast<DeclRefExpr>(BestExpr)->getDecl();

    {
      // Try to build a (member) call expression for the current best applicable
      // variant expression. We allow this to fail in which case we continue
      // with the next best variant expression. The fail case is part of the
      // implementation defined behavior in the OpenMP standard when it talks
      // about what differences in the function prototypes: "Any differences
      // that the specific OpenMP context requires in the prototype of the
      // variant from the base function prototype are implementation defined."
      // This wording is there to allow the specialized variant to have a
      // different type than the base function. This is intended and OK but if
      // we cannot create a call the difference is not in the "implementation
      // defined range" we allow.
      Sema::TentativeAnalysisScope Trap(*this);

      if (auto *SpecializedMethod = dyn_cast<CXXMethodDecl>(BestDecl)) {
        auto *MemberCall = dyn_cast<CXXMemberCallExpr>(CE);
        BestExpr = MemberExpr::CreateImplicit(
            Context, MemberCall->getImplicitObjectArgument(),
            /* IsArrow */ false, SpecializedMethod, Context.BoundMemberTy,
            MemberCall->getValueKind(), MemberCall->getObjectKind());
      }
      NewCall = BuildCallExpr(Scope, BestExpr, LParenLoc, ArgExprs, RParenLoc,
                              ExecConfig);
      if (NewCall.isUsable()) {
        if (CallExpr *NCE = dyn_cast<CallExpr>(NewCall.get())) {
          FunctionDecl *NewCalleeFnDecl = NCE->getDirectCallee();
          QualType NewType = Context.mergeFunctionTypes(
              CalleeFnType, NewCalleeFnDecl->getType(),
              /* OfBlockPointer */ false,
              /* Unqualified */ false, /* AllowCXX */ true);
          if (!NewType.isNull())
            break;
          // Don't use the call if the function type was not compatible.
          NewCall = nullptr;
        }
      }
    }

    VMIs.erase(VMIs.begin() + BestIdx);
    Exprs.erase(Exprs.begin() + BestIdx);
  } while (!VMIs.empty());

  if (!NewCall.isUsable())
    return Call;
  return PseudoObjectExpr::Create(Context, CE, {NewCall.get()}, 0);
}

Optional<std::pair<FunctionDecl *, Expr *>>
Sema::checkOpenMPDeclareVariantFunction(Sema::DeclGroupPtrTy DG,
                                        Expr *VariantRef, OMPTraitInfo &TI,
                                        unsigned NumAppendArgs,
                                        SourceRange SR) {
  if (!DG || DG.get().isNull())
    return None;

  const int VariantId = 1;
  // Must be applied only to single decl.
  if (!DG.get().isSingleDecl()) {
    Diag(SR.getBegin(), diag::err_omp_single_decl_in_declare_simd_variant)
        << VariantId << SR;
    return None;
  }
  Decl *ADecl = DG.get().getSingleDecl();
  if (auto *FTD = dyn_cast<FunctionTemplateDecl>(ADecl))
    ADecl = FTD->getTemplatedDecl();

  // Decl must be a function.
  auto *FD = dyn_cast<FunctionDecl>(ADecl);
  if (!FD) {
    Diag(ADecl->getLocation(), diag::err_omp_function_expected)
        << VariantId << SR;
    return None;
  }

  auto &&HasMultiVersionAttributes = [](const FunctionDecl *FD) {
    return FD->hasAttrs() &&
           (FD->hasAttr<CPUDispatchAttr>() || FD->hasAttr<CPUSpecificAttr>() ||
            FD->hasAttr<TargetAttr>());
  };
  // OpenMP is not compatible with CPU-specific attributes.
  if (HasMultiVersionAttributes(FD)) {
    Diag(FD->getLocation(), diag::err_omp_declare_variant_incompat_attributes)
        << SR;
    return None;
  }

  // Allow #pragma omp declare variant only if the function is not used.
  if (FD->isUsed(false))
    Diag(SR.getBegin(), diag::warn_omp_declare_variant_after_used)
        << FD->getLocation();

  // Check if the function was emitted already.
  const FunctionDecl *Definition;
  if (!FD->isThisDeclarationADefinition() && FD->isDefined(Definition) &&
      (LangOpts.EmitAllDecls || Context.DeclMustBeEmitted(Definition)))
    Diag(SR.getBegin(), diag::warn_omp_declare_variant_after_emitted)
        << FD->getLocation();

  // The VariantRef must point to function.
  if (!VariantRef) {
    Diag(SR.getBegin(), diag::err_omp_function_expected) << VariantId;
    return None;
  }

  auto ShouldDelayChecks = [](Expr *&E, bool) {
    return E && (E->isTypeDependent() || E->isValueDependent() ||
                 E->containsUnexpandedParameterPack() ||
                 E->isInstantiationDependent());
  };
  // Do not check templates, wait until instantiation.
  if (FD->isDependentContext() || ShouldDelayChecks(VariantRef, false) ||
      TI.anyScoreOrCondition(ShouldDelayChecks))
    return std::make_pair(FD, VariantRef);

  // Deal with non-constant score and user condition expressions.
  auto HandleNonConstantScoresAndConditions = [this](Expr *&E,
                                                     bool IsScore) -> bool {
    if (!E || E->isIntegerConstantExpr(Context))
      return false;

    if (IsScore) {
      // We warn on non-constant scores and pretend they were not present.
      Diag(E->getExprLoc(), diag::warn_omp_declare_variant_score_not_constant)
          << E;
      E = nullptr;
    } else {
      // We could replace a non-constant user condition with "false" but we
      // will soon need to handle these anyway for the dynamic version of
      // OpenMP context selectors.
      Diag(E->getExprLoc(),
           diag::err_omp_declare_variant_user_condition_not_constant)
          << E;
    }
    return true;
  };
  if (TI.anyScoreOrCondition(HandleNonConstantScoresAndConditions))
    return None;

  QualType AdjustedFnType = FD->getType();
  if (NumAppendArgs) {
    const auto *PTy = AdjustedFnType->getAsAdjusted<FunctionProtoType>();
    if (!PTy) {
      Diag(FD->getLocation(), diag::err_omp_declare_variant_prototype_required)
          << SR;
      return None;
    }
    // Adjust the function type to account for an extra omp_interop_t for each
    // specified in the append_args clause.
    const TypeDecl *TD = nullptr;
    LookupResult Result(*this, &Context.Idents.get("omp_interop_t"),
                        SR.getBegin(), Sema::LookupOrdinaryName);
    if (LookupName(Result, getCurScope())) {
      NamedDecl *ND = Result.getFoundDecl();
      TD = dyn_cast_or_null<TypeDecl>(ND);
    }
    if (!TD) {
      Diag(SR.getBegin(), diag::err_omp_interop_type_not_found) << SR;
      return None;
    }
    QualType InteropType = Context.getTypeDeclType(TD);
    if (PTy->isVariadic()) {
      Diag(FD->getLocation(), diag::err_omp_append_args_with_varargs) << SR;
      return None;
    }
    llvm::SmallVector<QualType, 8> Params;
    Params.append(PTy->param_type_begin(), PTy->param_type_end());
    Params.insert(Params.end(), NumAppendArgs, InteropType);
    AdjustedFnType = Context.getFunctionType(PTy->getReturnType(), Params,
                                             PTy->getExtProtoInfo());
  }

  // Convert VariantRef expression to the type of the original function to
  // resolve possible conflicts.
  ExprResult VariantRefCast = VariantRef;
  if (LangOpts.CPlusPlus) {
    QualType FnPtrType;
    auto *Method = dyn_cast<CXXMethodDecl>(FD);
    if (Method && !Method->isStatic()) {
      const Type *ClassType =
          Context.getTypeDeclType(Method->getParent()).getTypePtr();
      FnPtrType = Context.getMemberPointerType(AdjustedFnType, ClassType);
      ExprResult ER;
      {
        // Build adrr_of unary op to correctly handle type checks for member
        // functions.
        Sema::TentativeAnalysisScope Trap(*this);
        ER = CreateBuiltinUnaryOp(VariantRef->getBeginLoc(), UO_AddrOf,
                                  VariantRef);
      }
      if (!ER.isUsable()) {
        Diag(VariantRef->getExprLoc(), diag::err_omp_function_expected)
            << VariantId << VariantRef->getSourceRange();
        return None;
      }
      VariantRef = ER.get();
    } else {
      FnPtrType = Context.getPointerType(AdjustedFnType);
    }
    QualType VarianPtrType = Context.getPointerType(VariantRef->getType());
    if (VarianPtrType.getUnqualifiedType() != FnPtrType.getUnqualifiedType()) {
      ImplicitConversionSequence ICS = TryImplicitConversion(
          VariantRef, FnPtrType.getUnqualifiedType(),
          /*SuppressUserConversions=*/false, AllowedExplicit::None,
          /*InOverloadResolution=*/false,
          /*CStyle=*/false,
          /*AllowObjCWritebackConversion=*/false);
      if (ICS.isFailure()) {
        Diag(VariantRef->getExprLoc(),
             diag::err_omp_declare_variant_incompat_types)
            << VariantRef->getType()
            << ((Method && !Method->isStatic()) ? FnPtrType : FD->getType())
            << (NumAppendArgs ? 1 : 0) << VariantRef->getSourceRange();
        return None;
      }
      VariantRefCast = PerformImplicitConversion(
          VariantRef, FnPtrType.getUnqualifiedType(), AA_Converting);
      if (!VariantRefCast.isUsable())
        return None;
    }
    // Drop previously built artificial addr_of unary op for member functions.
    if (Method && !Method->isStatic()) {
      Expr *PossibleAddrOfVariantRef = VariantRefCast.get();
      if (auto *UO = dyn_cast<UnaryOperator>(
              PossibleAddrOfVariantRef->IgnoreImplicit()))
        VariantRefCast = UO->getSubExpr();
    }
  }

  ExprResult ER = CheckPlaceholderExpr(VariantRefCast.get());
  if (!ER.isUsable() ||
      !ER.get()->IgnoreParenImpCasts()->getType()->isFunctionType()) {
    Diag(VariantRef->getExprLoc(), diag::err_omp_function_expected)
        << VariantId << VariantRef->getSourceRange();
    return None;
  }

  // The VariantRef must point to function.
  auto *DRE = dyn_cast<DeclRefExpr>(ER.get()->IgnoreParenImpCasts());
  if (!DRE) {
    Diag(VariantRef->getExprLoc(), diag::err_omp_function_expected)
        << VariantId << VariantRef->getSourceRange();
    return None;
  }
  auto *NewFD = dyn_cast_or_null<FunctionDecl>(DRE->getDecl());
  if (!NewFD) {
    Diag(VariantRef->getExprLoc(), diag::err_omp_function_expected)
        << VariantId << VariantRef->getSourceRange();
    return None;
  }

  // Check if function types are compatible in C.
  if (!LangOpts.CPlusPlus) {
    QualType NewType =
        Context.mergeFunctionTypes(AdjustedFnType, NewFD->getType());
    if (NewType.isNull()) {
      Diag(VariantRef->getExprLoc(),
           diag::err_omp_declare_variant_incompat_types)
          << NewFD->getType() << FD->getType() << (NumAppendArgs ? 1 : 0)
          << VariantRef->getSourceRange();
      return None;
    }
    if (NewType->isFunctionProtoType()) {
      if (FD->getType()->isFunctionNoProtoType())
        setPrototype(*this, FD, NewFD, NewType);
      else if (NewFD->getType()->isFunctionNoProtoType())
        setPrototype(*this, NewFD, FD, NewType);
    }
  }

  // Check if variant function is not marked with declare variant directive.
  if (NewFD->hasAttrs() && NewFD->hasAttr<OMPDeclareVariantAttr>()) {
    Diag(VariantRef->getExprLoc(),
         diag::warn_omp_declare_variant_marked_as_declare_variant)
        << VariantRef->getSourceRange();
    SourceRange SR =
        NewFD->specific_attr_begin<OMPDeclareVariantAttr>()->getRange();
    Diag(SR.getBegin(), diag::note_omp_marked_declare_variant_here) << SR;
    return None;
  }

  enum DoesntSupport {
    VirtFuncs = 1,
    Constructors = 3,
    Destructors = 4,
    DeletedFuncs = 5,
    DefaultedFuncs = 6,
    ConstexprFuncs = 7,
    ConstevalFuncs = 8,
  };
  if (const auto *CXXFD = dyn_cast<CXXMethodDecl>(FD)) {
    if (CXXFD->isVirtual()) {
      Diag(FD->getLocation(), diag::err_omp_declare_variant_doesnt_support)
          << VirtFuncs;
      return None;
    }

    if (isa<CXXConstructorDecl>(FD)) {
      Diag(FD->getLocation(), diag::err_omp_declare_variant_doesnt_support)
          << Constructors;
      return None;
    }

    if (isa<CXXDestructorDecl>(FD)) {
      Diag(FD->getLocation(), diag::err_omp_declare_variant_doesnt_support)
          << Destructors;
      return None;
    }
  }

  if (FD->isDeleted()) {
    Diag(FD->getLocation(), diag::err_omp_declare_variant_doesnt_support)
        << DeletedFuncs;
    return None;
  }

  if (FD->isDefaulted()) {
    Diag(FD->getLocation(), diag::err_omp_declare_variant_doesnt_support)
        << DefaultedFuncs;
    return None;
  }

  if (FD->isConstexpr()) {
    Diag(FD->getLocation(), diag::err_omp_declare_variant_doesnt_support)
        << (NewFD->isConsteval() ? ConstevalFuncs : ConstexprFuncs);
    return None;
  }

  // Check general compatibility.
  if (areMultiversionVariantFunctionsCompatible(
          FD, NewFD, PartialDiagnostic::NullDiagnostic(),
          PartialDiagnosticAt(SourceLocation(),
                              PartialDiagnostic::NullDiagnostic()),
          PartialDiagnosticAt(
              VariantRef->getExprLoc(),
              PDiag(diag::err_omp_declare_variant_doesnt_support)),
          PartialDiagnosticAt(VariantRef->getExprLoc(),
                              PDiag(diag::err_omp_declare_variant_diff)
                                  << FD->getLocation()),
          /*TemplatesSupported=*/true, /*ConstexprSupported=*/false,
          /*CLinkageMayDiffer=*/true))
    return None;
  return std::make_pair(FD, cast<Expr>(DRE));
}

void Sema::ActOnOpenMPDeclareVariantDirective(
    FunctionDecl *FD, Expr *VariantRef, OMPTraitInfo &TI,
    ArrayRef<Expr *> AdjustArgsNothing,
    ArrayRef<Expr *> AdjustArgsNeedDevicePtr,
    ArrayRef<OMPDeclareVariantAttr::InteropType> AppendArgs,
    SourceLocation AdjustArgsLoc, SourceLocation AppendArgsLoc,
    SourceRange SR) {

  // OpenMP 5.1 [2.3.5, declare variant directive, Restrictions]
  // An adjust_args clause or append_args clause can only be specified if the
  // dispatch selector of the construct selector set appears in the match
  // clause.

  SmallVector<Expr *, 8> AllAdjustArgs;
  llvm::append_range(AllAdjustArgs, AdjustArgsNothing);
  llvm::append_range(AllAdjustArgs, AdjustArgsNeedDevicePtr);

  if (!AllAdjustArgs.empty() || !AppendArgs.empty()) {
    VariantMatchInfo VMI;
    TI.getAsVariantMatchInfo(Context, VMI);
    if (!llvm::is_contained(
            VMI.ConstructTraits,
            llvm::omp::TraitProperty::construct_dispatch_dispatch)) {
      if (!AllAdjustArgs.empty())
        Diag(AdjustArgsLoc, diag::err_omp_clause_requires_dispatch_construct)
            << getOpenMPClauseName(OMPC_adjust_args);
      if (!AppendArgs.empty())
        Diag(AppendArgsLoc, diag::err_omp_clause_requires_dispatch_construct)
            << getOpenMPClauseName(OMPC_append_args);
      return;
    }
  }

  // OpenMP 5.1 [2.3.5, declare variant directive, Restrictions]
  // Each argument can only appear in a single adjust_args clause for each
  // declare variant directive.
  llvm::SmallPtrSet<const VarDecl *, 4> AdjustVars;

  for (Expr *E : AllAdjustArgs) {
    E = E->IgnoreParenImpCasts();
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
        const VarDecl *CanonPVD = PVD->getCanonicalDecl();
        if (FD->getNumParams() > PVD->getFunctionScopeIndex() &&
            FD->getParamDecl(PVD->getFunctionScopeIndex())
                    ->getCanonicalDecl() == CanonPVD) {
          // It's a parameter of the function, check duplicates.
          if (!AdjustVars.insert(CanonPVD).second) {
            Diag(DRE->getLocation(), diag::err_omp_adjust_arg_multiple_clauses)
                << PVD;
            return;
          }
          continue;
        }
      }
    }
    // Anything that is not a function parameter is an error.
    Diag(E->getExprLoc(), diag::err_omp_param_or_this_in_clause) << FD << 0;
    return;
  }

  auto *NewAttr = OMPDeclareVariantAttr::CreateImplicit(
      Context, VariantRef, &TI, const_cast<Expr **>(AdjustArgsNothing.data()),
      AdjustArgsNothing.size(),
      const_cast<Expr **>(AdjustArgsNeedDevicePtr.data()),
      AdjustArgsNeedDevicePtr.size(),
      const_cast<OMPDeclareVariantAttr::InteropType *>(AppendArgs.data()),
      AppendArgs.size(), SR);
  FD->addAttr(NewAttr);
}

StmtResult Sema::ActOnOpenMPParallelDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  setFunctionHasBranchProtectedScope();

  return OMPParallelDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                      DSAStack->getTaskgroupReductionRef(),
                                      DSAStack->isCancelRegion());
}

namespace {
/// Iteration space of a single for loop.
struct LoopIterationSpace final {
  /// True if the condition operator is the strict compare operator (<, > or
  /// !=).
  bool IsStrictCompare = false;
  /// Condition of the loop.
  Expr *PreCond = nullptr;
  /// This expression calculates the number of iterations in the loop.
  /// It is always possible to calculate it before starting the loop.
  Expr *NumIterations = nullptr;
  /// The loop counter variable.
  Expr *CounterVar = nullptr;
  /// Private loop counter variable.
  Expr *PrivateCounterVar = nullptr;
  /// This is initializer for the initial value of #CounterVar.
  Expr *CounterInit = nullptr;
  /// This is step for the #CounterVar used to generate its update:
  /// #CounterVar = #CounterInit + #CounterStep * CurrentIteration.
  Expr *CounterStep = nullptr;
  /// Should step be subtracted?
  bool Subtract = false;
  /// Source range of the loop init.
  SourceRange InitSrcRange;
  /// Source range of the loop condition.
  SourceRange CondSrcRange;
  /// Source range of the loop increment.
  SourceRange IncSrcRange;
  /// Minimum value that can have the loop control variable. Used to support
  /// non-rectangular loops. Applied only for LCV with the non-iterator types,
  /// since only such variables can be used in non-loop invariant expressions.
  Expr *MinValue = nullptr;
  /// Maximum value that can have the loop control variable. Used to support
  /// non-rectangular loops. Applied only for LCV with the non-iterator type,
  /// since only such variables can be used in non-loop invariant expressions.
  Expr *MaxValue = nullptr;
  /// true, if the lower bound depends on the outer loop control var.
  bool IsNonRectangularLB = false;
  /// true, if the upper bound depends on the outer loop control var.
  bool IsNonRectangularUB = false;
  /// Index of the loop this loop depends on and forms non-rectangular loop
  /// nest.
  unsigned LoopDependentIdx = 0;
  /// Final condition for the non-rectangular loop nest support. It is used to
  /// check that the number of iterations for this particular counter must be
  /// finished.
  Expr *FinalCondition = nullptr;
};

/// Helper class for checking canonical form of the OpenMP loops and
/// extracting iteration space of each loop in the loop nest, that will be used
/// for IR generation.
class OpenMPIterationSpaceChecker {
  /// Reference to Sema.
  Sema &SemaRef;
  /// Does the loop associated directive support non-rectangular loops?
  bool SupportsNonRectangular;
  /// Data-sharing stack.
  DSAStackTy &Stack;
  /// A location for diagnostics (when there is no some better location).
  SourceLocation DefaultLoc;
  /// A location for diagnostics (when increment is not compatible).
  SourceLocation ConditionLoc;
  /// A source location for referring to loop init later.
  SourceRange InitSrcRange;
  /// A source location for referring to condition later.
  SourceRange ConditionSrcRange;
  /// A source location for referring to increment later.
  SourceRange IncrementSrcRange;
  /// Loop variable.
  ValueDecl *LCDecl = nullptr;
  /// Reference to loop variable.
  Expr *LCRef = nullptr;
  /// Lower bound (initializer for the var).
  Expr *LB = nullptr;
  /// Upper bound.
  Expr *UB = nullptr;
  /// Loop step (increment).
  Expr *Step = nullptr;
  /// This flag is true when condition is one of:
  ///   Var <  UB
  ///   Var <= UB
  ///   UB  >  Var
  ///   UB  >= Var
  /// This will have no value when the condition is !=
  llvm::Optional<bool> TestIsLessOp;
  /// This flag is true when condition is strict ( < or > ).
  bool TestIsStrictOp = false;
  /// This flag is true when step is subtracted on each iteration.
  bool SubtractStep = false;
  /// The outer loop counter this loop depends on (if any).
  const ValueDecl *DepDecl = nullptr;
  /// Contains number of loop (starts from 1) on which loop counter init
  /// expression of this loop depends on.
  Optional<unsigned> InitDependOnLC;
  /// Contains number of loop (starts from 1) on which loop counter condition
  /// expression of this loop depends on.
  Optional<unsigned> CondDependOnLC;
  /// Checks if the provide statement depends on the loop counter.
  Optional<unsigned> doesDependOnLoopCounter(const Stmt *S, bool IsInitializer);
  /// Original condition required for checking of the exit condition for
  /// non-rectangular loop.
  Expr *Condition = nullptr;

public:
  OpenMPIterationSpaceChecker(Sema &SemaRef, bool SupportsNonRectangular,
                              DSAStackTy &Stack, SourceLocation DefaultLoc)
      : SemaRef(SemaRef), SupportsNonRectangular(SupportsNonRectangular),
        Stack(Stack), DefaultLoc(DefaultLoc), ConditionLoc(DefaultLoc) {}
  /// Check init-expr for canonical loop form and save loop counter
  /// variable - #Var and its initialization value - #LB.
  bool checkAndSetInit(Stmt *S, bool EmitDiags = true);
  /// Check test-expr for canonical form, save upper-bound (#UB), flags
  /// for less/greater and for strict/non-strict comparison.
  bool checkAndSetCond(Expr *S);
  /// Check incr-expr for canonical loop form and return true if it
  /// does not conform, otherwise save loop step (#Step).
  bool checkAndSetInc(Expr *S);
  /// Return the loop counter variable.
  ValueDecl *getLoopDecl() const { return LCDecl; }
  /// Return the reference expression to loop counter variable.
  Expr *getLoopDeclRefExpr() const { return LCRef; }
  /// Source range of the loop init.
  SourceRange getInitSrcRange() const { return InitSrcRange; }
  /// Source range of the loop condition.
  SourceRange getConditionSrcRange() const { return ConditionSrcRange; }
  /// Source range of the loop increment.
  SourceRange getIncrementSrcRange() const { return IncrementSrcRange; }
  /// True if the step should be subtracted.
  bool shouldSubtractStep() const { return SubtractStep; }
  /// True, if the compare operator is strict (<, > or !=).
  bool isStrictTestOp() const { return TestIsStrictOp; }
  /// Build the expression to calculate the number of iterations.
  Expr *buildNumIterations(
      Scope *S, ArrayRef<LoopIterationSpace> ResultIterSpaces, bool LimitedType,
      llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) const;
  /// Build the precondition expression for the loops.
  Expr *
  buildPreCond(Scope *S, Expr *Cond,
               llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) const;
  /// Build reference expression to the counter be used for codegen.
  DeclRefExpr *
  buildCounterVar(llvm::MapVector<const Expr *, DeclRefExpr *> &Captures,
                  DSAStackTy &DSA) const;
  /// Build reference expression to the private counter be used for
  /// codegen.
  Expr *buildPrivateCounterVar() const;
  /// Build initialization of the counter be used for codegen.
  Expr *buildCounterInit() const;
  /// Build step of the counter be used for codegen.
  Expr *buildCounterStep() const;
  /// Build loop data with counter value for depend clauses in ordered
  /// directives.
  Expr *
  buildOrderedLoopData(Scope *S, Expr *Counter,
                       llvm::MapVector<const Expr *, DeclRefExpr *> &Captures,
                       SourceLocation Loc, Expr *Inc = nullptr,
                       OverloadedOperatorKind OOK = OO_Amp);
  /// Builds the minimum value for the loop counter.
  std::pair<Expr *, Expr *> buildMinMaxValues(
      Scope *S, llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) const;
  /// Builds final condition for the non-rectangular loops.
  Expr *buildFinalCondition(Scope *S) const;
  /// Return true if any expression is dependent.
  bool dependent() const;
  /// Returns true if the initializer forms non-rectangular loop.
  bool doesInitDependOnLC() const { return InitDependOnLC.hasValue(); }
  /// Returns true if the condition forms non-rectangular loop.
  bool doesCondDependOnLC() const { return CondDependOnLC.hasValue(); }
  /// Returns index of the loop we depend on (starting from 1), or 0 otherwise.
  unsigned getLoopDependentIdx() const {
    return InitDependOnLC.getValueOr(CondDependOnLC.getValueOr(0));
  }

private:
  /// Check the right-hand side of an assignment in the increment
  /// expression.
  bool checkAndSetIncRHS(Expr *RHS);
  /// Helper to set loop counter variable and its initializer.
  bool setLCDeclAndLB(ValueDecl *NewLCDecl, Expr *NewDeclRefExpr, Expr *NewLB,
                      bool EmitDiags);
  /// Helper to set upper bound.
  bool setUB(Expr *NewUB, llvm::Optional<bool> LessOp, bool StrictOp,
             SourceRange SR, SourceLocation SL);
  /// Helper to set loop increment.
  bool setStep(Expr *NewStep, bool Subtract);
};

bool OpenMPIterationSpaceChecker::dependent() const {
  if (!LCDecl) {
    assert(!LB && !UB && !Step);
    return false;
  }
  return LCDecl->getType()->isDependentType() ||
         (LB && LB->isValueDependent()) || (UB && UB->isValueDependent()) ||
         (Step && Step->isValueDependent());
}

bool OpenMPIterationSpaceChecker::setLCDeclAndLB(ValueDecl *NewLCDecl,
                                                 Expr *NewLCRefExpr,
                                                 Expr *NewLB, bool EmitDiags) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl == nullptr && LB == nullptr && LCRef == nullptr &&
         UB == nullptr && Step == nullptr && !TestIsLessOp && !TestIsStrictOp);
  if (!NewLCDecl || !NewLB || NewLB->containsErrors())
    return true;
  LCDecl = getCanonicalDecl(NewLCDecl);
  LCRef = NewLCRefExpr;
  if (auto *CE = dyn_cast_or_null<CXXConstructExpr>(NewLB))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        NewLB = CE->getArg(0)->IgnoreParenImpCasts();
  LB = NewLB;
  if (EmitDiags)
    InitDependOnLC = doesDependOnLoopCounter(LB, /*IsInitializer=*/true);
  return false;
}

bool OpenMPIterationSpaceChecker::setUB(Expr *NewUB,
                                        llvm::Optional<bool> LessOp,
                                        bool StrictOp, SourceRange SR,
                                        SourceLocation SL) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl != nullptr && LB != nullptr && UB == nullptr &&
         Step == nullptr && !TestIsLessOp && !TestIsStrictOp);
  if (!NewUB || NewUB->containsErrors())
    return true;
  UB = NewUB;
  if (LessOp)
    TestIsLessOp = LessOp;
  TestIsStrictOp = StrictOp;
  ConditionSrcRange = SR;
  ConditionLoc = SL;
  CondDependOnLC = doesDependOnLoopCounter(UB, /*IsInitializer=*/false);
  return false;
}

bool OpenMPIterationSpaceChecker::setStep(Expr *NewStep, bool Subtract) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl != nullptr && LB != nullptr && Step == nullptr);
  if (!NewStep || NewStep->containsErrors())
    return true;
  if (!NewStep->isValueDependent()) {
    // Check that the step is integer expression.
    SourceLocation StepLoc = NewStep->getBeginLoc();
    ExprResult Val = SemaRef.PerformOpenMPImplicitIntegerConversion(
        StepLoc, getExprAsWritten(NewStep));
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
    Optional<llvm::APSInt> Result =
        NewStep->getIntegerConstantExpr(SemaRef.Context);
    bool IsUnsigned = !NewStep->getType()->hasSignedIntegerRepresentation();
    bool IsConstNeg =
        Result && Result->isSigned() && (Subtract != Result->isNegative());
    bool IsConstPos =
        Result && Result->isSigned() && (Subtract == Result->isNegative());
    bool IsConstZero = Result && !Result->getBoolValue();

    // != with increment is treated as <; != with decrement is treated as >
    if (!TestIsLessOp.hasValue())
      TestIsLessOp = IsConstPos || (IsUnsigned && !Subtract);
    if (UB &&
        (IsConstZero || (TestIsLessOp.getValue()
                             ? (IsConstNeg || (IsUnsigned && Subtract))
                             : (IsConstPos || (IsUnsigned && !Subtract))))) {
      SemaRef.Diag(NewStep->getExprLoc(),
                   diag::err_omp_loop_incr_not_compatible)
          << LCDecl << TestIsLessOp.getValue() << NewStep->getSourceRange();
      SemaRef.Diag(ConditionLoc,
                   diag::note_omp_loop_cond_requres_compatible_incr)
          << TestIsLessOp.getValue() << ConditionSrcRange;
      return true;
    }
    if (TestIsLessOp.getValue() == Subtract) {
      NewStep =
          SemaRef.CreateBuiltinUnaryOp(NewStep->getExprLoc(), UO_Minus, NewStep)
              .get();
      Subtract = !Subtract;
    }
  }

  Step = NewStep;
  SubtractStep = Subtract;
  return false;
}

namespace {
/// Checker for the non-rectangular loops. Checks if the initializer or
/// condition expression references loop counter variable.
class LoopCounterRefChecker final
    : public ConstStmtVisitor<LoopCounterRefChecker, bool> {
  Sema &SemaRef;
  DSAStackTy &Stack;
  const ValueDecl *CurLCDecl = nullptr;
  const ValueDecl *DepDecl = nullptr;
  const ValueDecl *PrevDepDecl = nullptr;
  bool IsInitializer = true;
  bool SupportsNonRectangular;
  unsigned BaseLoopId = 0;
  bool checkDecl(const Expr *E, const ValueDecl *VD) {
    if (getCanonicalDecl(VD) == getCanonicalDecl(CurLCDecl)) {
      SemaRef.Diag(E->getExprLoc(), diag::err_omp_stmt_depends_on_loop_counter)
          << (IsInitializer ? 0 : 1);
      return false;
    }
    const auto &&Data = Stack.isLoopControlVariable(VD);
    // OpenMP, 2.9.1 Canonical Loop Form, Restrictions.
    // The type of the loop iterator on which we depend may not have a random
    // access iterator type.
    if (Data.first && VD->getType()->isRecordType()) {
      SmallString<128> Name;
      llvm::raw_svector_ostream OS(Name);
      VD->getNameForDiagnostic(OS, SemaRef.getPrintingPolicy(),
                               /*Qualified=*/true);
      SemaRef.Diag(E->getExprLoc(),
                   diag::err_omp_wrong_dependency_iterator_type)
          << OS.str();
      SemaRef.Diag(VD->getLocation(), diag::note_previous_decl) << VD;
      return false;
    }
    if (Data.first && !SupportsNonRectangular) {
      SemaRef.Diag(E->getExprLoc(), diag::err_omp_invariant_dependency);
      return false;
    }
    if (Data.first &&
        (DepDecl || (PrevDepDecl &&
                     getCanonicalDecl(VD) != getCanonicalDecl(PrevDepDecl)))) {
      if (!DepDecl && PrevDepDecl)
        DepDecl = PrevDepDecl;
      SmallString<128> Name;
      llvm::raw_svector_ostream OS(Name);
      DepDecl->getNameForDiagnostic(OS, SemaRef.getPrintingPolicy(),
                                    /*Qualified=*/true);
      SemaRef.Diag(E->getExprLoc(),
                   diag::err_omp_invariant_or_linear_dependency)
          << OS.str();
      return false;
    }
    if (Data.first) {
      DepDecl = VD;
      BaseLoopId = Data.first;
    }
    return Data.first;
  }

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    const ValueDecl *VD = E->getDecl();
    if (isa<VarDecl>(VD))
      return checkDecl(E, VD);
    return false;
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (isa<CXXThisExpr>(E->getBase()->IgnoreParens())) {
      const ValueDecl *VD = E->getMemberDecl();
      if (isa<VarDecl>(VD) || isa<FieldDecl>(VD))
        return checkDecl(E, VD);
    }
    return false;
  }
  bool VisitStmt(const Stmt *S) {
    bool Res = false;
    for (const Stmt *Child : S->children())
      Res = (Child && Visit(Child)) || Res;
    return Res;
  }
  explicit LoopCounterRefChecker(Sema &SemaRef, DSAStackTy &Stack,
                                 const ValueDecl *CurLCDecl, bool IsInitializer,
                                 const ValueDecl *PrevDepDecl = nullptr,
                                 bool SupportsNonRectangular = true)
      : SemaRef(SemaRef), Stack(Stack), CurLCDecl(CurLCDecl),
        PrevDepDecl(PrevDepDecl), IsInitializer(IsInitializer),
        SupportsNonRectangular(SupportsNonRectangular) {}
  unsigned getBaseLoopId() const {
    assert(CurLCDecl && "Expected loop dependency.");
    return BaseLoopId;
  }
  const ValueDecl *getDepDecl() const {
    assert(CurLCDecl && "Expected loop dependency.");
    return DepDecl;
  }
};
} // namespace

Optional<unsigned>
OpenMPIterationSpaceChecker::doesDependOnLoopCounter(const Stmt *S,
                                                     bool IsInitializer) {
  // Check for the non-rectangular loops.
  LoopCounterRefChecker LoopStmtChecker(SemaRef, Stack, LCDecl, IsInitializer,
                                        DepDecl, SupportsNonRectangular);
  if (LoopStmtChecker.Visit(S)) {
    DepDecl = LoopStmtChecker.getDepDecl();
    return LoopStmtChecker.getBaseLoopId();
  }
  return llvm::None;
}

bool OpenMPIterationSpaceChecker::checkAndSetInit(Stmt *S, bool EmitDiags) {
  // Check init-expr for canonical loop form and save loop counter
  // variable - #Var and its initialization value - #LB.
  // OpenMP [2.6] Canonical loop form. init-expr may be one of the following:
  //   var = lb
  //   integer-type var = lb
  //   random-access-iterator-type var = lb
  //   pointer-type var = lb
  //
  if (!S) {
    if (EmitDiags) {
      SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_init);
    }
    return true;
  }
  if (auto *ExprTemp = dyn_cast<ExprWithCleanups>(S))
    if (!ExprTemp->cleanupsHaveSideEffects())
      S = ExprTemp->getSubExpr();

  InitSrcRange = S->getSourceRange();
  if (Expr *E = dyn_cast<Expr>(S))
    S = E->IgnoreParens();
  if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() == BO_Assign) {
      Expr *LHS = BO->getLHS()->IgnoreParens();
      if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
        if (auto *CED = dyn_cast<OMPCapturedExprDecl>(DRE->getDecl()))
          if (auto *ME = dyn_cast<MemberExpr>(getExprAsWritten(CED->getInit())))
            return setLCDeclAndLB(ME->getMemberDecl(), ME, BO->getRHS(),
                                  EmitDiags);
        return setLCDeclAndLB(DRE->getDecl(), DRE, BO->getRHS(), EmitDiags);
      }
      if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
        if (ME->isArrow() &&
            isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
          return setLCDeclAndLB(ME->getMemberDecl(), ME, BO->getRHS(),
                                EmitDiags);
      }
    }
  } else if (auto *DS = dyn_cast<DeclStmt>(S)) {
    if (DS->isSingleDecl()) {
      if (auto *Var = dyn_cast_or_null<VarDecl>(DS->getSingleDecl())) {
        if (Var->hasInit() && !Var->getType()->isReferenceType()) {
          // Accept non-canonical init form here but emit ext. warning.
          if (Var->getInitStyle() != VarDecl::CInit && EmitDiags)
            SemaRef.Diag(S->getBeginLoc(),
                         diag::ext_omp_loop_not_canonical_init)
                << S->getSourceRange();
          return setLCDeclAndLB(
              Var,
              buildDeclRefExpr(SemaRef, Var,
                               Var->getType().getNonReferenceType(),
                               DS->getBeginLoc()),
              Var->getInit(), EmitDiags);
        }
      }
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getOperator() == OO_Equal) {
      Expr *LHS = CE->getArg(0);
      if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
        if (auto *CED = dyn_cast<OMPCapturedExprDecl>(DRE->getDecl()))
          if (auto *ME = dyn_cast<MemberExpr>(getExprAsWritten(CED->getInit())))
            return setLCDeclAndLB(ME->getMemberDecl(), ME, BO->getRHS(),
                                  EmitDiags);
        return setLCDeclAndLB(DRE->getDecl(), DRE, CE->getArg(1), EmitDiags);
      }
      if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
        if (ME->isArrow() &&
            isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
          return setLCDeclAndLB(ME->getMemberDecl(), ME, BO->getRHS(),
                                EmitDiags);
      }
    }
  }

  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  if (EmitDiags) {
    SemaRef.Diag(S->getBeginLoc(), diag::err_omp_loop_not_canonical_init)
        << S->getSourceRange();
  }
  return true;
}

/// Ignore parenthesizes, implicit casts, copy constructor and return the
/// variable (which may be the loop variable) if possible.
static const ValueDecl *getInitLCDecl(const Expr *E) {
  if (!E)
    return nullptr;
  E = getExprAsWritten(E);
  if (const auto *CE = dyn_cast_or_null<CXXConstructExpr>(E))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        E = CE->getArg(0)->IgnoreParenImpCasts();
  if (const auto *DRE = dyn_cast_or_null<DeclRefExpr>(E)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return getCanonicalDecl(VD);
  }
  if (const auto *ME = dyn_cast_or_null<MemberExpr>(E))
    if (ME->isArrow() && isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
      return getCanonicalDecl(ME->getMemberDecl());
  return nullptr;
}

bool OpenMPIterationSpaceChecker::checkAndSetCond(Expr *S) {
  // Check test-expr for canonical form, save upper-bound UB, flags for
  // less/greater and for strict/non-strict comparison.
  // OpenMP [2.9] Canonical loop form. Test-expr may be one of the following:
  //   var relational-op b
  //   b relational-op var
  //
  bool IneqCondIsCanonical = SemaRef.getLangOpts().OpenMP >= 50;
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_cond)
        << (IneqCondIsCanonical ? 1 : 0) << LCDecl;
    return true;
  }
  Condition = S;
  S = getExprAsWritten(S);
  SourceLocation CondLoc = S->getBeginLoc();
  auto &&CheckAndSetCond = [this, IneqCondIsCanonical](
                               BinaryOperatorKind Opcode, const Expr *LHS,
                               const Expr *RHS, SourceRange SR,
                               SourceLocation OpLoc) -> llvm::Optional<bool> {
    if (BinaryOperator::isRelationalOp(Opcode)) {
      if (getInitLCDecl(LHS) == LCDecl)
        return setUB(const_cast<Expr *>(RHS),
                     (Opcode == BO_LT || Opcode == BO_LE),
                     (Opcode == BO_LT || Opcode == BO_GT), SR, OpLoc);
      if (getInitLCDecl(RHS) == LCDecl)
        return setUB(const_cast<Expr *>(LHS),
                     (Opcode == BO_GT || Opcode == BO_GE),
                     (Opcode == BO_LT || Opcode == BO_GT), SR, OpLoc);
    } else if (IneqCondIsCanonical && Opcode == BO_NE) {
      return setUB(const_cast<Expr *>(getInitLCDecl(LHS) == LCDecl ? RHS : LHS),
                   /*LessOp=*/llvm::None,
                   /*StrictOp=*/true, SR, OpLoc);
    }
    return llvm::None;
  };
  llvm::Optional<bool> Res;
  if (auto *RBO = dyn_cast<CXXRewrittenBinaryOperator>(S)) {
    CXXRewrittenBinaryOperator::DecomposedForm DF = RBO->getDecomposedForm();
    Res = CheckAndSetCond(DF.Opcode, DF.LHS, DF.RHS, RBO->getSourceRange(),
                          RBO->getOperatorLoc());
  } else if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    Res = CheckAndSetCond(BO->getOpcode(), BO->getLHS(), BO->getRHS(),
                          BO->getSourceRange(), BO->getOperatorLoc());
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getNumArgs() == 2) {
      Res = CheckAndSetCond(
          BinaryOperator::getOverloadedOpcode(CE->getOperator()), CE->getArg(0),
          CE->getArg(1), CE->getSourceRange(), CE->getOperatorLoc());
    }
  }
  if (Res.hasValue())
    return *Res;
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(CondLoc, diag::err_omp_loop_not_canonical_cond)
      << (IneqCondIsCanonical ? 1 : 0) << S->getSourceRange() << LCDecl;
  return true;
}

bool OpenMPIterationSpaceChecker::checkAndSetIncRHS(Expr *RHS) {
  // RHS of canonical loop form increment can be:
  //   var + incr
  //   incr + var
  //   var - incr
  //
  RHS = RHS->IgnoreParenImpCasts();
  if (auto *BO = dyn_cast<BinaryOperator>(RHS)) {
    if (BO->isAdditiveOp()) {
      bool IsAdd = BO->getOpcode() == BO_Add;
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return setStep(BO->getRHS(), !IsAdd);
      if (IsAdd && getInitLCDecl(BO->getRHS()) == LCDecl)
        return setStep(BO->getLHS(), /*Subtract=*/false);
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(RHS)) {
    bool IsAdd = CE->getOperator() == OO_Plus;
    if ((IsAdd || CE->getOperator() == OO_Minus) && CE->getNumArgs() == 2) {
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(CE->getArg(1), !IsAdd);
      if (IsAdd && getInitLCDecl(CE->getArg(1)) == LCDecl)
        return setStep(CE->getArg(0), /*Subtract=*/false);
    }
  }
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(RHS->getBeginLoc(), diag::err_omp_loop_not_canonical_incr)
      << RHS->getSourceRange() << LCDecl;
  return true;
}

bool OpenMPIterationSpaceChecker::checkAndSetInc(Expr *S) {
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
    SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_incr) << LCDecl;
    return true;
  }
  if (auto *ExprTemp = dyn_cast<ExprWithCleanups>(S))
    if (!ExprTemp->cleanupsHaveSideEffects())
      S = ExprTemp->getSubExpr();

  IncrementSrcRange = S->getSourceRange();
  S = S->IgnoreParens();
  if (auto *UO = dyn_cast<UnaryOperator>(S)) {
    if (UO->isIncrementDecrementOp() &&
        getInitLCDecl(UO->getSubExpr()) == LCDecl)
      return setStep(SemaRef
                         .ActOnIntegerConstant(UO->getBeginLoc(),
                                               (UO->isDecrementOp() ? -1 : 1))
                         .get(),
                     /*Subtract=*/false);
  } else if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    switch (BO->getOpcode()) {
    case BO_AddAssign:
    case BO_SubAssign:
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return setStep(BO->getRHS(), BO->getOpcode() == BO_SubAssign);
      break;
    case BO_Assign:
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return checkAndSetIncRHS(BO->getRHS());
      break;
    default:
      break;
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    switch (CE->getOperator()) {
    case OO_PlusPlus:
    case OO_MinusMinus:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(SemaRef
                           .ActOnIntegerConstant(
                               CE->getBeginLoc(),
                               ((CE->getOperator() == OO_MinusMinus) ? -1 : 1))
                           .get(),
                       /*Subtract=*/false);
      break;
    case OO_PlusEqual:
    case OO_MinusEqual:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(CE->getArg(1), CE->getOperator() == OO_MinusEqual);
      break;
    case OO_Equal:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return checkAndSetIncRHS(CE->getArg(1));
      break;
    default:
      break;
    }
  }
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(S->getBeginLoc(), diag::err_omp_loop_not_canonical_incr)
      << S->getSourceRange() << LCDecl;
  return true;
}

static ExprResult
tryBuildCapture(Sema &SemaRef, Expr *Capture,
                llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) {
  if (SemaRef.CurContext->isDependentContext() || Capture->containsErrors())
    return Capture;
  if (Capture->isEvaluatable(SemaRef.Context, Expr::SE_AllowSideEffects))
    return SemaRef.PerformImplicitConversion(
        Capture->IgnoreImpCasts(), Capture->getType(), Sema::AA_Converting,
        /*AllowExplicit=*/true);
  auto I = Captures.find(Capture);
  if (I != Captures.end())
    return buildCapture(SemaRef, Capture, I->second);
  DeclRefExpr *Ref = nullptr;
  ExprResult Res = buildCapture(SemaRef, Capture, Ref);
  Captures[Capture] = Ref;
  return Res;
}

/// Calculate number of iterations, transforming to unsigned, if number of
/// iterations may be larger than the original type.
static Expr *
calculateNumIters(Sema &SemaRef, Scope *S, SourceLocation DefaultLoc,
                  Expr *Lower, Expr *Upper, Expr *Step, QualType LCTy,
                  bool TestIsStrictOp, bool RoundToStep,
                  llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) {
  ExprResult NewStep = tryBuildCapture(SemaRef, Step, Captures);
  if (!NewStep.isUsable())
    return nullptr;
  llvm::APSInt LRes, SRes;
  bool IsLowerConst = false, IsStepConst = false;
  if (Optional<llvm::APSInt> Res =
          Lower->getIntegerConstantExpr(SemaRef.Context)) {
    LRes = *Res;
    IsLowerConst = true;
  }
  if (Optional<llvm::APSInt> Res =
          Step->getIntegerConstantExpr(SemaRef.Context)) {
    SRes = *Res;
    IsStepConst = true;
  }
  bool NoNeedToConvert = IsLowerConst && !RoundToStep &&
                         ((!TestIsStrictOp && LRes.isNonNegative()) ||
                          (TestIsStrictOp && LRes.isStrictlyPositive()));
  bool NeedToReorganize = false;
  // Check if any subexpressions in Lower -Step [+ 1] lead to overflow.
  if (!NoNeedToConvert && IsLowerConst &&
      (TestIsStrictOp || (RoundToStep && IsStepConst))) {
    NoNeedToConvert = true;
    if (RoundToStep) {
      unsigned BW = LRes.getBitWidth() > SRes.getBitWidth()
                        ? LRes.getBitWidth()
                        : SRes.getBitWidth();
      LRes = LRes.extend(BW + 1);
      LRes.setIsSigned(true);
      SRes = SRes.extend(BW + 1);
      SRes.setIsSigned(true);
      LRes -= SRes;
      NoNeedToConvert = LRes.trunc(BW).extend(BW + 1) == LRes;
      LRes = LRes.trunc(BW);
    }
    if (TestIsStrictOp) {
      unsigned BW = LRes.getBitWidth();
      LRes = LRes.extend(BW + 1);
      LRes.setIsSigned(true);
      ++LRes;
      NoNeedToConvert =
          NoNeedToConvert && LRes.trunc(BW).extend(BW + 1) == LRes;
      // truncate to the original bitwidth.
      LRes = LRes.trunc(BW);
    }
    NeedToReorganize = NoNeedToConvert;
  }
  llvm::APSInt URes;
  bool IsUpperConst = false;
  if (Optional<llvm::APSInt> Res =
          Upper->getIntegerConstantExpr(SemaRef.Context)) {
    URes = *Res;
    IsUpperConst = true;
  }
  if (NoNeedToConvert && IsLowerConst && IsUpperConst &&
      (!RoundToStep || IsStepConst)) {
    unsigned BW = LRes.getBitWidth() > URes.getBitWidth() ? LRes.getBitWidth()
                                                          : URes.getBitWidth();
    LRes = LRes.extend(BW + 1);
    LRes.setIsSigned(true);
    URes = URes.extend(BW + 1);
    URes.setIsSigned(true);
    URes -= LRes;
    NoNeedToConvert = URes.trunc(BW).extend(BW + 1) == URes;
    NeedToReorganize = NoNeedToConvert;
  }
  // If the boundaries are not constant or (Lower - Step [+ 1]) is not constant
  // or less than zero (Upper - (Lower - Step [+ 1]) may overflow) - promote to
  // unsigned.
  if ((!NoNeedToConvert || (LRes.isNegative() && !IsUpperConst)) &&
      !LCTy->isDependentType() && LCTy->isIntegerType()) {
    QualType LowerTy = Lower->getType();
    QualType UpperTy = Upper->getType();
    uint64_t LowerSize = SemaRef.Context.getTypeSize(LowerTy);
    uint64_t UpperSize = SemaRef.Context.getTypeSize(UpperTy);
    if ((LowerSize <= UpperSize && UpperTy->hasSignedIntegerRepresentation()) ||
        (LowerSize > UpperSize && LowerTy->hasSignedIntegerRepresentation())) {
      QualType CastType = SemaRef.Context.getIntTypeForBitwidth(
          LowerSize > UpperSize ? LowerSize : UpperSize, /*Signed=*/0);
      Upper =
          SemaRef
              .PerformImplicitConversion(
                  SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Upper).get(),
                  CastType, Sema::AA_Converting)
              .get();
      Lower = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Lower).get();
      NewStep = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, NewStep.get());
    }
  }
  if (!Lower || !Upper || NewStep.isInvalid())
    return nullptr;

  ExprResult Diff;
  // If need to reorganize, then calculate the form as Upper - (Lower - Step [+
  // 1]).
  if (NeedToReorganize) {
    Diff = Lower;

    if (RoundToStep) {
      // Lower - Step
      Diff =
          SemaRef.BuildBinOp(S, DefaultLoc, BO_Sub, Diff.get(), NewStep.get());
      if (!Diff.isUsable())
        return nullptr;
    }

    // Lower - Step [+ 1]
    if (TestIsStrictOp)
      Diff = SemaRef.BuildBinOp(
          S, DefaultLoc, BO_Add, Diff.get(),
          SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!Diff.isUsable())
      return nullptr;

    Diff = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Diff.get());
    if (!Diff.isUsable())
      return nullptr;

    // Upper - (Lower - Step [+ 1]).
    Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Sub, Upper, Diff.get());
    if (!Diff.isUsable())
      return nullptr;
  } else {
    Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Sub, Upper, Lower);

    if (!Diff.isUsable() && LCTy->getAsCXXRecordDecl()) {
      // BuildBinOp already emitted error, this one is to point user to upper
      // and lower bound, and to tell what is passed to 'operator-'.
      SemaRef.Diag(Upper->getBeginLoc(), diag::err_omp_loop_diff_cxx)
          << Upper->getSourceRange() << Lower->getSourceRange();
      return nullptr;
    }

    if (!Diff.isUsable())
      return nullptr;

    // Upper - Lower [- 1]
    if (TestIsStrictOp)
      Diff = SemaRef.BuildBinOp(
          S, DefaultLoc, BO_Sub, Diff.get(),
          SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!Diff.isUsable())
      return nullptr;

    if (RoundToStep) {
      // Upper - Lower [- 1] + Step
      Diff =
          SemaRef.BuildBinOp(S, DefaultLoc, BO_Add, Diff.get(), NewStep.get());
      if (!Diff.isUsable())
        return nullptr;
    }
  }

  // Parentheses (for dumping/debugging purposes only).
  Diff = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Diff.get());
  if (!Diff.isUsable())
    return nullptr;

  // (Upper - Lower [- 1] + Step) / Step or (Upper - Lower) / Step
  Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Div, Diff.get(), NewStep.get());
  if (!Diff.isUsable())
    return nullptr;

  return Diff.get();
}

/// Build the expression to calculate the number of iterations.
Expr *OpenMPIterationSpaceChecker::buildNumIterations(
    Scope *S, ArrayRef<LoopIterationSpace> ResultIterSpaces, bool LimitedType,
    llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) const {
  QualType VarType = LCDecl->getType().getNonReferenceType();
  if (!VarType->isIntegerType() && !VarType->isPointerType() &&
      !SemaRef.getLangOpts().CPlusPlus)
    return nullptr;
  Expr *LBVal = LB;
  Expr *UBVal = UB;
  // LB = TestIsLessOp.getValue() ? min(LB(MinVal), LB(MaxVal)) :
  // max(LB(MinVal), LB(MaxVal))
  if (InitDependOnLC) {
    const LoopIterationSpace &IS = ResultIterSpaces[*InitDependOnLC - 1];
    if (!IS.MinValue || !IS.MaxValue)
      return nullptr;
    // OuterVar = Min
    ExprResult MinValue =
        SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, IS.MinValue);
    if (!MinValue.isUsable())
      return nullptr;

    ExprResult LBMinVal = SemaRef.BuildBinOp(S, DefaultLoc, BO_Assign,
                                             IS.CounterVar, MinValue.get());
    if (!LBMinVal.isUsable())
      return nullptr;
    // OuterVar = Min, LBVal
    LBMinVal =
        SemaRef.BuildBinOp(S, DefaultLoc, BO_Comma, LBMinVal.get(), LBVal);
    if (!LBMinVal.isUsable())
      return nullptr;
    // (OuterVar = Min, LBVal)
    LBMinVal = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, LBMinVal.get());
    if (!LBMinVal.isUsable())
      return nullptr;

    // OuterVar = Max
    ExprResult MaxValue =
        SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, IS.MaxValue);
    if (!MaxValue.isUsable())
      return nullptr;

    ExprResult LBMaxVal = SemaRef.BuildBinOp(S, DefaultLoc, BO_Assign,
                                             IS.CounterVar, MaxValue.get());
    if (!LBMaxVal.isUsable())
      return nullptr;
    // OuterVar = Max, LBVal
    LBMaxVal =
        SemaRef.BuildBinOp(S, DefaultLoc, BO_Comma, LBMaxVal.get(), LBVal);
    if (!LBMaxVal.isUsable())
      return nullptr;
    // (OuterVar = Max, LBVal)
    LBMaxVal = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, LBMaxVal.get());
    if (!LBMaxVal.isUsable())
      return nullptr;

    Expr *LBMin = tryBuildCapture(SemaRef, LBMinVal.get(), Captures).get();
    Expr *LBMax = tryBuildCapture(SemaRef, LBMaxVal.get(), Captures).get();
    if (!LBMin || !LBMax)
      return nullptr;
    // LB(MinVal) < LB(MaxVal)
    ExprResult MinLessMaxRes =
        SemaRef.BuildBinOp(S, DefaultLoc, BO_LT, LBMin, LBMax);
    if (!MinLessMaxRes.isUsable())
      return nullptr;
    Expr *MinLessMax =
        tryBuildCapture(SemaRef, MinLessMaxRes.get(), Captures).get();
    if (!MinLessMax)
      return nullptr;
    if (TestIsLessOp.getValue()) {
      // LB(MinVal) < LB(MaxVal) ? LB(MinVal) : LB(MaxVal) - min(LB(MinVal),
      // LB(MaxVal))
      ExprResult MinLB = SemaRef.ActOnConditionalOp(DefaultLoc, DefaultLoc,
                                                    MinLessMax, LBMin, LBMax);
      if (!MinLB.isUsable())
        return nullptr;
      LBVal = MinLB.get();
    } else {
      // LB(MinVal) < LB(MaxVal) ? LB(MaxVal) : LB(MinVal) - max(LB(MinVal),
      // LB(MaxVal))
      ExprResult MaxLB = SemaRef.ActOnConditionalOp(DefaultLoc, DefaultLoc,
                                                    MinLessMax, LBMax, LBMin);
      if (!MaxLB.isUsable())
        return nullptr;
      LBVal = MaxLB.get();
    }
  }
  // UB = TestIsLessOp.getValue() ? max(UB(MinVal), UB(MaxVal)) :
  // min(UB(MinVal), UB(MaxVal))
  if (CondDependOnLC) {
    const LoopIterationSpace &IS = ResultIterSpaces[*CondDependOnLC - 1];
    if (!IS.MinValue || !IS.MaxValue)
      return nullptr;
    // OuterVar = Min
    ExprResult MinValue =
        SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, IS.MinValue);
    if (!MinValue.isUsable())
      return nullptr;

    ExprResult UBMinVal = SemaRef.BuildBinOp(S, DefaultLoc, BO_Assign,
                                             IS.CounterVar, MinValue.get());
    if (!UBMinVal.isUsable())
      return nullptr;
    // OuterVar = Min, UBVal
    UBMinVal =
        SemaRef.BuildBinOp(S, DefaultLoc, BO_Comma, UBMinVal.get(), UBVal);
    if (!UBMinVal.isUsable())
      return nullptr;
    // (OuterVar = Min, UBVal)
    UBMinVal = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, UBMinVal.get());
    if (!UBMinVal.isUsable())
      return nullptr;

    // OuterVar = Max
    ExprResult MaxValue =
        SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, IS.MaxValue);
    if (!MaxValue.isUsable())
      return nullptr;

    ExprResult UBMaxVal = SemaRef.BuildBinOp(S, DefaultLoc, BO_Assign,
                                             IS.CounterVar, MaxValue.get());
    if (!UBMaxVal.isUsable())
      return nullptr;
    // OuterVar = Max, UBVal
    UBMaxVal =
        SemaRef.BuildBinOp(S, DefaultLoc, BO_Comma, UBMaxVal.get(), UBVal);
    if (!UBMaxVal.isUsable())
      return nullptr;
    // (OuterVar = Max, UBVal)
    UBMaxVal = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, UBMaxVal.get());
    if (!UBMaxVal.isUsable())
      return nullptr;

    Expr *UBMin = tryBuildCapture(SemaRef, UBMinVal.get(), Captures).get();
    Expr *UBMax = tryBuildCapture(SemaRef, UBMaxVal.get(), Captures).get();
    if (!UBMin || !UBMax)
      return nullptr;
    // UB(MinVal) > UB(MaxVal)
    ExprResult MinGreaterMaxRes =
        SemaRef.BuildBinOp(S, DefaultLoc, BO_GT, UBMin, UBMax);
    if (!MinGreaterMaxRes.isUsable())
      return nullptr;
    Expr *MinGreaterMax =
        tryBuildCapture(SemaRef, MinGreaterMaxRes.get(), Captures).get();
    if (!MinGreaterMax)
      return nullptr;
    if (TestIsLessOp.getValue()) {
      // UB(MinVal) > UB(MaxVal) ? UB(MinVal) : UB(MaxVal) - max(UB(MinVal),
      // UB(MaxVal))
      ExprResult MaxUB = SemaRef.ActOnConditionalOp(
          DefaultLoc, DefaultLoc, MinGreaterMax, UBMin, UBMax);
      if (!MaxUB.isUsable())
        return nullptr;
      UBVal = MaxUB.get();
    } else {
      // UB(MinVal) > UB(MaxVal) ? UB(MaxVal) : UB(MinVal) - min(UB(MinVal),
      // UB(MaxVal))
      ExprResult MinUB = SemaRef.ActOnConditionalOp(
          DefaultLoc, DefaultLoc, MinGreaterMax, UBMax, UBMin);
      if (!MinUB.isUsable())
        return nullptr;
      UBVal = MinUB.get();
    }
  }
  Expr *UBExpr = TestIsLessOp.getValue() ? UBVal : LBVal;
  Expr *LBExpr = TestIsLessOp.getValue() ? LBVal : UBVal;
  Expr *Upper = tryBuildCapture(SemaRef, UBExpr, Captures).get();
  Expr *Lower = tryBuildCapture(SemaRef, LBExpr, Captures).get();
  if (!Upper || !Lower)
    return nullptr;

  ExprResult Diff = calculateNumIters(SemaRef, S, DefaultLoc, Lower, Upper,
                                      Step, VarType, TestIsStrictOp,
                                      /*RoundToStep=*/true, Captures);
  if (!Diff.isUsable())
    return nullptr;

  // OpenMP runtime requires 32-bit or 64-bit loop variables.
  QualType Type = Diff.get()->getType();
  ASTContext &C = SemaRef.Context;
  bool UseVarType = VarType->hasIntegerRepresentation() &&
                    C.getTypeSize(Type) > C.getTypeSize(VarType);
  if (!Type->isIntegerType() || UseVarType) {
    unsigned NewSize =
        UseVarType ? C.getTypeSize(VarType) : C.getTypeSize(Type);
    bool IsSigned = UseVarType ? VarType->hasSignedIntegerRepresentation()
                               : Type->hasSignedIntegerRepresentation();
    Type = C.getIntTypeForBitwidth(NewSize, IsSigned);
    if (!SemaRef.Context.hasSameType(Diff.get()->getType(), Type)) {
      Diff = SemaRef.PerformImplicitConversion(
          Diff.get(), Type, Sema::AA_Converting, /*AllowExplicit=*/true);
      if (!Diff.isUsable())
        return nullptr;
    }
  }
  if (LimitedType) {
    unsigned NewSize = (C.getTypeSize(Type) > 32) ? 64 : 32;
    if (NewSize != C.getTypeSize(Type)) {
      if (NewSize < C.getTypeSize(Type)) {
        assert(NewSize == 64 && "incorrect loop var size");
        SemaRef.Diag(DefaultLoc, diag::warn_omp_loop_64_bit_var)
            << InitSrcRange << ConditionSrcRange;
      }
      QualType NewType = C.getIntTypeForBitwidth(
          NewSize, Type->hasSignedIntegerRepresentation() ||
                       C.getTypeSize(Type) < NewSize);
      if (!SemaRef.Context.hasSameType(Diff.get()->getType(), NewType)) {
        Diff = SemaRef.PerformImplicitConversion(Diff.get(), NewType,
                                                 Sema::AA_Converting, true);
        if (!Diff.isUsable())
          return nullptr;
      }
    }
  }

  return Diff.get();
}

std::pair<Expr *, Expr *> OpenMPIterationSpaceChecker::buildMinMaxValues(
    Scope *S, llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) const {
  // Do not build for iterators, they cannot be used in non-rectangular loop
  // nests.
  if (LCDecl->getType()->isRecordType())
    return std::make_pair(nullptr, nullptr);
  // If we subtract, the min is in the condition, otherwise the min is in the
  // init value.
  Expr *MinExpr = nullptr;
  Expr *MaxExpr = nullptr;
  Expr *LBExpr = TestIsLessOp.getValue() ? LB : UB;
  Expr *UBExpr = TestIsLessOp.getValue() ? UB : LB;
  bool LBNonRect = TestIsLessOp.getValue() ? InitDependOnLC.hasValue()
                                           : CondDependOnLC.hasValue();
  bool UBNonRect = TestIsLessOp.getValue() ? CondDependOnLC.hasValue()
                                           : InitDependOnLC.hasValue();
  Expr *Lower =
      LBNonRect ? LBExpr : tryBuildCapture(SemaRef, LBExpr, Captures).get();
  Expr *Upper =
      UBNonRect ? UBExpr : tryBuildCapture(SemaRef, UBExpr, Captures).get();
  if (!Upper || !Lower)
    return std::make_pair(nullptr, nullptr);

  if (TestIsLessOp.getValue())
    MinExpr = Lower;
  else
    MaxExpr = Upper;

  // Build minimum/maximum value based on number of iterations.
  QualType VarType = LCDecl->getType().getNonReferenceType();

  ExprResult Diff = calculateNumIters(SemaRef, S, DefaultLoc, Lower, Upper,
                                      Step, VarType, TestIsStrictOp,
                                      /*RoundToStep=*/false, Captures);
  if (!Diff.isUsable())
    return std::make_pair(nullptr, nullptr);

  // ((Upper - Lower [- 1]) / Step) * Step
  // Parentheses (for dumping/debugging purposes only).
  Diff = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Diff.get());
  if (!Diff.isUsable())
    return std::make_pair(nullptr, nullptr);

  ExprResult NewStep = tryBuildCapture(SemaRef, Step, Captures);
  if (!NewStep.isUsable())
    return std::make_pair(nullptr, nullptr);
  Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Mul, Diff.get(), NewStep.get());
  if (!Diff.isUsable())
    return std::make_pair(nullptr, nullptr);

  // Parentheses (for dumping/debugging purposes only).
  Diff = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Diff.get());
  if (!Diff.isUsable())
    return std::make_pair(nullptr, nullptr);

  // Convert to the ptrdiff_t, if original type is pointer.
  if (VarType->isAnyPointerType() &&
      !SemaRef.Context.hasSameType(
          Diff.get()->getType(),
          SemaRef.Context.getUnsignedPointerDiffType())) {
    Diff = SemaRef.PerformImplicitConversion(
        Diff.get(), SemaRef.Context.getUnsignedPointerDiffType(),
        Sema::AA_Converting, /*AllowExplicit=*/true);
  }
  if (!Diff.isUsable())
    return std::make_pair(nullptr, nullptr);

  if (TestIsLessOp.getValue()) {
    // MinExpr = Lower;
    // MaxExpr = Lower + (((Upper - Lower [- 1]) / Step) * Step)
    Diff = SemaRef.BuildBinOp(
        S, DefaultLoc, BO_Add,
        SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Lower).get(),
        Diff.get());
    if (!Diff.isUsable())
      return std::make_pair(nullptr, nullptr);
  } else {
    // MaxExpr = Upper;
    // MinExpr = Upper - (((Upper - Lower [- 1]) / Step) * Step)
    Diff = SemaRef.BuildBinOp(
        S, DefaultLoc, BO_Sub,
        SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Upper).get(),
        Diff.get());
    if (!Diff.isUsable())
      return std::make_pair(nullptr, nullptr);
  }

  // Convert to the original type.
  if (SemaRef.Context.hasSameType(Diff.get()->getType(), VarType))
    Diff = SemaRef.PerformImplicitConversion(Diff.get(), VarType,
                                             Sema::AA_Converting,
                                             /*AllowExplicit=*/true);
  if (!Diff.isUsable())
    return std::make_pair(nullptr, nullptr);

  Sema::TentativeAnalysisScope Trap(SemaRef);
  Diff = SemaRef.ActOnFinishFullExpr(Diff.get(), /*DiscardedValue=*/false);
  if (!Diff.isUsable())
    return std::make_pair(nullptr, nullptr);

  if (TestIsLessOp.getValue())
    MaxExpr = Diff.get();
  else
    MinExpr = Diff.get();

  return std::make_pair(MinExpr, MaxExpr);
}

Expr *OpenMPIterationSpaceChecker::buildFinalCondition(Scope *S) const {
  if (InitDependOnLC || CondDependOnLC)
    return Condition;
  return nullptr;
}

Expr *OpenMPIterationSpaceChecker::buildPreCond(
    Scope *S, Expr *Cond,
    llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) const {
  // Do not build a precondition when the condition/initialization is dependent
  // to prevent pessimistic early loop exit.
  // TODO: this can be improved by calculating min/max values but not sure that
  // it will be very effective.
  if (CondDependOnLC || InitDependOnLC)
    return SemaRef
        .PerformImplicitConversion(
            SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get(),
            SemaRef.Context.BoolTy, /*Action=*/Sema::AA_Casting,
            /*AllowExplicit=*/true)
        .get();

  // Try to build LB <op> UB, where <op> is <, >, <=, or >=.
  Sema::TentativeAnalysisScope Trap(SemaRef);

  ExprResult NewLB = tryBuildCapture(SemaRef, LB, Captures);
  ExprResult NewUB = tryBuildCapture(SemaRef, UB, Captures);
  if (!NewLB.isUsable() || !NewUB.isUsable())
    return nullptr;

  ExprResult CondExpr = SemaRef.BuildBinOp(
      S, DefaultLoc,
      TestIsLessOp.getValue() ? (TestIsStrictOp ? BO_LT : BO_LE)
                              : (TestIsStrictOp ? BO_GT : BO_GE),
      NewLB.get(), NewUB.get());
  if (CondExpr.isUsable()) {
    if (!SemaRef.Context.hasSameUnqualifiedType(CondExpr.get()->getType(),
                                                SemaRef.Context.BoolTy))
      CondExpr = SemaRef.PerformImplicitConversion(
          CondExpr.get(), SemaRef.Context.BoolTy, /*Action=*/Sema::AA_Casting,
          /*AllowExplicit=*/true);
  }

  // Otherwise use original loop condition and evaluate it in runtime.
  return CondExpr.isUsable() ? CondExpr.get() : Cond;
}

/// Build reference expression to the counter be used for codegen.
DeclRefExpr *OpenMPIterationSpaceChecker::buildCounterVar(
    llvm::MapVector<const Expr *, DeclRefExpr *> &Captures,
    DSAStackTy &DSA) const {
  auto *VD = dyn_cast<VarDecl>(LCDecl);
  if (!VD) {
    VD = SemaRef.isOpenMPCapturedDecl(LCDecl);
    DeclRefExpr *Ref = buildDeclRefExpr(
        SemaRef, VD, VD->getType().getNonReferenceType(), DefaultLoc);
    const DSAStackTy::DSAVarData Data =
        DSA.getTopDSA(LCDecl, /*FromParent=*/false);
    // If the loop control decl is explicitly marked as private, do not mark it
    // as captured again.
    if (!isOpenMPPrivate(Data.CKind) || !Data.RefExpr)
      Captures.insert(std::make_pair(LCRef, Ref));
    return Ref;
  }
  return cast<DeclRefExpr>(LCRef);
}

Expr *OpenMPIterationSpaceChecker::buildPrivateCounterVar() const {
  if (LCDecl && !LCDecl->isInvalidDecl()) {
    QualType Type = LCDecl->getType().getNonReferenceType();
    VarDecl *PrivateVar = buildVarDecl(
        SemaRef, DefaultLoc, Type, LCDecl->getName(),
        LCDecl->hasAttrs() ? &LCDecl->getAttrs() : nullptr,
        isa<VarDecl>(LCDecl)
            ? buildDeclRefExpr(SemaRef, cast<VarDecl>(LCDecl), Type, DefaultLoc)
            : nullptr);
    if (PrivateVar->isInvalidDecl())
      return nullptr;
    return buildDeclRefExpr(SemaRef, PrivateVar, Type, DefaultLoc);
  }
  return nullptr;
}

/// Build initialization of the counter to be used for codegen.
Expr *OpenMPIterationSpaceChecker::buildCounterInit() const { return LB; }

/// Build step of the counter be used for codegen.
Expr *OpenMPIterationSpaceChecker::buildCounterStep() const { return Step; }

Expr *OpenMPIterationSpaceChecker::buildOrderedLoopData(
    Scope *S, Expr *Counter,
    llvm::MapVector<const Expr *, DeclRefExpr *> &Captures, SourceLocation Loc,
    Expr *Inc, OverloadedOperatorKind OOK) {
  Expr *Cnt = SemaRef.DefaultLvalueConversion(Counter).get();
  if (!Cnt)
    return nullptr;
  if (Inc) {
    assert((OOK == OO_Plus || OOK == OO_Minus) &&
           "Expected only + or - operations for depend clauses.");
    BinaryOperatorKind BOK = (OOK == OO_Plus) ? BO_Add : BO_Sub;
    Cnt = SemaRef.BuildBinOp(S, Loc, BOK, Cnt, Inc).get();
    if (!Cnt)
      return nullptr;
  }
  QualType VarType = LCDecl->getType().getNonReferenceType();
  if (!VarType->isIntegerType() && !VarType->isPointerType() &&
      !SemaRef.getLangOpts().CPlusPlus)
    return nullptr;
  // Upper - Lower
  Expr *Upper = TestIsLessOp.getValue()
                    ? Cnt
                    : tryBuildCapture(SemaRef, LB, Captures).get();
  Expr *Lower = TestIsLessOp.getValue()
                    ? tryBuildCapture(SemaRef, LB, Captures).get()
                    : Cnt;
  if (!Upper || !Lower)
    return nullptr;

  ExprResult Diff = calculateNumIters(
      SemaRef, S, DefaultLoc, Lower, Upper, Step, VarType,
      /*TestIsStrictOp=*/false, /*RoundToStep=*/false, Captures);
  if (!Diff.isUsable())
    return nullptr;

  return Diff.get();
}
} // namespace

void Sema::ActOnOpenMPLoopInitialization(SourceLocation ForLoc, Stmt *Init) {
  assert(getLangOpts().OpenMP && "OpenMP is not active.");
  assert(Init && "Expected loop in canonical form.");
  unsigned AssociatedLoops = DSAStack->getAssociatedLoops();
  if (AssociatedLoops > 0 &&
      isOpenMPLoopDirective(DSAStack->getCurrentDirective())) {
    DSAStack->loopStart();
    OpenMPIterationSpaceChecker ISC(*this, /*SupportsNonRectangular=*/true,
                                    *DSAStack, ForLoc);
    if (!ISC.checkAndSetInit(Init, /*EmitDiags=*/false)) {
      if (ValueDecl *D = ISC.getLoopDecl()) {
        auto *VD = dyn_cast<VarDecl>(D);
        DeclRefExpr *PrivateRef = nullptr;
        if (!VD) {
          if (VarDecl *Private = isOpenMPCapturedDecl(D)) {
            VD = Private;
          } else {
            PrivateRef = buildCapture(*this, D, ISC.getLoopDeclRefExpr(),
                                      /*WithInit=*/false);
            VD = cast<VarDecl>(PrivateRef->getDecl());
          }
        }
        DSAStack->addLoopControlVariable(D, VD);
        const Decl *LD = DSAStack->getPossiblyLoopCunter();
        if (LD != D->getCanonicalDecl()) {
          DSAStack->resetPossibleLoopCounter();
          if (auto *Var = dyn_cast_or_null<VarDecl>(LD))
            MarkDeclarationsReferencedInExpr(
                buildDeclRefExpr(*this, const_cast<VarDecl *>(Var),
                                 Var->getType().getNonLValueExprType(Context),
                                 ForLoc, /*RefersToCapture=*/true));
        }
        OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
        // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables
        // Referenced in a Construct, C/C++]. The loop iteration variable in the
        // associated for-loop of a simd construct with just one associated
        // for-loop may be listed in a linear clause with a constant-linear-step
        // that is the increment of the associated for-loop. The loop iteration
        // variable(s) in the associated for-loop(s) of a for or parallel for
        // construct may be listed in a private or lastprivate clause.
        DSAStackTy::DSAVarData DVar =
            DSAStack->getTopDSA(D, /*FromParent=*/false);
        // If LoopVarRefExpr is nullptr it means the corresponding loop variable
        // is declared in the loop and it is predetermined as a private.
        Expr *LoopDeclRefExpr = ISC.getLoopDeclRefExpr();
        OpenMPClauseKind PredeterminedCKind =
            isOpenMPSimdDirective(DKind)
                ? (DSAStack->hasMutipleLoops() ? OMPC_lastprivate : OMPC_linear)
                : OMPC_private;
        if (((isOpenMPSimdDirective(DKind) && DVar.CKind != OMPC_unknown &&
              DVar.CKind != PredeterminedCKind && DVar.RefExpr &&
              (LangOpts.OpenMP <= 45 || (DVar.CKind != OMPC_lastprivate &&
                                         DVar.CKind != OMPC_private))) ||
             ((isOpenMPWorksharingDirective(DKind) || DKind == OMPD_taskloop ||
               DKind == OMPD_master_taskloop ||
               DKind == OMPD_parallel_master_taskloop ||
               isOpenMPDistributeDirective(DKind)) &&
              !isOpenMPSimdDirective(DKind) && DVar.CKind != OMPC_unknown &&
              DVar.CKind != OMPC_private && DVar.CKind != OMPC_lastprivate)) &&
            (DVar.CKind != OMPC_private || DVar.RefExpr)) {
          Diag(Init->getBeginLoc(), diag::err_omp_loop_var_dsa)
              << getOpenMPClauseName(DVar.CKind)
              << getOpenMPDirectiveName(DKind)
              << getOpenMPClauseName(PredeterminedCKind);
          if (DVar.RefExpr == nullptr)
            DVar.CKind = PredeterminedCKind;
          reportOriginalDsa(*this, DSAStack, D, DVar,
                            /*IsLoopIterVar=*/true);
        } else if (LoopDeclRefExpr) {
          // Make the loop iteration variable private (for worksharing
          // constructs), linear (for simd directives with the only one
          // associated loop) or lastprivate (for simd directives with several
          // collapsed or ordered loops).
          if (DVar.CKind == OMPC_unknown)
            DSAStack->addDSA(D, LoopDeclRefExpr, PredeterminedCKind,
                             PrivateRef);
        }
      }
    }
    DSAStack->setAssociatedLoops(AssociatedLoops - 1);
  }
}

/// Called on a for stmt to check and extract its iteration space
/// for further processing (such as collapsing).
static bool checkOpenMPIterationSpace(
    OpenMPDirectiveKind DKind, Stmt *S, Sema &SemaRef, DSAStackTy &DSA,
    unsigned CurrentNestedLoopCount, unsigned NestedLoopCount,
    unsigned TotalNestedLoopCount, Expr *CollapseLoopCountExpr,
    Expr *OrderedLoopCountExpr,
    Sema::VarsWithInheritedDSAType &VarsWithImplicitDSA,
    llvm::MutableArrayRef<LoopIterationSpace> ResultIterSpaces,
    llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) {
  bool SupportsNonRectangular = !isOpenMPLoopTransformationDirective(DKind);
  // OpenMP [2.9.1, Canonical Loop Form]
  //   for (init-expr; test-expr; incr-expr) structured-block
  //   for (range-decl: range-expr) structured-block
  if (auto *CanonLoop = dyn_cast_or_null<OMPCanonicalLoop>(S))
    S = CanonLoop->getLoopStmt();
  auto *For = dyn_cast_or_null<ForStmt>(S);
  auto *CXXFor = dyn_cast_or_null<CXXForRangeStmt>(S);
  // Ranged for is supported only in OpenMP 5.0.
  if (!For && (SemaRef.LangOpts.OpenMP <= 45 || !CXXFor)) {
    SemaRef.Diag(S->getBeginLoc(), diag::err_omp_not_for)
        << (CollapseLoopCountExpr != nullptr || OrderedLoopCountExpr != nullptr)
        << getOpenMPDirectiveName(DKind) << TotalNestedLoopCount
        << (CurrentNestedLoopCount > 0) << CurrentNestedLoopCount;
    if (TotalNestedLoopCount > 1) {
      if (CollapseLoopCountExpr && OrderedLoopCountExpr)
        SemaRef.Diag(DSA.getConstructLoc(),
                     diag::note_omp_collapse_ordered_expr)
            << 2 << CollapseLoopCountExpr->getSourceRange()
            << OrderedLoopCountExpr->getSourceRange();
      else if (CollapseLoopCountExpr)
        SemaRef.Diag(CollapseLoopCountExpr->getExprLoc(),
                     diag::note_omp_collapse_ordered_expr)
            << 0 << CollapseLoopCountExpr->getSourceRange();
      else
        SemaRef.Diag(OrderedLoopCountExpr->getExprLoc(),
                     diag::note_omp_collapse_ordered_expr)
            << 1 << OrderedLoopCountExpr->getSourceRange();
    }
    return true;
  }
  assert(((For && For->getBody()) || (CXXFor && CXXFor->getBody())) &&
         "No loop body.");
  // Postpone analysis in dependent contexts for ranged for loops.
  if (CXXFor && SemaRef.CurContext->isDependentContext())
    return false;

  OpenMPIterationSpaceChecker ISC(SemaRef, SupportsNonRectangular, DSA,
                                  For ? For->getForLoc() : CXXFor->getForLoc());

  // Check init.
  Stmt *Init = For ? For->getInit() : CXXFor->getBeginStmt();
  if (ISC.checkAndSetInit(Init))
    return true;

  bool HasErrors = false;

  // Check loop variable's type.
  if (ValueDecl *LCDecl = ISC.getLoopDecl()) {
    // OpenMP [2.6, Canonical Loop Form]
    // Var is one of the following:
    //   A variable of signed or unsigned integer type.
    //   For C++, a variable of a random access iterator type.
    //   For C, a variable of a pointer type.
    QualType VarType = LCDecl->getType().getNonReferenceType();
    if (!VarType->isDependentType() && !VarType->isIntegerType() &&
        !VarType->isPointerType() &&
        !(SemaRef.getLangOpts().CPlusPlus && VarType->isOverloadableType())) {
      SemaRef.Diag(Init->getBeginLoc(), diag::err_omp_loop_variable_type)
          << SemaRef.getLangOpts().CPlusPlus;
      HasErrors = true;
    }

    // OpenMP, 2.14.1.1 Data-sharing Attribute Rules for Variables Referenced in
    // a Construct
    // The loop iteration variable(s) in the associated for-loop(s) of a for or
    // parallel for construct is (are) private.
    // The loop iteration variable in the associated for-loop of a simd
    // construct with just one associated for-loop is linear with a
    // constant-linear-step that is the increment of the associated for-loop.
    // Exclude loop var from the list of variables with implicitly defined data
    // sharing attributes.
    VarsWithImplicitDSA.erase(LCDecl);

    assert(isOpenMPLoopDirective(DKind) && "DSA for non-loop vars");

    // Check test-expr.
    HasErrors |= ISC.checkAndSetCond(For ? For->getCond() : CXXFor->getCond());

    // Check incr-expr.
    HasErrors |= ISC.checkAndSetInc(For ? For->getInc() : CXXFor->getInc());
  }

  if (ISC.dependent() || SemaRef.CurContext->isDependentContext() || HasErrors)
    return HasErrors;

  // Build the loop's iteration space representation.
  ResultIterSpaces[CurrentNestedLoopCount].PreCond = ISC.buildPreCond(
      DSA.getCurScope(), For ? For->getCond() : CXXFor->getCond(), Captures);
  ResultIterSpaces[CurrentNestedLoopCount].NumIterations =
      ISC.buildNumIterations(DSA.getCurScope(), ResultIterSpaces,
                             (isOpenMPWorksharingDirective(DKind) ||
                              isOpenMPGenericLoopDirective(DKind) ||
                              isOpenMPTaskLoopDirective(DKind) ||
                              isOpenMPDistributeDirective(DKind) ||
                              isOpenMPLoopTransformationDirective(DKind)),
                             Captures);
  ResultIterSpaces[CurrentNestedLoopCount].CounterVar =
      ISC.buildCounterVar(Captures, DSA);
  ResultIterSpaces[CurrentNestedLoopCount].PrivateCounterVar =
      ISC.buildPrivateCounterVar();
  ResultIterSpaces[CurrentNestedLoopCount].CounterInit = ISC.buildCounterInit();
  ResultIterSpaces[CurrentNestedLoopCount].CounterStep = ISC.buildCounterStep();
  ResultIterSpaces[CurrentNestedLoopCount].InitSrcRange = ISC.getInitSrcRange();
  ResultIterSpaces[CurrentNestedLoopCount].CondSrcRange =
      ISC.getConditionSrcRange();
  ResultIterSpaces[CurrentNestedLoopCount].IncSrcRange =
      ISC.getIncrementSrcRange();
  ResultIterSpaces[CurrentNestedLoopCount].Subtract = ISC.shouldSubtractStep();
  ResultIterSpaces[CurrentNestedLoopCount].IsStrictCompare =
      ISC.isStrictTestOp();
  std::tie(ResultIterSpaces[CurrentNestedLoopCount].MinValue,
           ResultIterSpaces[CurrentNestedLoopCount].MaxValue) =
      ISC.buildMinMaxValues(DSA.getCurScope(), Captures);
  ResultIterSpaces[CurrentNestedLoopCount].FinalCondition =
      ISC.buildFinalCondition(DSA.getCurScope());
  ResultIterSpaces[CurrentNestedLoopCount].IsNonRectangularLB =
      ISC.doesInitDependOnLC();
  ResultIterSpaces[CurrentNestedLoopCount].IsNonRectangularUB =
      ISC.doesCondDependOnLC();
  ResultIterSpaces[CurrentNestedLoopCount].LoopDependentIdx =
      ISC.getLoopDependentIdx();

  HasErrors |=
      (ResultIterSpaces[CurrentNestedLoopCount].PreCond == nullptr ||
       ResultIterSpaces[CurrentNestedLoopCount].NumIterations == nullptr ||
       ResultIterSpaces[CurrentNestedLoopCount].CounterVar == nullptr ||
       ResultIterSpaces[CurrentNestedLoopCount].PrivateCounterVar == nullptr ||
       ResultIterSpaces[CurrentNestedLoopCount].CounterInit == nullptr ||
       ResultIterSpaces[CurrentNestedLoopCount].CounterStep == nullptr);
  if (!HasErrors && DSA.isOrderedRegion()) {
    if (DSA.getOrderedRegionParam().second->getNumForLoops()) {
      if (CurrentNestedLoopCount <
          DSA.getOrderedRegionParam().second->getLoopNumIterations().size()) {
        DSA.getOrderedRegionParam().second->setLoopNumIterations(
            CurrentNestedLoopCount,
            ResultIterSpaces[CurrentNestedLoopCount].NumIterations);
        DSA.getOrderedRegionParam().second->setLoopCounter(
            CurrentNestedLoopCount,
            ResultIterSpaces[CurrentNestedLoopCount].CounterVar);
      }
    }
    for (auto &Pair : DSA.getDoacrossDependClauses()) {
      if (CurrentNestedLoopCount >= Pair.first->getNumLoops()) {
        // Erroneous case - clause has some problems.
        continue;
      }
      if (Pair.first->getDependencyKind() == OMPC_DEPEND_sink &&
          Pair.second.size() <= CurrentNestedLoopCount) {
        // Erroneous case - clause has some problems.
        Pair.first->setLoopData(CurrentNestedLoopCount, nullptr);
        continue;
      }
      Expr *CntValue;
      if (Pair.first->getDependencyKind() == OMPC_DEPEND_source)
        CntValue = ISC.buildOrderedLoopData(
            DSA.getCurScope(),
            ResultIterSpaces[CurrentNestedLoopCount].CounterVar, Captures,
            Pair.first->getDependencyLoc());
      else
        CntValue = ISC.buildOrderedLoopData(
            DSA.getCurScope(),
            ResultIterSpaces[CurrentNestedLoopCount].CounterVar, Captures,
            Pair.first->getDependencyLoc(),
            Pair.second[CurrentNestedLoopCount].first,
            Pair.second[CurrentNestedLoopCount].second);
      Pair.first->setLoopData(CurrentNestedLoopCount, CntValue);
    }
  }

  return HasErrors;
}

/// Build 'VarRef = Start.
static ExprResult
buildCounterInit(Sema &SemaRef, Scope *S, SourceLocation Loc, ExprResult VarRef,
                 ExprResult Start, bool IsNonRectangularLB,
                 llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) {
  // Build 'VarRef = Start.
  ExprResult NewStart = IsNonRectangularLB
                            ? Start.get()
                            : tryBuildCapture(SemaRef, Start.get(), Captures);
  if (!NewStart.isUsable())
    return ExprError();
  if (!SemaRef.Context.hasSameType(NewStart.get()->getType(),
                                   VarRef.get()->getType())) {
    NewStart = SemaRef.PerformImplicitConversion(
        NewStart.get(), VarRef.get()->getType(), Sema::AA_Converting,
        /*AllowExplicit=*/true);
    if (!NewStart.isUsable())
      return ExprError();
  }

  ExprResult Init =
      SemaRef.BuildBinOp(S, Loc, BO_Assign, VarRef.get(), NewStart.get());
  return Init;
}

/// Build 'VarRef = Start + Iter * Step'.
static ExprResult buildCounterUpdate(
    Sema &SemaRef, Scope *S, SourceLocation Loc, ExprResult VarRef,
    ExprResult Start, ExprResult Iter, ExprResult Step, bool Subtract,
    bool IsNonRectangularLB,
    llvm::MapVector<const Expr *, DeclRefExpr *> *Captures = nullptr) {
  // Add parentheses (for debugging purposes only).
  Iter = SemaRef.ActOnParenExpr(Loc, Loc, Iter.get());
  if (!VarRef.isUsable() || !Start.isUsable() || !Iter.isUsable() ||
      !Step.isUsable())
    return ExprError();

  ExprResult NewStep = Step;
  if (Captures)
    NewStep = tryBuildCapture(SemaRef, Step.get(), *Captures);
  if (NewStep.isInvalid())
    return ExprError();
  ExprResult Update =
      SemaRef.BuildBinOp(S, Loc, BO_Mul, Iter.get(), NewStep.get());
  if (!Update.isUsable())
    return ExprError();

  // Try to build 'VarRef = Start, VarRef (+|-)= Iter * Step' or
  // 'VarRef = Start (+|-) Iter * Step'.
  if (!Start.isUsable())
    return ExprError();
  ExprResult NewStart = SemaRef.ActOnParenExpr(Loc, Loc, Start.get());
  if (!NewStart.isUsable())
    return ExprError();
  if (Captures && !IsNonRectangularLB)
    NewStart = tryBuildCapture(SemaRef, Start.get(), *Captures);
  if (NewStart.isInvalid())
    return ExprError();

  // First attempt: try to build 'VarRef = Start, VarRef += Iter * Step'.
  ExprResult SavedUpdate = Update;
  ExprResult UpdateVal;
  if (VarRef.get()->getType()->isOverloadableType() ||
      NewStart.get()->getType()->isOverloadableType() ||
      Update.get()->getType()->isOverloadableType()) {
    Sema::TentativeAnalysisScope Trap(SemaRef);

    Update =
        SemaRef.BuildBinOp(S, Loc, BO_Assign, VarRef.get(), NewStart.get());
    if (Update.isUsable()) {
      UpdateVal =
          SemaRef.BuildBinOp(S, Loc, Subtract ? BO_SubAssign : BO_AddAssign,
                             VarRef.get(), SavedUpdate.get());
      if (UpdateVal.isUsable()) {
        Update = SemaRef.CreateBuiltinBinOp(Loc, BO_Comma, Update.get(),
                                            UpdateVal.get());
      }
    }
  }

  // Second attempt: try to build 'VarRef = Start (+|-) Iter * Step'.
  if (!Update.isUsable() || !UpdateVal.isUsable()) {
    Update = SemaRef.BuildBinOp(S, Loc, Subtract ? BO_Sub : BO_Add,
                                NewStart.get(), SavedUpdate.get());
    if (!Update.isUsable())
      return ExprError();

    if (!SemaRef.Context.hasSameType(Update.get()->getType(),
                                     VarRef.get()->getType())) {
      Update = SemaRef.PerformImplicitConversion(
          Update.get(), VarRef.get()->getType(), Sema::AA_Converting, true);
      if (!Update.isUsable())
        return ExprError();
    }

    Update = SemaRef.BuildBinOp(S, Loc, BO_Assign, VarRef.get(), Update.get());
  }
  return Update;
}

/// Convert integer expression \a E to make it have at least \a Bits
/// bits.
static ExprResult widenIterationCount(unsigned Bits, Expr *E, Sema &SemaRef) {
  if (E == nullptr)
    return ExprError();
  ASTContext &C = SemaRef.Context;
  QualType OldType = E->getType();
  unsigned HasBits = C.getTypeSize(OldType);
  if (HasBits >= Bits)
    return ExprResult(E);
  // OK to convert to signed, because new type has more bits than old.
  QualType NewType = C.getIntTypeForBitwidth(Bits, /* Signed */ true);
  return SemaRef.PerformImplicitConversion(E, NewType, Sema::AA_Converting,
                                           true);
}

/// Check if the given expression \a E is a constant integer that fits
/// into \a Bits bits.
static bool fitsInto(unsigned Bits, bool Signed, const Expr *E, Sema &SemaRef) {
  if (E == nullptr)
    return false;
  if (Optional<llvm::APSInt> Result =
          E->getIntegerConstantExpr(SemaRef.Context))
    return Signed ? Result->isSignedIntN(Bits) : Result->isIntN(Bits);
  return false;
}

/// Build preinits statement for the given declarations.
static Stmt *buildPreInits(ASTContext &Context,
                           MutableArrayRef<Decl *> PreInits) {
  if (!PreInits.empty()) {
    return new (Context) DeclStmt(
        DeclGroupRef::Create(Context, PreInits.begin(), PreInits.size()),
        SourceLocation(), SourceLocation());
  }
  return nullptr;
}

/// Build preinits statement for the given declarations.
static Stmt *
buildPreInits(ASTContext &Context,
              const llvm::MapVector<const Expr *, DeclRefExpr *> &Captures) {
  if (!Captures.empty()) {
    SmallVector<Decl *, 16> PreInits;
    for (const auto &Pair : Captures)
      PreInits.push_back(Pair.second->getDecl());
    return buildPreInits(Context, PreInits);
  }
  return nullptr;
}

/// Build postupdate expression for the given list of postupdates expressions.
static Expr *buildPostUpdate(Sema &S, ArrayRef<Expr *> PostUpdates) {
  Expr *PostUpdate = nullptr;
  if (!PostUpdates.empty()) {
    for (Expr *E : PostUpdates) {
      Expr *ConvE = S.BuildCStyleCastExpr(
                         E->getExprLoc(),
                         S.Context.getTrivialTypeSourceInfo(S.Context.VoidTy),
                         E->getExprLoc(), E)
                        .get();
      PostUpdate = PostUpdate
                       ? S.CreateBuiltinBinOp(ConvE->getExprLoc(), BO_Comma,
                                              PostUpdate, ConvE)
                             .get()
                       : ConvE;
    }
  }
  return PostUpdate;
}

/// Called on a for stmt to check itself and nested loops (if any).
/// \return Returns 0 if one of the collapsed stmts is not canonical for loop,
/// number of collapsed loops otherwise.
static unsigned
checkOpenMPLoop(OpenMPDirectiveKind DKind, Expr *CollapseLoopCountExpr,
                Expr *OrderedLoopCountExpr, Stmt *AStmt, Sema &SemaRef,
                DSAStackTy &DSA,
                Sema::VarsWithInheritedDSAType &VarsWithImplicitDSA,
                OMPLoopBasedDirective::HelperExprs &Built) {
  unsigned NestedLoopCount = 1;
  bool SupportsNonPerfectlyNested = (SemaRef.LangOpts.OpenMP >= 50) &&
                                    !isOpenMPLoopTransformationDirective(DKind);

  if (CollapseLoopCountExpr) {
    // Found 'collapse' clause - calculate collapse number.
    Expr::EvalResult Result;
    if (!CollapseLoopCountExpr->isValueDependent() &&
        CollapseLoopCountExpr->EvaluateAsInt(Result, SemaRef.getASTContext())) {
      NestedLoopCount = Result.Val.getInt().getLimitedValue();
    } else {
      Built.clear(/*Size=*/1);
      return 1;
    }
  }
  unsigned OrderedLoopCount = 1;
  if (OrderedLoopCountExpr) {
    // Found 'ordered' clause - calculate collapse number.
    Expr::EvalResult EVResult;
    if (!OrderedLoopCountExpr->isValueDependent() &&
        OrderedLoopCountExpr->EvaluateAsInt(EVResult,
                                            SemaRef.getASTContext())) {
      llvm::APSInt Result = EVResult.Val.getInt();
      if (Result.getLimitedValue() < NestedLoopCount) {
        SemaRef.Diag(OrderedLoopCountExpr->getExprLoc(),
                     diag::err_omp_wrong_ordered_loop_count)
            << OrderedLoopCountExpr->getSourceRange();
        SemaRef.Diag(CollapseLoopCountExpr->getExprLoc(),
                     diag::note_collapse_loop_count)
            << CollapseLoopCountExpr->getSourceRange();
      }
      OrderedLoopCount = Result.getLimitedValue();
    } else {
      Built.clear(/*Size=*/1);
      return 1;
    }
  }
  // This is helper routine for loop directives (e.g., 'for', 'simd',
  // 'for simd', etc.).
  llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
  unsigned NumLoops = std::max(OrderedLoopCount, NestedLoopCount);
  SmallVector<LoopIterationSpace, 4> IterSpaces(NumLoops);
  if (!OMPLoopBasedDirective::doForAllLoops(
          AStmt->IgnoreContainers(!isOpenMPLoopTransformationDirective(DKind)),
          SupportsNonPerfectlyNested, NumLoops,
          [DKind, &SemaRef, &DSA, NumLoops, NestedLoopCount,
           CollapseLoopCountExpr, OrderedLoopCountExpr, &VarsWithImplicitDSA,
           &IterSpaces, &Captures](unsigned Cnt, Stmt *CurStmt) {
            if (checkOpenMPIterationSpace(
                    DKind, CurStmt, SemaRef, DSA, Cnt, NestedLoopCount,
                    NumLoops, CollapseLoopCountExpr, OrderedLoopCountExpr,
                    VarsWithImplicitDSA, IterSpaces, Captures))
              return true;
            if (Cnt > 0 && Cnt >= NestedLoopCount &&
                IterSpaces[Cnt].CounterVar) {
              // Handle initialization of captured loop iterator variables.
              auto *DRE = cast<DeclRefExpr>(IterSpaces[Cnt].CounterVar);
              if (isa<OMPCapturedExprDecl>(DRE->getDecl())) {
                Captures[DRE] = DRE;
              }
            }
            return false;
          },
          [&SemaRef, &Captures](OMPLoopTransformationDirective *Transform) {
            Stmt *DependentPreInits = Transform->getPreInits();
            if (!DependentPreInits)
              return;
            for (Decl *C : cast<DeclStmt>(DependentPreInits)->getDeclGroup()) {
              auto *D = cast<VarDecl>(C);
              DeclRefExpr *Ref = buildDeclRefExpr(SemaRef, D, D->getType(),
                                                  Transform->getBeginLoc());
              Captures[Ref] = Ref;
            }
          }))
    return 0;

  Built.clear(/* size */ NestedLoopCount);

  if (SemaRef.CurContext->isDependentContext())
    return NestedLoopCount;

  // An example of what is generated for the following code:
  //
  //   #pragma omp simd collapse(2) ordered(2)
  //   for (i = 0; i < NI; ++i)
  //     for (k = 0; k < NK; ++k)
  //       for (j = J0; j < NJ; j+=2) {
  //         <loop body>
  //       }
  //
  // We generate the code below.
  // Note: the loop body may be outlined in CodeGen.
  // Note: some counters may be C++ classes, operator- is used to find number of
  // iterations and operator+= to calculate counter value.
  // Note: decltype(NumIterations) must be integer type (in 'omp for', only i32
  // or i64 is currently supported).
  //
  //   #define NumIterations (NI * ((NJ - J0 - 1 + 2) / 2))
  //   for (int[32|64]_t IV = 0; IV < NumIterations; ++IV ) {
  //     .local.i = IV / ((NJ - J0 - 1 + 2) / 2);
  //     .local.j = J0 + (IV % ((NJ - J0 - 1 + 2) / 2)) * 2;
  //     // similar updates for vars in clauses (e.g. 'linear')
  //     <loop body (using local i and j)>
  //   }
  //   i = NI; // assign final values of counters
  //   j = NJ;
  //

  // Last iteration number is (I1 * I2 * ... In) - 1, where I1, I2 ... In are
  // the iteration counts of the collapsed for loops.
  // Precondition tests if there is at least one iteration (all conditions are
  // true).
  auto PreCond = ExprResult(IterSpaces[0].PreCond);
  Expr *N0 = IterSpaces[0].NumIterations;
  ExprResult LastIteration32 =
      widenIterationCount(/*Bits=*/32,
                          SemaRef
                              .PerformImplicitConversion(
                                  N0->IgnoreImpCasts(), N0->getType(),
                                  Sema::AA_Converting, /*AllowExplicit=*/true)
                              .get(),
                          SemaRef);
  ExprResult LastIteration64 = widenIterationCount(
      /*Bits=*/64,
      SemaRef
          .PerformImplicitConversion(N0->IgnoreImpCasts(), N0->getType(),
                                     Sema::AA_Converting,
                                     /*AllowExplicit=*/true)
          .get(),
      SemaRef);

  if (!LastIteration32.isUsable() || !LastIteration64.isUsable())
    return NestedLoopCount;

  ASTContext &C = SemaRef.Context;
  bool AllCountsNeedLessThan32Bits = C.getTypeSize(N0->getType()) < 32;

  Scope *CurScope = DSA.getCurScope();
  for (unsigned Cnt = 1; Cnt < NestedLoopCount; ++Cnt) {
    if (PreCond.isUsable()) {
      PreCond =
          SemaRef.BuildBinOp(CurScope, PreCond.get()->getExprLoc(), BO_LAnd,
                             PreCond.get(), IterSpaces[Cnt].PreCond);
    }
    Expr *N = IterSpaces[Cnt].NumIterations;
    SourceLocation Loc = N->getExprLoc();
    AllCountsNeedLessThan32Bits &= C.getTypeSize(N->getType()) < 32;
    if (LastIteration32.isUsable())
      LastIteration32 = SemaRef.BuildBinOp(
          CurScope, Loc, BO_Mul, LastIteration32.get(),
          SemaRef
              .PerformImplicitConversion(N->IgnoreImpCasts(), N->getType(),
                                         Sema::AA_Converting,
                                         /*AllowExplicit=*/true)
              .get());
    if (LastIteration64.isUsable())
      LastIteration64 = SemaRef.BuildBinOp(
          CurScope, Loc, BO_Mul, LastIteration64.get(),
          SemaRef
              .PerformImplicitConversion(N->IgnoreImpCasts(), N->getType(),
                                         Sema::AA_Converting,
                                         /*AllowExplicit=*/true)
              .get());
  }

  // Choose either the 32-bit or 64-bit version.
  ExprResult LastIteration = LastIteration64;
  if (SemaRef.getLangOpts().OpenMPOptimisticCollapse ||
      (LastIteration32.isUsable() &&
       C.getTypeSize(LastIteration32.get()->getType()) == 32 &&
       (AllCountsNeedLessThan32Bits || NestedLoopCount == 1 ||
        fitsInto(
            /*Bits=*/32,
            LastIteration32.get()->getType()->hasSignedIntegerRepresentation(),
            LastIteration64.get(), SemaRef))))
    LastIteration = LastIteration32;
  QualType VType = LastIteration.get()->getType();
  QualType RealVType = VType;
  QualType StrideVType = VType;
  if (isOpenMPTaskLoopDirective(DKind)) {
    VType =
        SemaRef.Context.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/0);
    StrideVType =
        SemaRef.Context.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1);
  }

  if (!LastIteration.isUsable())
    return 0;

  // Save the number of iterations.
  ExprResult NumIterations = LastIteration;
  {
    LastIteration = SemaRef.BuildBinOp(
        CurScope, LastIteration.get()->getExprLoc(), BO_Sub,
        LastIteration.get(),
        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!LastIteration.isUsable())
      return 0;
  }

  // Calculate the last iteration number beforehand instead of doing this on
  // each iteration. Do not do this if the number of iterations may be kfold-ed.
  bool IsConstant = LastIteration.get()->isIntegerConstantExpr(SemaRef.Context);
  ExprResult CalcLastIteration;
  if (!IsConstant) {
    ExprResult SaveRef =
        tryBuildCapture(SemaRef, LastIteration.get(), Captures);
    LastIteration = SaveRef;

    // Prepare SaveRef + 1.
    NumIterations = SemaRef.BuildBinOp(
        CurScope, SaveRef.get()->getExprLoc(), BO_Add, SaveRef.get(),
        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!NumIterations.isUsable())
      return 0;
  }

  SourceLocation InitLoc = IterSpaces[0].InitSrcRange.getBegin();

  // Build variables passed into runtime, necessary for worksharing directives.
  ExprResult LB, UB, IL, ST, EUB, CombLB, CombUB, PrevLB, PrevUB, CombEUB;
  if (isOpenMPWorksharingDirective(DKind) || isOpenMPTaskLoopDirective(DKind) ||
      isOpenMPDistributeDirective(DKind) ||
      isOpenMPGenericLoopDirective(DKind) ||
      isOpenMPLoopTransformationDirective(DKind)) {
    // Lower bound variable, initialized with zero.
    VarDecl *LBDecl = buildVarDecl(SemaRef, InitLoc, VType, ".omp.lb");
    LB = buildDeclRefExpr(SemaRef, LBDecl, VType, InitLoc);
    SemaRef.AddInitializerToDecl(LBDecl,
                                 SemaRef.ActOnIntegerConstant(InitLoc, 0).get(),
                                 /*DirectInit*/ false);

    // Upper bound variable, initialized with last iteration number.
    VarDecl *UBDecl = buildVarDecl(SemaRef, InitLoc, VType, ".omp.ub");
    UB = buildDeclRefExpr(SemaRef, UBDecl, VType, InitLoc);
    SemaRef.AddInitializerToDecl(UBDecl, LastIteration.get(),
                                 /*DirectInit*/ false);

    // A 32-bit variable-flag where runtime returns 1 for the last iteration.
    // This will be used to implement clause 'lastprivate'.
    QualType Int32Ty = SemaRef.Context.getIntTypeForBitwidth(32, true);
    VarDecl *ILDecl = buildVarDecl(SemaRef, InitLoc, Int32Ty, ".omp.is_last");
    IL = buildDeclRefExpr(SemaRef, ILDecl, Int32Ty, InitLoc);
    SemaRef.AddInitializerToDecl(ILDecl,
                                 SemaRef.ActOnIntegerConstant(InitLoc, 0).get(),
                                 /*DirectInit*/ false);

    // Stride variable returned by runtime (we initialize it to 1 by default).
    VarDecl *STDecl =
        buildVarDecl(SemaRef, InitLoc, StrideVType, ".omp.stride");
    ST = buildDeclRefExpr(SemaRef, STDecl, StrideVType, InitLoc);
    SemaRef.AddInitializerToDecl(STDecl,
                                 SemaRef.ActOnIntegerConstant(InitLoc, 1).get(),
                                 /*DirectInit*/ false);

    // Build expression: UB = min(UB, LastIteration)
    // It is necessary for CodeGen of directives with static scheduling.
    ExprResult IsUBGreater = SemaRef.BuildBinOp(CurScope, InitLoc, BO_GT,
                                                UB.get(), LastIteration.get());
    ExprResult CondOp = SemaRef.ActOnConditionalOp(
        LastIteration.get()->getExprLoc(), InitLoc, IsUBGreater.get(),
        LastIteration.get(), UB.get());
    EUB = SemaRef.BuildBinOp(CurScope, InitLoc, BO_Assign, UB.get(),
                             CondOp.get());
    EUB = SemaRef.ActOnFinishFullExpr(EUB.get(), /*DiscardedValue*/ false);

    // If we have a combined directive that combines 'distribute', 'for' or
    // 'simd' we need to be able to access the bounds of the schedule of the
    // enclosing region. E.g. in 'distribute parallel for' the bounds obtained
    // by scheduling 'distribute' have to be passed to the schedule of 'for'.
    if (isOpenMPLoopBoundSharingDirective(DKind)) {
      // Lower bound variable, initialized with zero.
      VarDecl *CombLBDecl =
          buildVarDecl(SemaRef, InitLoc, VType, ".omp.comb.lb");
      CombLB = buildDeclRefExpr(SemaRef, CombLBDecl, VType, InitLoc);
      SemaRef.AddInitializerToDecl(
          CombLBDecl, SemaRef.ActOnIntegerConstant(InitLoc, 0).get(),
          /*DirectInit*/ false);

      // Upper bound variable, initialized with last iteration number.
      VarDecl *CombUBDecl =
          buildVarDecl(SemaRef, InitLoc, VType, ".omp.comb.ub");
      CombUB = buildDeclRefExpr(SemaRef, CombUBDecl, VType, InitLoc);
      SemaRef.AddInitializerToDecl(CombUBDecl, LastIteration.get(),
                                   /*DirectInit*/ false);

      ExprResult CombIsUBGreater = SemaRef.BuildBinOp(
          CurScope, InitLoc, BO_GT, CombUB.get(), LastIteration.get());
      ExprResult CombCondOp =
          SemaRef.ActOnConditionalOp(InitLoc, InitLoc, CombIsUBGreater.get(),
                                     LastIteration.get(), CombUB.get());
      CombEUB = SemaRef.BuildBinOp(CurScope, InitLoc, BO_Assign, CombUB.get(),
                                   CombCondOp.get());
      CombEUB =
          SemaRef.ActOnFinishFullExpr(CombEUB.get(), /*DiscardedValue*/ false);

      const CapturedDecl *CD = cast<CapturedStmt>(AStmt)->getCapturedDecl();
      // We expect to have at least 2 more parameters than the 'parallel'
      // directive does - the lower and upper bounds of the previous schedule.
      assert(CD->getNumParams() >= 4 &&
             "Unexpected number of parameters in loop combined directive");

      // Set the proper type for the bounds given what we learned from the
      // enclosed loops.
      ImplicitParamDecl *PrevLBDecl = CD->getParam(/*PrevLB=*/2);
      ImplicitParamDecl *PrevUBDecl = CD->getParam(/*PrevUB=*/3);

      // Previous lower and upper bounds are obtained from the region
      // parameters.
      PrevLB =
          buildDeclRefExpr(SemaRef, PrevLBDecl, PrevLBDecl->getType(), InitLoc);
      PrevUB =
          buildDeclRefExpr(SemaRef, PrevUBDecl, PrevUBDecl->getType(), InitLoc);
    }
  }

  // Build the iteration variable and its initialization before loop.
  ExprResult IV;
  ExprResult Init, CombInit;
  {
    VarDecl *IVDecl = buildVarDecl(SemaRef, InitLoc, RealVType, ".omp.iv");
    IV = buildDeclRefExpr(SemaRef, IVDecl, RealVType, InitLoc);
    Expr *RHS = (isOpenMPWorksharingDirective(DKind) ||
                 isOpenMPGenericLoopDirective(DKind) ||
                 isOpenMPTaskLoopDirective(DKind) ||
                 isOpenMPDistributeDirective(DKind) ||
                 isOpenMPLoopTransformationDirective(DKind))
                    ? LB.get()
                    : SemaRef.ActOnIntegerConstant(SourceLocation(), 0).get();
    Init = SemaRef.BuildBinOp(CurScope, InitLoc, BO_Assign, IV.get(), RHS);
    Init = SemaRef.ActOnFinishFullExpr(Init.get(), /*DiscardedValue*/ false);

    if (isOpenMPLoopBoundSharingDirective(DKind)) {
      Expr *CombRHS =
          (isOpenMPWorksharingDirective(DKind) ||
           isOpenMPGenericLoopDirective(DKind) ||
           isOpenMPTaskLoopDirective(DKind) ||
           isOpenMPDistributeDirective(DKind))
              ? CombLB.get()
              : SemaRef.ActOnIntegerConstant(SourceLocation(), 0).get();
      CombInit =
          SemaRef.BuildBinOp(CurScope, InitLoc, BO_Assign, IV.get(), CombRHS);
      CombInit =
          SemaRef.ActOnFinishFullExpr(CombInit.get(), /*DiscardedValue*/ false);
    }
  }

  bool UseStrictCompare =
      RealVType->hasUnsignedIntegerRepresentation() &&
      llvm::all_of(IterSpaces, [](const LoopIterationSpace &LIS) {
        return LIS.IsStrictCompare;
      });
  // Loop condition (IV < NumIterations) or (IV <= UB or IV < UB + 1 (for
  // unsigned IV)) for worksharing loops.
  SourceLocation CondLoc = AStmt->getBeginLoc();
  Expr *BoundUB = UB.get();
  if (UseStrictCompare) {
    BoundUB =
        SemaRef
            .BuildBinOp(CurScope, CondLoc, BO_Add, BoundUB,
                        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get())
            .get();
    BoundUB =
        SemaRef.ActOnFinishFullExpr(BoundUB, /*DiscardedValue*/ false).get();
  }
  ExprResult Cond =
      (isOpenMPWorksharingDirective(DKind) ||
       isOpenMPGenericLoopDirective(DKind) ||
       isOpenMPTaskLoopDirective(DKind) || isOpenMPDistributeDirective(DKind) ||
       isOpenMPLoopTransformationDirective(DKind))
          ? SemaRef.BuildBinOp(CurScope, CondLoc,
                               UseStrictCompare ? BO_LT : BO_LE, IV.get(),
                               BoundUB)
          : SemaRef.BuildBinOp(CurScope, CondLoc, BO_LT, IV.get(),
                               NumIterations.get());
  ExprResult CombDistCond;
  if (isOpenMPLoopBoundSharingDirective(DKind)) {
    CombDistCond = SemaRef.BuildBinOp(CurScope, CondLoc, BO_LT, IV.get(),
                                      NumIterations.get());
  }

  ExprResult CombCond;
  if (isOpenMPLoopBoundSharingDirective(DKind)) {
    Expr *BoundCombUB = CombUB.get();
    if (UseStrictCompare) {
      BoundCombUB =
          SemaRef
              .BuildBinOp(
                  CurScope, CondLoc, BO_Add, BoundCombUB,
                  SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get())
              .get();
      BoundCombUB =
          SemaRef.ActOnFinishFullExpr(BoundCombUB, /*DiscardedValue*/ false)
              .get();
    }
    CombCond =
        SemaRef.BuildBinOp(CurScope, CondLoc, UseStrictCompare ? BO_LT : BO_LE,
                           IV.get(), BoundCombUB);
  }
  // Loop increment (IV = IV + 1)
  SourceLocation IncLoc = AStmt->getBeginLoc();
  ExprResult Inc =
      SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, IV.get(),
                         SemaRef.ActOnIntegerConstant(IncLoc, 1).get());
  if (!Inc.isUsable())
    return 0;
  Inc = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, IV.get(), Inc.get());
  Inc = SemaRef.ActOnFinishFullExpr(Inc.get(), /*DiscardedValue*/ false);
  if (!Inc.isUsable())
    return 0;

  // Increments for worksharing loops (LB = LB + ST; UB = UB + ST).
  // Used for directives with static scheduling.
  // In combined construct, add combined version that use CombLB and CombUB
  // base variables for the update
  ExprResult NextLB, NextUB, CombNextLB, CombNextUB;
  if (isOpenMPWorksharingDirective(DKind) || isOpenMPTaskLoopDirective(DKind) ||
      isOpenMPGenericLoopDirective(DKind) ||
      isOpenMPDistributeDirective(DKind) ||
      isOpenMPLoopTransformationDirective(DKind)) {
    // LB + ST
    NextLB = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, LB.get(), ST.get());
    if (!NextLB.isUsable())
      return 0;
    // LB = LB + ST
    NextLB =
        SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, LB.get(), NextLB.get());
    NextLB =
        SemaRef.ActOnFinishFullExpr(NextLB.get(), /*DiscardedValue*/ false);
    if (!NextLB.isUsable())
      return 0;
    // UB + ST
    NextUB = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, UB.get(), ST.get());
    if (!NextUB.isUsable())
      return 0;
    // UB = UB + ST
    NextUB =
        SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, UB.get(), NextUB.get());
    NextUB =
        SemaRef.ActOnFinishFullExpr(NextUB.get(), /*DiscardedValue*/ false);
    if (!NextUB.isUsable())
      return 0;
    if (isOpenMPLoopBoundSharingDirective(DKind)) {
      CombNextLB =
          SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, CombLB.get(), ST.get());
      if (!NextLB.isUsable())
        return 0;
      // LB = LB + ST
      CombNextLB = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, CombLB.get(),
                                      CombNextLB.get());
      CombNextLB = SemaRef.ActOnFinishFullExpr(CombNextLB.get(),
                                               /*DiscardedValue*/ false);
      if (!CombNextLB.isUsable())
        return 0;
      // UB + ST
      CombNextUB =
          SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, CombUB.get(), ST.get());
      if (!CombNextUB.isUsable())
        return 0;
      // UB = UB + ST
      CombNextUB = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, CombUB.get(),
                                      CombNextUB.get());
      CombNextUB = SemaRef.ActOnFinishFullExpr(CombNextUB.get(),
                                               /*DiscardedValue*/ false);
      if (!CombNextUB.isUsable())
        return 0;
    }
  }

  // Create increment expression for distribute loop when combined in a same
  // directive with for as IV = IV + ST; ensure upper bound expression based
  // on PrevUB instead of NumIterations - used to implement 'for' when found
  // in combination with 'distribute', like in 'distribute parallel for'
  SourceLocation DistIncLoc = AStmt->getBeginLoc();
  ExprResult DistCond, DistInc, PrevEUB, ParForInDistCond;
  if (isOpenMPLoopBoundSharingDirective(DKind)) {
    DistCond = SemaRef.BuildBinOp(
        CurScope, CondLoc, UseStrictCompare ? BO_LT : BO_LE, IV.get(), BoundUB);
    assert(DistCond.isUsable() && "distribute cond expr was not built");

    DistInc =
        SemaRef.BuildBinOp(CurScope, DistIncLoc, BO_Add, IV.get(), ST.get());
    assert(DistInc.isUsable() && "distribute inc expr was not built");
    DistInc = SemaRef.BuildBinOp(CurScope, DistIncLoc, BO_Assign, IV.get(),
                                 DistInc.get());
    DistInc =
        SemaRef.ActOnFinishFullExpr(DistInc.get(), /*DiscardedValue*/ false);
    assert(DistInc.isUsable() && "distribute inc expr was not built");

    // Build expression: UB = min(UB, prevUB) for #for in composite or combined
    // construct
    ExprResult NewPrevUB = PrevUB;
    SourceLocation DistEUBLoc = AStmt->getBeginLoc();
    if (!SemaRef.Context.hasSameType(UB.get()->getType(),
                                     PrevUB.get()->getType())) {
      NewPrevUB = SemaRef.BuildCStyleCastExpr(
          DistEUBLoc,
          SemaRef.Context.getTrivialTypeSourceInfo(UB.get()->getType()),
          DistEUBLoc, NewPrevUB.get());
      if (!NewPrevUB.isUsable())
        return 0;
    }
    ExprResult IsUBGreater = SemaRef.BuildBinOp(CurScope, DistEUBLoc, BO_GT,
                                                UB.get(), NewPrevUB.get());
    ExprResult CondOp = SemaRef.ActOnConditionalOp(
        DistEUBLoc, DistEUBLoc, IsUBGreater.get(), NewPrevUB.get(), UB.get());
    PrevEUB = SemaRef.BuildBinOp(CurScope, DistIncLoc, BO_Assign, UB.get(),
                                 CondOp.get());
    PrevEUB =
        SemaRef.ActOnFinishFullExpr(PrevEUB.get(), /*DiscardedValue*/ false);

    // Build IV <= PrevUB or IV < PrevUB + 1 for unsigned IV to be used in
    // parallel for is in combination with a distribute directive with
    // schedule(static, 1)
    Expr *BoundPrevUB = PrevUB.get();
    if (UseStrictCompare) {
      BoundPrevUB =
          SemaRef
              .BuildBinOp(
                  CurScope, CondLoc, BO_Add, BoundPrevUB,
                  SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get())
              .get();
      BoundPrevUB =
          SemaRef.ActOnFinishFullExpr(BoundPrevUB, /*DiscardedValue*/ false)
              .get();
    }
    ParForInDistCond =
        SemaRef.BuildBinOp(CurScope, CondLoc, UseStrictCompare ? BO_LT : BO_LE,
                           IV.get(), BoundPrevUB);
  }

  // Build updates and final values of the loop counters.
  bool HasErrors = false;
  Built.Counters.resize(NestedLoopCount);
  Built.Inits.resize(NestedLoopCount);
  Built.Updates.resize(NestedLoopCount);
  Built.Finals.resize(NestedLoopCount);
  Built.DependentCounters.resize(NestedLoopCount);
  Built.DependentInits.resize(NestedLoopCount);
  Built.FinalsConditions.resize(NestedLoopCount);
  {
    // We implement the following algorithm for obtaining the
    // original loop iteration variable values based on the
    // value of the collapsed loop iteration variable IV.
    //
    // Let n+1 be the number of collapsed loops in the nest.
    // Iteration variables (I0, I1, .... In)
    // Iteration counts (N0, N1, ... Nn)
    //
    // Acc = IV;
    //
    // To compute Ik for loop k, 0 <= k <= n, generate:
    //    Prod = N(k+1) * N(k+2) * ... * Nn;
    //    Ik = Acc / Prod;
    //    Acc -= Ik * Prod;
    //
    ExprResult Acc = IV;
    for (unsigned int Cnt = 0; Cnt < NestedLoopCount; ++Cnt) {
      LoopIterationSpace &IS = IterSpaces[Cnt];
      SourceLocation UpdLoc = IS.IncSrcRange.getBegin();
      ExprResult Iter;

      // Compute prod
      ExprResult Prod = SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get();
      for (unsigned int K = Cnt + 1; K < NestedLoopCount; ++K)
        Prod = SemaRef.BuildBinOp(CurScope, UpdLoc, BO_Mul, Prod.get(),
                                  IterSpaces[K].NumIterations);

      // Iter = Acc / Prod
      // If there is at least one more inner loop to avoid
      // multiplication by 1.
      if (Cnt + 1 < NestedLoopCount)
        Iter =
            SemaRef.BuildBinOp(CurScope, UpdLoc, BO_Div, Acc.get(), Prod.get());
      else
        Iter = Acc;
      if (!Iter.isUsable()) {
        HasErrors = true;
        break;
      }

      // Update Acc:
      // Acc -= Iter * Prod
      // Check if there is at least one more inner loop to avoid
      // multiplication by 1.
      if (Cnt + 1 < NestedLoopCount)
        Prod = SemaRef.BuildBinOp(CurScope, UpdLoc, BO_Mul, Iter.get(),
                                  Prod.get());
      else
        Prod = Iter;
      Acc = SemaRef.BuildBinOp(CurScope, UpdLoc, BO_Sub, Acc.get(), Prod.get());

      // Build update: IS.CounterVar(Private) = IS.Start + Iter * IS.Step
      auto *VD = cast<VarDecl>(cast<DeclRefExpr>(IS.CounterVar)->getDecl());
      DeclRefExpr *CounterVar = buildDeclRefExpr(
          SemaRef, VD, IS.CounterVar->getType(), IS.CounterVar->getExprLoc(),
          /*RefersToCapture=*/true);
      ExprResult Init =
          buildCounterInit(SemaRef, CurScope, UpdLoc, CounterVar,
                           IS.CounterInit, IS.IsNonRectangularLB, Captures);
      if (!Init.isUsable()) {
        HasErrors = true;
        break;
      }
      ExprResult Update = buildCounterUpdate(
          SemaRef, CurScope, UpdLoc, CounterVar, IS.CounterInit, Iter,
          IS.CounterStep, IS.Subtract, IS.IsNonRectangularLB, &Captures);
      if (!Update.isUsable()) {
        HasErrors = true;
        break;
      }

      // Build final: IS.CounterVar = IS.Start + IS.NumIters * IS.Step
      ExprResult Final =
          buildCounterUpdate(SemaRef, CurScope, UpdLoc, CounterVar,
                             IS.CounterInit, IS.NumIterations, IS.CounterStep,
                             IS.Subtract, IS.IsNonRectangularLB, &Captures);
      if (!Final.isUsable()) {
        HasErrors = true;
        break;
      }

      if (!Update.isUsable() || !Final.isUsable()) {
        HasErrors = true;
        break;
      }
      // Save results
      Built.Counters[Cnt] = IS.CounterVar;
      Built.PrivateCounters[Cnt] = IS.PrivateCounterVar;
      Built.Inits[Cnt] = Init.get();
      Built.Updates[Cnt] = Update.get();
      Built.Finals[Cnt] = Final.get();
      Built.DependentCounters[Cnt] = nullptr;
      Built.DependentInits[Cnt] = nullptr;
      Built.FinalsConditions[Cnt] = nullptr;
      if (IS.IsNonRectangularLB || IS.IsNonRectangularUB) {
        Built.DependentCounters[Cnt] =
            Built.Counters[NestedLoopCount - 1 - IS.LoopDependentIdx];
        Built.DependentInits[Cnt] =
            Built.Inits[NestedLoopCount - 1 - IS.LoopDependentIdx];
        Built.FinalsConditions[Cnt] = IS.FinalCondition;
      }
    }
  }

  if (HasErrors)
    return 0;

  // Save results
  Built.IterationVarRef = IV.get();
  Built.LastIteration = LastIteration.get();
  Built.NumIterations = NumIterations.get();
  Built.CalcLastIteration = SemaRef
                                .ActOnFinishFullExpr(CalcLastIteration.get(),
                                                     /*DiscardedValue=*/false)
                                .get();
  Built.PreCond = PreCond.get();
  Built.PreInits = buildPreInits(C, Captures);
  Built.Cond = Cond.get();
  Built.Init = Init.get();
  Built.Inc = Inc.get();
  Built.LB = LB.get();
  Built.UB = UB.get();
  Built.IL = IL.get();
  Built.ST = ST.get();
  Built.EUB = EUB.get();
  Built.NLB = NextLB.get();
  Built.NUB = NextUB.get();
  Built.PrevLB = PrevLB.get();
  Built.PrevUB = PrevUB.get();
  Built.DistInc = DistInc.get();
  Built.PrevEUB = PrevEUB.get();
  Built.DistCombinedFields.LB = CombLB.get();
  Built.DistCombinedFields.UB = CombUB.get();
  Built.DistCombinedFields.EUB = CombEUB.get();
  Built.DistCombinedFields.Init = CombInit.get();
  Built.DistCombinedFields.Cond = CombCond.get();
  Built.DistCombinedFields.NLB = CombNextLB.get();
  Built.DistCombinedFields.NUB = CombNextUB.get();
  Built.DistCombinedFields.DistCond = CombDistCond.get();
  Built.DistCombinedFields.ParForInDistCond = ParForInDistCond.get();

  return NestedLoopCount;
}

static Expr *getCollapseNumberExpr(ArrayRef<OMPClause *> Clauses) {
  auto CollapseClauses =
      OMPExecutableDirective::getClausesOfKind<OMPCollapseClause>(Clauses);
  if (CollapseClauses.begin() != CollapseClauses.end())
    return (*CollapseClauses.begin())->getNumForLoops();
  return nullptr;
}

static Expr *getOrderedNumberExpr(ArrayRef<OMPClause *> Clauses) {
  auto OrderedClauses =
      OMPExecutableDirective::getClausesOfKind<OMPOrderedClause>(Clauses);
  if (OrderedClauses.begin() != OrderedClauses.end())
    return (*OrderedClauses.begin())->getNumForLoops();
  return nullptr;
}

static bool checkSimdlenSafelenSpecified(Sema &S,
                                         const ArrayRef<OMPClause *> Clauses) {
  const OMPSafelenClause *Safelen = nullptr;
  const OMPSimdlenClause *Simdlen = nullptr;

  for (const OMPClause *Clause : Clauses) {
    if (Clause->getClauseKind() == OMPC_safelen)
      Safelen = cast<OMPSafelenClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_simdlen)
      Simdlen = cast<OMPSimdlenClause>(Clause);
    if (Safelen && Simdlen)
      break;
  }

  if (Simdlen && Safelen) {
    const Expr *SimdlenLength = Simdlen->getSimdlen();
    const Expr *SafelenLength = Safelen->getSafelen();
    if (SimdlenLength->isValueDependent() || SimdlenLength->isTypeDependent() ||
        SimdlenLength->isInstantiationDependent() ||
        SimdlenLength->containsUnexpandedParameterPack())
      return false;
    if (SafelenLength->isValueDependent() || SafelenLength->isTypeDependent() ||
        SafelenLength->isInstantiationDependent() ||
        SafelenLength->containsUnexpandedParameterPack())
      return false;
    Expr::EvalResult SimdlenResult, SafelenResult;
    SimdlenLength->EvaluateAsInt(SimdlenResult, S.Context);
    SafelenLength->EvaluateAsInt(SafelenResult, S.Context);
    llvm::APSInt SimdlenRes = SimdlenResult.Val.getInt();
    llvm::APSInt SafelenRes = SafelenResult.Val.getInt();
    // OpenMP 4.5 [2.8.1, simd Construct, Restrictions]
    // If both simdlen and safelen clauses are specified, the value of the
    // simdlen parameter must be less than or equal to the value of the safelen
    // parameter.
    if (SimdlenRes > SafelenRes) {
      S.Diag(SimdlenLength->getExprLoc(),
             diag::err_omp_wrong_simdlen_safelen_values)
          << SimdlenLength->getSourceRange() << SafelenLength->getSourceRange();
      return true;
    }
  }
  return false;
}

StmtResult
Sema::ActOnOpenMPSimdDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                               SourceLocation StartLoc, SourceLocation EndLoc,
                               VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_simd, getCollapseNumberExpr(Clauses), getOrderedNumberExpr(Clauses),
      AStmt, *this, *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPSimdDirective::Create(Context, StartLoc, EndLoc, NestedLoopCount,
                                  Clauses, AStmt, B);
}

StmtResult
Sema::ActOnOpenMPForDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                              SourceLocation StartLoc, SourceLocation EndLoc,
                              VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_for, getCollapseNumberExpr(Clauses), getOrderedNumberExpr(Clauses),
      AStmt, *this, *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  setFunctionHasBranchProtectedScope();
  return OMPForDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_for_simd, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPForSimdDirective::Create(Context, StartLoc, EndLoc, NestedLoopCount,
                                     Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto BaseStmt = AStmt;
  while (auto *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  if (auto *C = dyn_cast_or_null<CompoundStmt>(BaseStmt)) {
    auto S = C->children();
    if (S.begin() == S.end())
      return StmtError();
    // All associated statements must be '#pragma omp section' except for
    // the first one.
    for (Stmt *SectionStmt : llvm::drop_begin(S)) {
      if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
        if (SectionStmt)
          Diag(SectionStmt->getBeginLoc(),
               diag::err_omp_sections_substmt_not_section);
        return StmtError();
      }
      cast<OMPSectionDirective>(SectionStmt)
          ->setHasCancel(DSAStack->isCancelRegion());
    }
  } else {
    Diag(AStmt->getBeginLoc(), diag::err_omp_sections_not_compound_stmt);
    return StmtError();
  }

  setFunctionHasBranchProtectedScope();

  return OMPSectionsDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                      DSAStack->getTaskgroupReductionRef(),
                                      DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPSectionDirective(Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  setFunctionHasBranchProtectedScope();
  DSAStack->setParentCancelRegion(DSAStack->isCancelRegion());

  return OMPSectionDirective::Create(Context, StartLoc, EndLoc, AStmt,
                                     DSAStack->isCancelRegion());
}

static Expr *getDirectCallExpr(Expr *E) {
  E = E->IgnoreParenCasts()->IgnoreImplicit();
  if (auto *CE = dyn_cast<CallExpr>(E))
    if (CE->getDirectCallee())
      return E;
  return nullptr;
}

StmtResult Sema::ActOnOpenMPDispatchDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  Stmt *S = cast<CapturedStmt>(AStmt)->getCapturedStmt();

  // 5.1 OpenMP
  // expression-stmt : an expression statement with one of the following forms:
  //   expression = target-call ( [expression-list] );
  //   target-call ( [expression-list] );

  SourceLocation TargetCallLoc;

  if (!CurContext->isDependentContext()) {
    Expr *TargetCall = nullptr;

    auto *E = dyn_cast<Expr>(S);
    if (!E) {
      Diag(S->getBeginLoc(), diag::err_omp_dispatch_statement_call);
      return StmtError();
    }

    E = E->IgnoreParenCasts()->IgnoreImplicit();

    if (auto *BO = dyn_cast<BinaryOperator>(E)) {
      if (BO->getOpcode() == BO_Assign)
        TargetCall = getDirectCallExpr(BO->getRHS());
    } else {
      if (auto *COCE = dyn_cast<CXXOperatorCallExpr>(E))
        if (COCE->getOperator() == OO_Equal)
          TargetCall = getDirectCallExpr(COCE->getArg(1));
      if (!TargetCall)
        TargetCall = getDirectCallExpr(E);
    }
    if (!TargetCall) {
      Diag(E->getBeginLoc(), diag::err_omp_dispatch_statement_call);
      return StmtError();
    }
    TargetCallLoc = TargetCall->getExprLoc();
  }

  setFunctionHasBranchProtectedScope();

  return OMPDispatchDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                      TargetCallLoc);
}

StmtResult Sema::ActOnOpenMPGenericLoopDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  // OpenMP 5.1 [2.11.7, loop construct]
  // A list item may not appear in a lastprivate clause unless it is the
  // loop iteration variable of a loop that is associated with the construct.
  for (OMPClause *C : Clauses) {
    if (auto *LPC = dyn_cast<OMPLastprivateClause>(C)) {
      for (Expr *RefExpr : LPC->varlists()) {
        SourceLocation ELoc;
        SourceRange ERange;
        Expr *SimpleRefExpr = RefExpr;
        auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
        if (ValueDecl *D = Res.first) {
          auto &&Info = DSAStack->isLoopControlVariable(D);
          if (!Info.first) {
            Diag(ELoc, diag::err_omp_lastprivate_loop_var_non_loop_iteration);
            return StmtError();
          }
        }
      }
    }
  }

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse', it will define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_loop, getCollapseNumberExpr(Clauses), getOrderedNumberExpr(Clauses),
      AStmt, *this, *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp loop exprs were not built");

  setFunctionHasBranchProtectedScope();
  return OMPGenericLoopDirective::Create(Context, StartLoc, EndLoc,
                                         NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPSingleDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  setFunctionHasBranchProtectedScope();

  // OpenMP [2.7.3, single Construct, Restrictions]
  // The copyprivate clause must not be used with the nowait clause.
  const OMPClause *Nowait = nullptr;
  const OMPClause *Copyprivate = nullptr;
  for (const OMPClause *Clause : Clauses) {
    if (Clause->getClauseKind() == OMPC_nowait)
      Nowait = Clause;
    else if (Clause->getClauseKind() == OMPC_copyprivate)
      Copyprivate = Clause;
    if (Copyprivate && Nowait) {
      Diag(Copyprivate->getBeginLoc(),
           diag::err_omp_single_copyprivate_with_nowait);
      Diag(Nowait->getBeginLoc(), diag::note_omp_nowait_clause_here);
      return StmtError();
    }
  }

  return OMPSingleDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPMasterDirective(Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  setFunctionHasBranchProtectedScope();

  return OMPMasterDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

StmtResult Sema::ActOnOpenMPMaskedDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  setFunctionHasBranchProtectedScope();

  return OMPMaskedDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPCriticalDirective(
    const DeclarationNameInfo &DirName, ArrayRef<OMPClause *> Clauses,
    Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  bool ErrorFound = false;
  llvm::APSInt Hint;
  SourceLocation HintLoc;
  bool DependentHint = false;
  for (const OMPClause *C : Clauses) {
    if (C->getClauseKind() == OMPC_hint) {
      if (!DirName.getName()) {
        Diag(C->getBeginLoc(), diag::err_omp_hint_clause_no_name);
        ErrorFound = true;
      }
      Expr *E = cast<OMPHintClause>(C)->getHint();
      if (E->isTypeDependent() || E->isValueDependent() ||
          E->isInstantiationDependent()) {
        DependentHint = true;
      } else {
        Hint = E->EvaluateKnownConstInt(Context);
        HintLoc = C->getBeginLoc();
      }
    }
  }
  if (ErrorFound)
    return StmtError();
  const auto Pair = DSAStack->getCriticalWithHint(DirName);
  if (Pair.first && DirName.getName() && !DependentHint) {
    if (llvm::APSInt::compareValues(Hint, Pair.second) != 0) {
      Diag(StartLoc, diag::err_omp_critical_with_hint);
      if (HintLoc.isValid())
        Diag(HintLoc, diag::note_omp_critical_hint_here)
            << 0 << toString(Hint, /*Radix=*/10, /*Signed=*/false);
      else
        Diag(StartLoc, diag::note_omp_critical_no_hint) << 0;
      if (const auto *C = Pair.first->getSingleClause<OMPHintClause>()) {
        Diag(C->getBeginLoc(), diag::note_omp_critical_hint_here)
            << 1
            << toString(C->getHint()->EvaluateKnownConstInt(Context),
                        /*Radix=*/10, /*Signed=*/false);
      } else {
        Diag(Pair.first->getBeginLoc(), diag::note_omp_critical_no_hint) << 1;
      }
    }
  }

  setFunctionHasBranchProtectedScope();

  auto *Dir = OMPCriticalDirective::Create(Context, DirName, StartLoc, EndLoc,
                                           Clauses, AStmt);
  if (!Pair.first && DirName.getName() && !DependentHint)
    DSAStack->addCriticalWithHint(Dir, Hint);
  return Dir;
}

StmtResult Sema::ActOnOpenMPParallelForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_parallel_for, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp parallel for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  setFunctionHasBranchProtectedScope();
  return OMPParallelForDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPParallelForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_parallel_for_simd, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult
Sema::ActOnOpenMPParallelMasterDirective(ArrayRef<OMPClause *> Clauses,
                                         Stmt *AStmt, SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  setFunctionHasBranchProtectedScope();

  return OMPParallelMasterDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt,
      DSAStack->getTaskgroupReductionRef());
}

StmtResult
Sema::ActOnOpenMPParallelSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                           Stmt *AStmt, SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto BaseStmt = AStmt;
  while (auto *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  if (auto *C = dyn_cast_or_null<CompoundStmt>(BaseStmt)) {
    auto S = C->children();
    if (S.begin() == S.end())
      return StmtError();
    // All associated statements must be '#pragma omp section' except for
    // the first one.
    for (Stmt *SectionStmt : llvm::drop_begin(S)) {
      if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
        if (SectionStmt)
          Diag(SectionStmt->getBeginLoc(),
               diag::err_omp_parallel_sections_substmt_not_section);
        return StmtError();
      }
      cast<OMPSectionDirective>(SectionStmt)
          ->setHasCancel(DSAStack->isCancelRegion());
    }
  } else {
    Diag(AStmt->getBeginLoc(),
         diag::err_omp_parallel_sections_not_compound_stmt);
    return StmtError();
  }

  setFunctionHasBranchProtectedScope();

  return OMPParallelSectionsDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

/// Find and diagnose mutually exclusive clause kinds.
static bool checkMutuallyExclusiveClauses(
    Sema &S, ArrayRef<OMPClause *> Clauses,
    ArrayRef<OpenMPClauseKind> MutuallyExclusiveClauses) {
  const OMPClause *PrevClause = nullptr;
  bool ErrorFound = false;
  for (const OMPClause *C : Clauses) {
    if (llvm::is_contained(MutuallyExclusiveClauses, C->getClauseKind())) {
      if (!PrevClause) {
        PrevClause = C;
      } else if (PrevClause->getClauseKind() != C->getClauseKind()) {
        S.Diag(C->getBeginLoc(), diag::err_omp_clauses_mutually_exclusive)
            << getOpenMPClauseName(C->getClauseKind())
            << getOpenMPClauseName(PrevClause->getClauseKind());
        S.Diag(PrevClause->getBeginLoc(), diag::note_omp_previous_clause)
            << getOpenMPClauseName(PrevClause->getClauseKind());
        ErrorFound = true;
      }
    }
  }
  return ErrorFound;
}

StmtResult Sema::ActOnOpenMPTaskDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  // OpenMP 5.0, 2.10.1 task Construct
  // If a detach clause appears on the directive, then a mergeable clause cannot
  // appear on the same directive.
  if (checkMutuallyExclusiveClauses(*this, Clauses,
                                    {OMPC_detach, OMPC_mergeable}))
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  setFunctionHasBranchProtectedScope();

  return OMPTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                  DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPTaskyieldDirective(SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  return OMPTaskyieldDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPBarrierDirective(SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  return OMPBarrierDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPTaskwaitDirective(ArrayRef<OMPClause *> Clauses,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  return OMPTaskwaitDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOpenMPTaskgroupDirective(ArrayRef<OMPClause *> Clauses,
                                               Stmt *AStmt,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  setFunctionHasBranchProtectedScope();

  return OMPTaskgroupDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                       AStmt,
                                       DSAStack->getTaskgroupReductionRef());
}

StmtResult Sema::ActOnOpenMPFlushDirective(ArrayRef<OMPClause *> Clauses,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  OMPFlushClause *FC = nullptr;
  OMPClause *OrderClause = nullptr;
  for (OMPClause *C : Clauses) {
    if (C->getClauseKind() == OMPC_flush)
      FC = cast<OMPFlushClause>(C);
    else
      OrderClause = C;
  }
  OpenMPClauseKind MemOrderKind = OMPC_unknown;
  SourceLocation MemOrderLoc;
  for (const OMPClause *C : Clauses) {
    if (C->getClauseKind() == OMPC_acq_rel ||
        C->getClauseKind() == OMPC_acquire ||
        C->getClauseKind() == OMPC_release) {
      if (MemOrderKind != OMPC_unknown) {
        Diag(C->getBeginLoc(), diag::err_omp_several_mem_order_clauses)
            << getOpenMPDirectiveName(OMPD_flush) << 1
            << SourceRange(C->getBeginLoc(), C->getEndLoc());
        Diag(MemOrderLoc, diag::note_omp_previous_mem_order_clause)
            << getOpenMPClauseName(MemOrderKind);
      } else {
        MemOrderKind = C->getClauseKind();
        MemOrderLoc = C->getBeginLoc();
      }
    }
  }
  if (FC && OrderClause) {
    Diag(FC->getLParenLoc(), diag::err_omp_flush_order_clause_and_list)
        << getOpenMPClauseName(OrderClause->getClauseKind());
    Diag(OrderClause->getBeginLoc(), diag::note_omp_flush_order_clause_here)
        << getOpenMPClauseName(OrderClause->getClauseKind());
    return StmtError();
  }
  return OMPFlushDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOpenMPDepobjDirective(ArrayRef<OMPClause *> Clauses,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (Clauses.empty()) {
    Diag(StartLoc, diag::err_omp_depobj_expected);
    return StmtError();
  } else if (Clauses[0]->getClauseKind() != OMPC_depobj) {
    Diag(Clauses[0]->getBeginLoc(), diag::err_omp_depobj_expected);
    return StmtError();
  }
  // Only depobj expression and another single clause is allowed.
  if (Clauses.size() > 2) {
    Diag(Clauses[2]->getBeginLoc(),
         diag::err_omp_depobj_single_clause_expected);
    return StmtError();
  } else if (Clauses.size() < 1) {
    Diag(Clauses[0]->getEndLoc(), diag::err_omp_depobj_single_clause_expected);
    return StmtError();
  }
  return OMPDepobjDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOpenMPScanDirective(ArrayRef<OMPClause *> Clauses,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  // Check that exactly one clause is specified.
  if (Clauses.size() != 1) {
    Diag(Clauses.empty() ? EndLoc : Clauses[1]->getBeginLoc(),
         diag::err_omp_scan_single_clause_expected);
    return StmtError();
  }
  // Check that scan directive is used in the scopeof the OpenMP loop body.
  if (Scope *S = DSAStack->getCurScope()) {
    Scope *ParentS = S->getParent();
    if (!ParentS || ParentS->getParent() != ParentS->getBreakParent() ||
        !ParentS->getBreakParent()->isOpenMPLoopScope())
      return StmtError(Diag(StartLoc, diag::err_omp_orphaned_device_directive)
                       << getOpenMPDirectiveName(OMPD_scan) << 5);
  }
  // Check that only one instance of scan directives is used in the same outer
  // region.
  if (DSAStack->doesParentHasScanDirective()) {
    Diag(StartLoc, diag::err_omp_several_directives_in_region) << "scan";
    Diag(DSAStack->getParentScanDirectiveLoc(),
         diag::note_omp_previous_directive)
        << "scan";
    return StmtError();
  }
  DSAStack->setParentHasScanDirective(StartLoc);
  return OMPScanDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOpenMPOrderedDirective(ArrayRef<OMPClause *> Clauses,
                                             Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  const OMPClause *DependFound = nullptr;
  const OMPClause *DependSourceClause = nullptr;
  const OMPClause *DependSinkClause = nullptr;
  bool ErrorFound = false;
  const OMPThreadsClause *TC = nullptr;
  const OMPSIMDClause *SC = nullptr;
  for (const OMPClause *C : Clauses) {
    if (auto *DC = dyn_cast<OMPDependClause>(C)) {
      DependFound = C;
      if (DC->getDependencyKind() == OMPC_DEPEND_source) {
        if (DependSourceClause) {
          Diag(C->getBeginLoc(), diag::err_omp_more_one_clause)
              << getOpenMPDirectiveName(OMPD_ordered)
              << getOpenMPClauseName(OMPC_depend) << 2;
          ErrorFound = true;
        } else {
          DependSourceClause = C;
        }
        if (DependSinkClause) {
          Diag(C->getBeginLoc(), diag::err_omp_depend_sink_source_not_allowed)
              << 0;
          ErrorFound = true;
        }
      } else if (DC->getDependencyKind() == OMPC_DEPEND_sink) {
        if (DependSourceClause) {
          Diag(C->getBeginLoc(), diag::err_omp_depend_sink_source_not_allowed)
              << 1;
          ErrorFound = true;
        }
        DependSinkClause = C;
      }
    } else if (C->getClauseKind() == OMPC_threads) {
      TC = cast<OMPThreadsClause>(C);
    } else if (C->getClauseKind() == OMPC_simd) {
      SC = cast<OMPSIMDClause>(C);
    }
  }
  if (!ErrorFound && !SC &&
      isOpenMPSimdDirective(DSAStack->getParentDirective())) {
    // OpenMP [2.8.1,simd Construct, Restrictions]
    // An ordered construct with the simd clause is the only OpenMP construct
    // that can appear in the simd region.
    Diag(StartLoc, diag::err_omp_prohibited_region_simd)
        << (LangOpts.OpenMP >= 50 ? 1 : 0);
    ErrorFound = true;
  } else if (DependFound && (TC || SC)) {
    Diag(DependFound->getBeginLoc(), diag::err_omp_depend_clause_thread_simd)
        << getOpenMPClauseName(TC ? TC->getClauseKind() : SC->getClauseKind());
    ErrorFound = true;
  } else if (DependFound && !DSAStack->getParentOrderedRegionParam().first) {
    Diag(DependFound->getBeginLoc(),
         diag::err_omp_ordered_directive_without_param);
    ErrorFound = true;
  } else if (TC || Clauses.empty()) {
    if (const Expr *Param = DSAStack->getParentOrderedRegionParam().first) {
      SourceLocation ErrLoc = TC ? TC->getBeginLoc() : StartLoc;
      Diag(ErrLoc, diag::err_omp_ordered_directive_with_param)
          << (TC != nullptr);
      Diag(Param->getBeginLoc(), diag::note_omp_ordered_param) << 1;
      ErrorFound = true;
    }
  }
  if ((!AStmt && !DependFound) || ErrorFound)
    return StmtError();

  // OpenMP 5.0, 2.17.9, ordered Construct, Restrictions.
  // During execution of an iteration of a worksharing-loop or a loop nest
  // within a worksharing-loop, simd, or worksharing-loop SIMD region, a thread
  // must not execute more than one ordered region corresponding to an ordered
  // construct without a depend clause.
  if (!DependFound) {
    if (DSAStack->doesParentHasOrderedDirective()) {
      Diag(StartLoc, diag::err_omp_several_directives_in_region) << "ordered";
      Diag(DSAStack->getParentOrderedDirectiveLoc(),
           diag::note_omp_previous_directive)
          << "ordered";
      return StmtError();
    }
    DSAStack->setParentHasOrderedDirective(StartLoc);
  }

  if (AStmt) {
    assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

    setFunctionHasBranchProtectedScope();
  }

  return OMPOrderedDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

namespace {
/// Helper class for checking expression in 'omp atomic [update]'
/// construct.
class OpenMPAtomicUpdateChecker {
  /// Error results for atomic update expressions.
  enum ExprAnalysisErrorCode {
    /// A statement is not an expression statement.
    NotAnExpression,
    /// Expression is not builtin binary or unary operation.
    NotABinaryOrUnaryExpression,
    /// Unary operation is not post-/pre- increment/decrement operation.
    NotAnUnaryIncDecExpression,
    /// An expression is not of scalar type.
    NotAScalarType,
    /// A binary operation is not an assignment operation.
    NotAnAssignmentOp,
    /// RHS part of the binary operation is not a binary expression.
    NotABinaryExpression,
    /// RHS part is not additive/multiplicative/shift/biwise binary
    /// expression.
    NotABinaryOperator,
    /// RHS binary operation does not have reference to the updated LHS
    /// part.
    NotAnUpdateExpression,
    /// No errors is found.
    NoError
  };
  /// Reference to Sema.
  Sema &SemaRef;
  /// A location for note diagnostics (when error is found).
  SourceLocation NoteLoc;
  /// 'x' lvalue part of the source atomic expression.
  Expr *X;
  /// 'expr' rvalue part of the source atomic expression.
  Expr *E;
  /// Helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *UpdateExpr;
  /// Is 'x' a LHS in a RHS part of full update expression. It is
  /// important for non-associative operations.
  bool IsXLHSInRHSPart;
  BinaryOperatorKind Op;
  SourceLocation OpLoc;
  /// true if the source expression is a postfix unary operation, false
  /// if it is a prefix unary operation.
  bool IsPostfixUpdate;

public:
  OpenMPAtomicUpdateChecker(Sema &SemaRef)
      : SemaRef(SemaRef), X(nullptr), E(nullptr), UpdateExpr(nullptr),
        IsXLHSInRHSPart(false), Op(BO_PtrMemD), IsPostfixUpdate(false) {}
  /// Check specified statement that it is suitable for 'atomic update'
  /// constructs and extract 'x', 'expr' and Operation from the original
  /// expression. If DiagId and NoteId == 0, then only check is performed
  /// without error notification.
  /// \param DiagId Diagnostic which should be emitted if error is found.
  /// \param NoteId Diagnostic note for the main error message.
  /// \return true if statement is not an update expression, false otherwise.
  bool checkStatement(Stmt *S, unsigned DiagId = 0, unsigned NoteId = 0);
  /// Return the 'x' lvalue part of the source atomic expression.
  Expr *getX() const { return X; }
  /// Return the 'expr' rvalue part of the source atomic expression.
  Expr *getExpr() const { return E; }
  /// Return the update expression used in calculation of the updated
  /// value. Always has form 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *getUpdateExpr() const { return UpdateExpr; }
  /// Return true if 'x' is LHS in RHS part of full update expression,
  /// false otherwise.
  bool isXLHSInRHSPart() const { return IsXLHSInRHSPart; }

  /// true if the source expression is a postfix unary operation, false
  /// if it is a prefix unary operation.
  bool isPostfixUpdate() const { return IsPostfixUpdate; }

private:
  bool checkBinaryOperation(BinaryOperator *AtomicBinOp, unsigned DiagId = 0,
                            unsigned NoteId = 0);
};

bool OpenMPAtomicUpdateChecker::checkBinaryOperation(
    BinaryOperator *AtomicBinOp, unsigned DiagId, unsigned NoteId) {
  ExprAnalysisErrorCode ErrorFound = NoError;
  SourceLocation ErrorLoc, NoteLoc;
  SourceRange ErrorRange, NoteRange;
  // Allowed constructs are:
  //  x = x binop expr;
  //  x = expr binop x;
  if (AtomicBinOp->getOpcode() == BO_Assign) {
    X = AtomicBinOp->getLHS();
    if (const auto *AtomicInnerBinOp = dyn_cast<BinaryOperator>(
            AtomicBinOp->getRHS()->IgnoreParenImpCasts())) {
      if (AtomicInnerBinOp->isMultiplicativeOp() ||
          AtomicInnerBinOp->isAdditiveOp() || AtomicInnerBinOp->isShiftOp() ||
          AtomicInnerBinOp->isBitwiseOp()) {
        Op = AtomicInnerBinOp->getOpcode();
        OpLoc = AtomicInnerBinOp->getOperatorLoc();
        Expr *LHS = AtomicInnerBinOp->getLHS();
        Expr *RHS = AtomicInnerBinOp->getRHS();
        llvm::FoldingSetNodeID XId, LHSId, RHSId;
        X->IgnoreParenImpCasts()->Profile(XId, SemaRef.getASTContext(),
                                          /*Canonical=*/true);
        LHS->IgnoreParenImpCasts()->Profile(LHSId, SemaRef.getASTContext(),
                                            /*Canonical=*/true);
        RHS->IgnoreParenImpCasts()->Profile(RHSId, SemaRef.getASTContext(),
                                            /*Canonical=*/true);
        if (XId == LHSId) {
          E = RHS;
          IsXLHSInRHSPart = true;
        } else if (XId == RHSId) {
          E = LHS;
          IsXLHSInRHSPart = false;
        } else {
          ErrorLoc = AtomicInnerBinOp->getExprLoc();
          ErrorRange = AtomicInnerBinOp->getSourceRange();
          NoteLoc = X->getExprLoc();
          NoteRange = X->getSourceRange();
          ErrorFound = NotAnUpdateExpression;
        }
      } else {
        ErrorLoc = AtomicInnerBinOp->getExprLoc();
        ErrorRange = AtomicInnerBinOp->getSourceRange();
        NoteLoc = AtomicInnerBinOp->getOperatorLoc();
        NoteRange = SourceRange(NoteLoc, NoteLoc);
        ErrorFound = NotABinaryOperator;
      }
    } else {
      NoteLoc = ErrorLoc = AtomicBinOp->getRHS()->getExprLoc();
      NoteRange = ErrorRange = AtomicBinOp->getRHS()->getSourceRange();
      ErrorFound = NotABinaryExpression;
    }
  } else {
    ErrorLoc = AtomicBinOp->getExprLoc();
    ErrorRange = AtomicBinOp->getSourceRange();
    NoteLoc = AtomicBinOp->getOperatorLoc();
    NoteRange = SourceRange(NoteLoc, NoteLoc);
    ErrorFound = NotAnAssignmentOp;
  }
  if (ErrorFound != NoError && DiagId != 0 && NoteId != 0) {
    SemaRef.Diag(ErrorLoc, DiagId) << ErrorRange;
    SemaRef.Diag(NoteLoc, NoteId) << ErrorFound << NoteRange;
    return true;
  }
  if (SemaRef.CurContext->isDependentContext())
    E = X = UpdateExpr = nullptr;
  return ErrorFound != NoError;
}

bool OpenMPAtomicUpdateChecker::checkStatement(Stmt *S, unsigned DiagId,
                                               unsigned NoteId) {
  ExprAnalysisErrorCode ErrorFound = NoError;
  SourceLocation ErrorLoc, NoteLoc;
  SourceRange ErrorRange, NoteRange;
  // Allowed constructs are:
  //  x++;
  //  x--;
  //  ++x;
  //  --x;
  //  x binop= expr;
  //  x = x binop expr;
  //  x = expr binop x;
  if (auto *AtomicBody = dyn_cast<Expr>(S)) {
    AtomicBody = AtomicBody->IgnoreParenImpCasts();
    if (AtomicBody->getType()->isScalarType() ||
        AtomicBody->isInstantiationDependent()) {
      if (const auto *AtomicCompAssignOp = dyn_cast<CompoundAssignOperator>(
              AtomicBody->IgnoreParenImpCasts())) {
        // Check for Compound Assignment Operation
        Op = BinaryOperator::getOpForCompoundAssignment(
            AtomicCompAssignOp->getOpcode());
        OpLoc = AtomicCompAssignOp->getOperatorLoc();
        E = AtomicCompAssignOp->getRHS();
        X = AtomicCompAssignOp->getLHS()->IgnoreParens();
        IsXLHSInRHSPart = true;
      } else if (auto *AtomicBinOp = dyn_cast<BinaryOperator>(
                     AtomicBody->IgnoreParenImpCasts())) {
        // Check for Binary Operation
        if (checkBinaryOperation(AtomicBinOp, DiagId, NoteId))
          return true;
      } else if (const auto *AtomicUnaryOp = dyn_cast<UnaryOperator>(
                     AtomicBody->IgnoreParenImpCasts())) {
        // Check for Unary Operation
        if (AtomicUnaryOp->isIncrementDecrementOp()) {
          IsPostfixUpdate = AtomicUnaryOp->isPostfix();
          Op = AtomicUnaryOp->isIncrementOp() ? BO_Add : BO_Sub;
          OpLoc = AtomicUnaryOp->getOperatorLoc();
          X = AtomicUnaryOp->getSubExpr()->IgnoreParens();
          E = SemaRef.ActOnIntegerConstant(OpLoc, /*uint64_t Val=*/1).get();
          IsXLHSInRHSPart = true;
        } else {
          ErrorFound = NotAnUnaryIncDecExpression;
          ErrorLoc = AtomicUnaryOp->getExprLoc();
          ErrorRange = AtomicUnaryOp->getSourceRange();
          NoteLoc = AtomicUnaryOp->getOperatorLoc();
          NoteRange = SourceRange(NoteLoc, NoteLoc);
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotABinaryOrUnaryExpression;
        NoteLoc = ErrorLoc = AtomicBody->getExprLoc();
        NoteRange = ErrorRange = AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAScalarType;
      NoteLoc = ErrorLoc = AtomicBody->getBeginLoc();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
  } else {
    ErrorFound = NotAnExpression;
    NoteLoc = ErrorLoc = S->getBeginLoc();
    NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
  }
  if (ErrorFound != NoError && DiagId != 0 && NoteId != 0) {
    SemaRef.Diag(ErrorLoc, DiagId) << ErrorRange;
    SemaRef.Diag(NoteLoc, NoteId) << ErrorFound << NoteRange;
    return true;
  }
  if (SemaRef.CurContext->isDependentContext())
    E = X = UpdateExpr = nullptr;
  if (ErrorFound == NoError && E && X) {
    // Build an update expression of form 'OpaqueValueExpr(x) binop
    // OpaqueValueExpr(expr)' or 'OpaqueValueExpr(expr) binop
    // OpaqueValueExpr(x)' and then cast it to the type of the 'x' expression.
    auto *OVEX = new (SemaRef.getASTContext())
        OpaqueValueExpr(X->getExprLoc(), X->getType(), VK_PRValue);
    auto *OVEExpr = new (SemaRef.getASTContext())
        OpaqueValueExpr(E->getExprLoc(), E->getType(), VK_PRValue);
    ExprResult Update =
        SemaRef.CreateBuiltinBinOp(OpLoc, Op, IsXLHSInRHSPart ? OVEX : OVEExpr,
                                   IsXLHSInRHSPart ? OVEExpr : OVEX);
    if (Update.isInvalid())
      return true;
    Update = SemaRef.PerformImplicitConversion(Update.get(), X->getType(),
                                               Sema::AA_Casting);
    if (Update.isInvalid())
      return true;
    UpdateExpr = Update.get();
  }
  return ErrorFound != NoError;
}
} // namespace

StmtResult Sema::ActOnOpenMPAtomicDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  // Register location of the first atomic directive.
  DSAStack->addAtomicDirectiveLoc(StartLoc);
  if (!AStmt)
    return StmtError();

  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  OpenMPClauseKind AtomicKind = OMPC_unknown;
  SourceLocation AtomicKindLoc;
  OpenMPClauseKind MemOrderKind = OMPC_unknown;
  SourceLocation MemOrderLoc;
  for (const OMPClause *C : Clauses) {
    switch (C->getClauseKind()) {
    case OMPC_read:
    case OMPC_write:
    case OMPC_update:
    case OMPC_capture:
    case OMPC_compare: {
      if (AtomicKind != OMPC_unknown) {
        Diag(C->getBeginLoc(), diag::err_omp_atomic_several_clauses)
            << SourceRange(C->getBeginLoc(), C->getEndLoc());
        Diag(AtomicKindLoc, diag::note_omp_previous_mem_order_clause)
            << getOpenMPClauseName(AtomicKind);
      } else {
        AtomicKind = C->getClauseKind();
        AtomicKindLoc = C->getBeginLoc();
      }
      break;
    }
    case OMPC_seq_cst:
    case OMPC_acq_rel:
    case OMPC_acquire:
    case OMPC_release:
    case OMPC_relaxed: {
      if (MemOrderKind != OMPC_unknown) {
        Diag(C->getBeginLoc(), diag::err_omp_several_mem_order_clauses)
            << getOpenMPDirectiveName(OMPD_atomic) << 0
            << SourceRange(C->getBeginLoc(), C->getEndLoc());
        Diag(MemOrderLoc, diag::note_omp_previous_mem_order_clause)
            << getOpenMPClauseName(MemOrderKind);
      } else {
        MemOrderKind = C->getClauseKind();
        MemOrderLoc = C->getBeginLoc();
      }
      break;
    }
    // The following clauses are allowed, but we don't need to do anything here.
    case OMPC_hint:
      break;
    default:
      llvm_unreachable("unknown clause is encountered");
    }
  }
  // OpenMP 5.0, 2.17.7 atomic Construct, Restrictions
  // If atomic-clause is read then memory-order-clause must not be acq_rel or
  // release.
  // If atomic-clause is write then memory-order-clause must not be acq_rel or
  // acquire.
  // If atomic-clause is update or not present then memory-order-clause must not
  // be acq_rel or acquire.
  if ((AtomicKind == OMPC_read &&
       (MemOrderKind == OMPC_acq_rel || MemOrderKind == OMPC_release)) ||
      ((AtomicKind == OMPC_write || AtomicKind == OMPC_update ||
        AtomicKind == OMPC_unknown) &&
       (MemOrderKind == OMPC_acq_rel || MemOrderKind == OMPC_acquire))) {
    SourceLocation Loc = AtomicKindLoc;
    if (AtomicKind == OMPC_unknown)
      Loc = StartLoc;
    Diag(Loc, diag::err_omp_atomic_incompatible_mem_order_clause)
        << getOpenMPClauseName(AtomicKind)
        << (AtomicKind == OMPC_unknown ? 1 : 0)
        << getOpenMPClauseName(MemOrderKind);
    Diag(MemOrderLoc, diag::note_omp_previous_mem_order_clause)
        << getOpenMPClauseName(MemOrderKind);
  }

  Stmt *Body = AStmt;
  if (auto *EWC = dyn_cast<ExprWithCleanups>(Body))
    Body = EWC->getSubExpr();

  Expr *X = nullptr;
  Expr *V = nullptr;
  Expr *E = nullptr;
  Expr *UE = nullptr;
  bool IsXLHSInRHSPart = false;
  bool IsPostfixUpdate = false;
  // OpenMP [2.12.6, atomic Construct]
  // In the next expressions:
  // * x and v (as applicable) are both l-value expressions with scalar type.
  // * During the execution of an atomic region, multiple syntactic
  // occurrences of x must designate the same storage location.
  // * Neither of v and expr (as applicable) may access the storage location
  // designated by x.
  // * Neither of x and expr (as applicable) may access the storage location
  // designated by v.
  // * expr is an expression with scalar type.
  // * binop is one of +, *, -, /, &, ^, |, <<, or >>.
  // * binop, binop=, ++, and -- are not overloaded operators.
  // * The expression x binop expr must be numerically equivalent to x binop
  // (expr). This requirement is satisfied if the operators in expr have
  // precedence greater than binop, or by using parentheses around expr or
  // subexpressions of expr.
  // * The expression expr binop x must be numerically equivalent to (expr)
  // binop x. This requirement is satisfied if the operators in expr have
  // precedence equal to or greater than binop, or by using parentheses around
  // expr or subexpressions of expr.
  // * For forms that allow multiple occurrences of x, the number of times
  // that x is evaluated is unspecified.
  if (AtomicKind == OMPC_read) {
    enum {
      NotAnExpression,
      NotAnAssignmentOp,
      NotAScalarType,
      NotAnLValue,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    // If clause is read:
    //  v = x;
    if (const auto *AtomicBody = dyn_cast<Expr>(Body)) {
      const auto *AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        X = AtomicBinOp->getRHS()->IgnoreParenImpCasts();
        V = AtomicBinOp->getLHS()->IgnoreParenImpCasts();
        if ((X->isInstantiationDependent() || X->getType()->isScalarType()) &&
            (V->isInstantiationDependent() || V->getType()->isScalarType())) {
          if (!X->isLValue() || !V->isLValue()) {
            const Expr *NotLValueExpr = X->isLValue() ? V : X;
            ErrorFound = NotAnLValue;
            ErrorLoc = AtomicBinOp->getExprLoc();
            ErrorRange = AtomicBinOp->getSourceRange();
            NoteLoc = NotLValueExpr->getExprLoc();
            NoteRange = NotLValueExpr->getSourceRange();
          }
        } else if (!X->isInstantiationDependent() ||
                   !V->isInstantiationDependent()) {
          const Expr *NotScalarExpr =
              (X->isInstantiationDependent() || X->getType()->isScalarType())
                  ? V
                  : X;
          ErrorFound = NotAScalarType;
          ErrorLoc = AtomicBinOp->getExprLoc();
          ErrorRange = AtomicBinOp->getSourceRange();
          NoteLoc = NotScalarExpr->getExprLoc();
          NoteRange = NotScalarExpr->getSourceRange();
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotAnAssignmentOp;
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAnExpression;
      NoteLoc = ErrorLoc = Body->getBeginLoc();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_omp_atomic_read_not_expression_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_omp_atomic_read_write)
          << ErrorFound << NoteRange;
      return StmtError();
    }
    if (CurContext->isDependentContext())
      V = X = nullptr;
  } else if (AtomicKind == OMPC_write) {
    enum {
      NotAnExpression,
      NotAnAssignmentOp,
      NotAScalarType,
      NotAnLValue,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    // If clause is write:
    //  x = expr;
    if (const auto *AtomicBody = dyn_cast<Expr>(Body)) {
      const auto *AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        X = AtomicBinOp->getLHS();
        E = AtomicBinOp->getRHS();
        if ((X->isInstantiationDependent() || X->getType()->isScalarType()) &&
            (E->isInstantiationDependent() || E->getType()->isScalarType())) {
          if (!X->isLValue()) {
            ErrorFound = NotAnLValue;
            ErrorLoc = AtomicBinOp->getExprLoc();
            ErrorRange = AtomicBinOp->getSourceRange();
            NoteLoc = X->getExprLoc();
            NoteRange = X->getSourceRange();
          }
        } else if (!X->isInstantiationDependent() ||
                   !E->isInstantiationDependent()) {
          const Expr *NotScalarExpr =
              (X->isInstantiationDependent() || X->getType()->isScalarType())
                  ? E
                  : X;
          ErrorFound = NotAScalarType;
          ErrorLoc = AtomicBinOp->getExprLoc();
          ErrorRange = AtomicBinOp->getSourceRange();
          NoteLoc = NotScalarExpr->getExprLoc();
          NoteRange = NotScalarExpr->getSourceRange();
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotAnAssignmentOp;
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAnExpression;
      NoteLoc = ErrorLoc = Body->getBeginLoc();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_omp_atomic_write_not_expression_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_omp_atomic_read_write)
          << ErrorFound << NoteRange;
      return StmtError();
    }
    if (CurContext->isDependentContext())
      E = X = nullptr;
  } else if (AtomicKind == OMPC_update || AtomicKind == OMPC_unknown) {
    // If clause is update:
    //  x++;
    //  x--;
    //  ++x;
    //  --x;
    //  x binop= expr;
    //  x = x binop expr;
    //  x = expr binop x;
    OpenMPAtomicUpdateChecker Checker(*this);
    if (Checker.checkStatement(
            Body,
            (AtomicKind == OMPC_update)
                ? diag::err_omp_atomic_update_not_expression_statement
                : diag::err_omp_atomic_not_expression_statement,
            diag::note_omp_atomic_update))
      return StmtError();
    if (!CurContext->isDependentContext()) {
      E = Checker.getExpr();
      X = Checker.getX();
      UE = Checker.getUpdateExpr();
      IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
    }
  } else if (AtomicKind == OMPC_capture) {
    enum {
      NotAnAssignmentOp,
      NotACompoundStatement,
      NotTwoSubstatements,
      NotASpecificExpression,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    if (const auto *AtomicBody = dyn_cast<Expr>(Body)) {
      // If clause is a capture:
      //  v = x++;
      //  v = x--;
      //  v = ++x;
      //  v = --x;
      //  v = x binop= expr;
      //  v = x = x binop expr;
      //  v = x = expr binop x;
      const auto *AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        V = AtomicBinOp->getLHS();
        Body = AtomicBinOp->getRHS()->IgnoreParenImpCasts();
        OpenMPAtomicUpdateChecker Checker(*this);
        if (Checker.checkStatement(
                Body, diag::err_omp_atomic_capture_not_expression_statement,
                diag::note_omp_atomic_update))
          return StmtError();
        E = Checker.getExpr();
        X = Checker.getX();
        UE = Checker.getUpdateExpr();
        IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
        IsPostfixUpdate = Checker.isPostfixUpdate();
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
        ErrorFound = NotAnAssignmentOp;
      }
      if (ErrorFound != NoError) {
        Diag(ErrorLoc, diag::err_omp_atomic_capture_not_expression_statement)
            << ErrorRange;
        Diag(NoteLoc, diag::note_omp_atomic_capture) << ErrorFound << NoteRange;
        return StmtError();
      }
      if (CurContext->isDependentContext())
        UE = V = E = X = nullptr;
    } else {
      // If clause is a capture:
      //  { v = x; x = expr; }
      //  { v = x; x++; }
      //  { v = x; x--; }
      //  { v = x; ++x; }
      //  { v = x; --x; }
      //  { v = x; x binop= expr; }
      //  { v = x; x = x binop expr; }
      //  { v = x; x = expr binop x; }
      //  { x++; v = x; }
      //  { x--; v = x; }
      //  { ++x; v = x; }
      //  { --x; v = x; }
      //  { x binop= expr; v = x; }
      //  { x = x binop expr; v = x; }
      //  { x = expr binop x; v = x; }
      if (auto *CS = dyn_cast<CompoundStmt>(Body)) {
        // Check that this is { expr1; expr2; }
        if (CS->size() == 2) {
          Stmt *First = CS->body_front();
          Stmt *Second = CS->body_back();
          if (auto *EWC = dyn_cast<ExprWithCleanups>(First))
            First = EWC->getSubExpr()->IgnoreParenImpCasts();
          if (auto *EWC = dyn_cast<ExprWithCleanups>(Second))
            Second = EWC->getSubExpr()->IgnoreParenImpCasts();
          // Need to find what subexpression is 'v' and what is 'x'.
          OpenMPAtomicUpdateChecker Checker(*this);
          bool IsUpdateExprFound = !Checker.checkStatement(Second);
          BinaryOperator *BinOp = nullptr;
          if (IsUpdateExprFound) {
            BinOp = dyn_cast<BinaryOperator>(First);
            IsUpdateExprFound = BinOp && BinOp->getOpcode() == BO_Assign;
          }
          if (IsUpdateExprFound && !CurContext->isDependentContext()) {
            //  { v = x; x++; }
            //  { v = x; x--; }
            //  { v = x; ++x; }
            //  { v = x; --x; }
            //  { v = x; x binop= expr; }
            //  { v = x; x = x binop expr; }
            //  { v = x; x = expr binop x; }
            // Check that the first expression has form v = x.
            Expr *PossibleX = BinOp->getRHS()->IgnoreParenImpCasts();
            llvm::FoldingSetNodeID XId, PossibleXId;
            Checker.getX()->Profile(XId, Context, /*Canonical=*/true);
            PossibleX->Profile(PossibleXId, Context, /*Canonical=*/true);
            IsUpdateExprFound = XId == PossibleXId;
            if (IsUpdateExprFound) {
              V = BinOp->getLHS();
              X = Checker.getX();
              E = Checker.getExpr();
              UE = Checker.getUpdateExpr();
              IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
              IsPostfixUpdate = true;
            }
          }
          if (!IsUpdateExprFound) {
            IsUpdateExprFound = !Checker.checkStatement(First);
            BinOp = nullptr;
            if (IsUpdateExprFound) {
              BinOp = dyn_cast<BinaryOperator>(Second);
              IsUpdateExprFound = BinOp && BinOp->getOpcode() == BO_Assign;
            }
            if (IsUpdateExprFound && !CurContext->isDependentContext()) {
              //  { x++; v = x; }
              //  { x--; v = x; }
              //  { ++x; v = x; }
              //  { --x; v = x; }
              //  { x binop= expr; v = x; }
              //  { x = x binop expr; v = x; }
              //  { x = expr binop x; v = x; }
              // Check that the second expression has form v = x.
              Expr *PossibleX = BinOp->getRHS()->IgnoreParenImpCasts();
              llvm::FoldingSetNodeID XId, PossibleXId;
              Checker.getX()->Profile(XId, Context, /*Canonical=*/true);
              PossibleX->Profile(PossibleXId, Context, /*Canonical=*/true);
              IsUpdateExprFound = XId == PossibleXId;
              if (IsUpdateExprFound) {
                V = BinOp->getLHS();
                X = Checker.getX();
                E = Checker.getExpr();
                UE = Checker.getUpdateExpr();
                IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
                IsPostfixUpdate = false;
              }
            }
          }
          if (!IsUpdateExprFound) {
            //  { v = x; x = expr; }
            auto *FirstExpr = dyn_cast<Expr>(First);
            auto *SecondExpr = dyn_cast<Expr>(Second);
            if (!FirstExpr || !SecondExpr ||
                !(FirstExpr->isInstantiationDependent() ||
                  SecondExpr->isInstantiationDependent())) {
              auto *FirstBinOp = dyn_cast<BinaryOperator>(First);
              if (!FirstBinOp || FirstBinOp->getOpcode() != BO_Assign) {
                ErrorFound = NotAnAssignmentOp;
                NoteLoc = ErrorLoc = FirstBinOp ? FirstBinOp->getOperatorLoc()
                                                : First->getBeginLoc();
                NoteRange = ErrorRange = FirstBinOp
                                             ? FirstBinOp->getSourceRange()
                                             : SourceRange(ErrorLoc, ErrorLoc);
              } else {
                auto *SecondBinOp = dyn_cast<BinaryOperator>(Second);
                if (!SecondBinOp || SecondBinOp->getOpcode() != BO_Assign) {
                  ErrorFound = NotAnAssignmentOp;
                  NoteLoc = ErrorLoc = SecondBinOp
                                           ? SecondBinOp->getOperatorLoc()
                                           : Second->getBeginLoc();
                  NoteRange = ErrorRange =
                      SecondBinOp ? SecondBinOp->getSourceRange()
                                  : SourceRange(ErrorLoc, ErrorLoc);
                } else {
                  Expr *PossibleXRHSInFirst =
                      FirstBinOp->getRHS()->IgnoreParenImpCasts();
                  Expr *PossibleXLHSInSecond =
                      SecondBinOp->getLHS()->IgnoreParenImpCasts();
                  llvm::FoldingSetNodeID X1Id, X2Id;
                  PossibleXRHSInFirst->Profile(X1Id, Context,
                                               /*Canonical=*/true);
                  PossibleXLHSInSecond->Profile(X2Id, Context,
                                                /*Canonical=*/true);
                  IsUpdateExprFound = X1Id == X2Id;
                  if (IsUpdateExprFound) {
                    V = FirstBinOp->getLHS();
                    X = SecondBinOp->getLHS();
                    E = SecondBinOp->getRHS();
                    UE = nullptr;
                    IsXLHSInRHSPart = false;
                    IsPostfixUpdate = true;
                  } else {
                    ErrorFound = NotASpecificExpression;
                    ErrorLoc = FirstBinOp->getExprLoc();
                    ErrorRange = FirstBinOp->getSourceRange();
                    NoteLoc = SecondBinOp->getLHS()->getExprLoc();
                    NoteRange = SecondBinOp->getRHS()->getSourceRange();
                  }
                }
              }
            }
          }
        } else {
          NoteLoc = ErrorLoc = Body->getBeginLoc();
          NoteRange = ErrorRange =
              SourceRange(Body->getBeginLoc(), Body->getBeginLoc());
          ErrorFound = NotTwoSubstatements;
        }
      } else {
        NoteLoc = ErrorLoc = Body->getBeginLoc();
        NoteRange = ErrorRange =
            SourceRange(Body->getBeginLoc(), Body->getBeginLoc());
        ErrorFound = NotACompoundStatement;
      }
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_omp_atomic_capture_not_compound_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_omp_atomic_capture) << ErrorFound << NoteRange;
      return StmtError();
    }
    if (CurContext->isDependentContext())
      UE = V = E = X = nullptr;
  } else if (AtomicKind == OMPC_compare) {
    // TODO: For now we emit an error here and in emitOMPAtomicExpr we ignore
    // code gen.
    unsigned DiagID = Diags.getCustomDiagID(
        DiagnosticsEngine::Error, "atomic compare is not supported for now");
    Diag(AtomicKindLoc, DiagID);
  }

  setFunctionHasBranchProtectedScope();

  return OMPAtomicDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                    X, V, E, UE, IsXLHSInRHSPart,
                                    IsPostfixUpdate);
}

StmtResult Sema::ActOnOpenMPTargetDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  // OpenMP [2.16, Nesting of Regions]
  // If specified, a teams construct must be contained within a target
  // construct. That target construct must contain no statements or directives
  // outside of the teams construct.
  if (DSAStack->hasInnerTeamsRegion()) {
    const Stmt *S = CS->IgnoreContainers(/*IgnoreCaptured=*/true);
    bool OMPTeamsFound = true;
    if (const auto *CS = dyn_cast<CompoundStmt>(S)) {
      auto I = CS->body_begin();
      while (I != CS->body_end()) {
        const auto *OED = dyn_cast<OMPExecutableDirective>(*I);
        if (!OED || !isOpenMPTeamsDirective(OED->getDirectiveKind()) ||
            OMPTeamsFound) {

          OMPTeamsFound = false;
          break;
        }
        ++I;
      }
      assert(I != CS->body_end() && "Not found statement");
      S = *I;
    } else {
      const auto *OED = dyn_cast<OMPExecutableDirective>(S);
      OMPTeamsFound = OED && isOpenMPTeamsDirective(OED->getDirectiveKind());
    }
    if (!OMPTeamsFound) {
      Diag(StartLoc, diag::err_omp_target_contains_not_only_teams);
      Diag(DSAStack->getInnerTeamsRegionLoc(),
           diag::note_omp_nested_teams_construct_here);
      Diag(S->getBeginLoc(), diag::note_omp_nested_statement_here)
          << isa<OMPExecutableDirective>(S);
      return StmtError();
    }
  }

  setFunctionHasBranchProtectedScope();

  return OMPTargetDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult
Sema::ActOnOpenMPTargetParallelDirective(ArrayRef<OMPClause *> Clauses,
                                         Stmt *AStmt, SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_parallel);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  setFunctionHasBranchProtectedScope();

  return OMPTargetParallelDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPTargetParallelForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_parallel_for);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_target_parallel_for, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), CS, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp target parallel for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  setFunctionHasBranchProtectedScope();
  return OMPTargetParallelForDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

/// Check for existence of a map clause in the list of clauses.
static bool hasClauses(ArrayRef<OMPClause *> Clauses,
                       const OpenMPClauseKind K) {
  return llvm::any_of(
      Clauses, [K](const OMPClause *C) { return C->getClauseKind() == K; });
}

template <typename... Params>
static bool hasClauses(ArrayRef<OMPClause *> Clauses, const OpenMPClauseKind K,
                       const Params... ClauseTypes) {
  return hasClauses(Clauses, K) || hasClauses(Clauses, ClauseTypes...);
}

StmtResult Sema::ActOnOpenMPTargetDataDirective(ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  // OpenMP [2.12.2, target data Construct, Restrictions]
  // At least one map, use_device_addr or use_device_ptr clause must appear on
  // the directive.
  if (!hasClauses(Clauses, OMPC_map, OMPC_use_device_ptr) &&
      (LangOpts.OpenMP < 50 || !hasClauses(Clauses, OMPC_use_device_addr))) {
    StringRef Expected;
    if (LangOpts.OpenMP < 50)
      Expected = "'map' or 'use_device_ptr'";
    else
      Expected = "'map', 'use_device_ptr', or 'use_device_addr'";
    Diag(StartLoc, diag::err_omp_no_clause_for_directive)
        << Expected << getOpenMPDirectiveName(OMPD_target_data);
    return StmtError();
  }

  setFunctionHasBranchProtectedScope();

  return OMPTargetDataDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                        AStmt);
}

StmtResult
Sema::ActOnOpenMPTargetEnterDataDirective(ArrayRef<OMPClause *> Clauses,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc, Stmt *AStmt) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_enter_data);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  // OpenMP [2.10.2, Restrictions, p. 99]
  // At least one map clause must appear on the directive.
  if (!hasClauses(Clauses, OMPC_map)) {
    Diag(StartLoc, diag::err_omp_no_clause_for_directive)
        << "'map'" << getOpenMPDirectiveName(OMPD_target_enter_data);
    return StmtError();
  }

  return OMPTargetEnterDataDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                             AStmt);
}

StmtResult
Sema::ActOnOpenMPTargetExitDataDirective(ArrayRef<OMPClause *> Clauses,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc, Stmt *AStmt) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_exit_data);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  // OpenMP [2.10.3, Restrictions, p. 102]
  // At least one map clause must appear on the directive.
  if (!hasClauses(Clauses, OMPC_map)) {
    Diag(StartLoc, diag::err_omp_no_clause_for_directive)
        << "'map'" << getOpenMPDirectiveName(OMPD_target_exit_data);
    return StmtError();
  }

  return OMPTargetExitDataDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                            AStmt);
}

StmtResult Sema::ActOnOpenMPTargetUpdateDirective(ArrayRef<OMPClause *> Clauses,
                                                  SourceLocation StartLoc,
                                                  SourceLocation EndLoc,
                                                  Stmt *AStmt) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_update);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  if (!hasClauses(Clauses, OMPC_to, OMPC_from)) {
    Diag(StartLoc, diag::err_omp_at_least_one_motion_clause_required);
    return StmtError();
  }
  return OMPTargetUpdateDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                          AStmt);
}

StmtResult Sema::ActOnOpenMPTeamsDirective(ArrayRef<OMPClause *> Clauses,
                                           Stmt *AStmt, SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  setFunctionHasBranchProtectedScope();

  DSAStack->setParentTeamsRegionLoc(StartLoc);

  return OMPTeamsDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult
Sema::ActOnOpenMPCancellationPointDirective(SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            OpenMPDirectiveKind CancelRegion) {
  if (DSAStack->isParentNowaitRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_nowait) << 0;
    return StmtError();
  }
  if (DSAStack->isParentOrderedRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_ordered) << 0;
    return StmtError();
  }
  return OMPCancellationPointDirective::Create(Context, StartLoc, EndLoc,
                                               CancelRegion);
}

StmtResult Sema::ActOnOpenMPCancelDirective(ArrayRef<OMPClause *> Clauses,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            OpenMPDirectiveKind CancelRegion) {
  if (DSAStack->isParentNowaitRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_nowait) << 1;
    return StmtError();
  }
  if (DSAStack->isParentOrderedRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_ordered) << 1;
    return StmtError();
  }
  DSAStack->setParentCancelRegion(/*Cancel=*/true);
  return OMPCancelDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                    CancelRegion);
}

static bool checkReductionClauseWithNogroup(Sema &S,
                                            ArrayRef<OMPClause *> Clauses) {
  const OMPClause *ReductionClause = nullptr;
  const OMPClause *NogroupClause = nullptr;
  for (const OMPClause *C : Clauses) {
    if (C->getClauseKind() == OMPC_reduction) {
      ReductionClause = C;
      if (NogroupClause)
        break;
      continue;
    }
    if (C->getClauseKind() == OMPC_nogroup) {
      NogroupClause = C;
      if (ReductionClause)
        break;
      continue;
    }
  }
  if (ReductionClause && NogroupClause) {
    S.Diag(ReductionClause->getBeginLoc(), diag::err_omp_reduction_with_nogroup)
        << SourceRange(NogroupClause->getBeginLoc(),
                       NogroupClause->getEndLoc());
    return true;
  }
  return false;
}

StmtResult Sema::ActOnOpenMPTaskLoopDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_taskloop, getCollapseNumberExpr(Clauses),
                      /*OrderedLoopCountExpr=*/nullptr, AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkMutuallyExclusiveClauses(*this, Clauses,
                                    {OMPC_grainsize, OMPC_num_tasks}))
    return StmtError();
  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // If a reduction clause is present on the taskloop directive, the nogroup
  // clause must not be specified.
  if (checkReductionClauseWithNogroup(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPTaskLoopDirective::Create(Context, StartLoc, EndLoc,
                                      NestedLoopCount, Clauses, AStmt, B,
                                      DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPTaskLoopSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_taskloop_simd, getCollapseNumberExpr(Clauses),
                      /*OrderedLoopCountExpr=*/nullptr, AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkMutuallyExclusiveClauses(*this, Clauses,
                                    {OMPC_grainsize, OMPC_num_tasks}))
    return StmtError();
  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // If a reduction clause is present on the taskloop directive, the nogroup
  // clause must not be specified.
  if (checkReductionClauseWithNogroup(*this, Clauses))
    return StmtError();
  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPTaskLoopSimdDirective::Create(Context, StartLoc, EndLoc,
                                          NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPMasterTaskLoopDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_master_taskloop, getCollapseNumberExpr(Clauses),
                      /*OrderedLoopCountExpr=*/nullptr, AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkMutuallyExclusiveClauses(*this, Clauses,
                                    {OMPC_grainsize, OMPC_num_tasks}))
    return StmtError();
  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // If a reduction clause is present on the taskloop directive, the nogroup
  // clause must not be specified.
  if (checkReductionClauseWithNogroup(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPMasterTaskLoopDirective::Create(Context, StartLoc, EndLoc,
                                            NestedLoopCount, Clauses, AStmt, B,
                                            DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPMasterTaskLoopSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_master_taskloop_simd, getCollapseNumberExpr(Clauses),
                      /*OrderedLoopCountExpr=*/nullptr, AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkMutuallyExclusiveClauses(*this, Clauses,
                                    {OMPC_grainsize, OMPC_num_tasks}))
    return StmtError();
  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // If a reduction clause is present on the taskloop directive, the nogroup
  // clause must not be specified.
  if (checkReductionClauseWithNogroup(*this, Clauses))
    return StmtError();
  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPMasterTaskLoopSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPParallelMasterTaskLoopDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_parallel_master_taskloop);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_parallel_master_taskloop, getCollapseNumberExpr(Clauses),
      /*OrderedLoopCountExpr=*/nullptr, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkMutuallyExclusiveClauses(*this, Clauses,
                                    {OMPC_grainsize, OMPC_num_tasks}))
    return StmtError();
  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // If a reduction clause is present on the taskloop directive, the nogroup
  // clause must not be specified.
  if (checkReductionClauseWithNogroup(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPParallelMasterTaskLoopDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B,
      DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPParallelMasterTaskLoopSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_parallel_master_taskloop_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_parallel_master_taskloop_simd, getCollapseNumberExpr(Clauses),
      /*OrderedLoopCountExpr=*/nullptr, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkMutuallyExclusiveClauses(*this, Clauses,
                                    {OMPC_grainsize, OMPC_num_tasks}))
    return StmtError();
  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // If a reduction clause is present on the taskloop directive, the nogroup
  // clause must not be specified.
  if (checkReductionClauseWithNogroup(*this, Clauses))
    return StmtError();
  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPParallelMasterTaskLoopSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPDistributeDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_distribute, getCollapseNumberExpr(Clauses),
                      nullptr /*ordered not a clause on distribute*/, AStmt,
                      *this, *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  setFunctionHasBranchProtectedScope();
  return OMPDistributeDirective::Create(Context, StartLoc, EndLoc,
                                        NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPDistributeParallelForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_distribute_parallel_for);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_distribute_parallel_for, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  setFunctionHasBranchProtectedScope();
  return OMPDistributeParallelForDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPDistributeParallelForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_distribute_parallel_for_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_distribute_parallel_for_simd, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPDistributeParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPDistributeSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_distribute_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_distribute_simd, getCollapseNumberExpr(Clauses),
                      nullptr /*ordered not a clause on distribute*/, CS, *this,
                      *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPDistributeSimdDirective::Create(Context, StartLoc, EndLoc,
                                            NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTargetParallelForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_parallel_for);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_target_parallel_for_simd, getCollapseNumberExpr(Clauses),
      getOrderedNumberExpr(Clauses), CS, *this, *DSAStack, VarsWithImplicitDSA,
      B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp target parallel for simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }
  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPTargetParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTargetSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will define the
  // nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_target_simd, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), CS, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp target simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPTargetSimdDirective::Create(Context, StartLoc, EndLoc,
                                        NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTeamsDistributeDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_teams_distribute);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_teams_distribute, getCollapseNumberExpr(Clauses),
                      nullptr /*ordered not a clause on distribute*/, CS, *this,
                      *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp teams distribute loop exprs were not built");

  setFunctionHasBranchProtectedScope();

  DSAStack->setParentTeamsRegionLoc(StartLoc);

  return OMPTeamsDistributeDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTeamsDistributeSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_teams_distribute_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_teams_distribute_simd, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);

  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp teams distribute simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();

  DSAStack->setParentTeamsRegionLoc(StartLoc);

  return OMPTeamsDistributeSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTeamsDistributeParallelForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_teams_distribute_parallel_for_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_teams_distribute_parallel_for_simd, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);

  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();

  DSAStack->setParentTeamsRegionLoc(StartLoc);

  return OMPTeamsDistributeParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTeamsDistributeParallelForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_teams_distribute_parallel_for);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_teams_distribute_parallel_for, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);

  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  setFunctionHasBranchProtectedScope();

  DSAStack->setParentTeamsRegionLoc(StartLoc);

  return OMPTeamsDistributeParallelForDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPTargetTeamsDirective(ArrayRef<OMPClause *> Clauses,
                                                 Stmt *AStmt,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  for (int ThisCaptureLevel = getOpenMPCaptureLevels(OMPD_target_teams);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }
  setFunctionHasBranchProtectedScope();

  return OMPTargetTeamsDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                         AStmt);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_target_teams_distribute);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_target_teams_distribute, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp target teams distribute loop exprs were not built");

  setFunctionHasBranchProtectedScope();
  return OMPTargetTeamsDistributeDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeParallelForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_target_teams_distribute_parallel_for);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_target_teams_distribute_parallel_for, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp target teams distribute parallel for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  setFunctionHasBranchProtectedScope();
  return OMPTargetTeamsDistributeParallelForDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B,
      DSAStack->getTaskgroupReductionRef(), DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeParallelForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel = getOpenMPCaptureLevels(
           OMPD_target_teams_distribute_parallel_for_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      checkOpenMPLoop(OMPD_target_teams_distribute_parallel_for_simd,
                      getCollapseNumberExpr(Clauses),
                      nullptr /*ordered not a clause on distribute*/, CS, *this,
                      *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp target teams distribute parallel for simd loop exprs were not "
         "built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPTargetTeamsDistributeParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  auto *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();
  for (int ThisCaptureLevel =
           getOpenMPCaptureLevels(OMPD_target_teams_distribute_simd);
       ThisCaptureLevel > 1; --ThisCaptureLevel) {
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
    // 1.2.2 OpenMP Language Terminology
    // Structured block - An executable statement with a single entry at the
    // top and a single exit at the bottom.
    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CS->getCapturedDecl()->setNothrow();
  }

  OMPLoopBasedDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = checkOpenMPLoop(
      OMPD_target_teams_distribute_simd, getCollapseNumberExpr(Clauses),
      nullptr /*ordered not a clause on distribute*/, CS, *this, *DSAStack,
      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp target teams distribute simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (OMPClause *C : Clauses) {
      if (auto *LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope,
                                     DSAStack))
          return StmtError();
    }
  }

  if (checkSimdlenSafelenSpecified(*this, Clauses))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OMPTargetTeamsDistributeSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

bool Sema::checkTransformableLoopNest(
    OpenMPDirectiveKind Kind, Stmt *AStmt, int NumLoops,
    SmallVectorImpl<OMPLoopBasedDirective::HelperExprs> &LoopHelpers,
    Stmt *&Body,
    SmallVectorImpl<SmallVector<llvm::PointerUnion<Stmt *, Decl *>, 0>>
        &OriginalInits) {
  OriginalInits.emplace_back();
  bool Result = OMPLoopBasedDirective::doForAllLoops(
      AStmt->IgnoreContainers(), /*TryImperfectlyNestedLoops=*/false, NumLoops,
      [this, &LoopHelpers, &Body, &OriginalInits, Kind](unsigned Cnt,
                                                        Stmt *CurStmt) {
        VarsWithInheritedDSAType TmpDSA;
        unsigned SingleNumLoops =
            checkOpenMPLoop(Kind, nullptr, nullptr, CurStmt, *this, *DSAStack,
                            TmpDSA, LoopHelpers[Cnt]);
        if (SingleNumLoops == 0)
          return true;
        assert(SingleNumLoops == 1 && "Expect single loop iteration space");
        if (auto *For = dyn_cast<ForStmt>(CurStmt)) {
          OriginalInits.back().push_back(For->getInit());
          Body = For->getBody();
        } else {
          assert(isa<CXXForRangeStmt>(CurStmt) &&
                 "Expected canonical for or range-based for loops.");
          auto *CXXFor = cast<CXXForRangeStmt>(CurStmt);
          OriginalInits.back().push_back(CXXFor->getBeginStmt());
          Body = CXXFor->getBody();
        }
        OriginalInits.emplace_back();
        return false;
      },
      [&OriginalInits](OMPLoopBasedDirective *Transform) {
        Stmt *DependentPreInits;
        if (auto *Dir = dyn_cast<OMPTileDirective>(Transform))
          DependentPreInits = Dir->getPreInits();
        else if (auto *Dir = dyn_cast<OMPUnrollDirective>(Transform))
          DependentPreInits = Dir->getPreInits();
        else
          llvm_unreachable("Unhandled loop transformation");
        if (!DependentPreInits)
          return;
        for (Decl *C : cast<DeclStmt>(DependentPreInits)->getDeclGroup())
          OriginalInits.back().push_back(C);
      });
  assert(OriginalInits.back().empty() && "No preinit after innermost loop");
  OriginalInits.pop_back();
  return Result;
}

StmtResult Sema::ActOnOpenMPTileDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  auto SizesClauses =
      OMPExecutableDirective::getClausesOfKind<OMPSizesClause>(Clauses);
  if (SizesClauses.empty()) {
    // A missing 'sizes' clause is already reported by the parser.
    return StmtError();
  }
  const OMPSizesClause *SizesClause = *SizesClauses.begin();
  unsigned NumLoops = SizesClause->getNumSizes();

  // Empty statement should only be possible if there already was an error.
  if (!AStmt)
    return StmtError();

  // Verify and diagnose loop nest.
  SmallVector<OMPLoopBasedDirective::HelperExprs, 4> LoopHelpers(NumLoops);
  Stmt *Body = nullptr;
  SmallVector<SmallVector<llvm::PointerUnion<Stmt *, Decl *>, 0>, 4>
      OriginalInits;
  if (!checkTransformableLoopNest(OMPD_tile, AStmt, NumLoops, LoopHelpers, Body,
                                  OriginalInits))
    return StmtError();

  // Delay tiling to when template is completely instantiated.
  if (CurContext->isDependentContext())
    return OMPTileDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                    NumLoops, AStmt, nullptr, nullptr);

  SmallVector<Decl *, 4> PreInits;

  // Create iteration variables for the generated loops.
  SmallVector<VarDecl *, 4> FloorIndVars;
  SmallVector<VarDecl *, 4> TileIndVars;
  FloorIndVars.resize(NumLoops);
  TileIndVars.resize(NumLoops);
  for (unsigned I = 0; I < NumLoops; ++I) {
    OMPLoopBasedDirective::HelperExprs &LoopHelper = LoopHelpers[I];

    assert(LoopHelper.Counters.size() == 1 &&
           "Expect single-dimensional loop iteration space");
    auto *OrigCntVar = cast<DeclRefExpr>(LoopHelper.Counters.front());
    std::string OrigVarName = OrigCntVar->getNameInfo().getAsString();
    DeclRefExpr *IterVarRef = cast<DeclRefExpr>(LoopHelper.IterationVarRef);
    QualType CntTy = IterVarRef->getType();

    // Iteration variable for the floor (i.e. outer) loop.
    {
      std::string FloorCntName =
          (Twine(".floor_") + llvm::utostr(I) + ".iv." + OrigVarName).str();
      VarDecl *FloorCntDecl =
          buildVarDecl(*this, {}, CntTy, FloorCntName, nullptr, OrigCntVar);
      FloorIndVars[I] = FloorCntDecl;
    }

    // Iteration variable for the tile (i.e. inner) loop.
    {
      std::string TileCntName =
          (Twine(".tile_") + llvm::utostr(I) + ".iv." + OrigVarName).str();

      // Reuse the iteration variable created by checkOpenMPLoop. It is also
      // used by the expressions to derive the original iteration variable's
      // value from the logical iteration number.
      auto *TileCntDecl = cast<VarDecl>(IterVarRef->getDecl());
      TileCntDecl->setDeclName(&PP.getIdentifierTable().get(TileCntName));
      TileIndVars[I] = TileCntDecl;
    }
    for (auto &P : OriginalInits[I]) {
      if (auto *D = P.dyn_cast<Decl *>())
        PreInits.push_back(D);
      else if (auto *PI = dyn_cast_or_null<DeclStmt>(P.dyn_cast<Stmt *>()))
        PreInits.append(PI->decl_begin(), PI->decl_end());
    }
    if (auto *PI = cast_or_null<DeclStmt>(LoopHelper.PreInits))
      PreInits.append(PI->decl_begin(), PI->decl_end());
    // Gather declarations for the data members used as counters.
    for (Expr *CounterRef : LoopHelper.Counters) {
      auto *CounterDecl = cast<DeclRefExpr>(CounterRef)->getDecl();
      if (isa<OMPCapturedExprDecl>(CounterDecl))
        PreInits.push_back(CounterDecl);
    }
  }

  // Once the original iteration values are set, append the innermost body.
  Stmt *Inner = Body;

  // Create tile loops from the inside to the outside.
  for (int I = NumLoops - 1; I >= 0; --I) {
    OMPLoopBasedDirective::HelperExprs &LoopHelper = LoopHelpers[I];
    Expr *NumIterations = LoopHelper.NumIterations;
    auto *OrigCntVar = cast<DeclRefExpr>(LoopHelper.Counters[0]);
    QualType CntTy = OrigCntVar->getType();
    Expr *DimTileSize = SizesClause->getSizesRefs()[I];
    Scope *CurScope = getCurScope();

    // Commonly used variables.
    DeclRefExpr *TileIV = buildDeclRefExpr(*this, TileIndVars[I], CntTy,
                                           OrigCntVar->getExprLoc());
    DeclRefExpr *FloorIV = buildDeclRefExpr(*this, FloorIndVars[I], CntTy,
                                            OrigCntVar->getExprLoc());

    // For init-statement: auto .tile.iv = .floor.iv
    AddInitializerToDecl(TileIndVars[I], DefaultLvalueConversion(FloorIV).get(),
                         /*DirectInit=*/false);
    Decl *CounterDecl = TileIndVars[I];
    StmtResult InitStmt = new (Context)
        DeclStmt(DeclGroupRef::Create(Context, &CounterDecl, 1),
                 OrigCntVar->getBeginLoc(), OrigCntVar->getEndLoc());
    if (!InitStmt.isUsable())
      return StmtError();

    // For cond-expression: .tile.iv < min(.floor.iv + DimTileSize,
    // NumIterations)
    ExprResult EndOfTile = BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(),
                                      BO_Add, FloorIV, DimTileSize);
    if (!EndOfTile.isUsable())
      return StmtError();
    ExprResult IsPartialTile =
        BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(), BO_LT,
                   NumIterations, EndOfTile.get());
    if (!IsPartialTile.isUsable())
      return StmtError();
    ExprResult MinTileAndIterSpace = ActOnConditionalOp(
        LoopHelper.Cond->getBeginLoc(), LoopHelper.Cond->getEndLoc(),
        IsPartialTile.get(), NumIterations, EndOfTile.get());
    if (!MinTileAndIterSpace.isUsable())
      return StmtError();
    ExprResult CondExpr = BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(),
                                     BO_LT, TileIV, MinTileAndIterSpace.get());
    if (!CondExpr.isUsable())
      return StmtError();

    // For incr-statement: ++.tile.iv
    ExprResult IncrStmt =
        BuildUnaryOp(CurScope, LoopHelper.Inc->getExprLoc(), UO_PreInc, TileIV);
    if (!IncrStmt.isUsable())
      return StmtError();

    // Statements to set the original iteration variable's value from the
    // logical iteration number.
    // Generated for loop is:
    // Original_for_init;
    // for (auto .tile.iv = .floor.iv; .tile.iv < min(.floor.iv + DimTileSize,
    // NumIterations); ++.tile.iv) {
    //   Original_Body;
    //   Original_counter_update;
    // }
    // FIXME: If the innermost body is an loop itself, inserting these
    // statements stops it being recognized  as a perfectly nested loop (e.g.
    // for applying tiling again). If this is the case, sink the expressions
    // further into the inner loop.
    SmallVector<Stmt *, 4> BodyParts;
    BodyParts.append(LoopHelper.Updates.begin(), LoopHelper.Updates.end());
    BodyParts.push_back(Inner);
    Inner = CompoundStmt::Create(Context, BodyParts, Inner->getBeginLoc(),
                                 Inner->getEndLoc());
    Inner = new (Context)
        ForStmt(Context, InitStmt.get(), CondExpr.get(), nullptr,
                IncrStmt.get(), Inner, LoopHelper.Init->getBeginLoc(),
                LoopHelper.Init->getBeginLoc(), LoopHelper.Inc->getEndLoc());
  }

  // Create floor loops from the inside to the outside.
  for (int I = NumLoops - 1; I >= 0; --I) {
    auto &LoopHelper = LoopHelpers[I];
    Expr *NumIterations = LoopHelper.NumIterations;
    DeclRefExpr *OrigCntVar = cast<DeclRefExpr>(LoopHelper.Counters[0]);
    QualType CntTy = OrigCntVar->getType();
    Expr *DimTileSize = SizesClause->getSizesRefs()[I];
    Scope *CurScope = getCurScope();

    // Commonly used variables.
    DeclRefExpr *FloorIV = buildDeclRefExpr(*this, FloorIndVars[I], CntTy,
                                            OrigCntVar->getExprLoc());

    // For init-statement: auto .floor.iv = 0
    AddInitializerToDecl(
        FloorIndVars[I],
        ActOnIntegerConstant(LoopHelper.Init->getExprLoc(), 0).get(),
        /*DirectInit=*/false);
    Decl *CounterDecl = FloorIndVars[I];
    StmtResult InitStmt = new (Context)
        DeclStmt(DeclGroupRef::Create(Context, &CounterDecl, 1),
                 OrigCntVar->getBeginLoc(), OrigCntVar->getEndLoc());
    if (!InitStmt.isUsable())
      return StmtError();

    // For cond-expression: .floor.iv < NumIterations
    ExprResult CondExpr = BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(),
                                     BO_LT, FloorIV, NumIterations);
    if (!CondExpr.isUsable())
      return StmtError();

    // For incr-statement: .floor.iv += DimTileSize
    ExprResult IncrStmt = BuildBinOp(CurScope, LoopHelper.Inc->getExprLoc(),
                                     BO_AddAssign, FloorIV, DimTileSize);
    if (!IncrStmt.isUsable())
      return StmtError();

    Inner = new (Context)
        ForStmt(Context, InitStmt.get(), CondExpr.get(), nullptr,
                IncrStmt.get(), Inner, LoopHelper.Init->getBeginLoc(),
                LoopHelper.Init->getBeginLoc(), LoopHelper.Inc->getEndLoc());
  }

  return OMPTileDirective::Create(Context, StartLoc, EndLoc, Clauses, NumLoops,
                                  AStmt, Inner,
                                  buildPreInits(Context, PreInits));
}

StmtResult Sema::ActOnOpenMPUnrollDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  // Empty statement should only be possible if there already was an error.
  if (!AStmt)
    return StmtError();

  if (checkMutuallyExclusiveClauses(*this, Clauses, {OMPC_partial, OMPC_full}))
    return StmtError();

  const OMPFullClause *FullClause =
      OMPExecutableDirective::getSingleClause<OMPFullClause>(Clauses);
  const OMPPartialClause *PartialClause =
      OMPExecutableDirective::getSingleClause<OMPPartialClause>(Clauses);
  assert(!(FullClause && PartialClause) &&
         "mutual exclusivity must have been checked before");

  constexpr unsigned NumLoops = 1;
  Stmt *Body = nullptr;
  SmallVector<OMPLoopBasedDirective::HelperExprs, NumLoops> LoopHelpers(
      NumLoops);
  SmallVector<SmallVector<llvm::PointerUnion<Stmt *, Decl *>, 0>, NumLoops + 1>
      OriginalInits;
  if (!checkTransformableLoopNest(OMPD_unroll, AStmt, NumLoops, LoopHelpers,
                                  Body, OriginalInits))
    return StmtError();

  unsigned NumGeneratedLoops = PartialClause ? 1 : 0;

  // Delay unrolling to when template is completely instantiated.
  if (CurContext->isDependentContext())
    return OMPUnrollDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                      NumGeneratedLoops, nullptr, nullptr);

  OMPLoopBasedDirective::HelperExprs &LoopHelper = LoopHelpers.front();

  if (FullClause) {
    if (!VerifyPositiveIntegerConstantInClause(
             LoopHelper.NumIterations, OMPC_full, /*StrictlyPositive=*/false,
             /*SuppressExprDiags=*/true)
             .isUsable()) {
      Diag(AStmt->getBeginLoc(), diag::err_omp_unroll_full_variable_trip_count);
      Diag(FullClause->getBeginLoc(), diag::note_omp_directive_here)
          << "#pragma omp unroll full";
      return StmtError();
    }
  }

  // The generated loop may only be passed to other loop-associated directive
  // when a partial clause is specified. Without the requirement it is
  // sufficient to generate loop unroll metadata at code-generation.
  if (NumGeneratedLoops == 0)
    return OMPUnrollDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                      NumGeneratedLoops, nullptr, nullptr);

  // Otherwise, we need to provide a de-sugared/transformed AST that can be
  // associated with another loop directive.
  //
  // The canonical loop analysis return by checkTransformableLoopNest assumes
  // the following structure to be the same loop without transformations or
  // directives applied: \code OriginalInits; LoopHelper.PreInits;
  // LoopHelper.Counters;
  // for (; IV < LoopHelper.NumIterations; ++IV) {
  //   LoopHelper.Updates;
  //   Body;
  // }
  // \endcode
  // where IV is a variable declared and initialized to 0 in LoopHelper.PreInits
  // and referenced by LoopHelper.IterationVarRef.
  //
  // The unrolling directive transforms this into the following loop:
  // \code
  // OriginalInits;         \
  // LoopHelper.PreInits;    > NewPreInits
  // LoopHelper.Counters;   /
  // for (auto UIV = 0; UIV < LoopHelper.NumIterations; UIV+=Factor) {
  //   #pragma clang loop unroll_count(Factor)
  //   for (IV = UIV; IV < UIV + Factor && UIV < LoopHelper.NumIterations; ++IV)
  //   {
  //     LoopHelper.Updates;
  //     Body;
  //   }
  // }
  // \endcode
  // where UIV is a new logical iteration counter. IV must be the same VarDecl
  // as the original LoopHelper.IterationVarRef because LoopHelper.Updates
  // references it. If the partially unrolled loop is associated with another
  // loop directive (like an OMPForDirective), it will use checkOpenMPLoop to
  // analyze this loop, i.e. the outer loop must fulfill the constraints of an
  // OpenMP canonical loop. The inner loop is not an associable canonical loop
  // and only exists to defer its unrolling to LLVM's LoopUnroll instead of
  // doing it in the frontend (by adding loop metadata). NewPreInits becomes a
  // property of the OMPLoopBasedDirective instead of statements in
  // CompoundStatement. This is to allow the loop to become a non-outermost loop
  // of a canonical loop nest where these PreInits are emitted before the
  // outermost directive.

  // Determine the PreInit declarations.
  SmallVector<Decl *, 4> PreInits;
  assert(OriginalInits.size() == 1 &&
         "Expecting a single-dimensional loop iteration space");
  for (auto &P : OriginalInits[0]) {
    if (auto *D = P.dyn_cast<Decl *>())
      PreInits.push_back(D);
    else if (auto *PI = dyn_cast_or_null<DeclStmt>(P.dyn_cast<Stmt *>()))
      PreInits.append(PI->decl_begin(), PI->decl_end());
  }
  if (auto *PI = cast_or_null<DeclStmt>(LoopHelper.PreInits))
    PreInits.append(PI->decl_begin(), PI->decl_end());
  // Gather declarations for the data members used as counters.
  for (Expr *CounterRef : LoopHelper.Counters) {
    auto *CounterDecl = cast<DeclRefExpr>(CounterRef)->getDecl();
    if (isa<OMPCapturedExprDecl>(CounterDecl))
      PreInits.push_back(CounterDecl);
  }

  auto *IterationVarRef = cast<DeclRefExpr>(LoopHelper.IterationVarRef);
  QualType IVTy = IterationVarRef->getType();
  assert(LoopHelper.Counters.size() == 1 &&
         "Expecting a single-dimensional loop iteration space");
  auto *OrigVar = cast<DeclRefExpr>(LoopHelper.Counters.front());

  // Determine the unroll factor.
  uint64_t Factor;
  SourceLocation FactorLoc;
  if (Expr *FactorVal = PartialClause->getFactor()) {
    Factor =
        FactorVal->getIntegerConstantExpr(Context).getValue().getZExtValue();
    FactorLoc = FactorVal->getExprLoc();
  } else {
    // TODO: Use a better profitability model.
    Factor = 2;
  }
  assert(Factor > 0 && "Expected positive unroll factor");
  auto MakeFactorExpr = [this, Factor, IVTy, FactorLoc]() {
    return IntegerLiteral::Create(
        Context, llvm::APInt(Context.getIntWidth(IVTy), Factor), IVTy,
        FactorLoc);
  };

  // Iteration variable SourceLocations.
  SourceLocation OrigVarLoc = OrigVar->getExprLoc();
  SourceLocation OrigVarLocBegin = OrigVar->getBeginLoc();
  SourceLocation OrigVarLocEnd = OrigVar->getEndLoc();

  // Internal variable names.
  std::string OrigVarName = OrigVar->getNameInfo().getAsString();
  std::string OuterIVName = (Twine(".unrolled.iv.") + OrigVarName).str();
  std::string InnerIVName = (Twine(".unroll_inner.iv.") + OrigVarName).str();
  std::string InnerTripCountName =
      (Twine(".unroll_inner.tripcount.") + OrigVarName).str();

  // Create the iteration variable for the unrolled loop.
  VarDecl *OuterIVDecl =
      buildVarDecl(*this, {}, IVTy, OuterIVName, nullptr, OrigVar);
  auto MakeOuterRef = [this, OuterIVDecl, IVTy, OrigVarLoc]() {
    return buildDeclRefExpr(*this, OuterIVDecl, IVTy, OrigVarLoc);
  };

  // Iteration variable for the inner loop: Reuse the iteration variable created
  // by checkOpenMPLoop.
  auto *InnerIVDecl = cast<VarDecl>(IterationVarRef->getDecl());
  InnerIVDecl->setDeclName(&PP.getIdentifierTable().get(InnerIVName));
  auto MakeInnerRef = [this, InnerIVDecl, IVTy, OrigVarLoc]() {
    return buildDeclRefExpr(*this, InnerIVDecl, IVTy, OrigVarLoc);
  };

  // Make a copy of the NumIterations expression for each use: By the AST
  // constraints, every expression object in a DeclContext must be unique.
  CaptureVars CopyTransformer(*this);
  auto MakeNumIterations = [&CopyTransformer, &LoopHelper]() -> Expr * {
    return AssertSuccess(
        CopyTransformer.TransformExpr(LoopHelper.NumIterations));
  };

  // Inner For init-statement: auto .unroll_inner.iv = .unrolled.iv
  ExprResult LValueConv = DefaultLvalueConversion(MakeOuterRef());
  AddInitializerToDecl(InnerIVDecl, LValueConv.get(), /*DirectInit=*/false);
  StmtResult InnerInit = new (Context)
      DeclStmt(DeclGroupRef(InnerIVDecl), OrigVarLocBegin, OrigVarLocEnd);
  if (!InnerInit.isUsable())
    return StmtError();

  // Inner For cond-expression:
  // \code
  //   .unroll_inner.iv < .unrolled.iv + Factor &&
  //   .unroll_inner.iv < NumIterations
  // \endcode
  // This conjunction of two conditions allows ScalarEvolution to derive the
  // maximum trip count of the inner loop.
  ExprResult EndOfTile = BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(),
                                    BO_Add, MakeOuterRef(), MakeFactorExpr());
  if (!EndOfTile.isUsable())
    return StmtError();
  ExprResult InnerCond1 = BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(),
                                     BO_LE, MakeInnerRef(), EndOfTile.get());
  if (!InnerCond1.isUsable())
    return StmtError();
  ExprResult InnerCond2 =
      BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(), BO_LE, MakeInnerRef(),
                 MakeNumIterations());
  if (!InnerCond2.isUsable())
    return StmtError();
  ExprResult InnerCond =
      BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(), BO_LAnd,
                 InnerCond1.get(), InnerCond2.get());
  if (!InnerCond.isUsable())
    return StmtError();

  // Inner For incr-statement: ++.unroll_inner.iv
  ExprResult InnerIncr = BuildUnaryOp(CurScope, LoopHelper.Inc->getExprLoc(),
                                      UO_PreInc, MakeInnerRef());
  if (!InnerIncr.isUsable())
    return StmtError();

  // Inner For statement.
  SmallVector<Stmt *> InnerBodyStmts;
  InnerBodyStmts.append(LoopHelper.Updates.begin(), LoopHelper.Updates.end());
  InnerBodyStmts.push_back(Body);
  CompoundStmt *InnerBody = CompoundStmt::Create(
      Context, InnerBodyStmts, Body->getBeginLoc(), Body->getEndLoc());
  ForStmt *InnerFor = new (Context)
      ForStmt(Context, InnerInit.get(), InnerCond.get(), nullptr,
              InnerIncr.get(), InnerBody, LoopHelper.Init->getBeginLoc(),
              LoopHelper.Init->getBeginLoc(), LoopHelper.Inc->getEndLoc());

  // Unroll metadata for the inner loop.
  // This needs to take into account the remainder portion of the unrolled loop,
  // hence `unroll(full)` does not apply here, even though the LoopUnroll pass
  // supports multiple loop exits. Instead, unroll using a factor equivalent to
  // the maximum trip count, which will also generate a remainder loop. Just
  // `unroll(enable)` (which could have been useful if the user has not
  // specified a concrete factor; even though the outer loop cannot be
  // influenced anymore, would avoid more code bloat than necessary) will refuse
  // the loop because "Won't unroll; remainder loop could not be generated when
  // assuming runtime trip count". Even if it did work, it must not choose a
  // larger unroll factor than the maximum loop length, or it would always just
  // execute the remainder loop.
  LoopHintAttr *UnrollHintAttr =
      LoopHintAttr::CreateImplicit(Context, LoopHintAttr::UnrollCount,
                                   LoopHintAttr::Numeric, MakeFactorExpr());
  AttributedStmt *InnerUnrolled =
      AttributedStmt::Create(Context, StartLoc, {UnrollHintAttr}, InnerFor);

  // Outer For init-statement: auto .unrolled.iv = 0
  AddInitializerToDecl(
      OuterIVDecl, ActOnIntegerConstant(LoopHelper.Init->getExprLoc(), 0).get(),
      /*DirectInit=*/false);
  StmtResult OuterInit = new (Context)
      DeclStmt(DeclGroupRef(OuterIVDecl), OrigVarLocBegin, OrigVarLocEnd);
  if (!OuterInit.isUsable())
    return StmtError();

  // Outer For cond-expression: .unrolled.iv < NumIterations
  ExprResult OuterConde =
      BuildBinOp(CurScope, LoopHelper.Cond->getExprLoc(), BO_LT, MakeOuterRef(),
                 MakeNumIterations());
  if (!OuterConde.isUsable())
    return StmtError();

  // Outer For incr-statement: .unrolled.iv += Factor
  ExprResult OuterIncr =
      BuildBinOp(CurScope, LoopHelper.Inc->getExprLoc(), BO_AddAssign,
                 MakeOuterRef(), MakeFactorExpr());
  if (!OuterIncr.isUsable())
    return StmtError();

  // Outer For statement.
  ForStmt *OuterFor = new (Context)
      ForStmt(Context, OuterInit.get(), OuterConde.get(), nullptr,
              OuterIncr.get(), InnerUnrolled, LoopHelper.Init->getBeginLoc(),
              LoopHelper.Init->getBeginLoc(), LoopHelper.Inc->getEndLoc());

  return OMPUnrollDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                    NumGeneratedLoops, OuterFor,
                                    buildPreInits(Context, PreInits));
}

OMPClause *Sema::ActOnOpenMPSingleExprClause(OpenMPClauseKind Kind, Expr *Expr,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_final:
    Res = ActOnOpenMPFinalClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_num_threads:
    Res = ActOnOpenMPNumThreadsClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_safelen:
    Res = ActOnOpenMPSafelenClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_simdlen:
    Res = ActOnOpenMPSimdlenClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_allocator:
    Res = ActOnOpenMPAllocatorClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_collapse:
    Res = ActOnOpenMPCollapseClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_ordered:
    Res = ActOnOpenMPOrderedClause(StartLoc, EndLoc, LParenLoc, Expr);
    break;
  case OMPC_num_teams:
    Res = ActOnOpenMPNumTeamsClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_thread_limit:
    Res = ActOnOpenMPThreadLimitClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_priority:
    Res = ActOnOpenMPPriorityClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_grainsize:
    Res = ActOnOpenMPGrainsizeClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_num_tasks:
    Res = ActOnOpenMPNumTasksClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_hint:
    Res = ActOnOpenMPHintClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_depobj:
    Res = ActOnOpenMPDepobjClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_detach:
    Res = ActOnOpenMPDetachClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_novariants:
    Res = ActOnOpenMPNovariantsClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_nocontext:
    Res = ActOnOpenMPNocontextClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_filter:
    Res = ActOnOpenMPFilterClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_partial:
    Res = ActOnOpenMPPartialClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_align:
    Res = ActOnOpenMPAlignClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_device:
  case OMPC_if:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_task_reduction:
  case OMPC_in_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_sizes:
  case OMPC_allocate:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_compare:
  case OMPC_seq_cst:
  case OMPC_acq_rel:
  case OMPC_acquire:
  case OMPC_release:
  case OMPC_relaxed:
  case OMPC_depend:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_nogroup:
  case OMPC_dist_schedule:
  case OMPC_defaultmap:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_use_device_addr:
  case OMPC_is_device_ptr:
  case OMPC_unified_address:
  case OMPC_unified_shared_memory:
  case OMPC_reverse_offload:
  case OMPC_dynamic_allocators:
  case OMPC_atomic_default_mem_order:
  case OMPC_device_type:
  case OMPC_match:
  case OMPC_nontemporal:
  case OMPC_order:
  case OMPC_destroy:
  case OMPC_inclusive:
  case OMPC_exclusive:
  case OMPC_uses_allocators:
  case OMPC_affinity:
  case OMPC_when:
  case OMPC_bind:
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

// An OpenMP directive such as 'target parallel' has two captured regions:
// for the 'target' and 'parallel' respectively.  This function returns
// the region in which to capture expressions associated with a clause.
// A return value of OMPD_unknown signifies that the expression should not
// be captured.
static OpenMPDirectiveKind getOpenMPCaptureRegionForClause(
    OpenMPDirectiveKind DKind, OpenMPClauseKind CKind, unsigned OpenMPVersion,
    OpenMPDirectiveKind NameModifier = OMPD_unknown) {
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;
  switch (CKind) {
  case OMPC_if:
    switch (DKind) {
    case OMPD_target_parallel_for_simd:
      if (OpenMPVersion >= 50 &&
          (NameModifier == OMPD_unknown || NameModifier == OMPD_simd)) {
        CaptureRegion = OMPD_parallel;
        break;
      }
      LLVM_FALLTHROUGH;
    case OMPD_target_parallel:
    case OMPD_target_parallel_for:
      // If this clause applies to the nested 'parallel' region, capture within
      // the 'target' region, otherwise do not capture.
      if (NameModifier == OMPD_unknown || NameModifier == OMPD_parallel)
        CaptureRegion = OMPD_target;
      break;
    case OMPD_target_teams_distribute_parallel_for_simd:
      if (OpenMPVersion >= 50 &&
          (NameModifier == OMPD_unknown || NameModifier == OMPD_simd)) {
        CaptureRegion = OMPD_parallel;
        break;
      }
      LLVM_FALLTHROUGH;
    case OMPD_target_teams_distribute_parallel_for:
      // If this clause applies to the nested 'parallel' region, capture within
      // the 'teams' region, otherwise do not capture.
      if (NameModifier == OMPD_unknown || NameModifier == OMPD_parallel)
        CaptureRegion = OMPD_teams;
      break;
    case OMPD_teams_distribute_parallel_for_simd:
      if (OpenMPVersion >= 50 &&
          (NameModifier == OMPD_unknown || NameModifier == OMPD_simd)) {
        CaptureRegion = OMPD_parallel;
        break;
      }
      LLVM_FALLTHROUGH;
    case OMPD_teams_distribute_parallel_for:
      CaptureRegion = OMPD_teams;
      break;
    case OMPD_target_update:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
      CaptureRegion = OMPD_task;
      break;
    case OMPD_parallel_master_taskloop:
      if (NameModifier == OMPD_unknown || NameModifier == OMPD_taskloop)
        CaptureRegion = OMPD_parallel;
      break;
    case OMPD_parallel_master_taskloop_simd:
      if ((OpenMPVersion <= 45 && NameModifier == OMPD_unknown) ||
          NameModifier == OMPD_taskloop) {
        CaptureRegion = OMPD_parallel;
        break;
      }
      if (OpenMPVersion <= 45)
        break;
      if (NameModifier == OMPD_unknown || NameModifier == OMPD_simd)
        CaptureRegion = OMPD_taskloop;
      break;
    case OMPD_parallel_for_simd:
      if (OpenMPVersion <= 45)
        break;
      if (NameModifier == OMPD_unknown || NameModifier == OMPD_simd)
        CaptureRegion = OMPD_parallel;
      break;
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop_simd:
      if (OpenMPVersion <= 45)
        break;
      if (NameModifier == OMPD_unknown || NameModifier == OMPD_simd)
        CaptureRegion = OMPD_taskloop;
      break;
    case OMPD_distribute_parallel_for_simd:
      if (OpenMPVersion <= 45)
        break;
      if (NameModifier == OMPD_unknown || NameModifier == OMPD_simd)
        CaptureRegion = OMPD_parallel;
      break;
    case OMPD_target_simd:
      if (OpenMPVersion >= 50 &&
          (NameModifier == OMPD_unknown || NameModifier == OMPD_simd))
        CaptureRegion = OMPD_target;
      break;
    case OMPD_teams_distribute_simd:
    case OMPD_target_teams_distribute_simd:
      if (OpenMPVersion >= 50 &&
          (NameModifier == OMPD_unknown || NameModifier == OMPD_simd))
        CaptureRegion = OMPD_teams;
      break;
    case OMPD_cancel:
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_parallel_for:
    case OMPD_target:
    case OMPD_target_teams:
    case OMPD_target_teams_distribute:
    case OMPD_distribute_parallel_for:
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_master_taskloop:
    case OMPD_target_data:
    case OMPD_simd:
    case OMPD_for_simd:
    case OMPD_distribute_simd:
      // Do not capture if-clause expressions.
      break;
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_teams:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_for:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_distribute:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_teams_distribute:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with if-clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_num_threads:
    switch (DKind) {
    case OMPD_target_parallel:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
      CaptureRegion = OMPD_target;
      break;
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
      CaptureRegion = OMPD_teams;
      break;
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_parallel_for:
    case OMPD_parallel_for_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
      // Do not capture num_threads-clause expressions.
      break;
    case OMPD_target_data:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
    case OMPD_target_update:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_teams:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_cancel:
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_teams:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_for:
    case OMPD_for_simd:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_distribute:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_distribute_simd:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with num_threads-clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_num_teams:
    switch (DKind) {
    case OMPD_target_teams:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
      CaptureRegion = OMPD_target;
      break;
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_teams:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
      // Do not capture num_teams-clause expressions.
      break;
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_target_data:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
    case OMPD_target_update:
    case OMPD_cancel:
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_parallel_for:
    case OMPD_parallel_for_simd:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_parallel:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_for:
    case OMPD_for_simd:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_distribute:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_distribute_simd:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with num_teams-clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_thread_limit:
    switch (DKind) {
    case OMPD_target_teams:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
      CaptureRegion = OMPD_target;
      break;
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_teams:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
      // Do not capture thread_limit-clause expressions.
      break;
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_target_data:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
    case OMPD_target_update:
    case OMPD_cancel:
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_parallel_for:
    case OMPD_parallel_for_simd:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_parallel:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_for:
    case OMPD_for_simd:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_distribute:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_distribute_simd:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with thread_limit-clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_schedule:
    switch (DKind) {
    case OMPD_parallel_for:
    case OMPD_parallel_for_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
      CaptureRegion = OMPD_parallel;
      break;
    case OMPD_for:
    case OMPD_for_simd:
      // Do not capture schedule-clause expressions.
      break;
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_target_data:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
    case OMPD_target_update:
    case OMPD_teams:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_parallel:
    case OMPD_cancel:
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_distribute:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_distribute_simd:
    case OMPD_target_teams:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with schedule clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_dist_schedule:
    switch (DKind) {
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
      CaptureRegion = OMPD_teams;
      break;
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_distribute:
    case OMPD_distribute_simd:
      // Do not capture dist_schedule-clause expressions.
      break;
    case OMPD_parallel_for:
    case OMPD_parallel_for_simd:
    case OMPD_target_parallel_for_simd:
    case OMPD_target_parallel_for:
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_target_data:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
    case OMPD_target_update:
    case OMPD_teams:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_parallel:
    case OMPD_cancel:
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_for:
    case OMPD_for_simd:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_target_teams:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with dist_schedule clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_device:
    switch (DKind) {
    case OMPD_target_update:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_teams:
    case OMPD_target_parallel:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
    case OMPD_dispatch:
      CaptureRegion = OMPD_task;
      break;
    case OMPD_target_data:
    case OMPD_interop:
      // Do not capture device-clause expressions.
      break;
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_teams:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_cancel:
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_parallel_for:
    case OMPD_parallel_for_simd:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_for:
    case OMPD_for_simd:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_distribute:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_distribute_simd:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with device-clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_grainsize:
  case OMPC_num_tasks:
  case OMPC_final:
  case OMPC_priority:
    switch (DKind) {
    case OMPD_task:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
      break;
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
      CaptureRegion = OMPD_parallel;
      break;
    case OMPD_target_update:
    case OMPD_target_enter_data:
    case OMPD_target_exit_data:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_teams:
    case OMPD_target_parallel:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
    case OMPD_target_data:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_teams:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_cancel:
    case OMPD_parallel:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_parallel_for:
    case OMPD_parallel_for_simd:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_cancellation_point:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_loop:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_for:
    case OMPD_for_simd:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_masked:
    case OMPD_critical:
    case OMPD_taskgroup:
    case OMPD_distribute:
    case OMPD_ordered:
    case OMPD_atomic:
    case OMPD_distribute_simd:
    case OMPD_requires:
    case OMPD_metadirective:
      llvm_unreachable("Unexpected OpenMP directive with grainsize-clause");
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown OpenMP directive");
    }
    break;
  case OMPC_novariants:
  case OMPC_nocontext:
    switch (DKind) {
    case OMPD_dispatch:
      CaptureRegion = OMPD_task;
      break;
    default:
      llvm_unreachable("Unexpected OpenMP directive");
    }
    break;
  case OMPC_filter:
    // Do not capture filter-clause expressions.
    break;
  case OMPC_when:
    if (DKind == OMPD_metadirective) {
      CaptureRegion = OMPD_metadirective;
    } else if (DKind == OMPD_unknown) {
      llvm_unreachable("Unknown OpenMP directive");
    } else {
      llvm_unreachable("Unexpected OpenMP directive with when clause");
    }
    break;
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_reduction:
  case OMPC_task_reduction:
  case OMPC_in_reduction:
  case OMPC_linear:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_sizes:
  case OMPC_allocator:
  case OMPC_collapse:
  case OMPC_private:
  case OMPC_shared:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_allocate:
  case OMPC_flush:
  case OMPC_depobj:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_compare:
  case OMPC_seq_cst:
  case OMPC_acq_rel:
  case OMPC_acquire:
  case OMPC_release:
  case OMPC_relaxed:
  case OMPC_depend:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_nogroup:
  case OMPC_hint:
  case OMPC_defaultmap:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_use_device_addr:
  case OMPC_is_device_ptr:
  case OMPC_unified_address:
  case OMPC_unified_shared_memory:
  case OMPC_reverse_offload:
  case OMPC_dynamic_allocators:
  case OMPC_atomic_default_mem_order:
  case OMPC_device_type:
  case OMPC_match:
  case OMPC_nontemporal:
  case OMPC_order:
  case OMPC_destroy:
  case OMPC_detach:
  case OMPC_inclusive:
  case OMPC_exclusive:
  case OMPC_uses_allocators:
  case OMPC_affinity:
  case OMPC_bind:
  default:
    llvm_unreachable("Unexpected OpenMP clause.");
  }
  return CaptureRegion;
}

OMPClause *Sema::ActOnOpenMPIfClause(OpenMPDirectiveKind NameModifier,
                                     Expr *Condition, SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation NameModifierLoc,
                                     SourceLocation ColonLoc,
                                     SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  Stmt *HelperValStmt = nullptr;
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = Val.get();

    OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
    CaptureRegion = getOpenMPCaptureRegionForClause(
        DKind, OMPC_if, LangOpts.OpenMP, NameModifier);
    if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
      ValExpr = MakeFullExpr(ValExpr).get();
      llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
      ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
      HelperValStmt = buildPreInits(Context, Captures);
    }
  }

  return new (Context)
      OMPIfClause(NameModifier, ValExpr, HelperValStmt, CaptureRegion, StartLoc,
                  LParenLoc, NameModifierLoc, ColonLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPFinalClause(Expr *Condition,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  Stmt *HelperValStmt = nullptr;
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = MakeFullExpr(Val.get()).get();

    OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
    CaptureRegion =
        getOpenMPCaptureRegionForClause(DKind, OMPC_final, LangOpts.OpenMP);
    if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
      ValExpr = MakeFullExpr(ValExpr).get();
      llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
      ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
      HelperValStmt = buildPreInits(Context, Captures);
    }
  }

  return new (Context) OMPFinalClause(ValExpr, HelperValStmt, CaptureRegion,
                                      StartLoc, LParenLoc, EndLoc);
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

static bool
isNonNegativeIntegerValue(Expr *&ValExpr, Sema &SemaRef, OpenMPClauseKind CKind,
                          bool StrictlyPositive, bool BuildCapture = false,
                          OpenMPDirectiveKind DKind = OMPD_unknown,
                          OpenMPDirectiveKind *CaptureRegion = nullptr,
                          Stmt **HelperValStmt = nullptr) {
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        SemaRef.PerformOpenMPImplicitIntegerConversion(Loc, ValExpr);
    if (Value.isInvalid())
      return false;

    ValExpr = Value.get();
    // The expression must evaluate to a non-negative integer value.
    if (Optional<llvm::APSInt> Result =
            ValExpr->getIntegerConstantExpr(SemaRef.Context)) {
      if (Result->isSigned() &&
          !((!StrictlyPositive && Result->isNonNegative()) ||
            (StrictlyPositive && Result->isStrictlyPositive()))) {
        SemaRef.Diag(Loc, diag::err_omp_negative_expression_in_clause)
            << getOpenMPClauseName(CKind) << (StrictlyPositive ? 1 : 0)
            << ValExpr->getSourceRange();
        return false;
      }
    }
    if (!BuildCapture)
      return true;
    *CaptureRegion =
        getOpenMPCaptureRegionForClause(DKind, CKind, SemaRef.LangOpts.OpenMP);
    if (*CaptureRegion != OMPD_unknown &&
        !SemaRef.CurContext->isDependentContext()) {
      ValExpr = SemaRef.MakeFullExpr(ValExpr).get();
      llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
      ValExpr = tryBuildCapture(SemaRef, ValExpr, Captures).get();
      *HelperValStmt = buildPreInits(SemaRef.Context, Captures);
    }
  }
  return true;
}

OMPClause *Sema::ActOnOpenMPNumThreadsClause(Expr *NumThreads,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  Expr *ValExpr = NumThreads;
  Stmt *HelperValStmt = nullptr;

  // OpenMP [2.5, Restrictions]
  //  The num_threads expression must evaluate to a positive integer value.
  if (!isNonNegativeIntegerValue(ValExpr, *this, OMPC_num_threads,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
  OpenMPDirectiveKind CaptureRegion =
      getOpenMPCaptureRegionForClause(DKind, OMPC_num_threads, LangOpts.OpenMP);
  if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
    ValExpr = MakeFullExpr(ValExpr).get();
    llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
    ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
    HelperValStmt = buildPreInits(Context, Captures);
  }

  return new (Context) OMPNumThreadsClause(
      ValExpr, HelperValStmt, CaptureRegion, StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::VerifyPositiveIntegerConstantInClause(Expr *E,
                                                       OpenMPClauseKind CKind,
                                                       bool StrictlyPositive,
                                                       bool SuppressExprDiags) {
  if (!E)
    return ExprError();
  if (E->isValueDependent() || E->isTypeDependent() ||
      E->isInstantiationDependent() || E->containsUnexpandedParameterPack())
    return E;

  llvm::APSInt Result;
  ExprResult ICE;
  if (SuppressExprDiags) {
    // Use a custom diagnoser that suppresses 'note' diagnostics about the
    // expression.
    struct SuppressedDiagnoser : public Sema::VerifyICEDiagnoser {
      SuppressedDiagnoser() : VerifyICEDiagnoser(/*Suppress=*/true) {}
      Sema::SemaDiagnosticBuilder diagnoseNotICE(Sema &S,
                                                 SourceLocation Loc) override {
        llvm_unreachable("Diagnostic suppressed");
      }
    } Diagnoser;
    ICE = VerifyIntegerConstantExpression(E, &Result, Diagnoser, AllowFold);
  } else {
    ICE = VerifyIntegerConstantExpression(E, &Result, /*FIXME*/ AllowFold);
  }
  if (ICE.isInvalid())
    return ExprError();

  if ((StrictlyPositive && !Result.isStrictlyPositive()) ||
      (!StrictlyPositive && !Result.isNonNegative())) {
    Diag(E->getExprLoc(), diag::err_omp_negative_expression_in_clause)
        << getOpenMPClauseName(CKind) << (StrictlyPositive ? 1 : 0)
        << E->getSourceRange();
    return ExprError();
  }
  if ((CKind == OMPC_aligned || CKind == OMPC_align) && !Result.isPowerOf2()) {
    Diag(E->getExprLoc(), diag::warn_omp_alignment_not_power_of_two)
        << E->getSourceRange();
    return ExprError();
  }
  if (CKind == OMPC_collapse && DSAStack->getAssociatedLoops() == 1)
    DSAStack->setAssociatedLoops(Result.getExtValue());
  else if (CKind == OMPC_ordered)
    DSAStack->setAssociatedLoops(Result.getExtValue());
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

OMPClause *Sema::ActOnOpenMPSimdlenClause(Expr *Len, SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  // OpenMP [2.8.1, simd construct, Description]
  // The parameter of the simdlen clause must be a constant
  // positive integer expression.
  ExprResult Simdlen = VerifyPositiveIntegerConstantInClause(Len, OMPC_simdlen);
  if (Simdlen.isInvalid())
    return nullptr;
  return new (Context)
      OMPSimdlenClause(Simdlen.get(), StartLoc, LParenLoc, EndLoc);
}

/// Tries to find omp_allocator_handle_t type.
static bool findOMPAllocatorHandleT(Sema &S, SourceLocation Loc,
                                    DSAStackTy *Stack) {
  QualType OMPAllocatorHandleT = Stack->getOMPAllocatorHandleT();
  if (!OMPAllocatorHandleT.isNull())
    return true;
  // Build the predefined allocator expressions.
  bool ErrorFound = false;
  for (int I = 0; I < OMPAllocateDeclAttr::OMPUserDefinedMemAlloc; ++I) {
    auto AllocatorKind = static_cast<OMPAllocateDeclAttr::AllocatorTypeTy>(I);
    StringRef Allocator =
        OMPAllocateDeclAttr::ConvertAllocatorTypeTyToStr(AllocatorKind);
    DeclarationName AllocatorName = &S.getASTContext().Idents.get(Allocator);
    auto *VD = dyn_cast_or_null<ValueDecl>(
        S.LookupSingleName(S.TUScope, AllocatorName, Loc, Sema::LookupAnyName));
    if (!VD) {
      ErrorFound = true;
      break;
    }
    QualType AllocatorType =
        VD->getType().getNonLValueExprType(S.getASTContext());
    ExprResult Res = S.BuildDeclRefExpr(VD, AllocatorType, VK_LValue, Loc);
    if (!Res.isUsable()) {
      ErrorFound = true;
      break;
    }
    if (OMPAllocatorHandleT.isNull())
      OMPAllocatorHandleT = AllocatorType;
    if (!S.getASTContext().hasSameType(OMPAllocatorHandleT, AllocatorType)) {
      ErrorFound = true;
      break;
    }
    Stack->setAllocator(AllocatorKind, Res.get());
  }
  if (ErrorFound) {
    S.Diag(Loc, diag::err_omp_implied_type_not_found)
        << "omp_allocator_handle_t";
    return false;
  }
  OMPAllocatorHandleT.addConst();
  Stack->setOMPAllocatorHandleT(OMPAllocatorHandleT);
  return true;
}

OMPClause *Sema::ActOnOpenMPAllocatorClause(Expr *A, SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  // OpenMP [2.11.3, allocate Directive, Description]
  // allocator is an expression of omp_allocator_handle_t type.
  if (!findOMPAllocatorHandleT(*this, A->getExprLoc(), DSAStack))
    return nullptr;

  ExprResult Allocator = DefaultLvalueConversion(A);
  if (Allocator.isInvalid())
    return nullptr;
  Allocator = PerformImplicitConversion(Allocator.get(),
                                        DSAStack->getOMPAllocatorHandleT(),
                                        Sema::AA_Initializing,
                                        /*AllowExplicit=*/true);
  if (Allocator.isInvalid())
    return nullptr;
  return new (Context)
      OMPAllocatorClause(Allocator.get(), StartLoc, LParenLoc, EndLoc);
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

OMPClause *Sema::ActOnOpenMPOrderedClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc,
                                          SourceLocation LParenLoc,
                                          Expr *NumForLoops) {
  // OpenMP [2.7.1, loop construct, Description]
  // OpenMP [2.8.1, simd construct, Description]
  // OpenMP [2.9.6, distribute construct, Description]
  // The parameter of the ordered clause must be a constant
  // positive integer expression if any.
  if (NumForLoops && LParenLoc.isValid()) {
    ExprResult NumForLoopsResult =
        VerifyPositiveIntegerConstantInClause(NumForLoops, OMPC_ordered);
    if (NumForLoopsResult.isInvalid())
      return nullptr;
    NumForLoops = NumForLoopsResult.get();
  } else {
    NumForLoops = nullptr;
  }
  auto *Clause = OMPOrderedClause::Create(
      Context, NumForLoops, NumForLoops ? DSAStack->getAssociatedLoops() : 0,
      StartLoc, LParenLoc, EndLoc);
  DSAStack->setOrderedRegion(/*IsOrdered=*/true, NumForLoops, Clause);
  return Clause;
}

OMPClause *Sema::ActOnOpenMPSimpleClause(
    OpenMPClauseKind Kind, unsigned Argument, SourceLocation ArgumentLoc,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_default:
    Res = ActOnOpenMPDefaultClause(static_cast<DefaultKind>(Argument),
                                   ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_proc_bind:
    Res = ActOnOpenMPProcBindClause(static_cast<ProcBindKind>(Argument),
                                    ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_atomic_default_mem_order:
    Res = ActOnOpenMPAtomicDefaultMemOrderClause(
        static_cast<OpenMPAtomicDefaultMemOrderClauseKind>(Argument),
        ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_order:
    Res = ActOnOpenMPOrderClause(static_cast<OpenMPOrderClauseKind>(Argument),
                                 ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_update:
    Res = ActOnOpenMPUpdateClause(static_cast<OpenMPDependClauseKind>(Argument),
                                  ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_bind:
    Res = ActOnOpenMPBindClause(static_cast<OpenMPBindClauseKind>(Argument),
                                ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_sizes:
  case OMPC_allocator:
  case OMPC_collapse:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_task_reduction:
  case OMPC_in_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_allocate:
  case OMPC_flush:
  case OMPC_depobj:
  case OMPC_read:
  case OMPC_write:
  case OMPC_capture:
  case OMPC_compare:
  case OMPC_seq_cst:
  case OMPC_acq_rel:
  case OMPC_acquire:
  case OMPC_release:
  case OMPC_relaxed:
  case OMPC_depend:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_dist_schedule:
  case OMPC_defaultmap:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_use_device_addr:
  case OMPC_is_device_ptr:
  case OMPC_unified_address:
  case OMPC_unified_shared_memory:
  case OMPC_reverse_offload:
  case OMPC_dynamic_allocators:
  case OMPC_device_type:
  case OMPC_match:
  case OMPC_nontemporal:
  case OMPC_destroy:
  case OMPC_novariants:
  case OMPC_nocontext:
  case OMPC_detach:
  case OMPC_inclusive:
  case OMPC_exclusive:
  case OMPC_uses_allocators:
  case OMPC_affinity:
  case OMPC_when:
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

static std::string
getListOfPossibleValues(OpenMPClauseKind K, unsigned First, unsigned Last,
                        ArrayRef<unsigned> Exclude = llvm::None) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  unsigned Skipped = Exclude.size();
  auto S = Exclude.begin(), E = Exclude.end();
  for (unsigned I = First; I < Last; ++I) {
    if (std::find(S, E, I) != E) {
      --Skipped;
      continue;
    }
    Out << "'" << getOpenMPSimpleClauseTypeName(K, I) << "'";
    if (I + Skipped + 2 == Last)
      Out << " or ";
    else if (I + Skipped + 1 != Last)
      Out << ", ";
  }
  return std::string(Out.str());
}

OMPClause *Sema::ActOnOpenMPDefaultClause(DefaultKind Kind,
                                          SourceLocation KindKwLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  if (Kind == OMP_DEFAULT_unknown) {
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_default, /*First=*/0,
                                   /*Last=*/unsigned(OMP_DEFAULT_unknown))
        << getOpenMPClauseName(OMPC_default);
    return nullptr;
  }

  switch (Kind) {
  case OMP_DEFAULT_none:
    DSAStack->setDefaultDSANone(KindKwLoc);
    break;
  case OMP_DEFAULT_shared:
    DSAStack->setDefaultDSAShared(KindKwLoc);
    break;
  case OMP_DEFAULT_firstprivate:
    DSAStack->setDefaultDSAFirstPrivate(KindKwLoc);
    break;
  default:
    llvm_unreachable("DSA unexpected in OpenMP default clause");
  }

  return new (Context)
      OMPDefaultClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPProcBindClause(ProcBindKind Kind,
                                           SourceLocation KindKwLoc,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  if (Kind == OMP_PROC_BIND_unknown) {
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_proc_bind,
                                   /*First=*/unsigned(OMP_PROC_BIND_master),
                                   /*Last=*/
                                   unsigned(LangOpts.OpenMP > 50
                                                ? OMP_PROC_BIND_primary
                                                : OMP_PROC_BIND_spread) +
                                       1)
        << getOpenMPClauseName(OMPC_proc_bind);
    return nullptr;
  }
  if (Kind == OMP_PROC_BIND_primary && LangOpts.OpenMP < 51)
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_proc_bind,
                                   /*First=*/unsigned(OMP_PROC_BIND_master),
                                   /*Last=*/
                                   unsigned(OMP_PROC_BIND_spread) + 1)
        << getOpenMPClauseName(OMPC_proc_bind);
  return new (Context)
      OMPProcBindClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPAtomicDefaultMemOrderClause(
    OpenMPAtomicDefaultMemOrderClauseKind Kind, SourceLocation KindKwLoc,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc) {
  if (Kind == OMPC_ATOMIC_DEFAULT_MEM_ORDER_unknown) {
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(
               OMPC_atomic_default_mem_order, /*First=*/0,
               /*Last=*/OMPC_ATOMIC_DEFAULT_MEM_ORDER_unknown)
        << getOpenMPClauseName(OMPC_atomic_default_mem_order);
    return nullptr;
  }
  return new (Context) OMPAtomicDefaultMemOrderClause(Kind, KindKwLoc, StartLoc,
                                                      LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPOrderClause(OpenMPOrderClauseKind Kind,
                                        SourceLocation KindKwLoc,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc) {
  if (Kind == OMPC_ORDER_unknown) {
    static_assert(OMPC_ORDER_unknown > 0,
                  "OMPC_ORDER_unknown not greater than 0");
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_order, /*First=*/0,
                                   /*Last=*/OMPC_ORDER_unknown)
        << getOpenMPClauseName(OMPC_order);
    return nullptr;
  }
  return new (Context)
      OMPOrderClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUpdateClause(OpenMPDependClauseKind Kind,
                                         SourceLocation KindKwLoc,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  if (Kind == OMPC_DEPEND_unknown || Kind == OMPC_DEPEND_source ||
      Kind == OMPC_DEPEND_sink || Kind == OMPC_DEPEND_depobj) {
    unsigned Except[] = {OMPC_DEPEND_source, OMPC_DEPEND_sink,
                         OMPC_DEPEND_depobj};
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_depend, /*First=*/0,
                                   /*Last=*/OMPC_DEPEND_unknown, Except)
        << getOpenMPClauseName(OMPC_update);
    return nullptr;
  }
  return OMPUpdateClause::Create(Context, StartLoc, LParenLoc, KindKwLoc, Kind,
                                 EndLoc);
}

OMPClause *Sema::ActOnOpenMPSizesClause(ArrayRef<Expr *> SizeExprs,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc) {
  for (Expr *SizeExpr : SizeExprs) {
    ExprResult NumForLoopsResult = VerifyPositiveIntegerConstantInClause(
        SizeExpr, OMPC_sizes, /*StrictlyPositive=*/true);
    if (!NumForLoopsResult.isUsable())
      return nullptr;
  }

  DSAStack->setAssociatedLoops(SizeExprs.size());
  return OMPSizesClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                SizeExprs);
}

OMPClause *Sema::ActOnOpenMPFullClause(SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  return OMPFullClause::Create(Context, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPPartialClause(Expr *FactorExpr,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  if (FactorExpr) {
    // If an argument is specified, it must be a constant (or an unevaluated
    // template expression).
    ExprResult FactorResult = VerifyPositiveIntegerConstantInClause(
        FactorExpr, OMPC_partial, /*StrictlyPositive=*/true);
    if (FactorResult.isInvalid())
      return nullptr;
    FactorExpr = FactorResult.get();
  }

  return OMPPartialClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                  FactorExpr);
}

OMPClause *Sema::ActOnOpenMPAlignClause(Expr *A, SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc) {
  ExprResult AlignVal;
  AlignVal = VerifyPositiveIntegerConstantInClause(A, OMPC_align);
  if (AlignVal.isInvalid())
    return nullptr;
  return OMPAlignClause::Create(Context, AlignVal.get(), StartLoc, LParenLoc,
                                EndLoc);
}

OMPClause *Sema::ActOnOpenMPSingleExprWithArgClause(
    OpenMPClauseKind Kind, ArrayRef<unsigned> Argument, Expr *Expr,
    SourceLocation StartLoc, SourceLocation LParenLoc,
    ArrayRef<SourceLocation> ArgumentLoc, SourceLocation DelimLoc,
    SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_schedule:
    enum { Modifier1, Modifier2, ScheduleKind, NumberOfElements };
    assert(Argument.size() == NumberOfElements &&
           ArgumentLoc.size() == NumberOfElements);
    Res = ActOnOpenMPScheduleClause(
        static_cast<OpenMPScheduleClauseModifier>(Argument[Modifier1]),
        static_cast<OpenMPScheduleClauseModifier>(Argument[Modifier2]),
        static_cast<OpenMPScheduleClauseKind>(Argument[ScheduleKind]), Expr,
        StartLoc, LParenLoc, ArgumentLoc[Modifier1], ArgumentLoc[Modifier2],
        ArgumentLoc[ScheduleKind], DelimLoc, EndLoc);
    break;
  case OMPC_if:
    assert(Argument.size() == 1 && ArgumentLoc.size() == 1);
    Res = ActOnOpenMPIfClause(static_cast<OpenMPDirectiveKind>(Argument.back()),
                              Expr, StartLoc, LParenLoc, ArgumentLoc.back(),
                              DelimLoc, EndLoc);
    break;
  case OMPC_dist_schedule:
    Res = ActOnOpenMPDistScheduleClause(
        static_cast<OpenMPDistScheduleClauseKind>(Argument.back()), Expr,
        StartLoc, LParenLoc, ArgumentLoc.back(), DelimLoc, EndLoc);
    break;
  case OMPC_defaultmap:
    enum { Modifier, DefaultmapKind };
    Res = ActOnOpenMPDefaultmapClause(
        static_cast<OpenMPDefaultmapClauseModifier>(Argument[Modifier]),
        static_cast<OpenMPDefaultmapClauseKind>(Argument[DefaultmapKind]),
        StartLoc, LParenLoc, ArgumentLoc[Modifier], ArgumentLoc[DefaultmapKind],
        EndLoc);
    break;
  case OMPC_device:
    assert(Argument.size() == 1 && ArgumentLoc.size() == 1);
    Res = ActOnOpenMPDeviceClause(
        static_cast<OpenMPDeviceClauseModifier>(Argument.back()), Expr,
        StartLoc, LParenLoc, ArgumentLoc.back(), EndLoc);
    break;
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_sizes:
  case OMPC_allocator:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_task_reduction:
  case OMPC_in_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_allocate:
  case OMPC_flush:
  case OMPC_depobj:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_compare:
  case OMPC_seq_cst:
  case OMPC_acq_rel:
  case OMPC_acquire:
  case OMPC_release:
  case OMPC_relaxed:
  case OMPC_depend:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_use_device_addr:
  case OMPC_is_device_ptr:
  case OMPC_unified_address:
  case OMPC_unified_shared_memory:
  case OMPC_reverse_offload:
  case OMPC_dynamic_allocators:
  case OMPC_atomic_default_mem_order:
  case OMPC_device_type:
  case OMPC_match:
  case OMPC_nontemporal:
  case OMPC_order:
  case OMPC_destroy:
  case OMPC_novariants:
  case OMPC_nocontext:
  case OMPC_detach:
  case OMPC_inclusive:
  case OMPC_exclusive:
  case OMPC_uses_allocators:
  case OMPC_affinity:
  case OMPC_when:
  case OMPC_bind:
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

static bool checkScheduleModifiers(Sema &S, OpenMPScheduleClauseModifier M1,
                                   OpenMPScheduleClauseModifier M2,
                                   SourceLocation M1Loc, SourceLocation M2Loc) {
  if (M1 == OMPC_SCHEDULE_MODIFIER_unknown && M1Loc.isValid()) {
    SmallVector<unsigned, 2> Excluded;
    if (M2 != OMPC_SCHEDULE_MODIFIER_unknown)
      Excluded.push_back(M2);
    if (M2 == OMPC_SCHEDULE_MODIFIER_nonmonotonic)
      Excluded.push_back(OMPC_SCHEDULE_MODIFIER_monotonic);
    if (M2 == OMPC_SCHEDULE_MODIFIER_monotonic)
      Excluded.push_back(OMPC_SCHEDULE_MODIFIER_nonmonotonic);
    S.Diag(M1Loc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_schedule,
                                   /*First=*/OMPC_SCHEDULE_MODIFIER_unknown + 1,
                                   /*Last=*/OMPC_SCHEDULE_MODIFIER_last,
                                   Excluded)
        << getOpenMPClauseName(OMPC_schedule);
    return true;
  }
  return false;
}

OMPClause *Sema::ActOnOpenMPScheduleClause(
    OpenMPScheduleClauseModifier M1, OpenMPScheduleClauseModifier M2,
    OpenMPScheduleClauseKind Kind, Expr *ChunkSize, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation M1Loc, SourceLocation M2Loc,
    SourceLocation KindLoc, SourceLocation CommaLoc, SourceLocation EndLoc) {
  if (checkScheduleModifiers(*this, M1, M2, M1Loc, M2Loc) ||
      checkScheduleModifiers(*this, M2, M1, M2Loc, M1Loc))
    return nullptr;
  // OpenMP, 2.7.1, Loop Construct, Restrictions
  // Either the monotonic modifier or the nonmonotonic modifier can be specified
  // but not both.
  if ((M1 == M2 && M1 != OMPC_SCHEDULE_MODIFIER_unknown) ||
      (M1 == OMPC_SCHEDULE_MODIFIER_monotonic &&
       M2 == OMPC_SCHEDULE_MODIFIER_nonmonotonic) ||
      (M1 == OMPC_SCHEDULE_MODIFIER_nonmonotonic &&
       M2 == OMPC_SCHEDULE_MODIFIER_monotonic)) {
    Diag(M2Loc, diag::err_omp_unexpected_schedule_modifier)
        << getOpenMPSimpleClauseTypeName(OMPC_schedule, M2)
        << getOpenMPSimpleClauseTypeName(OMPC_schedule, M1);
    return nullptr;
  }
  if (Kind == OMPC_SCHEDULE_unknown) {
    std::string Values;
    if (M1Loc.isInvalid() && M2Loc.isInvalid()) {
      unsigned Exclude[] = {OMPC_SCHEDULE_unknown};
      Values = getListOfPossibleValues(OMPC_schedule, /*First=*/0,
                                       /*Last=*/OMPC_SCHEDULE_MODIFIER_last,
                                       Exclude);
    } else {
      Values = getListOfPossibleValues(OMPC_schedule, /*First=*/0,
                                       /*Last=*/OMPC_SCHEDULE_unknown);
    }
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_schedule);
    return nullptr;
  }
  // OpenMP, 2.7.1, Loop Construct, Restrictions
  // The nonmonotonic modifier can only be specified with schedule(dynamic) or
  // schedule(guided).
  // OpenMP 5.0 does not have this restriction.
  if (LangOpts.OpenMP < 50 &&
      (M1 == OMPC_SCHEDULE_MODIFIER_nonmonotonic ||
       M2 == OMPC_SCHEDULE_MODIFIER_nonmonotonic) &&
      Kind != OMPC_SCHEDULE_dynamic && Kind != OMPC_SCHEDULE_guided) {
    Diag(M1 == OMPC_SCHEDULE_MODIFIER_nonmonotonic ? M1Loc : M2Loc,
         diag::err_omp_schedule_nonmonotonic_static);
    return nullptr;
  }
  Expr *ValExpr = ChunkSize;
  Stmt *HelperValStmt = nullptr;
  if (ChunkSize) {
    if (!ChunkSize->isValueDependent() && !ChunkSize->isTypeDependent() &&
        !ChunkSize->isInstantiationDependent() &&
        !ChunkSize->containsUnexpandedParameterPack()) {
      SourceLocation ChunkSizeLoc = ChunkSize->getBeginLoc();
      ExprResult Val =
          PerformOpenMPImplicitIntegerConversion(ChunkSizeLoc, ChunkSize);
      if (Val.isInvalid())
        return nullptr;

      ValExpr = Val.get();

      // OpenMP [2.7.1, Restrictions]
      //  chunk_size must be a loop invariant integer expression with a positive
      //  value.
      if (Optional<llvm::APSInt> Result =
              ValExpr->getIntegerConstantExpr(Context)) {
        if (Result->isSigned() && !Result->isStrictlyPositive()) {
          Diag(ChunkSizeLoc, diag::err_omp_negative_expression_in_clause)
              << "schedule" << 1 << ChunkSize->getSourceRange();
          return nullptr;
        }
      } else if (getOpenMPCaptureRegionForClause(
                     DSAStack->getCurrentDirective(), OMPC_schedule,
                     LangOpts.OpenMP) != OMPD_unknown &&
                 !CurContext->isDependentContext()) {
        ValExpr = MakeFullExpr(ValExpr).get();
        llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
        ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
        HelperValStmt = buildPreInits(Context, Captures);
      }
    }
  }

  return new (Context)
      OMPScheduleClause(StartLoc, LParenLoc, KindLoc, CommaLoc, EndLoc, Kind,
                        ValExpr, HelperValStmt, M1, M1Loc, M2, M2Loc);
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
  case OMPC_untied:
    Res = ActOnOpenMPUntiedClause(StartLoc, EndLoc);
    break;
  case OMPC_mergeable:
    Res = ActOnOpenMPMergeableClause(StartLoc, EndLoc);
    break;
  case OMPC_read:
    Res = ActOnOpenMPReadClause(StartLoc, EndLoc);
    break;
  case OMPC_write:
    Res = ActOnOpenMPWriteClause(StartLoc, EndLoc);
    break;
  case OMPC_update:
    Res = ActOnOpenMPUpdateClause(StartLoc, EndLoc);
    break;
  case OMPC_capture:
    Res = ActOnOpenMPCaptureClause(StartLoc, EndLoc);
    break;
  case OMPC_compare:
    Res = ActOnOpenMPCompareClause(StartLoc, EndLoc);
    break;
  case OMPC_seq_cst:
    Res = ActOnOpenMPSeqCstClause(StartLoc, EndLoc);
    break;
  case OMPC_acq_rel:
    Res = ActOnOpenMPAcqRelClause(StartLoc, EndLoc);
    break;
  case OMPC_acquire:
    Res = ActOnOpenMPAcquireClause(StartLoc, EndLoc);
    break;
  case OMPC_release:
    Res = ActOnOpenMPReleaseClause(StartLoc, EndLoc);
    break;
  case OMPC_relaxed:
    Res = ActOnOpenMPRelaxedClause(StartLoc, EndLoc);
    break;
  case OMPC_threads:
    Res = ActOnOpenMPThreadsClause(StartLoc, EndLoc);
    break;
  case OMPC_simd:
    Res = ActOnOpenMPSIMDClause(StartLoc, EndLoc);
    break;
  case OMPC_nogroup:
    Res = ActOnOpenMPNogroupClause(StartLoc, EndLoc);
    break;
  case OMPC_unified_address:
    Res = ActOnOpenMPUnifiedAddressClause(StartLoc, EndLoc);
    break;
  case OMPC_unified_shared_memory:
    Res = ActOnOpenMPUnifiedSharedMemoryClause(StartLoc, EndLoc);
    break;
  case OMPC_reverse_offload:
    Res = ActOnOpenMPReverseOffloadClause(StartLoc, EndLoc);
    break;
  case OMPC_dynamic_allocators:
    Res = ActOnOpenMPDynamicAllocatorsClause(StartLoc, EndLoc);
    break;
  case OMPC_destroy:
    Res = ActOnOpenMPDestroyClause(/*InteropVar=*/nullptr, StartLoc,
                                   /*LParenLoc=*/SourceLocation(),
                                   /*VarLoc=*/SourceLocation(), EndLoc);
    break;
  case OMPC_full:
    Res = ActOnOpenMPFullClause(StartLoc, EndLoc);
    break;
  case OMPC_partial:
    Res = ActOnOpenMPPartialClause(nullptr, StartLoc, /*LParenLoc=*/{}, EndLoc);
    break;
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_sizes:
  case OMPC_allocator:
  case OMPC_collapse:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_task_reduction:
  case OMPC_in_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_threadprivate:
  case OMPC_allocate:
  case OMPC_flush:
  case OMPC_depobj:
  case OMPC_depend:
  case OMPC_device:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_dist_schedule:
  case OMPC_defaultmap:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_use_device_addr:
  case OMPC_is_device_ptr:
  case OMPC_atomic_default_mem_order:
  case OMPC_device_type:
  case OMPC_match:
  case OMPC_nontemporal:
  case OMPC_order:
  case OMPC_novariants:
  case OMPC_nocontext:
  case OMPC_detach:
  case OMPC_inclusive:
  case OMPC_exclusive:
  case OMPC_uses_allocators:
  case OMPC_affinity:
  case OMPC_when:
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPNowaitClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  DSAStack->setNowaitRegion();
  return new (Context) OMPNowaitClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUntiedClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPUntiedClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPMergeableClause(SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  return new (Context) OMPMergeableClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPReadClause(SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  return new (Context) OMPReadClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPWriteClause(SourceLocation StartLoc,
                                        SourceLocation EndLoc) {
  return new (Context) OMPWriteClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUpdateClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return OMPUpdateClause::Create(Context, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPCaptureClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPCaptureClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPCompareClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPCompareClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSeqCstClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPSeqCstClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPAcqRelClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPAcqRelClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPAcquireClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPAcquireClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPReleaseClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPReleaseClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPRelaxedClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPRelaxedClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPThreadsClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPThreadsClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSIMDClause(SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  return new (Context) OMPSIMDClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNogroupClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPNogroupClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUnifiedAddressClause(SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {
  return new (Context) OMPUnifiedAddressClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUnifiedSharedMemoryClause(SourceLocation StartLoc,
                                                      SourceLocation EndLoc) {
  return new (Context) OMPUnifiedSharedMemoryClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPReverseOffloadClause(SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {
  return new (Context) OMPReverseOffloadClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPDynamicAllocatorsClause(SourceLocation StartLoc,
                                                    SourceLocation EndLoc) {
  return new (Context) OMPDynamicAllocatorsClause(StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPInteropDirective(ArrayRef<OMPClause *> Clauses,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {

  // OpenMP 5.1 [2.15.1, interop Construct, Restrictions]
  // At least one action-clause must appear on a directive.
  if (!hasClauses(Clauses, OMPC_init, OMPC_use, OMPC_destroy, OMPC_nowait)) {
    StringRef Expected = "'init', 'use', 'destroy', or 'nowait'";
    Diag(StartLoc, diag::err_omp_no_clause_for_directive)
        << Expected << getOpenMPDirectiveName(OMPD_interop);
    return StmtError();
  }

  // OpenMP 5.1 [2.15.1, interop Construct, Restrictions]
  // A depend clause can only appear on the directive if a targetsync
  // interop-type is present or the interop-var was initialized with
  // the targetsync interop-type.

  // If there is any 'init' clause diagnose if there is no 'init' clause with
  // interop-type of 'targetsync'. Cases involving other directives cannot be
  // diagnosed.
  const OMPDependClause *DependClause = nullptr;
  bool HasInitClause = false;
  bool IsTargetSync = false;
  for (const OMPClause *C : Clauses) {
    if (IsTargetSync)
      break;
    if (const auto *InitClause = dyn_cast<OMPInitClause>(C)) {
      HasInitClause = true;
      if (InitClause->getIsTargetSync())
        IsTargetSync = true;
    } else if (const auto *DC = dyn_cast<OMPDependClause>(C)) {
      DependClause = DC;
    }
  }
  if (DependClause && HasInitClause && !IsTargetSync) {
    Diag(DependClause->getBeginLoc(), diag::err_omp_interop_bad_depend_clause);
    return StmtError();
  }

  // OpenMP 5.1 [2.15.1, interop Construct, Restrictions]
  // Each interop-var may be specified for at most one action-clause of each
  // interop construct.
  llvm::SmallPtrSet<const VarDecl *, 4> InteropVars;
  for (const OMPClause *C : Clauses) {
    OpenMPClauseKind ClauseKind = C->getClauseKind();
    const DeclRefExpr *DRE = nullptr;
    SourceLocation VarLoc;

    if (ClauseKind == OMPC_init) {
      const auto *IC = cast<OMPInitClause>(C);
      VarLoc = IC->getVarLoc();
      DRE = dyn_cast_or_null<DeclRefExpr>(IC->getInteropVar());
    } else if (ClauseKind == OMPC_use) {
      const auto *UC = cast<OMPUseClause>(C);
      VarLoc = UC->getVarLoc();
      DRE = dyn_cast_or_null<DeclRefExpr>(UC->getInteropVar());
    } else if (ClauseKind == OMPC_destroy) {
      const auto *DC = cast<OMPDestroyClause>(C);
      VarLoc = DC->getVarLoc();
      DRE = dyn_cast_or_null<DeclRefExpr>(DC->getInteropVar());
    }

    if (!DRE)
      continue;

    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (!InteropVars.insert(VD->getCanonicalDecl()).second) {
        Diag(VarLoc, diag::err_omp_interop_var_multiple_actions) << VD;
        return StmtError();
      }
    }
  }

  return OMPInteropDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

static bool isValidInteropVariable(Sema &SemaRef, Expr *InteropVarExpr,
                                   SourceLocation VarLoc,
                                   OpenMPClauseKind Kind) {
  if (InteropVarExpr->isValueDependent() || InteropVarExpr->isTypeDependent() ||
      InteropVarExpr->isInstantiationDependent() ||
      InteropVarExpr->containsUnexpandedParameterPack())
    return true;

  const auto *DRE = dyn_cast<DeclRefExpr>(InteropVarExpr);
  if (!DRE || !isa<VarDecl>(DRE->getDecl())) {
    SemaRef.Diag(VarLoc, diag::err_omp_interop_variable_expected) << 0;
    return false;
  }

  // Interop variable should be of type omp_interop_t.
  bool HasError = false;
  QualType InteropType;
  LookupResult Result(SemaRef, &SemaRef.Context.Idents.get("omp_interop_t"),
                      VarLoc, Sema::LookupOrdinaryName);
  if (SemaRef.LookupName(Result, SemaRef.getCurScope())) {
    NamedDecl *ND = Result.getFoundDecl();
    if (const auto *TD = dyn_cast<TypeDecl>(ND)) {
      InteropType = QualType(TD->getTypeForDecl(), 0);
    } else {
      HasError = true;
    }
  } else {
    HasError = true;
  }

  if (HasError) {
    SemaRef.Diag(VarLoc, diag::err_omp_implied_type_not_found)
        << "omp_interop_t";
    return false;
  }

  QualType VarType = InteropVarExpr->getType().getUnqualifiedType();
  if (!SemaRef.Context.hasSameType(InteropType, VarType)) {
    SemaRef.Diag(VarLoc, diag::err_omp_interop_variable_wrong_type);
    return false;
  }

  // OpenMP 5.1 [2.15.1, interop Construct, Restrictions]
  // The interop-var passed to init or destroy must be non-const.
  if ((Kind == OMPC_init || Kind == OMPC_destroy) &&
      isConstNotMutableType(SemaRef, InteropVarExpr->getType())) {
    SemaRef.Diag(VarLoc, diag::err_omp_interop_variable_expected)
        << /*non-const*/ 1;
    return false;
  }
  return true;
}

OMPClause *
Sema::ActOnOpenMPInitClause(Expr *InteropVar, ArrayRef<Expr *> PrefExprs,
                            bool IsTarget, bool IsTargetSync,
                            SourceLocation StartLoc, SourceLocation LParenLoc,
                            SourceLocation VarLoc, SourceLocation EndLoc) {

  if (!isValidInteropVariable(*this, InteropVar, VarLoc, OMPC_init))
    return nullptr;

  // Check prefer_type values.  These foreign-runtime-id values are either
  // string literals or constant integral expressions.
  for (const Expr *E : PrefExprs) {
    if (E->isValueDependent() || E->isTypeDependent() ||
        E->isInstantiationDependent() || E->containsUnexpandedParameterPack())
      continue;
    if (E->isIntegerConstantExpr(Context))
      continue;
    if (isa<StringLiteral>(E))
      continue;
    Diag(E->getExprLoc(), diag::err_omp_interop_prefer_type);
    return nullptr;
  }

  return OMPInitClause::Create(Context, InteropVar, PrefExprs, IsTarget,
                               IsTargetSync, StartLoc, LParenLoc, VarLoc,
                               EndLoc);
}

OMPClause *Sema::ActOnOpenMPUseClause(Expr *InteropVar, SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation VarLoc,
                                      SourceLocation EndLoc) {

  if (!isValidInteropVariable(*this, InteropVar, VarLoc, OMPC_use))
    return nullptr;

  return new (Context)
      OMPUseClause(InteropVar, StartLoc, LParenLoc, VarLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPDestroyClause(Expr *InteropVar,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation VarLoc,
                                          SourceLocation EndLoc) {
  if (InteropVar &&
      !isValidInteropVariable(*this, InteropVar, VarLoc, OMPC_destroy))
    return nullptr;

  return new (Context)
      OMPDestroyClause(InteropVar, StartLoc, LParenLoc, VarLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNovariantsClause(Expr *Condition,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  Stmt *HelperValStmt = nullptr;
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = MakeFullExpr(Val.get()).get();

    OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
    CaptureRegion = getOpenMPCaptureRegionForClause(DKind, OMPC_novariants,
                                                    LangOpts.OpenMP);
    if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
      ValExpr = MakeFullExpr(ValExpr).get();
      llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
      ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
      HelperValStmt = buildPreInits(Context, Captures);
    }
  }

  return new (Context) OMPNovariantsClause(
      ValExpr, HelperValStmt, CaptureRegion, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNocontextClause(Expr *Condition,
                                            SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  Stmt *HelperValStmt = nullptr;
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = MakeFullExpr(Val.get()).get();

    OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
    CaptureRegion =
        getOpenMPCaptureRegionForClause(DKind, OMPC_nocontext, LangOpts.OpenMP);
    if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
      ValExpr = MakeFullExpr(ValExpr).get();
      llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
      ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
      HelperValStmt = buildPreInits(Context, Captures);
    }
  }

  return new (Context) OMPNocontextClause(ValExpr, HelperValStmt, CaptureRegion,
                                          StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPFilterClause(Expr *ThreadID,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  Expr *ValExpr = ThreadID;
  Stmt *HelperValStmt = nullptr;

  OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
  OpenMPDirectiveKind CaptureRegion =
      getOpenMPCaptureRegionForClause(DKind, OMPC_filter, LangOpts.OpenMP);
  if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
    ValExpr = MakeFullExpr(ValExpr).get();
    llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
    ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
    HelperValStmt = buildPreInits(Context, Captures);
  }

  return new (Context) OMPFilterClause(ValExpr, HelperValStmt, CaptureRegion,
                                       StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPVarListClause(
    OpenMPClauseKind Kind, ArrayRef<Expr *> VarList, Expr *DepModOrTailExpr,
    const OMPVarListLocTy &Locs, SourceLocation ColonLoc,
    CXXScopeSpec &ReductionOrMapperIdScopeSpec,
    DeclarationNameInfo &ReductionOrMapperId, int ExtraModifier,
    ArrayRef<OpenMPMapModifierKind> MapTypeModifiers,
    ArrayRef<SourceLocation> MapTypeModifiersLoc, bool IsMapTypeImplicit,
    SourceLocation ExtraModifierLoc,
    ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
    ArrayRef<SourceLocation> MotionModifiersLoc) {
  SourceLocation StartLoc = Locs.StartLoc;
  SourceLocation LParenLoc = Locs.LParenLoc;
  SourceLocation EndLoc = Locs.EndLoc;
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_private:
    Res = ActOnOpenMPPrivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_firstprivate:
    Res = ActOnOpenMPFirstprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_lastprivate:
    assert(0 <= ExtraModifier && ExtraModifier <= OMPC_LASTPRIVATE_unknown &&
           "Unexpected lastprivate modifier.");
    Res = ActOnOpenMPLastprivateClause(
        VarList, static_cast<OpenMPLastprivateModifier>(ExtraModifier),
        ExtraModifierLoc, ColonLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_shared:
    Res = ActOnOpenMPSharedClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_reduction:
    assert(0 <= ExtraModifier && ExtraModifier <= OMPC_REDUCTION_unknown &&
           "Unexpected lastprivate modifier.");
    Res = ActOnOpenMPReductionClause(
        VarList, static_cast<OpenMPReductionClauseModifier>(ExtraModifier),
        StartLoc, LParenLoc, ExtraModifierLoc, ColonLoc, EndLoc,
        ReductionOrMapperIdScopeSpec, ReductionOrMapperId);
    break;
  case OMPC_task_reduction:
    Res = ActOnOpenMPTaskReductionClause(VarList, StartLoc, LParenLoc, ColonLoc,
                                         EndLoc, ReductionOrMapperIdScopeSpec,
                                         ReductionOrMapperId);
    break;
  case OMPC_in_reduction:
    Res = ActOnOpenMPInReductionClause(VarList, StartLoc, LParenLoc, ColonLoc,
                                       EndLoc, ReductionOrMapperIdScopeSpec,
                                       ReductionOrMapperId);
    break;
  case OMPC_linear:
    assert(0 <= ExtraModifier && ExtraModifier <= OMPC_LINEAR_unknown &&
           "Unexpected linear modifier.");
    Res = ActOnOpenMPLinearClause(
        VarList, DepModOrTailExpr, StartLoc, LParenLoc,
        static_cast<OpenMPLinearClauseKind>(ExtraModifier), ExtraModifierLoc,
        ColonLoc, EndLoc);
    break;
  case OMPC_aligned:
    Res = ActOnOpenMPAlignedClause(VarList, DepModOrTailExpr, StartLoc,
                                   LParenLoc, ColonLoc, EndLoc);
    break;
  case OMPC_copyin:
    Res = ActOnOpenMPCopyinClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_copyprivate:
    Res = ActOnOpenMPCopyprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_flush:
    Res = ActOnOpenMPFlushClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_depend:
    assert(0 <= ExtraModifier && ExtraModifier <= OMPC_DEPEND_unknown &&
           "Unexpected depend modifier.");
    Res = ActOnOpenMPDependClause(
        DepModOrTailExpr, static_cast<OpenMPDependClauseKind>(ExtraModifier),
        ExtraModifierLoc, ColonLoc, VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_map:
    assert(0 <= ExtraModifier && ExtraModifier <= OMPC_MAP_unknown &&
           "Unexpected map modifier.");
    Res = ActOnOpenMPMapClause(
        MapTypeModifiers, MapTypeModifiersLoc, ReductionOrMapperIdScopeSpec,
        ReductionOrMapperId, static_cast<OpenMPMapClauseKind>(ExtraModifier),
        IsMapTypeImplicit, ExtraModifierLoc, ColonLoc, VarList, Locs);
    break;
  case OMPC_to:
    Res = ActOnOpenMPToClause(MotionModifiers, MotionModifiersLoc,
                              ReductionOrMapperIdScopeSpec, ReductionOrMapperId,
                              ColonLoc, VarList, Locs);
    break;
  case OMPC_from:
    Res = ActOnOpenMPFromClause(MotionModifiers, MotionModifiersLoc,
                                ReductionOrMapperIdScopeSpec,
                                ReductionOrMapperId, ColonLoc, VarList, Locs);
    break;
  case OMPC_use_device_ptr:
    Res = ActOnOpenMPUseDevicePtrClause(VarList, Locs);
    break;
  case OMPC_use_device_addr:
    Res = ActOnOpenMPUseDeviceAddrClause(VarList, Locs);
    break;
  case OMPC_is_device_ptr:
    Res = ActOnOpenMPIsDevicePtrClause(VarList, Locs);
    break;
  case OMPC_allocate:
    Res = ActOnOpenMPAllocateClause(DepModOrTailExpr, VarList, StartLoc,
                                    LParenLoc, ColonLoc, EndLoc);
    break;
  case OMPC_nontemporal:
    Res = ActOnOpenMPNontemporalClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_inclusive:
    Res = ActOnOpenMPInclusiveClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_exclusive:
    Res = ActOnOpenMPExclusiveClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_affinity:
    Res = ActOnOpenMPAffinityClause(StartLoc, LParenLoc, ColonLoc, EndLoc,
                                    DepModOrTailExpr, VarList);
    break;
  case OMPC_if:
  case OMPC_depobj:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_sizes:
  case OMPC_allocator:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_compare:
  case OMPC_seq_cst:
  case OMPC_acq_rel:
  case OMPC_acquire:
  case OMPC_release:
  case OMPC_relaxed:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_dist_schedule:
  case OMPC_defaultmap:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_unified_address:
  case OMPC_unified_shared_memory:
  case OMPC_reverse_offload:
  case OMPC_dynamic_allocators:
  case OMPC_atomic_default_mem_order:
  case OMPC_device_type:
  case OMPC_match:
  case OMPC_order:
  case OMPC_destroy:
  case OMPC_novariants:
  case OMPC_nocontext:
  case OMPC_detach:
  case OMPC_uses_allocators:
  case OMPC_when:
  case OMPC_bind:
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

ExprResult Sema::getOpenMPCapturedExpr(VarDecl *Capture, ExprValueKind VK,
                                       ExprObjectKind OK, SourceLocation Loc) {
  ExprResult Res = BuildDeclRefExpr(
      Capture, Capture->getType().getNonReferenceType(), VK_LValue, Loc);
  if (!Res.isUsable())
    return ExprError();
  if (OK == OK_Ordinary && !getLangOpts().CPlusPlus) {
    Res = CreateBuiltinUnaryOp(Loc, UO_Deref, Res.get());
    if (!Res.isUsable())
      return ExprError();
  }
  if (VK != VK_LValue && Res.get()->isGLValue()) {
    Res = DefaultLvalueConversion(Res.get());
    if (!Res.isUsable())
      return ExprError();
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPPrivateClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> PrivateCopies;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP private clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    QualType Type = D->getType();
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type, diag::err_omp_private_incomplete_type))
      continue;
    Type = Type.getNonReferenceType();

    // OpenMP 5.0 [2.19.3, List Item Privatization, Restrictions]
    // A variable that is privatized must not have a const-qualified type
    // unless it is of class type with a mutable member. This restriction does
    // not apply to the firstprivate clause.
    //
    // OpenMP 3.1 [2.9.3.3, private clause, Restrictions]
    // A variable that appears in a private clause must not have a
    // const-qualified type unless it is of class type with a mutable member.
    if (rejectConstNotMutableType(*this, D, Type, OMPC_private, ELoc))
      continue;

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(D, /*FromParent=*/false);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_private) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_private);
      reportOriginalDsa(*this, DSAStack, D, DVar);
      continue;
    }

    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    // Variably modified types are not supported for tasks.
    if (!Type->isAnyPointerType() && Type->isVariablyModifiedType() &&
        isOpenMPTaskingDirective(CurrDir)) {
      Diag(ELoc, diag::err_omp_variably_modified_type_not_supported)
          << getOpenMPClauseName(OMPC_private) << Type
          << getOpenMPDirectiveName(CurrDir);
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      Diag(D->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }

    // OpenMP 4.5 [2.15.5.1, Restrictions, p.3]
    // A list item cannot appear in both a map clause and a data-sharing
    // attribute clause on the same construct
    //
    // OpenMP 5.0 [2.19.7.1, Restrictions, p.7]
    // A list item cannot appear in both a map clause and a data-sharing
    // attribute clause on the same construct unless the construct is a
    // combined construct.
    if ((LangOpts.OpenMP <= 45 && isOpenMPTargetExecutionDirective(CurrDir)) ||
        CurrDir == OMPD_target) {
      OpenMPClauseKind ConflictKind;
      if (DSAStack->checkMappableExprComponentListsForDecl(
              VD, /*CurrentRegionOnly=*/true,
              [&](OMPClauseMappableExprCommon::MappableExprComponentListRef,
                  OpenMPClauseKind WhereFoundClauseKind) -> bool {
                ConflictKind = WhereFoundClauseKind;
                return true;
              })) {
        Diag(ELoc, diag::err_omp_variable_in_given_clause_and_dsa)
            << getOpenMPClauseName(OMPC_private)
            << getOpenMPClauseName(ConflictKind)
            << getOpenMPDirectiveName(CurrDir);
        reportOriginalDsa(*this, DSAStack, D, DVar);
        continue;
      }
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accessible, unambiguous default constructor for the
    //  class type.
    // Generate helper private variable and initialize it with the default
    // value. The address of the original variable is replaced by the address of
    // the new private variable in CodeGen. This new variable is not added to
    // IdResolver, so the code in the OpenMP region uses original variable for
    // proper diagnostics.
    Type = Type.getUnqualifiedType();
    VarDecl *VDPrivate =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr,
                     VD ? cast<DeclRefExpr>(SimpleRefExpr) : nullptr);
    ActOnUninitializedDecl(VDPrivate);
    if (VDPrivate->isInvalidDecl())
      continue;
    DeclRefExpr *VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, RefExpr->getType().getUnqualifiedType(), ELoc);

    DeclRefExpr *Ref = nullptr;
    if (!VD && !CurContext->isDependentContext())
      Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/false);
    DSAStack->addDSA(D, RefExpr->IgnoreParens(), OMPC_private, Ref);
    Vars.push_back((VD || CurContext->isDependentContext())
                       ? RefExpr->IgnoreParens()
                       : Ref);
    PrivateCopies.push_back(VDPrivateRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OMPPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars,
                                  PrivateCopies);
}

OMPClause *Sema::ActOnOpenMPFirstprivateClause(ArrayRef<Expr *> VarList,
                                               SourceLocation StartLoc,
                                               SourceLocation LParenLoc,
                                               SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> PrivateCopies;
  SmallVector<Expr *, 8> Inits;
  SmallVector<Decl *, 4> ExprCaptures;
  bool IsImplicitClause =
      StartLoc.isInvalid() && LParenLoc.isInvalid() && EndLoc.isInvalid();
  SourceLocation ImplicitClauseLoc = DSAStack->getConstructLoc();

  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP firstprivate clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
      Inits.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    ELoc = IsImplicitClause ? ImplicitClauseLoc : ELoc;
    QualType Type = D->getType();
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_firstprivate_incomplete_type))
      continue;
    Type = Type.getNonReferenceType();

    // OpenMP [2.9.3.4, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accessible, unambiguous copy constructor for the
    //  class type.
    QualType ElemType = Context.getBaseElementType(Type).getNonReferenceType();

    // If an implicit firstprivate variable found it was checked already.
    DSAStackTy::DSAVarData TopDVar;
    if (!IsImplicitClause) {
      DSAStackTy::DSAVarData DVar =
          DSAStack->getTopDSA(D, /*FromParent=*/false);
      TopDVar = DVar;
      OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
      bool IsConstant = ElemType.isConstant(Context);
      // OpenMP [2.4.13, Data-sharing Attribute Clauses]
      //  A list item that specifies a given variable may not appear in more
      // than one clause on the same directive, except that a variable may be
      //  specified in both firstprivate and lastprivate clauses.
      // OpenMP 4.5 [2.10.8, Distribute Construct, p.3]
      // A list item may appear in a firstprivate or lastprivate clause but not
      // both.
      if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_firstprivate &&
          (isOpenMPDistributeDirective(CurrDir) ||
           DVar.CKind != OMPC_lastprivate) &&
          DVar.RefExpr) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_firstprivate);
        reportOriginalDsa(*this, DSAStack, D, DVar);
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
      if (!(IsConstant || (VD && VD->isStaticDataMember())) && !DVar.RefExpr &&
          DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_firstprivate);
        reportOriginalDsa(*this, DSAStack, D, DVar);
        continue;
      }

      // OpenMP [2.9.3.4, Restrictions, p.2]
      //  A list item that is private within a parallel region must not appear
      //  in a firstprivate clause on a worksharing construct if any of the
      //  worksharing regions arising from the worksharing construct ever bind
      //  to any of the parallel regions arising from the parallel construct.
      // OpenMP 4.5 [2.15.3.4, Restrictions, p.3]
      // A list item that is private within a teams region must not appear in a
      // firstprivate clause on a distribute construct if any of the distribute
      // regions arising from the distribute construct ever bind to any of the
      // teams regions arising from the teams construct.
      // OpenMP 4.5 [2.15.3.4, Restrictions, p.3]
      // A list item that appears in a reduction clause of a teams construct
      // must not appear in a firstprivate clause on a distribute construct if
      // any of the distribute regions arising from the distribute construct
      // ever bind to any of the teams regions arising from the teams construct.
      if ((isOpenMPWorksharingDirective(CurrDir) ||
           isOpenMPDistributeDirective(CurrDir)) &&
          !isOpenMPParallelDirective(CurrDir) &&
          !isOpenMPTeamsDirective(CurrDir)) {
        DVar = DSAStack->getImplicitDSA(D, true);
        if (DVar.CKind != OMPC_shared &&
            (isOpenMPParallelDirective(DVar.DKind) ||
             isOpenMPTeamsDirective(DVar.DKind) ||
             DVar.DKind == OMPD_unknown)) {
          Diag(ELoc, diag::err_omp_required_access)
              << getOpenMPClauseName(OMPC_firstprivate)
              << getOpenMPClauseName(OMPC_shared);
          reportOriginalDsa(*this, DSAStack, D, DVar);
          continue;
        }
      }
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
      if (isOpenMPTaskingDirective(CurrDir)) {
        DVar = DSAStack->hasInnermostDSA(
            D,
            [](OpenMPClauseKind C, bool AppliedToPointee) {
              return C == OMPC_reduction && !AppliedToPointee;
            },
            [](OpenMPDirectiveKind K) {
              return isOpenMPParallelDirective(K) ||
                     isOpenMPWorksharingDirective(K) ||
                     isOpenMPTeamsDirective(K);
            },
            /*FromParent=*/true);
        if (DVar.CKind == OMPC_reduction &&
            (isOpenMPParallelDirective(DVar.DKind) ||
             isOpenMPWorksharingDirective(DVar.DKind) ||
             isOpenMPTeamsDirective(DVar.DKind))) {
          Diag(ELoc, diag::err_omp_parallel_reduction_in_task_firstprivate)
              << getOpenMPDirectiveName(DVar.DKind);
          reportOriginalDsa(*this, DSAStack, D, DVar);
          continue;
        }
      }

      // OpenMP 4.5 [2.15.5.1, Restrictions, p.3]
      // A list item cannot appear in both a map clause and a data-sharing
      // attribute clause on the same construct
      //
      // OpenMP 5.0 [2.19.7.1, Restrictions, p.7]
      // A list item cannot appear in both a map clause and a data-sharing
      // attribute clause on the same construct unless the construct is a
      // combined construct.
      if ((LangOpts.OpenMP <= 45 &&
           isOpenMPTargetExecutionDirective(CurrDir)) ||
          CurrDir == OMPD_target) {
        OpenMPClauseKind ConflictKind;
        if (DSAStack->checkMappableExprComponentListsForDecl(
                VD, /*CurrentRegionOnly=*/true,
                [&ConflictKind](
                    OMPClauseMappableExprCommon::MappableExprComponentListRef,
                    OpenMPClauseKind WhereFoundClauseKind) {
                  ConflictKind = WhereFoundClauseKind;
                  return true;
                })) {
          Diag(ELoc, diag::err_omp_variable_in_given_clause_and_dsa)
              << getOpenMPClauseName(OMPC_firstprivate)
              << getOpenMPClauseName(ConflictKind)
              << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
          reportOriginalDsa(*this, DSAStack, D, DVar);
          continue;
        }
      }
    }

    // Variably modified types are not supported for tasks.
    if (!Type->isAnyPointerType() && Type->isVariablyModifiedType() &&
        isOpenMPTaskingDirective(DSAStack->getCurrentDirective())) {
      Diag(ELoc, diag::err_omp_variably_modified_type_not_supported)
          << getOpenMPClauseName(OMPC_firstprivate) << Type
          << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      Diag(D->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }

    Type = Type.getUnqualifiedType();
    VarDecl *VDPrivate =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr,
                     VD ? cast<DeclRefExpr>(SimpleRefExpr) : nullptr);
    // Generate helper private variable and initialize it with the value of the
    // original variable. The address of the original variable is replaced by
    // the address of the new private variable in the CodeGen. This new variable
    // is not added to IdResolver, so the code in the OpenMP region uses
    // original variable for proper diagnostics and variable capturing.
    Expr *VDInitRefExpr = nullptr;
    // For arrays generate initializer for single element and replace it by the
    // original array element in CodeGen.
    if (Type->isArrayType()) {
      VarDecl *VDInit =
          buildVarDecl(*this, RefExpr->getExprLoc(), ElemType, D->getName());
      VDInitRefExpr = buildDeclRefExpr(*this, VDInit, ElemType, ELoc);
      Expr *Init = DefaultLvalueConversion(VDInitRefExpr).get();
      ElemType = ElemType.getUnqualifiedType();
      VarDecl *VDInitTemp = buildVarDecl(*this, RefExpr->getExprLoc(), ElemType,
                                         ".firstprivate.temp");
      InitializedEntity Entity =
          InitializedEntity::InitializeVariable(VDInitTemp);
      InitializationKind Kind = InitializationKind::CreateCopy(ELoc, ELoc);

      InitializationSequence InitSeq(*this, Entity, Kind, Init);
      ExprResult Result = InitSeq.Perform(*this, Entity, Kind, Init);
      if (Result.isInvalid())
        VDPrivate->setInvalidDecl();
      else
        VDPrivate->setInit(Result.getAs<Expr>());
      // Remove temp variable declaration.
      Context.Deallocate(VDInitTemp);
    } else {
      VarDecl *VDInit = buildVarDecl(*this, RefExpr->getExprLoc(), Type,
                                     ".firstprivate.temp");
      VDInitRefExpr = buildDeclRefExpr(*this, VDInit, RefExpr->getType(),
                                       RefExpr->getExprLoc());
      AddInitializerToDecl(VDPrivate,
                           DefaultLvalueConversion(VDInitRefExpr).get(),
                           /*DirectInit=*/false);
    }
    if (VDPrivate->isInvalidDecl()) {
      if (IsImplicitClause) {
        Diag(RefExpr->getExprLoc(),
             diag::note_omp_task_predetermined_firstprivate_here);
      }
      continue;
    }
    CurContext->addDecl(VDPrivate);
    DeclRefExpr *VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, RefExpr->getType().getUnqualifiedType(),
        RefExpr->getExprLoc());
    DeclRefExpr *Ref = nullptr;
    if (!VD && !CurContext->isDependentContext()) {
      if (TopDVar.CKind == OMPC_lastprivate) {
        Ref = TopDVar.PrivateCopy;
      } else {
        Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/true);
        if (!isOpenMPCapturedDecl(D))
          ExprCaptures.push_back(Ref->getDecl());
      }
    }
    if (!IsImplicitClause)
      DSAStack->addDSA(D, RefExpr->IgnoreParens(), OMPC_firstprivate, Ref);
    Vars.push_back((VD || CurContext->isDependentContext())
                       ? RefExpr->IgnoreParens()
                       : Ref);
    PrivateCopies.push_back(VDPrivateRefExpr);
    Inits.push_back(VDInitRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OMPFirstprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                       Vars, PrivateCopies, Inits,
                                       buildPreInits(Context, ExprCaptures));
}

OMPClause *Sema::ActOnOpenMPLastprivateClause(
    ArrayRef<Expr *> VarList, OpenMPLastprivateModifier LPKind,
    SourceLocation LPKindLoc, SourceLocation ColonLoc, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation EndLoc) {
  if (LPKind == OMPC_LASTPRIVATE_unknown && LPKindLoc.isValid()) {
    assert(ColonLoc.isValid() && "Colon location must be valid.");
    Diag(LPKindLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_lastprivate, /*First=*/0,
                                   /*Last=*/OMPC_LASTPRIVATE_unknown)
        << getOpenMPClauseName(OMPC_lastprivate);
    return nullptr;
  }

  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> SrcExprs;
  SmallVector<Expr *, 8> DstExprs;
  SmallVector<Expr *, 8> AssignmentOps;
  SmallVector<Decl *, 4> ExprCaptures;
  SmallVector<Expr *, 4> ExprPostUpdates;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP lastprivate clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    QualType Type = D->getType();
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP [2.14.3.5, Restrictions, C/C++, p.2]
    //  A variable that appears in a lastprivate clause must not have an
    //  incomplete type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_lastprivate_incomplete_type))
      continue;
    Type = Type.getNonReferenceType();

    // OpenMP 5.0 [2.19.3, List Item Privatization, Restrictions]
    // A variable that is privatized must not have a const-qualified type
    // unless it is of class type with a mutable member. This restriction does
    // not apply to the firstprivate clause.
    //
    // OpenMP 3.1 [2.9.3.5, lastprivate clause, Restrictions]
    // A variable that appears in a lastprivate clause must not have a
    // const-qualified type unless it is of class type with a mutable member.
    if (rejectConstNotMutableType(*this, D, Type, OMPC_lastprivate, ELoc))
      continue;

    // OpenMP 5.0 [2.19.4.5 lastprivate Clause, Restrictions]
    // A list item that appears in a lastprivate clause with the conditional
    // modifier must be a scalar variable.
    if (LPKind == OMPC_LASTPRIVATE_conditional && !Type->isScalarType()) {
      Diag(ELoc, diag::err_omp_lastprivate_conditional_non_scalar);
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      Diag(D->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }

    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below.
    // OpenMP 4.5 [2.10.8, Distribute Construct, p.3]
    // A list item may appear in a firstprivate or lastprivate clause but not
    // both.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(D, /*FromParent=*/false);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_lastprivate &&
        (isOpenMPDistributeDirective(CurrDir) ||
         DVar.CKind != OMPC_firstprivate) &&
        (DVar.CKind != OMPC_private || DVar.RefExpr != nullptr)) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(DVar.CKind)
          << getOpenMPClauseName(OMPC_lastprivate);
      reportOriginalDsa(*this, DSAStack, D, DVar);
      continue;
    }

    // OpenMP [2.14.3.5, Restrictions, p.2]
    // A list item that is private within a parallel region, or that appears in
    // the reduction clause of a parallel construct, must not appear in a
    // lastprivate clause on a worksharing construct if any of the corresponding
    // worksharing regions ever binds to any of the corresponding parallel
    // regions.
    DSAStackTy::DSAVarData TopDVar = DVar;
    if (isOpenMPWorksharingDirective(CurrDir) &&
        !isOpenMPParallelDirective(CurrDir) &&
        !isOpenMPTeamsDirective(CurrDir)) {
      DVar = DSAStack->getImplicitDSA(D, true);
      if (DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_lastprivate)
            << getOpenMPClauseName(OMPC_shared);
        reportOriginalDsa(*this, DSAStack, D, DVar);
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
    Type = Context.getBaseElementType(Type).getNonReferenceType();
    VarDecl *SrcVD = buildVarDecl(*this, ERange.getBegin(),
                                  Type.getUnqualifiedType(), ".lastprivate.src",
                                  D->hasAttrs() ? &D->getAttrs() : nullptr);
    DeclRefExpr *PseudoSrcExpr =
        buildDeclRefExpr(*this, SrcVD, Type.getUnqualifiedType(), ELoc);
    VarDecl *DstVD =
        buildVarDecl(*this, ERange.getBegin(), Type, ".lastprivate.dst",
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
    DeclRefExpr *PseudoDstExpr = buildDeclRefExpr(*this, DstVD, Type, ELoc);
    // For arrays generate assignment operation for single element and replace
    // it by the original array element in CodeGen.
    ExprResult AssignmentOp = BuildBinOp(/*S=*/nullptr, ELoc, BO_Assign,
                                         PseudoDstExpr, PseudoSrcExpr);
    if (AssignmentOp.isInvalid())
      continue;
    AssignmentOp =
        ActOnFinishFullExpr(AssignmentOp.get(), ELoc, /*DiscardedValue*/ false);
    if (AssignmentOp.isInvalid())
      continue;

    DeclRefExpr *Ref = nullptr;
    if (!VD && !CurContext->isDependentContext()) {
      if (TopDVar.CKind == OMPC_firstprivate) {
        Ref = TopDVar.PrivateCopy;
      } else {
        Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/false);
        if (!isOpenMPCapturedDecl(D))
          ExprCaptures.push_back(Ref->getDecl());
      }
      if ((TopDVar.CKind == OMPC_firstprivate && !TopDVar.PrivateCopy) ||
          (!isOpenMPCapturedDecl(D) &&
           Ref->getDecl()->hasAttr<OMPCaptureNoInitAttr>())) {
        ExprResult RefRes = DefaultLvalueConversion(Ref);
        if (!RefRes.isUsable())
          continue;
        ExprResult PostUpdateRes =
            BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign, SimpleRefExpr,
                       RefRes.get());
        if (!PostUpdateRes.isUsable())
          continue;
        ExprPostUpdates.push_back(
            IgnoredValueConversions(PostUpdateRes.get()).get());
      }
    }
    DSAStack->addDSA(D, RefExpr->IgnoreParens(), OMPC_lastprivate, Ref);
    Vars.push_back((VD || CurContext->isDependentContext())
                       ? RefExpr->IgnoreParens()
                       : Ref);
    SrcExprs.push_back(PseudoSrcExpr);
    DstExprs.push_back(PseudoDstExpr);
    AssignmentOps.push_back(AssignmentOp.get());
  }

  if (Vars.empty())
    return nullptr;

  return OMPLastprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                      Vars, SrcExprs, DstExprs, AssignmentOps,
                                      LPKind, LPKindLoc, ColonLoc,
                                      buildPreInits(Context, ExprCaptures),
                                      buildPostUpdate(*this, ExprPostUpdates));
}

OMPClause *Sema::ActOnOpenMPSharedClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP lastprivate clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    auto *VD = dyn_cast<VarDecl>(D);
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(D, /*FromParent=*/false);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_shared &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_shared);
      reportOriginalDsa(*this, DSAStack, D, DVar);
      continue;
    }

    DeclRefExpr *Ref = nullptr;
    if (!VD && isOpenMPCapturedDecl(D) && !CurContext->isDependentContext())
      Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/true);
    DSAStack->addDSA(D, RefExpr->IgnoreParens(), OMPC_shared, Ref);
    Vars.push_back((VD || !Ref || CurContext->isDependentContext())
                       ? RefExpr->IgnoreParens()
                       : Ref);
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
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(VD, /*FromParent=*/false);
      if (DVar.CKind == OMPC_shared && !DVar.RefExpr)
        return false;
      if (DVar.CKind != OMPC_unknown)
        return true;
      DSAStackTy::DSAVarData DVarPrivate = Stack->hasDSA(
          VD,
          [](OpenMPClauseKind C, bool AppliedToPointee) {
            return isOpenMPPrivate(C) && !AppliedToPointee;
          },
          [](OpenMPDirectiveKind) { return true; },
          /*FromParent=*/true);
      return DVarPrivate.CKind != OMPC_unknown;
    }
    return false;
  }
  bool VisitStmt(Stmt *S) {
    for (Stmt *Child : S->children()) {
      if (Child && Visit(Child))
        return true;
    }
    return false;
  }
  explicit DSARefChecker(DSAStackTy *S) : Stack(S) {}
};
} // namespace

namespace {
// Transform MemberExpression for specified FieldDecl of current class to
// DeclRefExpr to specified OMPCapturedExprDecl.
class TransformExprToCaptures : public TreeTransform<TransformExprToCaptures> {
  typedef TreeTransform<TransformExprToCaptures> BaseTransform;
  ValueDecl *Field = nullptr;
  DeclRefExpr *CapturedExpr = nullptr;

public:
  TransformExprToCaptures(Sema &SemaRef, ValueDecl *FieldDecl)
      : BaseTransform(SemaRef), Field(FieldDecl), CapturedExpr(nullptr) {}

  ExprResult TransformMemberExpr(MemberExpr *E) {
    if (isa<CXXThisExpr>(E->getBase()->IgnoreParenImpCasts()) &&
        E->getMemberDecl() == Field) {
      CapturedExpr = buildCapture(SemaRef, Field, E, /*WithInit=*/false);
      return CapturedExpr;
    }
    return BaseTransform::TransformMemberExpr(E);
  }
  DeclRefExpr *getCapturedExpr() { return CapturedExpr; }
};
} // namespace

template <typename T, typename U>
static T filterLookupForUDReductionAndMapper(
    SmallVectorImpl<U> &Lookups, const llvm::function_ref<T(ValueDecl *)> Gen) {
  for (U &Set : Lookups) {
    for (auto *D : Set) {
      if (T Res = Gen(cast<ValueDecl>(D)))
        return Res;
    }
  }
  return T();
}

static NamedDecl *findAcceptableDecl(Sema &SemaRef, NamedDecl *D) {
  assert(!LookupResult::isVisible(SemaRef, D) && "not in slow case");

  for (auto RD : D->redecls()) {
    // Don't bother with extra checks if we already know this one isn't visible.
    if (RD == D)
      continue;

    auto ND = cast<NamedDecl>(RD);
    if (LookupResult::isVisible(SemaRef, ND))
      return ND;
  }

  return nullptr;
}

static void
argumentDependentLookup(Sema &SemaRef, const DeclarationNameInfo &Id,
                        SourceLocation Loc, QualType Ty,
                        SmallVectorImpl<UnresolvedSet<8>> &Lookups) {
  // Find all of the associated namespaces and classes based on the
  // arguments we have.
  Sema::AssociatedNamespaceSet AssociatedNamespaces;
  Sema::AssociatedClassSet AssociatedClasses;
  OpaqueValueExpr OVE(Loc, Ty, VK_LValue);
  SemaRef.FindAssociatedClassesAndNamespaces(Loc, &OVE, AssociatedNamespaces,
                                             AssociatedClasses);

  // C++ [basic.lookup.argdep]p3:
  //   Let X be the lookup set produced by unqualified lookup (3.4.1)
  //   and let Y be the lookup set produced by argument dependent
  //   lookup (defined as follows). If X contains [...] then Y is
  //   empty. Otherwise Y is the set of declarations found in the
  //   namespaces associated with the argument types as described
  //   below. The set of declarations found by the lookup of the name
  //   is the union of X and Y.
  //
  // Here, we compute Y and add its members to the overloaded
  // candidate set.
  for (auto *NS : AssociatedNamespaces) {
    //   When considering an associated namespace, the lookup is the
    //   same as the lookup performed when the associated namespace is
    //   used as a qualifier (3.4.3.2) except that:
    //
    //     -- Any using-directives in the associated namespace are
    //        ignored.
    //
    //     -- Any namespace-scope friend functions declared in
    //        associated classes are visible within their respective
    //        namespaces even if they are not visible during an ordinary
    //        lookup (11.4).
    DeclContext::lookup_result R = NS->lookup(Id.getName());
    for (auto *D : R) {
      auto *Underlying = D;
      if (auto *USD = dyn_cast<UsingShadowDecl>(D))
        Underlying = USD->getTargetDecl();

      if (!isa<OMPDeclareReductionDecl>(Underlying) &&
          !isa<OMPDeclareMapperDecl>(Underlying))
        continue;

      if (!SemaRef.isVisible(D)) {
        D = findAcceptableDecl(SemaRef, D);
        if (!D)
          continue;
        if (auto *USD = dyn_cast<UsingShadowDecl>(D))
          Underlying = USD->getTargetDecl();
      }
      Lookups.emplace_back();
      Lookups.back().addDecl(Underlying);
    }
  }
}

static ExprResult
buildDeclareReductionRef(Sema &SemaRef, SourceLocation Loc, SourceRange Range,
                         Scope *S, CXXScopeSpec &ReductionIdScopeSpec,
                         const DeclarationNameInfo &ReductionId, QualType Ty,
                         CXXCastPath &BasePath, Expr *UnresolvedReduction) {
  if (ReductionIdScopeSpec.isInvalid())
    return ExprError();
  SmallVector<UnresolvedSet<8>, 4> Lookups;
  if (S) {
    LookupResult Lookup(SemaRef, ReductionId, Sema::LookupOMPReductionName);
    Lookup.suppressDiagnostics();
    while (S && SemaRef.LookupParsedName(Lookup, S, &ReductionIdScopeSpec)) {
      NamedDecl *D = Lookup.getRepresentativeDecl();
      do {
        S = S->getParent();
      } while (S && !S->isDeclScope(D));
      if (S)
        S = S->getParent();
      Lookups.emplace_back();
      Lookups.back().append(Lookup.begin(), Lookup.end());
      Lookup.clear();
    }
  } else if (auto *ULE =
                 cast_or_null<UnresolvedLookupExpr>(UnresolvedReduction)) {
    Lookups.push_back(UnresolvedSet<8>());
    Decl *PrevD = nullptr;
    for (NamedDecl *D : ULE->decls()) {
      if (D == PrevD)
        Lookups.push_back(UnresolvedSet<8>());
      else if (auto *DRD = dyn_cast<OMPDeclareReductionDecl>(D))
        Lookups.back().addDecl(DRD);
      PrevD = D;
    }
  }
  if (SemaRef.CurContext->isDependentContext() || Ty->isDependentType() ||
      Ty->isInstantiationDependentType() ||
      Ty->containsUnexpandedParameterPack() ||
      filterLookupForUDReductionAndMapper<bool>(Lookups, [](ValueDecl *D) {
        return !D->isInvalidDecl() &&
               (D->getType()->isDependentType() ||
                D->getType()->isInstantiationDependentType() ||
                D->getType()->containsUnexpandedParameterPack());
      })) {
    UnresolvedSet<8> ResSet;
    for (const UnresolvedSet<8> &Set : Lookups) {
      if (Set.empty())
        continue;
      ResSet.append(Set.begin(), Set.end());
      // The last item marks the end of all declarations at the specified scope.
      ResSet.addDecl(Set[Set.size() - 1]);
    }
    return UnresolvedLookupExpr::Create(
        SemaRef.Context, /*NamingClass=*/nullptr,
        ReductionIdScopeSpec.getWithLocInContext(SemaRef.Context), ReductionId,
        /*ADL=*/true, /*Overloaded=*/true, ResSet.begin(), ResSet.end());
  }
  // Lookup inside the classes.
  // C++ [over.match.oper]p3:
  //   For a unary operator @ with an operand of a type whose
  //   cv-unqualified version is T1, and for a binary operator @ with
  //   a left operand of a type whose cv-unqualified version is T1 and
  //   a right operand of a type whose cv-unqualified version is T2,
  //   three sets of candidate functions, designated member
  //   candidates, non-member candidates and built-in candidates, are
  //   constructed as follows:
  //     -- If T1 is a complete class type or a class currently being
  //        defined, the set of member candidates is the result of the
  //        qualified lookup of T1::operator@ (13.3.1.1.1); otherwise,
  //        the set of member candidates is empty.
  LookupResult Lookup(SemaRef, ReductionId, Sema::LookupOMPReductionName);
  Lookup.suppressDiagnostics();
  if (const auto *TyRec = Ty->getAs<RecordType>()) {
    // Complete the type if it can be completed.
    // If the type is neither complete nor being defined, bail out now.
    if (SemaRef.isCompleteType(Loc, Ty) || TyRec->isBeingDefined() ||
        TyRec->getDecl()->getDefinition()) {
      Lookup.clear();
      SemaRef.LookupQualifiedName(Lookup, TyRec->getDecl());
      if (Lookup.empty()) {
        Lookups.emplace_back();
        Lookups.back().append(Lookup.begin(), Lookup.end());
      }
    }
  }
  // Perform ADL.
  if (SemaRef.getLangOpts().CPlusPlus)
    argumentDependentLookup(SemaRef, ReductionId, Loc, Ty, Lookups);
  if (auto *VD = filterLookupForUDReductionAndMapper<ValueDecl *>(
          Lookups, [&SemaRef, Ty](ValueDecl *D) -> ValueDecl * {
            if (!D->isInvalidDecl() &&
                SemaRef.Context.hasSameType(D->getType(), Ty))
              return D;
            return nullptr;
          }))
    return SemaRef.BuildDeclRefExpr(VD, VD->getType().getNonReferenceType(),
                                    VK_LValue, Loc);
  if (SemaRef.getLangOpts().CPlusPlus) {
    if (auto *VD = filterLookupForUDReductionAndMapper<ValueDecl *>(
            Lookups, [&SemaRef, Ty, Loc](ValueDecl *D) -> ValueDecl * {
              if (!D->isInvalidDecl() &&
                  SemaRef.IsDerivedFrom(Loc, Ty, D->getType()) &&
                  !Ty.isMoreQualifiedThan(D->getType()))
                return D;
              return nullptr;
            })) {
      CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                         /*DetectVirtual=*/false);
      if (SemaRef.IsDerivedFrom(Loc, Ty, VD->getType(), Paths)) {
        if (!Paths.isAmbiguous(SemaRef.Context.getCanonicalType(
                VD->getType().getUnqualifiedType()))) {
          if (SemaRef.CheckBaseClassAccess(
                  Loc, VD->getType(), Ty, Paths.front(),
                  /*DiagID=*/0) != Sema::AR_inaccessible) {
            SemaRef.BuildBasePathArray(Paths, BasePath);
            return SemaRef.BuildDeclRefExpr(
                VD, VD->getType().getNonReferenceType(), VK_LValue, Loc);
          }
        }
      }
    }
  }
  if (ReductionIdScopeSpec.isSet()) {
    SemaRef.Diag(Loc, diag::err_omp_not_resolved_reduction_identifier)
        << Ty << Range;
    return ExprError();
  }
  return ExprEmpty();
}

namespace {
/// Data for the reduction-based clauses.
struct ReductionData {
  /// List of original reduction items.
  SmallVector<Expr *, 8> Vars;
  /// List of private copies of the reduction items.
  SmallVector<Expr *, 8> Privates;
  /// LHS expressions for the reduction_op expressions.
  SmallVector<Expr *, 8> LHSs;
  /// RHS expressions for the reduction_op expressions.
  SmallVector<Expr *, 8> RHSs;
  /// Reduction operation expression.
  SmallVector<Expr *, 8> ReductionOps;
  /// inscan copy operation expressions.
  SmallVector<Expr *, 8> InscanCopyOps;
  /// inscan copy temp array expressions for prefix sums.
  SmallVector<Expr *, 8> InscanCopyArrayTemps;
  /// inscan copy temp array element expressions for prefix sums.
  SmallVector<Expr *, 8> InscanCopyArrayElems;
  /// Taskgroup descriptors for the corresponding reduction items in
  /// in_reduction clauses.
  SmallVector<Expr *, 8> TaskgroupDescriptors;
  /// List of captures for clause.
  SmallVector<Decl *, 4> ExprCaptures;
  /// List of postupdate expressions.
  SmallVector<Expr *, 4> ExprPostUpdates;
  /// Reduction modifier.
  unsigned RedModifier = 0;
  ReductionData() = delete;
  /// Reserves required memory for the reduction data.
  ReductionData(unsigned Size, unsigned Modifier = 0) : RedModifier(Modifier) {
    Vars.reserve(Size);
    Privates.reserve(Size);
    LHSs.reserve(Size);
    RHSs.reserve(Size);
    ReductionOps.reserve(Size);
    if (RedModifier == OMPC_REDUCTION_inscan) {
      InscanCopyOps.reserve(Size);
      InscanCopyArrayTemps.reserve(Size);
      InscanCopyArrayElems.reserve(Size);
    }
    TaskgroupDescriptors.reserve(Size);
    ExprCaptures.reserve(Size);
    ExprPostUpdates.reserve(Size);
  }
  /// Stores reduction item and reduction operation only (required for dependent
  /// reduction item).
  void push(Expr *Item, Expr *ReductionOp) {
    Vars.emplace_back(Item);
    Privates.emplace_back(nullptr);
    LHSs.emplace_back(nullptr);
    RHSs.emplace_back(nullptr);
    ReductionOps.emplace_back(ReductionOp);
    TaskgroupDescriptors.emplace_back(nullptr);
    if (RedModifier == OMPC_REDUCTION_inscan) {
      InscanCopyOps.push_back(nullptr);
      InscanCopyArrayTemps.push_back(nullptr);
      InscanCopyArrayElems.push_back(nullptr);
    }
  }
  /// Stores reduction data.
  void push(Expr *Item, Expr *Private, Expr *LHS, Expr *RHS, Expr *ReductionOp,
            Expr *TaskgroupDescriptor, Expr *CopyOp, Expr *CopyArrayTemp,
            Expr *CopyArrayElem) {
    Vars.emplace_back(Item);
    Privates.emplace_back(Private);
    LHSs.emplace_back(LHS);
    RHSs.emplace_back(RHS);
    ReductionOps.emplace_back(ReductionOp);
    TaskgroupDescriptors.emplace_back(TaskgroupDescriptor);
    if (RedModifier == OMPC_REDUCTION_inscan) {
      InscanCopyOps.push_back(CopyOp);
      InscanCopyArrayTemps.push_back(CopyArrayTemp);
      InscanCopyArrayElems.push_back(CopyArrayElem);
    } else {
      assert(CopyOp == nullptr && CopyArrayTemp == nullptr &&
             CopyArrayElem == nullptr &&
             "Copy operation must be used for inscan reductions only.");
    }
  }
};
} // namespace

static bool checkOMPArraySectionConstantForReduction(
    ASTContext &Context, const OMPArraySectionExpr *OASE, bool &SingleElement,
    SmallVectorImpl<llvm::APSInt> &ArraySizes) {
  const Expr *Length = OASE->getLength();
  if (Length == nullptr) {
    // For array sections of the form [1:] or [:], we would need to analyze
    // the lower bound...
    if (OASE->getColonLocFirst().isValid())
      return false;

    // This is an array subscript which has implicit length 1!
    SingleElement = true;
    ArraySizes.push_back(llvm::APSInt::get(1));
  } else {
    Expr::EvalResult Result;
    if (!Length->EvaluateAsInt(Result, Context))
      return false;

    llvm::APSInt ConstantLengthValue = Result.Val.getInt();
    SingleElement = (ConstantLengthValue.getSExtValue() == 1);
    ArraySizes.push_back(ConstantLengthValue);
  }

  // Get the base of this array section and walk up from there.
  const Expr *Base = OASE->getBase()->IgnoreParenImpCasts();

  // We require length = 1 for all array sections except the right-most to
  // guarantee that the memory region is contiguous and has no holes in it.
  while (const auto *TempOASE = dyn_cast<OMPArraySectionExpr>(Base)) {
    Length = TempOASE->getLength();
    if (Length == nullptr) {
      // For array sections of the form [1:] or [:], we would need to analyze
      // the lower bound...
      if (OASE->getColonLocFirst().isValid())
        return false;

      // This is an array subscript which has implicit length 1!
      ArraySizes.push_back(llvm::APSInt::get(1));
    } else {
      Expr::EvalResult Result;
      if (!Length->EvaluateAsInt(Result, Context))
        return false;

      llvm::APSInt ConstantLengthValue = Result.Val.getInt();
      if (ConstantLengthValue.getSExtValue() != 1)
        return false;

      ArraySizes.push_back(ConstantLengthValue);
    }
    Base = TempOASE->getBase()->IgnoreParenImpCasts();
  }

  // If we have a single element, we don't need to add the implicit lengths.
  if (!SingleElement) {
    while (const auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base)) {
      // Has implicit length 1!
      ArraySizes.push_back(llvm::APSInt::get(1));
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    }
  }

  // This array section can be privatized as a single value or as a constant
  // sized array.
  return true;
}

static BinaryOperatorKind
getRelatedCompoundReductionOp(BinaryOperatorKind BOK) {
  if (BOK == BO_Add)
    return BO_AddAssign;
  if (BOK == BO_Mul)
    return BO_MulAssign;
  if (BOK == BO_And)
    return BO_AndAssign;
  if (BOK == BO_Or)
    return BO_OrAssign;
  if (BOK == BO_Xor)
    return BO_XorAssign;
  return BOK;
}

static bool actOnOMPReductionKindClause(
    Sema &S, DSAStackTy *Stack, OpenMPClauseKind ClauseKind,
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec, const DeclarationNameInfo &ReductionId,
    ArrayRef<Expr *> UnresolvedReductions, ReductionData &RD) {
  DeclarationName DN = ReductionId.getName();
  OverloadedOperatorKind OOK = DN.getCXXOverloadedOperator();
  BinaryOperatorKind BOK = BO_Comma;

  ASTContext &Context = S.Context;
  // OpenMP [2.14.3.6, reduction clause]
  // C
  // reduction-identifier is either an identifier or one of the following
  // operators: +, -, *,  &, |, ^, && and ||
  // C++
  // reduction-identifier is either an id-expression or one of the following
  // operators: +, -, *, &, |, ^, && and ||
  switch (OOK) {
  case OO_Plus:
  case OO_Minus:
    BOK = BO_Add;
    break;
  case OO_Star:
    BOK = BO_Mul;
    break;
  case OO_Amp:
    BOK = BO_And;
    break;
  case OO_Pipe:
    BOK = BO_Or;
    break;
  case OO_Caret:
    BOK = BO_Xor;
    break;
  case OO_AmpAmp:
    BOK = BO_LAnd;
    break;
  case OO_PipePipe:
    BOK = BO_LOr;
    break;
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Slash:
  case OO_Percent:
  case OO_Tilde:
  case OO_Exclaim:
  case OO_Equal:
  case OO_Less:
  case OO_Greater:
  case OO_LessEqual:
  case OO_GreaterEqual:
  case OO_PlusEqual:
  case OO_MinusEqual:
  case OO_StarEqual:
  case OO_SlashEqual:
  case OO_PercentEqual:
  case OO_CaretEqual:
  case OO_AmpEqual:
  case OO_PipeEqual:
  case OO_LessLess:
  case OO_GreaterGreater:
  case OO_LessLessEqual:
  case OO_GreaterGreaterEqual:
  case OO_EqualEqual:
  case OO_ExclaimEqual:
  case OO_Spaceship:
  case OO_PlusPlus:
  case OO_MinusMinus:
  case OO_Comma:
  case OO_ArrowStar:
  case OO_Arrow:
  case OO_Call:
  case OO_Subscript:
  case OO_Conditional:
  case OO_Coawait:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("Unexpected reduction identifier");
  case OO_None:
    if (IdentifierInfo *II = DN.getAsIdentifierInfo()) {
      if (II->isStr("max"))
        BOK = BO_GT;
      else if (II->isStr("min"))
        BOK = BO_LT;
    }
    break;
  }
  SourceRange ReductionIdRange;
  if (ReductionIdScopeSpec.isValid())
    ReductionIdRange.setBegin(ReductionIdScopeSpec.getBeginLoc());
  else
    ReductionIdRange.setBegin(ReductionId.getBeginLoc());
  ReductionIdRange.setEnd(ReductionId.getEndLoc());

  auto IR = UnresolvedReductions.begin(), ER = UnresolvedReductions.end();
  bool FirstIter = true;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "nullptr expr in OpenMP reduction clause.");
    // OpenMP [2.1, C/C++]
    //  A list item is a variable or array section, subject to the restrictions
    //  specified in Section 2.4 on page 42 and in each of the sections
    // describing clauses and directives for which a list appears.
    // OpenMP  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    if (!FirstIter && IR != ER)
      ++IR;
    FirstIter = false;
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(S, SimpleRefExpr, ELoc, ERange,
                              /*AllowArraySection=*/true);
    if (Res.second) {
      // Try to find 'declare reduction' corresponding construct before using
      // builtin/overloaded operators.
      QualType Type = Context.DependentTy;
      CXXCastPath BasePath;
      ExprResult DeclareReductionRef = buildDeclareReductionRef(
          S, ELoc, ERange, Stack->getCurScope(), ReductionIdScopeSpec,
          ReductionId, Type, BasePath, IR == ER ? nullptr : *IR);
      Expr *ReductionOp = nullptr;
      if (S.CurContext->isDependentContext() &&
          (DeclareReductionRef.isUnset() ||
           isa<UnresolvedLookupExpr>(DeclareReductionRef.get())))
        ReductionOp = DeclareReductionRef.get();
      // It will be analyzed later.
      RD.push(RefExpr, ReductionOp);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    Expr *TaskgroupDescriptor = nullptr;
    QualType Type;
    auto *ASE = dyn_cast<ArraySubscriptExpr>(RefExpr->IgnoreParens());
    auto *OASE = dyn_cast<OMPArraySectionExpr>(RefExpr->IgnoreParens());
    if (ASE) {
      Type = ASE->getType().getNonReferenceType();
    } else if (OASE) {
      QualType BaseType =
          OMPArraySectionExpr::getBaseOriginalType(OASE->getBase());
      if (const auto *ATy = BaseType->getAsArrayTypeUnsafe())
        Type = ATy->getElementType();
      else
        Type = BaseType->getPointeeType();
      Type = Type.getNonReferenceType();
    } else {
      Type = Context.getBaseElementType(D->getType().getNonReferenceType());
    }
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (S.RequireCompleteType(ELoc, D->getType(),
                              diag::err_omp_reduction_incomplete_type))
      continue;
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // A list item that appears in a reduction clause must not be
    // const-qualified.
    if (rejectConstNotMutableType(S, D, Type, ClauseKind, ELoc,
                                  /*AcceptIfMutable*/ false, ASE || OASE))
      continue;

    OpenMPDirectiveKind CurrDir = Stack->getCurrentDirective();
    // OpenMP [2.9.3.6, Restrictions, C/C++, p.4]
    //  If a list-item is a reference type then it must bind to the same object
    //  for all threads of the team.
    if (!ASE && !OASE) {
      if (VD) {
        VarDecl *VDDef = VD->getDefinition();
        if (VD->getType()->isReferenceType() && VDDef && VDDef->hasInit()) {
          DSARefChecker Check(Stack);
          if (Check.Visit(VDDef->getInit())) {
            S.Diag(ELoc, diag::err_omp_reduction_ref_type_arg)
                << getOpenMPClauseName(ClauseKind) << ERange;
            S.Diag(VDDef->getLocation(), diag::note_defined_here) << VDDef;
            continue;
          }
        }
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
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(D, /*FromParent=*/false);
      if (DVar.CKind == OMPC_reduction) {
        S.Diag(ELoc, diag::err_omp_once_referenced)
            << getOpenMPClauseName(ClauseKind);
        if (DVar.RefExpr)
          S.Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_referenced);
        continue;
      }
      if (DVar.CKind != OMPC_unknown) {
        S.Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_reduction);
        reportOriginalDsa(S, Stack, D, DVar);
        continue;
      }

      // OpenMP [2.14.3.6, Restrictions, p.1]
      //  A list item that appears in a reduction clause of a worksharing
      //  construct must be shared in the parallel regions to which any of the
      //  worksharing regions arising from the worksharing construct bind.
      if (isOpenMPWorksharingDirective(CurrDir) &&
          !isOpenMPParallelDirective(CurrDir) &&
          !isOpenMPTeamsDirective(CurrDir)) {
        DVar = Stack->getImplicitDSA(D, true);
        if (DVar.CKind != OMPC_shared) {
          S.Diag(ELoc, diag::err_omp_required_access)
              << getOpenMPClauseName(OMPC_reduction)
              << getOpenMPClauseName(OMPC_shared);
          reportOriginalDsa(S, Stack, D, DVar);
          continue;
        }
      }
    } else {
      // Threadprivates cannot be shared between threads, so dignose if the base
      // is a threadprivate variable.
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(D, /*FromParent=*/false);
      if (DVar.CKind == OMPC_threadprivate) {
        S.Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_reduction);
        reportOriginalDsa(S, Stack, D, DVar);
        continue;
      }
    }

    // Try to find 'declare reduction' corresponding construct before using
    // builtin/overloaded operators.
    CXXCastPath BasePath;
    ExprResult DeclareReductionRef = buildDeclareReductionRef(
        S, ELoc, ERange, Stack->getCurScope(), ReductionIdScopeSpec,
        ReductionId, Type, BasePath, IR == ER ? nullptr : *IR);
    if (DeclareReductionRef.isInvalid())
      continue;
    if (S.CurContext->isDependentContext() &&
        (DeclareReductionRef.isUnset() ||
         isa<UnresolvedLookupExpr>(DeclareReductionRef.get()))) {
      RD.push(RefExpr, DeclareReductionRef.get());
      continue;
    }
    if (BOK == BO_Comma && DeclareReductionRef.isUnset()) {
      // Not allowed reduction identifier is found.
      S.Diag(ReductionId.getBeginLoc(),
             diag::err_omp_unknown_reduction_identifier)
          << Type << ReductionIdRange;
      continue;
    }

    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // The type of a list item that appears in a reduction clause must be valid
    // for the reduction-identifier. For a max or min reduction in C, the type
    // of the list item must be an allowed arithmetic data type: char, int,
    // float, double, or _Bool, possibly modified with long, short, signed, or
    // unsigned. For a max or min reduction in C++, the type of the list item
    // must be an allowed arithmetic data type: char, wchar_t, int, float,
    // double, or bool, possibly modified with long, short, signed, or unsigned.
    if (DeclareReductionRef.isUnset()) {
      if ((BOK == BO_GT || BOK == BO_LT) &&
          !(Type->isScalarType() ||
            (S.getLangOpts().CPlusPlus && Type->isArithmeticType()))) {
        S.Diag(ELoc, diag::err_omp_clause_not_arithmetic_type_arg)
            << getOpenMPClauseName(ClauseKind) << S.getLangOpts().CPlusPlus;
        if (!ASE && !OASE) {
          bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                                   VarDecl::DeclarationOnly;
          S.Diag(D->getLocation(),
                 IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << D;
        }
        continue;
      }
      if ((BOK == BO_OrAssign || BOK == BO_AndAssign || BOK == BO_XorAssign) &&
          !S.getLangOpts().CPlusPlus && Type->isFloatingType()) {
        S.Diag(ELoc, diag::err_omp_clause_floating_type_arg)
            << getOpenMPClauseName(ClauseKind);
        if (!ASE && !OASE) {
          bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                                   VarDecl::DeclarationOnly;
          S.Diag(D->getLocation(),
                 IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << D;
        }
        continue;
      }
    }

    Type = Type.getNonLValueExprType(Context).getUnqualifiedType();
    VarDecl *LHSVD = buildVarDecl(S, ELoc, Type, ".reduction.lhs",
                                  D->hasAttrs() ? &D->getAttrs() : nullptr);
    VarDecl *RHSVD = buildVarDecl(S, ELoc, Type, D->getName(),
                                  D->hasAttrs() ? &D->getAttrs() : nullptr);
    QualType PrivateTy = Type;

    // Try if we can determine constant lengths for all array sections and avoid
    // the VLA.
    bool ConstantLengthOASE = false;
    if (OASE) {
      bool SingleElement;
      llvm::SmallVector<llvm::APSInt, 4> ArraySizes;
      ConstantLengthOASE = checkOMPArraySectionConstantForReduction(
          Context, OASE, SingleElement, ArraySizes);

      // If we don't have a single element, we must emit a constant array type.
      if (ConstantLengthOASE && !SingleElement) {
        for (llvm::APSInt &Size : ArraySizes)
          PrivateTy = Context.getConstantArrayType(PrivateTy, Size, nullptr,
                                                   ArrayType::Normal,
                                                   /*IndexTypeQuals=*/0);
      }
    }

    if ((OASE && !ConstantLengthOASE) ||
        (!OASE && !ASE &&
         D->getType().getNonReferenceType()->isVariablyModifiedType())) {
      if (!Context.getTargetInfo().isVLASupported()) {
        if (isOpenMPTargetExecutionDirective(Stack->getCurrentDirective())) {
          S.Diag(ELoc, diag::err_omp_reduction_vla_unsupported) << !!OASE;
          S.Diag(ELoc, diag::note_vla_unsupported);
          continue;
        } else {
          S.targetDiag(ELoc, diag::err_omp_reduction_vla_unsupported) << !!OASE;
          S.targetDiag(ELoc, diag::note_vla_unsupported);
        }
      }
      // For arrays/array sections only:
      // Create pseudo array type for private copy. The size for this array will
      // be generated during codegen.
      // For array subscripts or single variables Private Ty is the same as Type
      // (type of the variable or single array element).
      PrivateTy = Context.getVariableArrayType(
          Type,
          new (Context)
              OpaqueValueExpr(ELoc, Context.getSizeType(), VK_PRValue),
          ArrayType::Normal, /*IndexTypeQuals=*/0, SourceRange());
    } else if (!ASE && !OASE &&
               Context.getAsArrayType(D->getType().getNonReferenceType())) {
      PrivateTy = D->getType().getNonReferenceType();
    }
    // Private copy.
    VarDecl *PrivateVD =
        buildVarDecl(S, ELoc, PrivateTy, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr,
                     VD ? cast<DeclRefExpr>(SimpleRefExpr) : nullptr);
    // Add initializer for private variable.
    Expr *Init = nullptr;
    DeclRefExpr *LHSDRE = buildDeclRefExpr(S, LHSVD, Type, ELoc);
    DeclRefExpr *RHSDRE = buildDeclRefExpr(S, RHSVD, Type, ELoc);
    if (DeclareReductionRef.isUsable()) {
      auto *DRDRef = DeclareReductionRef.getAs<DeclRefExpr>();
      auto *DRD = cast<OMPDeclareReductionDecl>(DRDRef->getDecl());
      if (DRD->getInitializer()) {
        Init = DRDRef;
        RHSVD->setInit(DRDRef);
        RHSVD->setInitStyle(VarDecl::CallInit);
      }
    } else {
      switch (BOK) {
      case BO_Add:
      case BO_Xor:
      case BO_Or:
      case BO_LOr:
        // '+', '-', '^', '|', '||' reduction ops - initializer is '0'.
        if (Type->isScalarType() || Type->isAnyComplexType())
          Init = S.ActOnIntegerConstant(ELoc, /*Val=*/0).get();
        break;
      case BO_Mul:
      case BO_LAnd:
        if (Type->isScalarType() || Type->isAnyComplexType()) {
          // '*' and '&&' reduction ops - initializer is '1'.
          Init = S.ActOnIntegerConstant(ELoc, /*Val=*/1).get();
        }
        break;
      case BO_And: {
        // '&' reduction op - initializer is '~0'.
        QualType OrigType = Type;
        if (auto *ComplexTy = OrigType->getAs<ComplexType>())
          Type = ComplexTy->getElementType();
        if (Type->isRealFloatingType()) {
          llvm::APFloat InitValue = llvm::APFloat::getAllOnesValue(
              Context.getFloatTypeSemantics(Type));
          Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                         Type, ELoc);
        } else if (Type->isScalarType()) {
          uint64_t Size = Context.getTypeSize(Type);
          QualType IntTy = Context.getIntTypeForBitwidth(Size, /*Signed=*/0);
          llvm::APInt InitValue = llvm::APInt::getAllOnes(Size);
          Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
        }
        if (Init && OrigType->isAnyComplexType()) {
          // Init = 0xFFFF + 0xFFFFi;
          auto *Im = new (Context) ImaginaryLiteral(Init, OrigType);
          Init = S.CreateBuiltinBinOp(ELoc, BO_Add, Init, Im).get();
        }
        Type = OrigType;
        break;
      }
      case BO_LT:
      case BO_GT: {
        // 'min' reduction op - initializer is 'Largest representable number in
        // the reduction list item type'.
        // 'max' reduction op - initializer is 'Least representable number in
        // the reduction list item type'.
        if (Type->isIntegerType() || Type->isPointerType()) {
          bool IsSigned = Type->hasSignedIntegerRepresentation();
          uint64_t Size = Context.getTypeSize(Type);
          QualType IntTy =
              Context.getIntTypeForBitwidth(Size, /*Signed=*/IsSigned);
          llvm::APInt InitValue =
              (BOK != BO_LT) ? IsSigned ? llvm::APInt::getSignedMinValue(Size)
                                        : llvm::APInt::getMinValue(Size)
              : IsSigned ? llvm::APInt::getSignedMaxValue(Size)
                             : llvm::APInt::getMaxValue(Size);
          Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
          if (Type->isPointerType()) {
            // Cast to pointer type.
            ExprResult CastExpr = S.BuildCStyleCastExpr(
                ELoc, Context.getTrivialTypeSourceInfo(Type, ELoc), ELoc, Init);
            if (CastExpr.isInvalid())
              continue;
            Init = CastExpr.get();
          }
        } else if (Type->isRealFloatingType()) {
          llvm::APFloat InitValue = llvm::APFloat::getLargest(
              Context.getFloatTypeSemantics(Type), BOK != BO_LT);
          Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                         Type, ELoc);
        }
        break;
      }
      case BO_PtrMemD:
      case BO_PtrMemI:
      case BO_MulAssign:
      case BO_Div:
      case BO_Rem:
      case BO_Sub:
      case BO_Shl:
      case BO_Shr:
      case BO_LE:
      case BO_GE:
      case BO_EQ:
      case BO_NE:
      case BO_Cmp:
      case BO_AndAssign:
      case BO_XorAssign:
      case BO_OrAssign:
      case BO_Assign:
      case BO_AddAssign:
      case BO_SubAssign:
      case BO_DivAssign:
      case BO_RemAssign:
      case BO_ShlAssign:
      case BO_ShrAssign:
      case BO_Comma:
        llvm_unreachable("Unexpected reduction operation");
      }
    }
    if (Init && DeclareReductionRef.isUnset()) {
      S.AddInitializerToDecl(RHSVD, Init, /*DirectInit=*/false);
      // Store initializer for single element in private copy. Will be used
      // during codegen.
      PrivateVD->setInit(RHSVD->getInit());
      PrivateVD->setInitStyle(RHSVD->getInitStyle());
    } else if (!Init) {
      S.ActOnUninitializedDecl(RHSVD);
      // Store initializer for single element in private copy. Will be used
      // during codegen.
      PrivateVD->setInit(RHSVD->getInit());
      PrivateVD->setInitStyle(RHSVD->getInitStyle());
    }
    if (RHSVD->isInvalidDecl())
      continue;
    if (!RHSVD->hasInit() && DeclareReductionRef.isUnset()) {
      S.Diag(ELoc, diag::err_omp_reduction_id_not_compatible)
          << Type << ReductionIdRange;
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      S.Diag(D->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }
    DeclRefExpr *PrivateDRE = buildDeclRefExpr(S, PrivateVD, PrivateTy, ELoc);
    ExprResult ReductionOp;
    if (DeclareReductionRef.isUsable()) {
      QualType RedTy = DeclareReductionRef.get()->getType();
      QualType PtrRedTy = Context.getPointerType(RedTy);
      ExprResult LHS = S.CreateBuiltinUnaryOp(ELoc, UO_AddrOf, LHSDRE);
      ExprResult RHS = S.CreateBuiltinUnaryOp(ELoc, UO_AddrOf, RHSDRE);
      if (!BasePath.empty()) {
        LHS = S.DefaultLvalueConversion(LHS.get());
        RHS = S.DefaultLvalueConversion(RHS.get());
        LHS = ImplicitCastExpr::Create(
            Context, PtrRedTy, CK_UncheckedDerivedToBase, LHS.get(), &BasePath,
            LHS.get()->getValueKind(), FPOptionsOverride());
        RHS = ImplicitCastExpr::Create(
            Context, PtrRedTy, CK_UncheckedDerivedToBase, RHS.get(), &BasePath,
            RHS.get()->getValueKind(), FPOptionsOverride());
      }
      FunctionProtoType::ExtProtoInfo EPI;
      QualType Params[] = {PtrRedTy, PtrRedTy};
      QualType FnTy = Context.getFunctionType(Context.VoidTy, Params, EPI);
      auto *OVE = new (Context) OpaqueValueExpr(
          ELoc, Context.getPointerType(FnTy), VK_PRValue, OK_Ordinary,
          S.DefaultLvalueConversion(DeclareReductionRef.get()).get());
      Expr *Args[] = {LHS.get(), RHS.get()};
      ReductionOp =
          CallExpr::Create(Context, OVE, Args, Context.VoidTy, VK_PRValue, ELoc,
                           S.CurFPFeatureOverrides());
    } else {
      BinaryOperatorKind CombBOK = getRelatedCompoundReductionOp(BOK);
      if (Type->isRecordType() && CombBOK != BOK) {
        Sema::TentativeAnalysisScope Trap(S);
        ReductionOp =
            S.BuildBinOp(Stack->getCurScope(), ReductionId.getBeginLoc(),
                         CombBOK, LHSDRE, RHSDRE);
      }
      if (!ReductionOp.isUsable()) {
        ReductionOp =
            S.BuildBinOp(Stack->getCurScope(), ReductionId.getBeginLoc(), BOK,
                         LHSDRE, RHSDRE);
        if (ReductionOp.isUsable()) {
          if (BOK != BO_LT && BOK != BO_GT) {
            ReductionOp =
                S.BuildBinOp(Stack->getCurScope(), ReductionId.getBeginLoc(),
                             BO_Assign, LHSDRE, ReductionOp.get());
          } else {
            auto *ConditionalOp = new (Context)
                ConditionalOperator(ReductionOp.get(), ELoc, LHSDRE, ELoc,
                                    RHSDRE, Type, VK_LValue, OK_Ordinary);
            ReductionOp =
                S.BuildBinOp(Stack->getCurScope(), ReductionId.getBeginLoc(),
                             BO_Assign, LHSDRE, ConditionalOp);
          }
        }
      }
      if (ReductionOp.isUsable())
        ReductionOp = S.ActOnFinishFullExpr(ReductionOp.get(),
                                            /*DiscardedValue*/ false);
      if (!ReductionOp.isUsable())
        continue;
    }

    // Add copy operations for inscan reductions.
    // LHS = RHS;
    ExprResult CopyOpRes, TempArrayRes, TempArrayElem;
    if (ClauseKind == OMPC_reduction &&
        RD.RedModifier == OMPC_REDUCTION_inscan) {
      ExprResult RHS = S.DefaultLvalueConversion(RHSDRE);
      CopyOpRes = S.BuildBinOp(Stack->getCurScope(), ELoc, BO_Assign, LHSDRE,
                               RHS.get());
      if (!CopyOpRes.isUsable())
        continue;
      CopyOpRes =
          S.ActOnFinishFullExpr(CopyOpRes.get(), /*DiscardedValue=*/true);
      if (!CopyOpRes.isUsable())
        continue;
      // For simd directive and simd-based directives in simd mode no need to
      // construct temp array, need just a single temp element.
      if (Stack->getCurrentDirective() == OMPD_simd ||
          (S.getLangOpts().OpenMPSimd &&
           isOpenMPSimdDirective(Stack->getCurrentDirective()))) {
        VarDecl *TempArrayVD =
            buildVarDecl(S, ELoc, PrivateTy, D->getName(),
                         D->hasAttrs() ? &D->getAttrs() : nullptr);
        // Add a constructor to the temp decl.
        S.ActOnUninitializedDecl(TempArrayVD);
        TempArrayRes = buildDeclRefExpr(S, TempArrayVD, PrivateTy, ELoc);
      } else {
        // Build temp array for prefix sum.
        auto *Dim = new (S.Context)
            OpaqueValueExpr(ELoc, S.Context.getSizeType(), VK_PRValue);
        QualType ArrayTy =
            S.Context.getVariableArrayType(PrivateTy, Dim, ArrayType::Normal,
                                           /*IndexTypeQuals=*/0, {ELoc, ELoc});
        VarDecl *TempArrayVD =
            buildVarDecl(S, ELoc, ArrayTy, D->getName(),
                         D->hasAttrs() ? &D->getAttrs() : nullptr);
        // Add a constructor to the temp decl.
        S.ActOnUninitializedDecl(TempArrayVD);
        TempArrayRes = buildDeclRefExpr(S, TempArrayVD, ArrayTy, ELoc);
        TempArrayElem =
            S.DefaultFunctionArrayLvalueConversion(TempArrayRes.get());
        auto *Idx = new (S.Context)
            OpaqueValueExpr(ELoc, S.Context.getSizeType(), VK_PRValue);
        TempArrayElem = S.CreateBuiltinArraySubscriptExpr(TempArrayElem.get(),
                                                          ELoc, Idx, ELoc);
      }
    }

    // OpenMP [2.15.4.6, Restrictions, p.2]
    // A list item that appears in an in_reduction clause of a task construct
    // must appear in a task_reduction clause of a construct associated with a
    // taskgroup region that includes the participating task in its taskgroup
    // set. The construct associated with the innermost region that meets this
    // condition must specify the same reduction-identifier as the in_reduction
    // clause.
    if (ClauseKind == OMPC_in_reduction) {
      SourceRange ParentSR;
      BinaryOperatorKind ParentBOK;
      const Expr *ParentReductionOp = nullptr;
      Expr *ParentBOKTD = nullptr, *ParentReductionOpTD = nullptr;
      DSAStackTy::DSAVarData ParentBOKDSA =
          Stack->getTopMostTaskgroupReductionData(D, ParentSR, ParentBOK,
                                                  ParentBOKTD);
      DSAStackTy::DSAVarData ParentReductionOpDSA =
          Stack->getTopMostTaskgroupReductionData(
              D, ParentSR, ParentReductionOp, ParentReductionOpTD);
      bool IsParentBOK = ParentBOKDSA.DKind != OMPD_unknown;
      bool IsParentReductionOp = ParentReductionOpDSA.DKind != OMPD_unknown;
      if ((DeclareReductionRef.isUnset() && IsParentReductionOp) ||
          (DeclareReductionRef.isUsable() && IsParentBOK) ||
          (IsParentBOK && BOK != ParentBOK) || IsParentReductionOp) {
        bool EmitError = true;
        if (IsParentReductionOp && DeclareReductionRef.isUsable()) {
          llvm::FoldingSetNodeID RedId, ParentRedId;
          ParentReductionOp->Profile(ParentRedId, Context, /*Canonical=*/true);
          DeclareReductionRef.get()->Profile(RedId, Context,
                                             /*Canonical=*/true);
          EmitError = RedId != ParentRedId;
        }
        if (EmitError) {
          S.Diag(ReductionId.getBeginLoc(),
                 diag::err_omp_reduction_identifier_mismatch)
              << ReductionIdRange << RefExpr->getSourceRange();
          S.Diag(ParentSR.getBegin(),
                 diag::note_omp_previous_reduction_identifier)
              << ParentSR
              << (IsParentBOK ? ParentBOKDSA.RefExpr
                              : ParentReductionOpDSA.RefExpr)
                     ->getSourceRange();
          continue;
        }
      }
      TaskgroupDescriptor = IsParentBOK ? ParentBOKTD : ParentReductionOpTD;
    }

    DeclRefExpr *Ref = nullptr;
    Expr *VarsExpr = RefExpr->IgnoreParens();
    if (!VD && !S.CurContext->isDependentContext()) {
      if (ASE || OASE) {
        TransformExprToCaptures RebuildToCapture(S, D);
        VarsExpr =
            RebuildToCapture.TransformExpr(RefExpr->IgnoreParens()).get();
        Ref = RebuildToCapture.getCapturedExpr();
      } else {
        VarsExpr = Ref = buildCapture(S, D, SimpleRefExpr, /*WithInit=*/false);
      }
      if (!S.isOpenMPCapturedDecl(D)) {
        RD.ExprCaptures.emplace_back(Ref->getDecl());
        if (Ref->getDecl()->hasAttr<OMPCaptureNoInitAttr>()) {
          ExprResult RefRes = S.DefaultLvalueConversion(Ref);
          if (!RefRes.isUsable())
            continue;
          ExprResult PostUpdateRes =
              S.BuildBinOp(Stack->getCurScope(), ELoc, BO_Assign, SimpleRefExpr,
                           RefRes.get());
          if (!PostUpdateRes.isUsable())
            continue;
          if (isOpenMPTaskingDirective(Stack->getCurrentDirective()) ||
              Stack->getCurrentDirective() == OMPD_taskgroup) {
            S.Diag(RefExpr->getExprLoc(),
                   diag::err_omp_reduction_non_addressable_expression)
                << RefExpr->getSourceRange();
            continue;
          }
          RD.ExprPostUpdates.emplace_back(
              S.IgnoredValueConversions(PostUpdateRes.get()).get());
        }
      }
    }
    // All reduction items are still marked as reduction (to do not increase
    // code base size).
    unsigned Modifier = RD.RedModifier;
    // Consider task_reductions as reductions with task modifier. Required for
    // correct analysis of in_reduction clauses.
    if (CurrDir == OMPD_taskgroup && ClauseKind == OMPC_task_reduction)
      Modifier = OMPC_REDUCTION_task;
    Stack->addDSA(D, RefExpr->IgnoreParens(), OMPC_reduction, Ref, Modifier,
                  ASE || OASE);
    if (Modifier == OMPC_REDUCTION_task &&
        (CurrDir == OMPD_taskgroup ||
         ((isOpenMPParallelDirective(CurrDir) ||
           isOpenMPWorksharingDirective(CurrDir)) &&
          !isOpenMPSimdDirective(CurrDir)))) {
      if (DeclareReductionRef.isUsable())
        Stack->addTaskgroupReductionData(D, ReductionIdRange,
                                         DeclareReductionRef.get());
      else
        Stack->addTaskgroupReductionData(D, ReductionIdRange, BOK);
    }
    RD.push(VarsExpr, PrivateDRE, LHSDRE, RHSDRE, ReductionOp.get(),
            TaskgroupDescriptor, CopyOpRes.get(), TempArrayRes.get(),
            TempArrayElem.get());
  }
  return RD.Vars.empty();
}

OMPClause *Sema::ActOnOpenMPReductionClause(
    ArrayRef<Expr *> VarList, OpenMPReductionClauseModifier Modifier,
    SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ModifierLoc, SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec, const DeclarationNameInfo &ReductionId,
    ArrayRef<Expr *> UnresolvedReductions) {
  if (ModifierLoc.isValid() && Modifier == OMPC_REDUCTION_unknown) {
    Diag(LParenLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_reduction, /*First=*/0,
                                   /*Last=*/OMPC_REDUCTION_unknown)
        << getOpenMPClauseName(OMPC_reduction);
    return nullptr;
  }
  // OpenMP 5.0, 2.19.5.4 reduction Clause, Restrictions
  // A reduction clause with the inscan reduction-modifier may only appear on a
  // worksharing-loop construct, a worksharing-loop SIMD construct, a simd
  // construct, a parallel worksharing-loop construct or a parallel
  // worksharing-loop SIMD construct.
  if (Modifier == OMPC_REDUCTION_inscan &&
      (DSAStack->getCurrentDirective() != OMPD_for &&
       DSAStack->getCurrentDirective() != OMPD_for_simd &&
       DSAStack->getCurrentDirective() != OMPD_simd &&
       DSAStack->getCurrentDirective() != OMPD_parallel_for &&
       DSAStack->getCurrentDirective() != OMPD_parallel_for_simd)) {
    Diag(ModifierLoc, diag::err_omp_wrong_inscan_reduction);
    return nullptr;
  }

  ReductionData RD(VarList.size(), Modifier);
  if (actOnOMPReductionKindClause(*this, DSAStack, OMPC_reduction, VarList,
                                  StartLoc, LParenLoc, ColonLoc, EndLoc,
                                  ReductionIdScopeSpec, ReductionId,
                                  UnresolvedReductions, RD))
    return nullptr;

  return OMPReductionClause::Create(
      Context, StartLoc, LParenLoc, ModifierLoc, ColonLoc, EndLoc, Modifier,
      RD.Vars, ReductionIdScopeSpec.getWithLocInContext(Context), ReductionId,
      RD.Privates, RD.LHSs, RD.RHSs, RD.ReductionOps, RD.InscanCopyOps,
      RD.InscanCopyArrayTemps, RD.InscanCopyArrayElems,
      buildPreInits(Context, RD.ExprCaptures),
      buildPostUpdate(*this, RD.ExprPostUpdates));
}

OMPClause *Sema::ActOnOpenMPTaskReductionClause(
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec, const DeclarationNameInfo &ReductionId,
    ArrayRef<Expr *> UnresolvedReductions) {
  ReductionData RD(VarList.size());
  if (actOnOMPReductionKindClause(*this, DSAStack, OMPC_task_reduction, VarList,
                                  StartLoc, LParenLoc, ColonLoc, EndLoc,
                                  ReductionIdScopeSpec, ReductionId,
                                  UnresolvedReductions, RD))
    return nullptr;

  return OMPTaskReductionClause::Create(
      Context, StartLoc, LParenLoc, ColonLoc, EndLoc, RD.Vars,
      ReductionIdScopeSpec.getWithLocInContext(Context), ReductionId,
      RD.Privates, RD.LHSs, RD.RHSs, RD.ReductionOps,
      buildPreInits(Context, RD.ExprCaptures),
      buildPostUpdate(*this, RD.ExprPostUpdates));
}

OMPClause *Sema::ActOnOpenMPInReductionClause(
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec, const DeclarationNameInfo &ReductionId,
    ArrayRef<Expr *> UnresolvedReductions) {
  ReductionData RD(VarList.size());
  if (actOnOMPReductionKindClause(*this, DSAStack, OMPC_in_reduction, VarList,
                                  StartLoc, LParenLoc, ColonLoc, EndLoc,
                                  ReductionIdScopeSpec, ReductionId,
                                  UnresolvedReductions, RD))
    return nullptr;

  return OMPInReductionClause::Create(
      Context, StartLoc, LParenLoc, ColonLoc, EndLoc, RD.Vars,
      ReductionIdScopeSpec.getWithLocInContext(Context), ReductionId,
      RD.Privates, RD.LHSs, RD.RHSs, RD.ReductionOps, RD.TaskgroupDescriptors,
      buildPreInits(Context, RD.ExprCaptures),
      buildPostUpdate(*this, RD.ExprPostUpdates));
}

bool Sema::CheckOpenMPLinearModifier(OpenMPLinearClauseKind LinKind,
                                     SourceLocation LinLoc) {
  if ((!LangOpts.CPlusPlus && LinKind != OMPC_LINEAR_val) ||
      LinKind == OMPC_LINEAR_unknown) {
    Diag(LinLoc, diag::err_omp_wrong_linear_modifier) << LangOpts.CPlusPlus;
    return true;
  }
  return false;
}

bool Sema::CheckOpenMPLinearDecl(const ValueDecl *D, SourceLocation ELoc,
                                 OpenMPLinearClauseKind LinKind, QualType Type,
                                 bool IsDeclareSimd) {
  const auto *VD = dyn_cast_or_null<VarDecl>(D);
  // A variable must not have an incomplete type or a reference type.
  if (RequireCompleteType(ELoc, Type, diag::err_omp_linear_incomplete_type))
    return true;
  if ((LinKind == OMPC_LINEAR_uval || LinKind == OMPC_LINEAR_ref) &&
      !Type->isReferenceType()) {
    Diag(ELoc, diag::err_omp_wrong_linear_modifier_non_reference)
        << Type << getOpenMPSimpleClauseTypeName(OMPC_linear, LinKind);
    return true;
  }
  Type = Type.getNonReferenceType();

  // OpenMP 5.0 [2.19.3, List Item Privatization, Restrictions]
  // A variable that is privatized must not have a const-qualified type
  // unless it is of class type with a mutable member. This restriction does
  // not apply to the firstprivate clause, nor to the linear clause on
  // declarative directives (like declare simd).
  if (!IsDeclareSimd &&
      rejectConstNotMutableType(*this, D, Type, OMPC_linear, ELoc))
    return true;

  // A list item must be of integral or pointer type.
  Type = Type.getUnqualifiedType().getCanonicalType();
  const auto *Ty = Type.getTypePtrOrNull();
  if (!Ty || (LinKind != OMPC_LINEAR_ref && !Ty->isDependentType() &&
              !Ty->isIntegralType(Context) && !Ty->isPointerType())) {
    Diag(ELoc, diag::err_omp_linear_expected_int_or_ptr) << Type;
    if (D) {
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      Diag(D->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
    }
    return true;
  }
  return false;
}

OMPClause *Sema::ActOnOpenMPLinearClause(
    ArrayRef<Expr *> VarList, Expr *Step, SourceLocation StartLoc,
    SourceLocation LParenLoc, OpenMPLinearClauseKind LinKind,
    SourceLocation LinLoc, SourceLocation ColonLoc, SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> Privates;
  SmallVector<Expr *, 8> Inits;
  SmallVector<Decl *, 4> ExprCaptures;
  SmallVector<Expr *, 4> ExprPostUpdates;
  if (CheckOpenMPLinearModifier(LinKind, LinLoc))
    LinKind = OMPC_LINEAR_val;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP linear clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      Privates.push_back(nullptr);
      Inits.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    QualType Type = D->getType();
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP [2.14.3.7, linear clause]
    //  A list-item cannot appear in more than one linear clause.
    //  A list-item that appears in a linear clause cannot appear in any
    //  other data-sharing attribute clause.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(D, /*FromParent=*/false);
    if (DVar.RefExpr) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_linear);
      reportOriginalDsa(*this, DSAStack, D, DVar);
      continue;
    }

    if (CheckOpenMPLinearDecl(D, ELoc, LinKind, Type))
      continue;
    Type = Type.getNonReferenceType().getUnqualifiedType().getCanonicalType();

    // Build private copy of original var.
    VarDecl *Private =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr,
                     VD ? cast<DeclRefExpr>(SimpleRefExpr) : nullptr);
    DeclRefExpr *PrivateRef = buildDeclRefExpr(*this, Private, Type, ELoc);
    // Build var to save initial value.
    VarDecl *Init = buildVarDecl(*this, ELoc, Type, ".linear.start");
    Expr *InitExpr;
    DeclRefExpr *Ref = nullptr;
    if (!VD && !CurContext->isDependentContext()) {
      Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/false);
      if (!isOpenMPCapturedDecl(D)) {
        ExprCaptures.push_back(Ref->getDecl());
        if (Ref->getDecl()->hasAttr<OMPCaptureNoInitAttr>()) {
          ExprResult RefRes = DefaultLvalueConversion(Ref);
          if (!RefRes.isUsable())
            continue;
          ExprResult PostUpdateRes =
              BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign,
                         SimpleRefExpr, RefRes.get());
          if (!PostUpdateRes.isUsable())
            continue;
          ExprPostUpdates.push_back(
              IgnoredValueConversions(PostUpdateRes.get()).get());
        }
      }
    }
    if (LinKind == OMPC_LINEAR_uval)
      InitExpr = VD ? VD->getInit() : SimpleRefExpr;
    else
      InitExpr = VD ? SimpleRefExpr : Ref;
    AddInitializerToDecl(Init, DefaultLvalueConversion(InitExpr).get(),
                         /*DirectInit=*/false);
    DeclRefExpr *InitRef = buildDeclRefExpr(*this, Init, Type, ELoc);

    DSAStack->addDSA(D, RefExpr->IgnoreParens(), OMPC_linear, Ref);
    Vars.push_back((VD || CurContext->isDependentContext())
                       ? RefExpr->IgnoreParens()
                       : Ref);
    Privates.push_back(PrivateRef);
    Inits.push_back(InitRef);
  }

  if (Vars.empty())
    return nullptr;

  Expr *StepExpr = Step;
  Expr *CalcStepExpr = nullptr;
  if (Step && !Step->isValueDependent() && !Step->isTypeDependent() &&
      !Step->isInstantiationDependent() &&
      !Step->containsUnexpandedParameterPack()) {
    SourceLocation StepLoc = Step->getBeginLoc();
    ExprResult Val = PerformOpenMPImplicitIntegerConversion(StepLoc, Step);
    if (Val.isInvalid())
      return nullptr;
    StepExpr = Val.get();

    // Build var to save the step value.
    VarDecl *SaveVar =
        buildVarDecl(*this, StepLoc, StepExpr->getType(), ".linear.step");
    ExprResult SaveRef =
        buildDeclRefExpr(*this, SaveVar, StepExpr->getType(), StepLoc);
    ExprResult CalcStep =
        BuildBinOp(CurScope, StepLoc, BO_Assign, SaveRef.get(), StepExpr);
    CalcStep = ActOnFinishFullExpr(CalcStep.get(), /*DiscardedValue*/ false);

    // Warn about zero linear step (it would be probably better specified as
    // making corresponding variables 'const').
    if (Optional<llvm::APSInt> Result =
            StepExpr->getIntegerConstantExpr(Context)) {
      if (!Result->isNegative() && !Result->isStrictlyPositive())
        Diag(StepLoc, diag::warn_omp_linear_step_zero)
            << Vars[0] << (Vars.size() > 1);
    } else if (CalcStep.isUsable()) {
      // Calculate the step beforehand instead of doing this on each iteration.
      // (This is not used if the number of iterations may be kfold-ed).
      CalcStepExpr = CalcStep.get();
    }
  }

  return OMPLinearClause::Create(Context, StartLoc, LParenLoc, LinKind, LinLoc,
                                 ColonLoc, EndLoc, Vars, Privates, Inits,
                                 StepExpr, CalcStepExpr,
                                 buildPreInits(Context, ExprCaptures),
                                 buildPostUpdate(*this, ExprPostUpdates));
}

static bool FinishOpenMPLinearClause(OMPLinearClause &Clause, DeclRefExpr *IV,
                                     Expr *NumIterations, Sema &SemaRef,
                                     Scope *S, DSAStackTy *Stack) {
  // Walk the vars and build update/final expressions for the CodeGen.
  SmallVector<Expr *, 8> Updates;
  SmallVector<Expr *, 8> Finals;
  SmallVector<Expr *, 8> UsedExprs;
  Expr *Step = Clause.getStep();
  Expr *CalcStep = Clause.getCalcStep();
  // OpenMP [2.14.3.7, linear clause]
  // If linear-step is not specified it is assumed to be 1.
  if (!Step)
    Step = SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get();
  else if (CalcStep)
    Step = cast<BinaryOperator>(CalcStep)->getLHS();
  bool HasErrors = false;
  auto CurInit = Clause.inits().begin();
  auto CurPrivate = Clause.privates().begin();
  OpenMPLinearClauseKind LinKind = Clause.getModifier();
  for (Expr *RefExpr : Clause.varlists()) {
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(SemaRef, SimpleRefExpr, ELoc, ERange);
    ValueDecl *D = Res.first;
    if (Res.second || !D) {
      Updates.push_back(nullptr);
      Finals.push_back(nullptr);
      HasErrors = true;
      continue;
    }
    auto &&Info = Stack->isLoopControlVariable(D);
    // OpenMP [2.15.11, distribute simd Construct]
    // A list item may not appear in a linear clause, unless it is the loop
    // iteration variable.
    if (isOpenMPDistributeDirective(Stack->getCurrentDirective()) &&
        isOpenMPSimdDirective(Stack->getCurrentDirective()) && !Info.first) {
      SemaRef.Diag(ELoc,
                   diag::err_omp_linear_distribute_var_non_loop_iteration);
      Updates.push_back(nullptr);
      Finals.push_back(nullptr);
      HasErrors = true;
      continue;
    }
    Expr *InitExpr = *CurInit;

    // Build privatized reference to the current linear var.
    auto *DE = cast<DeclRefExpr>(SimpleRefExpr);
    Expr *CapturedRef;
    if (LinKind == OMPC_LINEAR_uval)
      CapturedRef = cast<VarDecl>(DE->getDecl())->getInit();
    else
      CapturedRef =
          buildDeclRefExpr(SemaRef, cast<VarDecl>(DE->getDecl()),
                           DE->getType().getUnqualifiedType(), DE->getExprLoc(),
                           /*RefersToCapture=*/true);

    // Build update: Var = InitExpr + IV * Step
    ExprResult Update;
    if (!Info.first)
      Update = buildCounterUpdate(
          SemaRef, S, RefExpr->getExprLoc(), *CurPrivate, InitExpr, IV, Step,
          /*Subtract=*/false, /*IsNonRectangularLB=*/false);
    else
      Update = *CurPrivate;
    Update = SemaRef.ActOnFinishFullExpr(Update.get(), DE->getBeginLoc(),
                                         /*DiscardedValue*/ false);

    // Build final: Var = PrivCopy;
    ExprResult Final;
    if (!Info.first)
      Final = SemaRef.BuildBinOp(
          S, RefExpr->getExprLoc(), BO_Assign, CapturedRef,
          SemaRef.DefaultLvalueConversion(*CurPrivate).get());
    else
      Final = *CurPrivate;
    Final = SemaRef.ActOnFinishFullExpr(Final.get(), DE->getBeginLoc(),
                                        /*DiscardedValue*/ false);

    if (!Update.isUsable() || !Final.isUsable()) {
      Updates.push_back(nullptr);
      Finals.push_back(nullptr);
      UsedExprs.push_back(nullptr);
      HasErrors = true;
    } else {
      Updates.push_back(Update.get());
      Finals.push_back(Final.get());
      if (!Info.first)
        UsedExprs.push_back(SimpleRefExpr);
    }
    ++CurInit;
    ++CurPrivate;
  }
  if (Expr *S = Clause.getStep())
    UsedExprs.push_back(S);
  // Fill the remaining part with the nullptr.
  UsedExprs.append(Clause.varlist_size() + 1 - UsedExprs.size(), nullptr);
  Clause.setUpdates(Updates);
  Clause.setFinals(Finals);
  Clause.setUsedExprs(UsedExprs);
  return HasErrors;
}

OMPClause *Sema::ActOnOpenMPAlignedClause(
    ArrayRef<Expr *> VarList, Expr *Alignment, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP linear clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    QualType QType = D->getType();
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP  [2.8.1, simd construct, Restrictions]
    // The type of list items appearing in the aligned clause must be
    // array, pointer, reference to array, or reference to pointer.
    QType = QType.getNonReferenceType().getUnqualifiedType().getCanonicalType();
    const Type *Ty = QType.getTypePtrOrNull();
    if (!Ty || (!Ty->isArrayType() && !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_aligned_expected_array_or_ptr)
          << QType << getLangOpts().CPlusPlus << ERange;
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      Diag(D->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }

    // OpenMP  [2.8.1, simd construct, Restrictions]
    // A list-item cannot appear in more than one aligned clause.
    if (const Expr *PrevRef = DSAStack->addUniqueAligned(D, SimpleRefExpr)) {
      Diag(ELoc, diag::err_omp_used_in_clause_twice)
          << 0 << getOpenMPClauseName(OMPC_aligned) << ERange;
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(OMPC_aligned);
      continue;
    }

    DeclRefExpr *Ref = nullptr;
    if (!VD && isOpenMPCapturedDecl(D))
      Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/true);
    Vars.push_back(DefaultFunctionArrayConversion(
                       (VD || !Ref) ? RefExpr->IgnoreParens() : Ref)
                       .get());
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
  SmallVector<Expr *, 8> SrcExprs;
  SmallVector<Expr *, 8> DstExprs;
  SmallVector<Expr *, 8> AssignmentOps;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP copyin clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.4.1, Restrictions, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    auto *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name_member_expr)
          << 0 << RefExpr->getSourceRange();
      continue;
    }

    Decl *D = DE->getDecl();
    auto *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
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
    QualType ElemType = Context.getBaseElementType(Type).getNonReferenceType();
    VarDecl *SrcVD =
        buildVarDecl(*this, DE->getBeginLoc(), ElemType.getUnqualifiedType(),
                     ".copyin.src", VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    DeclRefExpr *PseudoSrcExpr = buildDeclRefExpr(
        *this, SrcVD, ElemType.getUnqualifiedType(), DE->getExprLoc());
    VarDecl *DstVD =
        buildVarDecl(*this, DE->getBeginLoc(), ElemType, ".copyin.dst",
                     VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    DeclRefExpr *PseudoDstExpr =
        buildDeclRefExpr(*this, DstVD, ElemType, DE->getExprLoc());
    // For arrays generate assignment operation for single element and replace
    // it by the original array element in CodeGen.
    ExprResult AssignmentOp =
        BuildBinOp(/*S=*/nullptr, DE->getExprLoc(), BO_Assign, PseudoDstExpr,
                   PseudoSrcExpr);
    if (AssignmentOp.isInvalid())
      continue;
    AssignmentOp = ActOnFinishFullExpr(AssignmentOp.get(), DE->getExprLoc(),
                                       /*DiscardedValue*/ false);
    if (AssignmentOp.isInvalid())
      continue;

    DSAStack->addDSA(VD, DE, OMPC_copyin);
    Vars.push_back(DE);
    SrcExprs.push_back(PseudoSrcExpr);
    DstExprs.push_back(PseudoDstExpr);
    AssignmentOps.push_back(AssignmentOp.get());
  }

  if (Vars.empty())
    return nullptr;

  return OMPCopyinClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars,
                                 SrcExprs, DstExprs, AssignmentOps);
}

OMPClause *Sema::ActOnOpenMPCopyprivateClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> SrcExprs;
  SmallVector<Expr *, 8> DstExprs;
  SmallVector<Expr *, 8> AssignmentOps;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP linear clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    QualType Type = D->getType();
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP [2.14.4.2, Restrictions, p.2]
    //  A list item that appears in a copyprivate clause may not appear in a
    //  private or firstprivate clause on the single construct.
    if (!VD || !DSAStack->isThreadPrivate(VD)) {
      DSAStackTy::DSAVarData DVar =
          DSAStack->getTopDSA(D, /*FromParent=*/false);
      if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_copyprivate &&
          DVar.RefExpr) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_copyprivate);
        reportOriginalDsa(*this, DSAStack, D, DVar);
        continue;
      }

      // OpenMP [2.11.4.2, Restrictions, p.1]
      //  All list items that appear in a copyprivate clause must be either
      //  threadprivate or private in the enclosing context.
      if (DVar.CKind == OMPC_unknown) {
        DVar = DSAStack->getImplicitDSA(D, false);
        if (DVar.CKind == OMPC_shared) {
          Diag(ELoc, diag::err_omp_required_access)
              << getOpenMPClauseName(OMPC_copyprivate)
              << "threadprivate or private in the enclosing context";
          reportOriginalDsa(*this, DSAStack, D, DVar);
          continue;
        }
      }
    }

    // Variably modified types are not supported.
    if (!Type->isAnyPointerType() && Type->isVariablyModifiedType()) {
      Diag(ELoc, diag::err_omp_variably_modified_type_not_supported)
          << getOpenMPClauseName(OMPC_copyprivate) << Type
          << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      Diag(D->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }

    // OpenMP [2.14.4.1, Restrictions, C/C++, p.2]
    //  A variable of class type (or array thereof) that appears in a
    //  copyin clause requires an accessible, unambiguous copy assignment
    //  operator for the class type.
    Type = Context.getBaseElementType(Type.getNonReferenceType())
               .getUnqualifiedType();
    VarDecl *SrcVD =
        buildVarDecl(*this, RefExpr->getBeginLoc(), Type, ".copyprivate.src",
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
    DeclRefExpr *PseudoSrcExpr = buildDeclRefExpr(*this, SrcVD, Type, ELoc);
    VarDecl *DstVD =
        buildVarDecl(*this, RefExpr->getBeginLoc(), Type, ".copyprivate.dst",
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
    DeclRefExpr *PseudoDstExpr = buildDeclRefExpr(*this, DstVD, Type, ELoc);
    ExprResult AssignmentOp = BuildBinOp(
        DSAStack->getCurScope(), ELoc, BO_Assign, PseudoDstExpr, PseudoSrcExpr);
    if (AssignmentOp.isInvalid())
      continue;
    AssignmentOp =
        ActOnFinishFullExpr(AssignmentOp.get(), ELoc, /*DiscardedValue*/ false);
    if (AssignmentOp.isInvalid())
      continue;

    // No need to mark vars as copyprivate, they are already threadprivate or
    // implicitly private.
    assert(VD || isOpenMPCapturedDecl(D));
    Vars.push_back(
        VD ? RefExpr->IgnoreParens()
           : buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/false));
    SrcExprs.push_back(PseudoSrcExpr);
    DstExprs.push_back(PseudoDstExpr);
    AssignmentOps.push_back(AssignmentOp.get());
  }

  if (Vars.empty())
    return nullptr;

  return OMPCopyprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                      Vars, SrcExprs, DstExprs, AssignmentOps);
}

OMPClause *Sema::ActOnOpenMPFlushClause(ArrayRef<Expr *> VarList,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc) {
  if (VarList.empty())
    return nullptr;

  return OMPFlushClause::Create(Context, StartLoc, LParenLoc, EndLoc, VarList);
}

/// Tries to find omp_depend_t. type.
static bool findOMPDependT(Sema &S, SourceLocation Loc, DSAStackTy *Stack,
                           bool Diagnose = true) {
  QualType OMPDependT = Stack->getOMPDependT();
  if (!OMPDependT.isNull())
    return true;
  IdentifierInfo *II = &S.PP.getIdentifierTable().get("omp_depend_t");
  ParsedType PT = S.getTypeName(*II, Loc, S.getCurScope());
  if (!PT.getAsOpaquePtr() || PT.get().isNull()) {
    if (Diagnose)
      S.Diag(Loc, diag::err_omp_implied_type_not_found) << "omp_depend_t";
    return false;
  }
  Stack->setOMPDependT(PT.get());
  return true;
}

OMPClause *Sema::ActOnOpenMPDepobjClause(Expr *Depobj, SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  if (!Depobj)
    return nullptr;

  bool OMPDependTFound = findOMPDependT(*this, StartLoc, DSAStack);

  // OpenMP 5.0, 2.17.10.1 depobj Construct
  // depobj is an lvalue expression of type omp_depend_t.
  if (!Depobj->isTypeDependent() && !Depobj->isValueDependent() &&
      !Depobj->isInstantiationDependent() &&
      !Depobj->containsUnexpandedParameterPack() &&
      (OMPDependTFound &&
       !Context.typesAreCompatible(DSAStack->getOMPDependT(), Depobj->getType(),
                                   /*CompareUnqualified=*/true))) {
    Diag(Depobj->getExprLoc(), diag::err_omp_expected_omp_depend_t_lvalue)
        << 0 << Depobj->getType() << Depobj->getSourceRange();
  }

  if (!Depobj->isLValue()) {
    Diag(Depobj->getExprLoc(), diag::err_omp_expected_omp_depend_t_lvalue)
        << 1 << Depobj->getSourceRange();
  }

  return OMPDepobjClause::Create(Context, StartLoc, LParenLoc, EndLoc, Depobj);
}

OMPClause *
Sema::ActOnOpenMPDependClause(Expr *DepModifier, OpenMPDependClauseKind DepKind,
                              SourceLocation DepLoc, SourceLocation ColonLoc,
                              ArrayRef<Expr *> VarList, SourceLocation StartLoc,
                              SourceLocation LParenLoc, SourceLocation EndLoc) {
  if (DSAStack->getCurrentDirective() == OMPD_ordered &&
      DepKind != OMPC_DEPEND_source && DepKind != OMPC_DEPEND_sink) {
    Diag(DepLoc, diag::err_omp_unexpected_clause_value)
        << "'source' or 'sink'" << getOpenMPClauseName(OMPC_depend);
    return nullptr;
  }
  if (DSAStack->getCurrentDirective() == OMPD_taskwait &&
      DepKind == OMPC_DEPEND_mutexinoutset) {
    Diag(DepLoc, diag::err_omp_taskwait_depend_mutexinoutset_not_allowed);
    return nullptr;
  }
  if ((DSAStack->getCurrentDirective() != OMPD_ordered ||
       DSAStack->getCurrentDirective() == OMPD_depobj) &&
      (DepKind == OMPC_DEPEND_unknown || DepKind == OMPC_DEPEND_source ||
       DepKind == OMPC_DEPEND_sink ||
       ((LangOpts.OpenMP < 50 ||
         DSAStack->getCurrentDirective() == OMPD_depobj) &&
        DepKind == OMPC_DEPEND_depobj))) {
    SmallVector<unsigned, 3> Except;
    Except.push_back(OMPC_DEPEND_source);
    Except.push_back(OMPC_DEPEND_sink);
    if (LangOpts.OpenMP < 50 || DSAStack->getCurrentDirective() == OMPD_depobj)
      Except.push_back(OMPC_DEPEND_depobj);
    std::string Expected = (LangOpts.OpenMP >= 50 && !DepModifier)
                               ? "depend modifier(iterator) or "
                               : "";
    Diag(DepLoc, diag::err_omp_unexpected_clause_value)
        << Expected + getListOfPossibleValues(OMPC_depend, /*First=*/0,
                                              /*Last=*/OMPC_DEPEND_unknown,
                                              Except)
        << getOpenMPClauseName(OMPC_depend);
    return nullptr;
  }
  if (DepModifier &&
      (DepKind == OMPC_DEPEND_source || DepKind == OMPC_DEPEND_sink)) {
    Diag(DepModifier->getExprLoc(),
         diag::err_omp_depend_sink_source_with_modifier);
    return nullptr;
  }
  if (DepModifier &&
      !DepModifier->getType()->isSpecificBuiltinType(BuiltinType::OMPIterator))
    Diag(DepModifier->getExprLoc(), diag::err_omp_depend_modifier_not_iterator);

  SmallVector<Expr *, 8> Vars;
  DSAStackTy::OperatorOffsetTy OpsOffs;
  llvm::APSInt DepCounter(/*BitWidth=*/32);
  llvm::APSInt TotalDepCount(/*BitWidth=*/32);
  if (DepKind == OMPC_DEPEND_sink || DepKind == OMPC_DEPEND_source) {
    if (const Expr *OrderedCountExpr =
            DSAStack->getParentOrderedRegionParam().first) {
      TotalDepCount = OrderedCountExpr->EvaluateKnownConstInt(Context);
      TotalDepCount.setIsUnsigned(/*Val=*/true);
    }
  }
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP shared clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    Expr *SimpleExpr = RefExpr->IgnoreParenCasts();
    if (DepKind == OMPC_DEPEND_sink) {
      if (DSAStack->getParentOrderedRegionParam().first &&
          DepCounter >= TotalDepCount) {
        Diag(ELoc, diag::err_omp_depend_sink_unexpected_expr);
        continue;
      }
      ++DepCounter;
      // OpenMP  [2.13.9, Summary]
      // depend(dependence-type : vec), where dependence-type is:
      // 'sink' and where vec is the iteration vector, which has the form:
      //  x1 [+- d1], x2 [+- d2 ], . . . , xn [+- dn]
      // where n is the value specified by the ordered clause in the loop
      // directive, xi denotes the loop iteration variable of the i-th nested
      // loop associated with the loop directive, and di is a constant
      // non-negative integer.
      if (CurContext->isDependentContext()) {
        // It will be analyzed later.
        Vars.push_back(RefExpr);
        continue;
      }
      SimpleExpr = SimpleExpr->IgnoreImplicit();
      OverloadedOperatorKind OOK = OO_None;
      SourceLocation OOLoc;
      Expr *LHS = SimpleExpr;
      Expr *RHS = nullptr;
      if (auto *BO = dyn_cast<BinaryOperator>(SimpleExpr)) {
        OOK = BinaryOperator::getOverloadedOperator(BO->getOpcode());
        OOLoc = BO->getOperatorLoc();
        LHS = BO->getLHS()->IgnoreParenImpCasts();
        RHS = BO->getRHS()->IgnoreParenImpCasts();
      } else if (auto *OCE = dyn_cast<CXXOperatorCallExpr>(SimpleExpr)) {
        OOK = OCE->getOperator();
        OOLoc = OCE->getOperatorLoc();
        LHS = OCE->getArg(/*Arg=*/0)->IgnoreParenImpCasts();
        RHS = OCE->getArg(/*Arg=*/1)->IgnoreParenImpCasts();
      } else if (auto *MCE = dyn_cast<CXXMemberCallExpr>(SimpleExpr)) {
        OOK = MCE->getMethodDecl()
                  ->getNameInfo()
                  .getName()
                  .getCXXOverloadedOperator();
        OOLoc = MCE->getCallee()->getExprLoc();
        LHS = MCE->getImplicitObjectArgument()->IgnoreParenImpCasts();
        RHS = MCE->getArg(/*Arg=*/0)->IgnoreParenImpCasts();
      }
      SourceLocation ELoc;
      SourceRange ERange;
      auto Res = getPrivateItem(*this, LHS, ELoc, ERange);
      if (Res.second) {
        // It will be analyzed later.
        Vars.push_back(RefExpr);
      }
      ValueDecl *D = Res.first;
      if (!D)
        continue;

      if (OOK != OO_Plus && OOK != OO_Minus && (RHS || OOK != OO_None)) {
        Diag(OOLoc, diag::err_omp_depend_sink_expected_plus_minus);
        continue;
      }
      if (RHS) {
        ExprResult RHSRes = VerifyPositiveIntegerConstantInClause(
            RHS, OMPC_depend, /*StrictlyPositive=*/false);
        if (RHSRes.isInvalid())
          continue;
      }
      if (!CurContext->isDependentContext() &&
          DSAStack->getParentOrderedRegionParam().first &&
          DepCounter != DSAStack->isParentLoopControlVariable(D).first) {
        const ValueDecl *VD =
            DSAStack->getParentLoopControlVariable(DepCounter.getZExtValue());
        if (VD)
          Diag(ELoc, diag::err_omp_depend_sink_expected_loop_iteration)
              << 1 << VD;
        else
          Diag(ELoc, diag::err_omp_depend_sink_expected_loop_iteration) << 0;
        continue;
      }
      OpsOffs.emplace_back(RHS, OOK);
    } else {
      bool OMPDependTFound = LangOpts.OpenMP >= 50;
      if (OMPDependTFound)
        OMPDependTFound = findOMPDependT(*this, StartLoc, DSAStack,
                                         DepKind == OMPC_DEPEND_depobj);
      if (DepKind == OMPC_DEPEND_depobj) {
        // OpenMP 5.0, 2.17.11 depend Clause, Restrictions, C/C++
        // List items used in depend clauses with the depobj dependence type
        // must be expressions of the omp_depend_t type.
        if (!RefExpr->isValueDependent() && !RefExpr->isTypeDependent() &&
            !RefExpr->isInstantiationDependent() &&
            !RefExpr->containsUnexpandedParameterPack() &&
            (OMPDependTFound &&
             !Context.hasSameUnqualifiedType(DSAStack->getOMPDependT(),
                                             RefExpr->getType()))) {
          Diag(ELoc, diag::err_omp_expected_omp_depend_t_lvalue)
              << 0 << RefExpr->getType() << RefExpr->getSourceRange();
          continue;
        }
        if (!RefExpr->isLValue()) {
          Diag(ELoc, diag::err_omp_expected_omp_depend_t_lvalue)
              << 1 << RefExpr->getType() << RefExpr->getSourceRange();
          continue;
        }
      } else {
        // OpenMP 5.0 [2.17.11, Restrictions]
        // List items used in depend clauses cannot be zero-length array
        // sections.
        QualType ExprTy = RefExpr->getType().getNonReferenceType();
        const auto *OASE = dyn_cast<OMPArraySectionExpr>(SimpleExpr);
        if (OASE) {
          QualType BaseType =
              OMPArraySectionExpr::getBaseOriginalType(OASE->getBase());
          if (const auto *ATy = BaseType->getAsArrayTypeUnsafe())
            ExprTy = ATy->getElementType();
          else
            ExprTy = BaseType->getPointeeType();
          ExprTy = ExprTy.getNonReferenceType();
          const Expr *Length = OASE->getLength();
          Expr::EvalResult Result;
          if (Length && !Length->isValueDependent() &&
              Length->EvaluateAsInt(Result, Context) &&
              Result.Val.getInt().isZero()) {
            Diag(ELoc,
                 diag::err_omp_depend_zero_length_array_section_not_allowed)
                << SimpleExpr->getSourceRange();
            continue;
          }
        }

        // OpenMP 5.0, 2.17.11 depend Clause, Restrictions, C/C++
        // List items used in depend clauses with the in, out, inout or
        // mutexinoutset dependence types cannot be expressions of the
        // omp_depend_t type.
        if (!RefExpr->isValueDependent() && !RefExpr->isTypeDependent() &&
            !RefExpr->isInstantiationDependent() &&
            !RefExpr->containsUnexpandedParameterPack() &&
            (!RefExpr->IgnoreParenImpCasts()->isLValue() ||
             (OMPDependTFound &&
              DSAStack->getOMPDependT().getTypePtr() == ExprTy.getTypePtr()))) {
          Diag(ELoc, diag::err_omp_expected_addressable_lvalue_or_array_item)
              << (LangOpts.OpenMP >= 50 ? 1 : 0)
              << (LangOpts.OpenMP >= 50 ? 1 : 0) << RefExpr->getSourceRange();
          continue;
        }

        auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
        if (ASE && !ASE->getBase()->isTypeDependent() &&
            !ASE->getBase()->getType().getNonReferenceType()->isPointerType() &&
            !ASE->getBase()->getType().getNonReferenceType()->isArrayType()) {
          Diag(ELoc, diag::err_omp_expected_addressable_lvalue_or_array_item)
              << (LangOpts.OpenMP >= 50 ? 1 : 0)
              << (LangOpts.OpenMP >= 50 ? 1 : 0) << RefExpr->getSourceRange();
          continue;
        }

        ExprResult Res;
        {
          Sema::TentativeAnalysisScope Trap(*this);
          Res = CreateBuiltinUnaryOp(ELoc, UO_AddrOf,
                                     RefExpr->IgnoreParenImpCasts());
        }
        if (!Res.isUsable() && !isa<OMPArraySectionExpr>(SimpleExpr) &&
            !isa<OMPArrayShapingExpr>(SimpleExpr)) {
          Diag(ELoc, diag::err_omp_expected_addressable_lvalue_or_array_item)
              << (LangOpts.OpenMP >= 50 ? 1 : 0)
              << (LangOpts.OpenMP >= 50 ? 1 : 0) << RefExpr->getSourceRange();
          continue;
        }
      }
    }
    Vars.push_back(RefExpr->IgnoreParenImpCasts());
  }

  if (!CurContext->isDependentContext() && DepKind == OMPC_DEPEND_sink &&
      TotalDepCount > VarList.size() &&
      DSAStack->getParentOrderedRegionParam().first &&
      DSAStack->getParentLoopControlVariable(VarList.size() + 1)) {
    Diag(EndLoc, diag::err_omp_depend_sink_expected_loop_iteration)
        << 1 << DSAStack->getParentLoopControlVariable(VarList.size() + 1);
  }
  if (DepKind != OMPC_DEPEND_source && DepKind != OMPC_DEPEND_sink &&
      Vars.empty())
    return nullptr;

  auto *C = OMPDependClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                    DepModifier, DepKind, DepLoc, ColonLoc,
                                    Vars, TotalDepCount.getZExtValue());
  if ((DepKind == OMPC_DEPEND_sink || DepKind == OMPC_DEPEND_source) &&
      DSAStack->isParentOrderedRegion())
    DSAStack->addDoacrossDependClause(C, OpsOffs);
  return C;
}

OMPClause *Sema::ActOnOpenMPDeviceClause(OpenMPDeviceClauseModifier Modifier,
                                         Expr *Device, SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation ModifierLoc,
                                         SourceLocation EndLoc) {
  assert((ModifierLoc.isInvalid() || LangOpts.OpenMP >= 50) &&
         "Unexpected device modifier in OpenMP < 50.");

  bool ErrorFound = false;
  if (ModifierLoc.isValid() && Modifier == OMPC_DEVICE_unknown) {
    std::string Values =
        getListOfPossibleValues(OMPC_device, /*First=*/0, OMPC_DEVICE_unknown);
    Diag(ModifierLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_device);
    ErrorFound = true;
  }

  Expr *ValExpr = Device;
  Stmt *HelperValStmt = nullptr;

  // OpenMP [2.9.1, Restrictions]
  // The device expression must evaluate to a non-negative integer value.
  ErrorFound = !isNonNegativeIntegerValue(ValExpr, *this, OMPC_device,
                                          /*StrictlyPositive=*/false) ||
               ErrorFound;
  if (ErrorFound)
    return nullptr;

  OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
  OpenMPDirectiveKind CaptureRegion =
      getOpenMPCaptureRegionForClause(DKind, OMPC_device, LangOpts.OpenMP);
  if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
    ValExpr = MakeFullExpr(ValExpr).get();
    llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
    ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
    HelperValStmt = buildPreInits(Context, Captures);
  }

  return new (Context)
      OMPDeviceClause(Modifier, ValExpr, HelperValStmt, CaptureRegion, StartLoc,
                      LParenLoc, ModifierLoc, EndLoc);
}

static bool checkTypeMappable(SourceLocation SL, SourceRange SR, Sema &SemaRef,
                              DSAStackTy *Stack, QualType QTy,
                              bool FullCheck = true) {
  if (SemaRef.RequireCompleteType(SL, QTy, diag::err_incomplete_type))
    return false;
  if (FullCheck && !SemaRef.CurContext->isDependentContext() &&
      !QTy.isTriviallyCopyableType(SemaRef.Context))
    SemaRef.Diag(SL, diag::warn_omp_non_trivial_type_mapped) << QTy << SR;
  return true;
}

/// Return true if it can be proven that the provided array expression
/// (array section or array subscript) does NOT specify the whole size of the
/// array whose base type is \a BaseQTy.
static bool checkArrayExpressionDoesNotReferToWholeSize(Sema &SemaRef,
                                                        const Expr *E,
                                                        QualType BaseQTy) {
  const auto *OASE = dyn_cast<OMPArraySectionExpr>(E);

  // If this is an array subscript, it refers to the whole size if the size of
  // the dimension is constant and equals 1. Also, an array section assumes the
  // format of an array subscript if no colon is used.
  if (isa<ArraySubscriptExpr>(E) ||
      (OASE && OASE->getColonLocFirst().isInvalid())) {
    if (const auto *ATy = dyn_cast<ConstantArrayType>(BaseQTy.getTypePtr()))
      return ATy->getSize().getSExtValue() != 1;
    // Size can't be evaluated statically.
    return false;
  }

  assert(OASE && "Expecting array section if not an array subscript.");
  const Expr *LowerBound = OASE->getLowerBound();
  const Expr *Length = OASE->getLength();

  // If there is a lower bound that does not evaluates to zero, we are not
  // covering the whole dimension.
  if (LowerBound) {
    Expr::EvalResult Result;
    if (!LowerBound->EvaluateAsInt(Result, SemaRef.getASTContext()))
      return false; // Can't get the integer value as a constant.

    llvm::APSInt ConstLowerBound = Result.Val.getInt();
    if (ConstLowerBound.getSExtValue())
      return true;
  }

  // If we don't have a length we covering the whole dimension.
  if (!Length)
    return false;

  // If the base is a pointer, we don't have a way to get the size of the
  // pointee.
  if (BaseQTy->isPointerType())
    return false;

  // We can only check if the length is the same as the size of the dimension
  // if we have a constant array.
  const auto *CATy = dyn_cast<ConstantArrayType>(BaseQTy.getTypePtr());
  if (!CATy)
    return false;

  Expr::EvalResult Result;
  if (!Length->EvaluateAsInt(Result, SemaRef.getASTContext()))
    return false; // Can't get the integer value as a constant.

  llvm::APSInt ConstLength = Result.Val.getInt();
  return CATy->getSize().getSExtValue() != ConstLength.getSExtValue();
}

// Return true if it can be proven that the provided array expression (array
// section or array subscript) does NOT specify a single element of the array
// whose base type is \a BaseQTy.
static bool checkArrayExpressionDoesNotReferToUnitySize(Sema &SemaRef,
                                                        const Expr *E,
                                                        QualType BaseQTy) {
  const auto *OASE = dyn_cast<OMPArraySectionExpr>(E);

  // An array subscript always refer to a single element. Also, an array section
  // assumes the format of an array subscript if no colon is used.
  if (isa<ArraySubscriptExpr>(E) ||
      (OASE && OASE->getColonLocFirst().isInvalid()))
    return false;

  assert(OASE && "Expecting array section if not an array subscript.");
  const Expr *Length = OASE->getLength();

  // If we don't have a length we have to check if the array has unitary size
  // for this dimension. Also, we should always expect a length if the base type
  // is pointer.
  if (!Length) {
    if (const auto *ATy = dyn_cast<ConstantArrayType>(BaseQTy.getTypePtr()))
      return ATy->getSize().getSExtValue() != 1;
    // We cannot assume anything.
    return false;
  }

  // Check if the length evaluates to 1.
  Expr::EvalResult Result;
  if (!Length->EvaluateAsInt(Result, SemaRef.getASTContext()))
    return false; // Can't get the integer value as a constant.

  llvm::APSInt ConstLength = Result.Val.getInt();
  return ConstLength.getSExtValue() != 1;
}

// The base of elements of list in a map clause have to be either:
//  - a reference to variable or field.
//  - a member expression.
//  - an array expression.
//
// E.g. if we have the expression 'r.S.Arr[:12]', we want to retrieve the
// reference to 'r'.
//
// If we have:
//
// struct SS {
//   Bla S;
//   foo() {
//     #pragma omp target map (S.Arr[:12]);
//   }
// }
//
// We want to retrieve the member expression 'this->S';

// OpenMP 5.0 [2.19.7.1, map Clause, Restrictions, p.2]
//  If a list item is an array section, it must specify contiguous storage.
//
// For this restriction it is sufficient that we make sure only references
// to variables or fields and array expressions, and that no array sections
// exist except in the rightmost expression (unless they cover the whole
// dimension of the array). E.g. these would be invalid:
//
//   r.ArrS[3:5].Arr[6:7]
//
//   r.ArrS[3:5].x
//
// but these would be valid:
//   r.ArrS[3].Arr[6:7]
//
//   r.ArrS[3].x
namespace {
class MapBaseChecker final : public StmtVisitor<MapBaseChecker, bool> {
  Sema &SemaRef;
  OpenMPClauseKind CKind = OMPC_unknown;
  OpenMPDirectiveKind DKind = OMPD_unknown;
  OMPClauseMappableExprCommon::MappableExprComponentList &Components;
  bool IsNonContiguous = false;
  bool NoDiagnose = false;
  const Expr *RelevantExpr = nullptr;
  bool AllowUnitySizeArraySection = true;
  bool AllowWholeSizeArraySection = true;
  bool AllowAnotherPtr = true;
  SourceLocation ELoc;
  SourceRange ERange;

  void emitErrorMsg() {
    // If nothing else worked, this is not a valid map clause expression.
    if (SemaRef.getLangOpts().OpenMP < 50) {
      SemaRef.Diag(ELoc,
                   diag::err_omp_expected_named_var_member_or_array_expression)
          << ERange;
    } else {
      SemaRef.Diag(ELoc, diag::err_omp_non_lvalue_in_map_or_motion_clauses)
          << getOpenMPClauseName(CKind) << ERange;
    }
  }

public:
  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (!isa<VarDecl>(DRE->getDecl())) {
      emitErrorMsg();
      return false;
    }
    assert(!RelevantExpr && "RelevantExpr is expected to be nullptr");
    RelevantExpr = DRE;
    // Record the component.
    Components.emplace_back(DRE, DRE->getDecl(), IsNonContiguous);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    Expr *E = ME;
    Expr *BaseE = ME->getBase()->IgnoreParenCasts();

    if (isa<CXXThisExpr>(BaseE)) {
      assert(!RelevantExpr && "RelevantExpr is expected to be nullptr");
      // We found a base expression: this->Val.
      RelevantExpr = ME;
    } else {
      E = BaseE;
    }

    if (!isa<FieldDecl>(ME->getMemberDecl())) {
      if (!NoDiagnose) {
        SemaRef.Diag(ELoc, diag::err_omp_expected_access_to_data_field)
            << ME->getSourceRange();
        return false;
      }
      if (RelevantExpr)
        return false;
      return Visit(E);
    }

    auto *FD = cast<FieldDecl>(ME->getMemberDecl());

    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C/C++, p.3]
    //  A bit-field cannot appear in a map clause.
    //
    if (FD->isBitField()) {
      if (!NoDiagnose) {
        SemaRef.Diag(ELoc, diag::err_omp_bit_fields_forbidden_in_clause)
            << ME->getSourceRange() << getOpenMPClauseName(CKind);
        return false;
      }
      if (RelevantExpr)
        return false;
      return Visit(E);
    }

    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C++, p.1]
    //  If the type of a list item is a reference to a type T then the type
    //  will be considered to be T for all purposes of this clause.
    QualType CurType = BaseE->getType().getNonReferenceType();

    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C/C++, p.2]
    //  A list item cannot be a variable that is a member of a structure with
    //  a union type.
    //
    if (CurType->isUnionType()) {
      if (!NoDiagnose) {
        SemaRef.Diag(ELoc, diag::err_omp_union_type_not_allowed)
            << ME->getSourceRange();
        return false;
      }
      return RelevantExpr || Visit(E);
    }

    // If we got a member expression, we should not expect any array section
    // before that:
    //
    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.7]
    //  If a list item is an element of a structure, only the rightmost symbol
    //  of the variable reference can be an array section.
    //
    AllowUnitySizeArraySection = false;
    AllowWholeSizeArraySection = false;

    // Record the component.
    Components.emplace_back(ME, FD, IsNonContiguous);
    return RelevantExpr || Visit(E);
  }

  bool VisitArraySubscriptExpr(ArraySubscriptExpr *AE) {
    Expr *E = AE->getBase()->IgnoreParenImpCasts();

    if (!E->getType()->isAnyPointerType() && !E->getType()->isArrayType()) {
      if (!NoDiagnose) {
        SemaRef.Diag(ELoc, diag::err_omp_expected_base_var_name)
            << 0 << AE->getSourceRange();
        return false;
      }
      return RelevantExpr || Visit(E);
    }

    // If we got an array subscript that express the whole dimension we
    // can have any array expressions before. If it only expressing part of
    // the dimension, we can only have unitary-size array expressions.
    if (checkArrayExpressionDoesNotReferToWholeSize(SemaRef, AE, E->getType()))
      AllowWholeSizeArraySection = false;

    if (const auto *TE = dyn_cast<CXXThisExpr>(E->IgnoreParenCasts())) {
      Expr::EvalResult Result;
      if (!AE->getIdx()->isValueDependent() &&
          AE->getIdx()->EvaluateAsInt(Result, SemaRef.getASTContext()) &&
          !Result.Val.getInt().isZero()) {
        SemaRef.Diag(AE->getIdx()->getExprLoc(),
                     diag::err_omp_invalid_map_this_expr);
        SemaRef.Diag(AE->getIdx()->getExprLoc(),
                     diag::note_omp_invalid_subscript_on_this_ptr_map);
      }
      assert(!RelevantExpr && "RelevantExpr is expected to be nullptr");
      RelevantExpr = TE;
    }

    // Record the component - we don't have any declaration associated.
    Components.emplace_back(AE, nullptr, IsNonContiguous);

    return RelevantExpr || Visit(E);
  }

  bool VisitOMPArraySectionExpr(OMPArraySectionExpr *OASE) {
    // After OMP 5.0  Array section in reduction clause will be implicitly
    // mapped
    assert(!(SemaRef.getLangOpts().OpenMP < 50 && NoDiagnose) &&
           "Array sections cannot be implicitly mapped.");
    Expr *E = OASE->getBase()->IgnoreParenImpCasts();
    QualType CurType =
        OMPArraySectionExpr::getBaseOriginalType(E).getCanonicalType();

    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C++, p.1]
    //  If the type of a list item is a reference to a type T then the type
    //  will be considered to be T for all purposes of this clause.
    if (CurType->isReferenceType())
      CurType = CurType->getPointeeType();

    bool IsPointer = CurType->isAnyPointerType();

    if (!IsPointer && !CurType->isArrayType()) {
      SemaRef.Diag(ELoc, diag::err_omp_expected_base_var_name)
          << 0 << OASE->getSourceRange();
      return false;
    }

    bool NotWhole =
        checkArrayExpressionDoesNotReferToWholeSize(SemaRef, OASE, CurType);
    bool NotUnity =
        checkArrayExpressionDoesNotReferToUnitySize(SemaRef, OASE, CurType);

    if (AllowWholeSizeArraySection) {
      // Any array section is currently allowed. Allowing a whole size array
      // section implies allowing a unity array section as well.
      //
      // If this array section refers to the whole dimension we can still
      // accept other array sections before this one, except if the base is a
      // pointer. Otherwise, only unitary sections are accepted.
      if (NotWhole || IsPointer)
        AllowWholeSizeArraySection = false;
    } else if (DKind == OMPD_target_update &&
               SemaRef.getLangOpts().OpenMP >= 50) {
      if (IsPointer && !AllowAnotherPtr)
        SemaRef.Diag(ELoc, diag::err_omp_section_length_undefined)
            << /*array of unknown bound */ 1;
      else
        IsNonContiguous = true;
    } else if (AllowUnitySizeArraySection && NotUnity) {
      // A unity or whole array section is not allowed and that is not
      // compatible with the properties of the current array section.
      if (NoDiagnose)
        return false;
      SemaRef.Diag(ELoc,
                   diag::err_array_section_does_not_specify_contiguous_storage)
          << OASE->getSourceRange();
      return false;
    }

    if (IsPointer)
      AllowAnotherPtr = false;

    if (const auto *TE = dyn_cast<CXXThisExpr>(E)) {
      Expr::EvalResult ResultR;
      Expr::EvalResult ResultL;
      if (!OASE->getLength()->isValueDependent() &&
          OASE->getLength()->EvaluateAsInt(ResultR, SemaRef.getASTContext()) &&
          !ResultR.Val.getInt().isOne()) {
        SemaRef.Diag(OASE->getLength()->getExprLoc(),
                     diag::err_omp_invalid_map_this_expr);
        SemaRef.Diag(OASE->getLength()->getExprLoc(),
                     diag::note_omp_invalid_length_on_this_ptr_mapping);
      }
      if (OASE->getLowerBound() && !OASE->getLowerBound()->isValueDependent() &&
          OASE->getLowerBound()->EvaluateAsInt(ResultL,
                                               SemaRef.getASTContext()) &&
          !ResultL.Val.getInt().isZero()) {
        SemaRef.Diag(OASE->getLowerBound()->getExprLoc(),
                     diag::err_omp_invalid_map_this_expr);
        SemaRef.Diag(OASE->getLowerBound()->getExprLoc(),
                     diag::note_omp_invalid_lower_bound_on_this_ptr_mapping);
      }
      assert(!RelevantExpr && "RelevantExpr is expected to be nullptr");
      RelevantExpr = TE;
    }

    // Record the component - we don't have any declaration associated.
    Components.emplace_back(OASE, nullptr, /*IsNonContiguous=*/false);
    return RelevantExpr || Visit(E);
  }
  bool VisitOMPArrayShapingExpr(OMPArrayShapingExpr *E) {
    Expr *Base = E->getBase();

    // Record the component - we don't have any declaration associated.
    Components.emplace_back(E, nullptr, IsNonContiguous);

    return Visit(Base->IgnoreParenImpCasts());
  }

  bool VisitUnaryOperator(UnaryOperator *UO) {
    if (SemaRef.getLangOpts().OpenMP < 50 || !UO->isLValue() ||
        UO->getOpcode() != UO_Deref) {
      emitErrorMsg();
      return false;
    }
    if (!RelevantExpr) {
      // Record the component if haven't found base decl.
      Components.emplace_back(UO, nullptr, /*IsNonContiguous=*/false);
    }
    return RelevantExpr || Visit(UO->getSubExpr()->IgnoreParenImpCasts());
  }
  bool VisitBinaryOperator(BinaryOperator *BO) {
    if (SemaRef.getLangOpts().OpenMP < 50 || !BO->getType()->isPointerType()) {
      emitErrorMsg();
      return false;
    }

    // Pointer arithmetic is the only thing we expect to happen here so after we
    // make sure the binary operator is a pointer type, the we only thing need
    // to to is to visit the subtree that has the same type as root (so that we
    // know the other subtree is just an offset)
    Expr *LE = BO->getLHS()->IgnoreParenImpCasts();
    Expr *RE = BO->getRHS()->IgnoreParenImpCasts();
    Components.emplace_back(BO, nullptr, false);
    assert((LE->getType().getTypePtr() == BO->getType().getTypePtr() ||
            RE->getType().getTypePtr() == BO->getType().getTypePtr()) &&
           "Either LHS or RHS have base decl inside");
    if (BO->getType().getTypePtr() == LE->getType().getTypePtr())
      return RelevantExpr || Visit(LE);
    return RelevantExpr || Visit(RE);
  }
  bool VisitCXXThisExpr(CXXThisExpr *CTE) {
    assert(!RelevantExpr && "RelevantExpr is expected to be nullptr");
    RelevantExpr = CTE;
    Components.emplace_back(CTE, nullptr, IsNonContiguous);
    return true;
  }
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *COCE) {
    assert(!RelevantExpr && "RelevantExpr is expected to be nullptr");
    Components.emplace_back(COCE, nullptr, IsNonContiguous);
    return true;
  }
  bool VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    Expr *Source = E->getSourceExpr();
    if (!Source) {
      emitErrorMsg();
      return false;
    }
    return Visit(Source);
  }
  bool VisitStmt(Stmt *) {
    emitErrorMsg();
    return false;
  }
  const Expr *getFoundBase() const { return RelevantExpr; }
  explicit MapBaseChecker(
      Sema &SemaRef, OpenMPClauseKind CKind, OpenMPDirectiveKind DKind,
      OMPClauseMappableExprCommon::MappableExprComponentList &Components,
      bool NoDiagnose, SourceLocation &ELoc, SourceRange &ERange)
      : SemaRef(SemaRef), CKind(CKind), DKind(DKind), Components(Components),
        NoDiagnose(NoDiagnose), ELoc(ELoc), ERange(ERange) {}
};
} // namespace

/// Return the expression of the base of the mappable expression or null if it
/// cannot be determined and do all the necessary checks to see if the
/// expression is valid as a standalone mappable expression. In the process,
/// record all the components of the expression.
static const Expr *checkMapClauseExpressionBase(
    Sema &SemaRef, Expr *E,
    OMPClauseMappableExprCommon::MappableExprComponentList &CurComponents,
    OpenMPClauseKind CKind, OpenMPDirectiveKind DKind, bool NoDiagnose) {
  SourceLocation ELoc = E->getExprLoc();
  SourceRange ERange = E->getSourceRange();
  MapBaseChecker Checker(SemaRef, CKind, DKind, CurComponents, NoDiagnose, ELoc,
                         ERange);
  if (Checker.Visit(E->IgnoreParens())) {
    // Check if the highest dimension array section has length specified
    if (SemaRef.getLangOpts().OpenMP >= 50 && !CurComponents.empty() &&
        (CKind == OMPC_to || CKind == OMPC_from)) {
      auto CI = CurComponents.rbegin();
      auto CE = CurComponents.rend();
      for (; CI != CE; ++CI) {
        const auto *OASE =
            dyn_cast<OMPArraySectionExpr>(CI->getAssociatedExpression());
        if (!OASE)
          continue;
        if (OASE && OASE->getLength())
          break;
        SemaRef.Diag(ELoc, diag::err_array_section_does_not_specify_length)
            << ERange;
      }
    }
    return Checker.getFoundBase();
  }
  return nullptr;
}

// Return true if expression E associated with value VD has conflicts with other
// map information.
static bool checkMapConflicts(
    Sema &SemaRef, DSAStackTy *DSAS, const ValueDecl *VD, const Expr *E,
    bool CurrentRegionOnly,
    OMPClauseMappableExprCommon::MappableExprComponentListRef CurComponents,
    OpenMPClauseKind CKind) {
  assert(VD && E);
  SourceLocation ELoc = E->getExprLoc();
  SourceRange ERange = E->getSourceRange();

  // In order to easily check the conflicts we need to match each component of
  // the expression under test with the components of the expressions that are
  // already in the stack.

  assert(!CurComponents.empty() && "Map clause expression with no components!");
  assert(CurComponents.back().getAssociatedDeclaration() == VD &&
         "Map clause expression with unexpected base!");

  // Variables to help detecting enclosing problems in data environment nests.
  bool IsEnclosedByDataEnvironmentExpr = false;
  const Expr *EnclosingExpr = nullptr;

  bool FoundError = DSAS->checkMappableExprComponentListsForDecl(
      VD, CurrentRegionOnly,
      [&IsEnclosedByDataEnvironmentExpr, &SemaRef, VD, CurrentRegionOnly, ELoc,
       ERange, CKind, &EnclosingExpr,
       CurComponents](OMPClauseMappableExprCommon::MappableExprComponentListRef
                          StackComponents,
                      OpenMPClauseKind Kind) {
        if (CKind == Kind && SemaRef.LangOpts.OpenMP >= 50)
          return false;
        assert(!StackComponents.empty() &&
               "Map clause expression with no components!");
        assert(StackComponents.back().getAssociatedDeclaration() == VD &&
               "Map clause expression with unexpected base!");
        (void)VD;

        // The whole expression in the stack.
        const Expr *RE = StackComponents.front().getAssociatedExpression();

        // Expressions must start from the same base. Here we detect at which
        // point both expressions diverge from each other and see if we can
        // detect if the memory referred to both expressions is contiguous and
        // do not overlap.
        auto CI = CurComponents.rbegin();
        auto CE = CurComponents.rend();
        auto SI = StackComponents.rbegin();
        auto SE = StackComponents.rend();
        for (; CI != CE && SI != SE; ++CI, ++SI) {

          // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.3]
          //  At most one list item can be an array item derived from a given
          //  variable in map clauses of the same construct.
          if (CurrentRegionOnly &&
              (isa<ArraySubscriptExpr>(CI->getAssociatedExpression()) ||
               isa<OMPArraySectionExpr>(CI->getAssociatedExpression()) ||
               isa<OMPArrayShapingExpr>(CI->getAssociatedExpression())) &&
              (isa<ArraySubscriptExpr>(SI->getAssociatedExpression()) ||
               isa<OMPArraySectionExpr>(SI->getAssociatedExpression()) ||
               isa<OMPArrayShapingExpr>(SI->getAssociatedExpression()))) {
            SemaRef.Diag(CI->getAssociatedExpression()->getExprLoc(),
                         diag::err_omp_multiple_array_items_in_map_clause)
                << CI->getAssociatedExpression()->getSourceRange();
            SemaRef.Diag(SI->getAssociatedExpression()->getExprLoc(),
                         diag::note_used_here)
                << SI->getAssociatedExpression()->getSourceRange();
            return true;
          }

          // Do both expressions have the same kind?
          if (CI->getAssociatedExpression()->getStmtClass() !=
              SI->getAssociatedExpression()->getStmtClass())
            break;

          // Are we dealing with different variables/fields?
          if (CI->getAssociatedDeclaration() != SI->getAssociatedDeclaration())
            break;
        }
        // Check if the extra components of the expressions in the enclosing
        // data environment are redundant for the current base declaration.
        // If they are, the maps completely overlap, which is legal.
        for (; SI != SE; ++SI) {
          QualType Type;
          if (const auto *ASE =
                  dyn_cast<ArraySubscriptExpr>(SI->getAssociatedExpression())) {
            Type = ASE->getBase()->IgnoreParenImpCasts()->getType();
          } else if (const auto *OASE = dyn_cast<OMPArraySectionExpr>(
                         SI->getAssociatedExpression())) {
            const Expr *E = OASE->getBase()->IgnoreParenImpCasts();
            Type =
                OMPArraySectionExpr::getBaseOriginalType(E).getCanonicalType();
          } else if (const auto *OASE = dyn_cast<OMPArrayShapingExpr>(
                         SI->getAssociatedExpression())) {
            Type = OASE->getBase()->getType()->getPointeeType();
          }
          if (Type.isNull() || Type->isAnyPointerType() ||
              checkArrayExpressionDoesNotReferToWholeSize(
                  SemaRef, SI->getAssociatedExpression(), Type))
            break;
        }

        // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.4]
        //  List items of map clauses in the same construct must not share
        //  original storage.
        //
        // If the expressions are exactly the same or one is a subset of the
        // other, it means they are sharing storage.
        if (CI == CE && SI == SE) {
          if (CurrentRegionOnly) {
            if (CKind == OMPC_map) {
              SemaRef.Diag(ELoc, diag::err_omp_map_shared_storage) << ERange;
            } else {
              assert(CKind == OMPC_to || CKind == OMPC_from);
              SemaRef.Diag(ELoc, diag::err_omp_once_referenced_in_target_update)
                  << ERange;
            }
            SemaRef.Diag(RE->getExprLoc(), diag::note_used_here)
                << RE->getSourceRange();
            return true;
          }
          // If we find the same expression in the enclosing data environment,
          // that is legal.
          IsEnclosedByDataEnvironmentExpr = true;
          return false;
        }

        QualType DerivedType =
            std::prev(CI)->getAssociatedDeclaration()->getType();
        SourceLocation DerivedLoc =
            std::prev(CI)->getAssociatedExpression()->getExprLoc();

        // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C++, p.1]
        //  If the type of a list item is a reference to a type T then the type
        //  will be considered to be T for all purposes of this clause.
        DerivedType = DerivedType.getNonReferenceType();

        // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C/C++, p.1]
        //  A variable for which the type is pointer and an array section
        //  derived from that variable must not appear as list items of map
        //  clauses of the same construct.
        //
        // Also, cover one of the cases in:
        // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.5]
        //  If any part of the original storage of a list item has corresponding
        //  storage in the device data environment, all of the original storage
        //  must have corresponding storage in the device data environment.
        //
        if (DerivedType->isAnyPointerType()) {
          if (CI == CE || SI == SE) {
            SemaRef.Diag(
                DerivedLoc,
                diag::err_omp_pointer_mapped_along_with_derived_section)
                << DerivedLoc;
            SemaRef.Diag(RE->getExprLoc(), diag::note_used_here)
                << RE->getSourceRange();
            return true;
          }
          if (CI->getAssociatedExpression()->getStmtClass() !=
                  SI->getAssociatedExpression()->getStmtClass() ||
              CI->getAssociatedDeclaration()->getCanonicalDecl() ==
                  SI->getAssociatedDeclaration()->getCanonicalDecl()) {
            assert(CI != CE && SI != SE);
            SemaRef.Diag(DerivedLoc, diag::err_omp_same_pointer_dereferenced)
                << DerivedLoc;
            SemaRef.Diag(RE->getExprLoc(), diag::note_used_here)
                << RE->getSourceRange();
            return true;
          }
        }

        // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.4]
        //  List items of map clauses in the same construct must not share
        //  original storage.
        //
        // An expression is a subset of the other.
        if (CurrentRegionOnly && (CI == CE || SI == SE)) {
          if (CKind == OMPC_map) {
            if (CI != CE || SI != SE) {
              // Allow constructs like this: map(s, s.ptr[0:1]), where s.ptr is
              // a pointer.
              auto Begin =
                  CI != CE ? CurComponents.begin() : StackComponents.begin();
              auto End = CI != CE ? CurComponents.end() : StackComponents.end();
              auto It = Begin;
              while (It != End && !It->getAssociatedDeclaration())
                std::advance(It, 1);
              assert(It != End &&
                     "Expected at least one component with the declaration.");
              if (It != Begin && It->getAssociatedDeclaration()
                                     ->getType()
                                     .getCanonicalType()
                                     ->isAnyPointerType()) {
                IsEnclosedByDataEnvironmentExpr = false;
                EnclosingExpr = nullptr;
                return false;
              }
            }
            SemaRef.Diag(ELoc, diag::err_omp_map_shared_storage) << ERange;
          } else {
            assert(CKind == OMPC_to || CKind == OMPC_from);
            SemaRef.Diag(ELoc, diag::err_omp_once_referenced_in_target_update)
                << ERange;
          }
          SemaRef.Diag(RE->getExprLoc(), diag::note_used_here)
              << RE->getSourceRange();
          return true;
        }

        // The current expression uses the same base as other expression in the
        // data environment but does not contain it completely.
        if (!CurrentRegionOnly && SI != SE)
          EnclosingExpr = RE;

        // The current expression is a subset of the expression in the data
        // environment.
        IsEnclosedByDataEnvironmentExpr |=
            (!CurrentRegionOnly && CI != CE && SI == SE);

        return false;
      });

  if (CurrentRegionOnly)
    return FoundError;

  // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.5]
  //  If any part of the original storage of a list item has corresponding
  //  storage in the device data environment, all of the original storage must
  //  have corresponding storage in the device data environment.
  // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.6]
  //  If a list item is an element of a structure, and a different element of
  //  the structure has a corresponding list item in the device data environment
  //  prior to a task encountering the construct associated with the map clause,
  //  then the list item must also have a corresponding list item in the device
  //  data environment prior to the task encountering the construct.
  //
  if (EnclosingExpr && !IsEnclosedByDataEnvironmentExpr) {
    SemaRef.Diag(ELoc,
                 diag::err_omp_original_storage_is_shared_and_does_not_contain)
        << ERange;
    SemaRef.Diag(EnclosingExpr->getExprLoc(), diag::note_used_here)
        << EnclosingExpr->getSourceRange();
    return true;
  }

  return FoundError;
}

// Look up the user-defined mapper given the mapper name and mapped type, and
// build a reference to it.
static ExprResult buildUserDefinedMapperRef(Sema &SemaRef, Scope *S,
                                            CXXScopeSpec &MapperIdScopeSpec,
                                            const DeclarationNameInfo &MapperId,
                                            QualType Type,
                                            Expr *UnresolvedMapper) {
  if (MapperIdScopeSpec.isInvalid())
    return ExprError();
  // Get the actual type for the array type.
  if (Type->isArrayType()) {
    assert(Type->getAsArrayTypeUnsafe() && "Expect to get a valid array type");
    Type = Type->getAsArrayTypeUnsafe()->getElementType().getCanonicalType();
  }
  // Find all user-defined mappers with the given MapperId.
  SmallVector<UnresolvedSet<8>, 4> Lookups;
  LookupResult Lookup(SemaRef, MapperId, Sema::LookupOMPMapperName);
  Lookup.suppressDiagnostics();
  if (S) {
    while (S && SemaRef.LookupParsedName(Lookup, S, &MapperIdScopeSpec)) {
      NamedDecl *D = Lookup.getRepresentativeDecl();
      while (S && !S->isDeclScope(D))
        S = S->getParent();
      if (S)
        S = S->getParent();
      Lookups.emplace_back();
      Lookups.back().append(Lookup.begin(), Lookup.end());
      Lookup.clear();
    }
  } else if (auto *ULE = cast_or_null<UnresolvedLookupExpr>(UnresolvedMapper)) {
    // Extract the user-defined mappers with the given MapperId.
    Lookups.push_back(UnresolvedSet<8>());
    for (NamedDecl *D : ULE->decls()) {
      auto *DMD = cast<OMPDeclareMapperDecl>(D);
      assert(DMD && "Expect valid OMPDeclareMapperDecl during instantiation.");
      Lookups.back().addDecl(DMD);
    }
  }
  // Defer the lookup for dependent types. The results will be passed through
  // UnresolvedMapper on instantiation.
  if (SemaRef.CurContext->isDependentContext() || Type->isDependentType() ||
      Type->isInstantiationDependentType() ||
      Type->containsUnexpandedParameterPack() ||
      filterLookupForUDReductionAndMapper<bool>(Lookups, [](ValueDecl *D) {
        return !D->isInvalidDecl() &&
               (D->getType()->isDependentType() ||
                D->getType()->isInstantiationDependentType() ||
                D->getType()->containsUnexpandedParameterPack());
      })) {
    UnresolvedSet<8> URS;
    for (const UnresolvedSet<8> &Set : Lookups) {
      if (Set.empty())
        continue;
      URS.append(Set.begin(), Set.end());
    }
    return UnresolvedLookupExpr::Create(
        SemaRef.Context, /*NamingClass=*/nullptr,
        MapperIdScopeSpec.getWithLocInContext(SemaRef.Context), MapperId,
        /*ADL=*/false, /*Overloaded=*/true, URS.begin(), URS.end());
  }
  SourceLocation Loc = MapperId.getLoc();
  // [OpenMP 5.0], 2.19.7.3 declare mapper Directive, Restrictions
  //  The type must be of struct, union or class type in C and C++
  if (!Type->isStructureOrClassType() && !Type->isUnionType() &&
      (MapperIdScopeSpec.isSet() || MapperId.getAsString() != "default")) {
    SemaRef.Diag(Loc, diag::err_omp_mapper_wrong_type);
    return ExprError();
  }
  // Perform argument dependent lookup.
  if (SemaRef.getLangOpts().CPlusPlus && !MapperIdScopeSpec.isSet())
    argumentDependentLookup(SemaRef, MapperId, Loc, Type, Lookups);
  // Return the first user-defined mapper with the desired type.
  if (auto *VD = filterLookupForUDReductionAndMapper<ValueDecl *>(
          Lookups, [&SemaRef, Type](ValueDecl *D) -> ValueDecl * {
            if (!D->isInvalidDecl() &&
                SemaRef.Context.hasSameType(D->getType(), Type))
              return D;
            return nullptr;
          }))
    return SemaRef.BuildDeclRefExpr(VD, Type, VK_LValue, Loc);
  // Find the first user-defined mapper with a type derived from the desired
  // type.
  if (auto *VD = filterLookupForUDReductionAndMapper<ValueDecl *>(
          Lookups, [&SemaRef, Type, Loc](ValueDecl *D) -> ValueDecl * {
            if (!D->isInvalidDecl() &&
                SemaRef.IsDerivedFrom(Loc, Type, D->getType()) &&
                !Type.isMoreQualifiedThan(D->getType()))
              return D;
            return nullptr;
          })) {
    CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                       /*DetectVirtual=*/false);
    if (SemaRef.IsDerivedFrom(Loc, Type, VD->getType(), Paths)) {
      if (!Paths.isAmbiguous(SemaRef.Context.getCanonicalType(
              VD->getType().getUnqualifiedType()))) {
        if (SemaRef.CheckBaseClassAccess(
                Loc, VD->getType(), Type, Paths.front(),
                /*DiagID=*/0) != Sema::AR_inaccessible) {
          return SemaRef.BuildDeclRefExpr(VD, Type, VK_LValue, Loc);
        }
      }
    }
  }
  // Report error if a mapper is specified, but cannot be found.
  if (MapperIdScopeSpec.isSet() || MapperId.getAsString() != "default") {
    SemaRef.Diag(Loc, diag::err_omp_invalid_mapper)
        << Type << MapperId.getName();
    return ExprError();
  }
  return ExprEmpty();
}

namespace {
// Utility struct that gathers all the related lists associated with a mappable
// expression.
struct MappableVarListInfo {
  // The list of expressions.
  ArrayRef<Expr *> VarList;
  // The list of processed expressions.
  SmallVector<Expr *, 16> ProcessedVarList;
  // The mappble components for each expression.
  OMPClauseMappableExprCommon::MappableExprComponentLists VarComponents;
  // The base declaration of the variable.
  SmallVector<ValueDecl *, 16> VarBaseDeclarations;
  // The reference to the user-defined mapper associated with every expression.
  SmallVector<Expr *, 16> UDMapperList;

  MappableVarListInfo(ArrayRef<Expr *> VarList) : VarList(VarList) {
    // We have a list of components and base declarations for each entry in the
    // variable list.
    VarComponents.reserve(VarList.size());
    VarBaseDeclarations.reserve(VarList.size());
  }
};
} // namespace

// Check the validity of the provided variable list for the provided clause kind
// \a CKind. In the check process the valid expressions, mappable expression
// components, variables, and user-defined mappers are extracted and used to
// fill \a ProcessedVarList, \a VarComponents, \a VarBaseDeclarations, and \a
// UDMapperList in MVLI. \a MapType, \a IsMapTypeImplicit, \a MapperIdScopeSpec,
// and \a MapperId are expected to be valid if the clause kind is 'map'.
static void checkMappableExpressionList(
    Sema &SemaRef, DSAStackTy *DSAS, OpenMPClauseKind CKind,
    MappableVarListInfo &MVLI, SourceLocation StartLoc,
    CXXScopeSpec &MapperIdScopeSpec, DeclarationNameInfo MapperId,
    ArrayRef<Expr *> UnresolvedMappers,
    OpenMPMapClauseKind MapType = OMPC_MAP_unknown,
    ArrayRef<OpenMPMapModifierKind> Modifiers = None,
    bool IsMapTypeImplicit = false, bool NoDiagnose = false) {
  // We only expect mappable expressions in 'to', 'from', and 'map' clauses.
  assert((CKind == OMPC_map || CKind == OMPC_to || CKind == OMPC_from) &&
         "Unexpected clause kind with mappable expressions!");

  // If the identifier of user-defined mapper is not specified, it is "default".
  // We do not change the actual name in this clause to distinguish whether a
  // mapper is specified explicitly, i.e., it is not explicitly specified when
  // MapperId.getName() is empty.
  if (!MapperId.getName() || MapperId.getName().isEmpty()) {
    auto &DeclNames = SemaRef.getASTContext().DeclarationNames;
    MapperId.setName(DeclNames.getIdentifier(
        &SemaRef.getASTContext().Idents.get("default")));
    MapperId.setLoc(StartLoc);
  }

  // Iterators to find the current unresolved mapper expression.
  auto UMIt = UnresolvedMappers.begin(), UMEnd = UnresolvedMappers.end();
  bool UpdateUMIt = false;
  Expr *UnresolvedMapper = nullptr;

  bool HasHoldModifier =
      llvm::is_contained(Modifiers, OMPC_MAP_MODIFIER_ompx_hold);

  // Keep track of the mappable components and base declarations in this clause.
  // Each entry in the list is going to have a list of components associated. We
  // record each set of the components so that we can build the clause later on.
  // In the end we should have the same amount of declarations and component
  // lists.

  for (Expr *RE : MVLI.VarList) {
    assert(RE && "Null expr in omp to/from/map clause");
    SourceLocation ELoc = RE->getExprLoc();

    // Find the current unresolved mapper expression.
    if (UpdateUMIt && UMIt != UMEnd) {
      UMIt++;
      assert(
          UMIt != UMEnd &&
          "Expect the size of UnresolvedMappers to match with that of VarList");
    }
    UpdateUMIt = true;
    if (UMIt != UMEnd)
      UnresolvedMapper = *UMIt;

    const Expr *VE = RE->IgnoreParenLValueCasts();

    if (VE->isValueDependent() || VE->isTypeDependent() ||
        VE->isInstantiationDependent() ||
        VE->containsUnexpandedParameterPack()) {
      // Try to find the associated user-defined mapper.
      ExprResult ER = buildUserDefinedMapperRef(
          SemaRef, DSAS->getCurScope(), MapperIdScopeSpec, MapperId,
          VE->getType().getCanonicalType(), UnresolvedMapper);
      if (ER.isInvalid())
        continue;
      MVLI.UDMapperList.push_back(ER.get());
      // We can only analyze this information once the missing information is
      // resolved.
      MVLI.ProcessedVarList.push_back(RE);
      continue;
    }

    Expr *SimpleExpr = RE->IgnoreParenCasts();

    if (!RE->isLValue()) {
      if (SemaRef.getLangOpts().OpenMP < 50) {
        SemaRef.Diag(
            ELoc, diag::err_omp_expected_named_var_member_or_array_expression)
            << RE->getSourceRange();
      } else {
        SemaRef.Diag(ELoc, diag::err_omp_non_lvalue_in_map_or_motion_clauses)
            << getOpenMPClauseName(CKind) << RE->getSourceRange();
      }
      continue;
    }

    OMPClauseMappableExprCommon::MappableExprComponentList CurComponents;
    ValueDecl *CurDeclaration = nullptr;

    // Obtain the array or member expression bases if required. Also, fill the
    // components array with all the components identified in the process.
    const Expr *BE =
        checkMapClauseExpressionBase(SemaRef, SimpleExpr, CurComponents, CKind,
                                     DSAS->getCurrentDirective(), NoDiagnose);
    if (!BE)
      continue;

    assert(!CurComponents.empty() &&
           "Invalid mappable expression information.");

    if (const auto *TE = dyn_cast<CXXThisExpr>(BE)) {
      // Add store "this" pointer to class in DSAStackTy for future checking
      DSAS->addMappedClassesQualTypes(TE->getType());
      // Try to find the associated user-defined mapper.
      ExprResult ER = buildUserDefinedMapperRef(
          SemaRef, DSAS->getCurScope(), MapperIdScopeSpec, MapperId,
          VE->getType().getCanonicalType(), UnresolvedMapper);
      if (ER.isInvalid())
        continue;
      MVLI.UDMapperList.push_back(ER.get());
      // Skip restriction checking for variable or field declarations
      MVLI.ProcessedVarList.push_back(RE);
      MVLI.VarComponents.resize(MVLI.VarComponents.size() + 1);
      MVLI.VarComponents.back().append(CurComponents.begin(),
                                       CurComponents.end());
      MVLI.VarBaseDeclarations.push_back(nullptr);
      continue;
    }

    // For the following checks, we rely on the base declaration which is
    // expected to be associated with the last component. The declaration is
    // expected to be a variable or a field (if 'this' is being mapped).
    CurDeclaration = CurComponents.back().getAssociatedDeclaration();
    assert(CurDeclaration && "Null decl on map clause.");
    assert(
        CurDeclaration->isCanonicalDecl() &&
        "Expecting components to have associated only canonical declarations.");

    auto *VD = dyn_cast<VarDecl>(CurDeclaration);
    const auto *FD = dyn_cast<FieldDecl>(CurDeclaration);

    assert((VD || FD) && "Only variables or fields are expected here!");
    (void)FD;

    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.10]
    // threadprivate variables cannot appear in a map clause.
    // OpenMP 4.5 [2.10.5, target update Construct]
    // threadprivate variables cannot appear in a from clause.
    if (VD && DSAS->isThreadPrivate(VD)) {
      if (NoDiagnose)
        continue;
      DSAStackTy::DSAVarData DVar = DSAS->getTopDSA(VD, /*FromParent=*/false);
      SemaRef.Diag(ELoc, diag::err_omp_threadprivate_in_clause)
          << getOpenMPClauseName(CKind);
      reportOriginalDsa(SemaRef, DSAS, VD, DVar);
      continue;
    }

    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.9]
    //  A list item cannot appear in both a map clause and a data-sharing
    //  attribute clause on the same construct.

    // Check conflicts with other map clause expressions. We check the conflicts
    // with the current construct separately from the enclosing data
    // environment, because the restrictions are different. We only have to
    // check conflicts across regions for the map clauses.
    if (checkMapConflicts(SemaRef, DSAS, CurDeclaration, SimpleExpr,
                          /*CurrentRegionOnly=*/true, CurComponents, CKind))
      break;
    if (CKind == OMPC_map &&
        (SemaRef.getLangOpts().OpenMP <= 45 || StartLoc.isValid()) &&
        checkMapConflicts(SemaRef, DSAS, CurDeclaration, SimpleExpr,
                          /*CurrentRegionOnly=*/false, CurComponents, CKind))
      break;

    // OpenMP 4.5 [2.10.5, target update Construct]
    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, C++, p.1]
    //  If the type of a list item is a reference to a type T then the type will
    //  be considered to be T for all purposes of this clause.
    auto I = llvm::find_if(
        CurComponents,
        [](const OMPClauseMappableExprCommon::MappableComponent &MC) {
          return MC.getAssociatedDeclaration();
        });
    assert(I != CurComponents.end() && "Null decl on map clause.");
    (void)I;
    QualType Type;
    auto *ASE = dyn_cast<ArraySubscriptExpr>(VE->IgnoreParens());
    auto *OASE = dyn_cast<OMPArraySectionExpr>(VE->IgnoreParens());
    auto *OAShE = dyn_cast<OMPArrayShapingExpr>(VE->IgnoreParens());
    if (ASE) {
      Type = ASE->getType().getNonReferenceType();
    } else if (OASE) {
      QualType BaseType =
          OMPArraySectionExpr::getBaseOriginalType(OASE->getBase());
      if (const auto *ATy = BaseType->getAsArrayTypeUnsafe())
        Type = ATy->getElementType();
      else
        Type = BaseType->getPointeeType();
      Type = Type.getNonReferenceType();
    } else if (OAShE) {
      Type = OAShE->getBase()->getType()->getPointeeType();
    } else {
      Type = VE->getType();
    }

    // OpenMP 4.5 [2.10.5, target update Construct, Restrictions, p.4]
    // A list item in a to or from clause must have a mappable type.
    // OpenMP 4.5 [2.15.5.1, map Clause, Restrictions, p.9]
    //  A list item must have a mappable type.
    if (!checkTypeMappable(VE->getExprLoc(), VE->getSourceRange(), SemaRef,
                           DSAS, Type, /*FullCheck=*/true))
      continue;

    if (CKind == OMPC_map) {
      // target enter data
      // OpenMP [2.10.2, Restrictions, p. 99]
      // A map-type must be specified in all map clauses and must be either
      // to or alloc.
      OpenMPDirectiveKind DKind = DSAS->getCurrentDirective();
      if (DKind == OMPD_target_enter_data &&
          !(MapType == OMPC_MAP_to || MapType == OMPC_MAP_alloc)) {
        SemaRef.Diag(StartLoc, diag::err_omp_invalid_map_type_for_directive)
            << (IsMapTypeImplicit ? 1 : 0)
            << getOpenMPSimpleClauseTypeName(OMPC_map, MapType)
            << getOpenMPDirectiveName(DKind);
        continue;
      }

      // target exit_data
      // OpenMP [2.10.3, Restrictions, p. 102]
      // A map-type must be specified in all map clauses and must be either
      // from, release, or delete.
      if (DKind == OMPD_target_exit_data &&
          !(MapType == OMPC_MAP_from || MapType == OMPC_MAP_release ||
            MapType == OMPC_MAP_delete)) {
        SemaRef.Diag(StartLoc, diag::err_omp_invalid_map_type_for_directive)
            << (IsMapTypeImplicit ? 1 : 0)
            << getOpenMPSimpleClauseTypeName(OMPC_map, MapType)
            << getOpenMPDirectiveName(DKind);
        continue;
      }

      // The 'ompx_hold' modifier is specifically intended to be used on a
      // 'target' or 'target data' directive to prevent data from being unmapped
      // during the associated statement.  It is not permitted on a 'target
      // enter data' or 'target exit data' directive, which have no associated
      // statement.
      if ((DKind == OMPD_target_enter_data || DKind == OMPD_target_exit_data) &&
          HasHoldModifier) {
        SemaRef.Diag(StartLoc,
                     diag::err_omp_invalid_map_type_modifier_for_directive)
            << getOpenMPSimpleClauseTypeName(OMPC_map,
                                             OMPC_MAP_MODIFIER_ompx_hold)
            << getOpenMPDirectiveName(DKind);
        continue;
      }

      // target, target data
      // OpenMP 5.0 [2.12.2, Restrictions, p. 163]
      // OpenMP 5.0 [2.12.5, Restrictions, p. 174]
      // A map-type in a map clause must be to, from, tofrom or alloc
      if ((DKind == OMPD_target_data ||
           isOpenMPTargetExecutionDirective(DKind)) &&
          !(MapType == OMPC_MAP_to || MapType == OMPC_MAP_from ||
            MapType == OMPC_MAP_tofrom || MapType == OMPC_MAP_alloc)) {
        SemaRef.Diag(StartLoc, diag::err_omp_invalid_map_type_for_directive)
            << (IsMapTypeImplicit ? 1 : 0)
            << getOpenMPSimpleClauseTypeName(OMPC_map, MapType)
            << getOpenMPDirectiveName(DKind);
        continue;
      }

      // OpenMP 4.5 [2.15.5.1, Restrictions, p.3]
      // A list item cannot appear in both a map clause and a data-sharing
      // attribute clause on the same construct
      //
      // OpenMP 5.0 [2.19.7.1, Restrictions, p.7]
      // A list item cannot appear in both a map clause and a data-sharing
      // attribute clause on the same construct unless the construct is a
      // combined construct.
      if (VD && ((SemaRef.LangOpts.OpenMP <= 45 &&
                  isOpenMPTargetExecutionDirective(DKind)) ||
                 DKind == OMPD_target)) {
        DSAStackTy::DSAVarData DVar = DSAS->getTopDSA(VD, /*FromParent=*/false);
        if (isOpenMPPrivate(DVar.CKind)) {
          SemaRef.Diag(ELoc, diag::err_omp_variable_in_given_clause_and_dsa)
              << getOpenMPClauseName(DVar.CKind)
              << getOpenMPClauseName(OMPC_map)
              << getOpenMPDirectiveName(DSAS->getCurrentDirective());
          reportOriginalDsa(SemaRef, DSAS, CurDeclaration, DVar);
          continue;
        }
      }
    }

    // Try to find the associated user-defined mapper.
    ExprResult ER = buildUserDefinedMapperRef(
        SemaRef, DSAS->getCurScope(), MapperIdScopeSpec, MapperId,
        Type.getCanonicalType(), UnresolvedMapper);
    if (ER.isInvalid())
      continue;
    MVLI.UDMapperList.push_back(ER.get());

    // Save the current expression.
    MVLI.ProcessedVarList.push_back(RE);

    // Store the components in the stack so that they can be used to check
    // against other clauses later on.
    DSAS->addMappableExpressionComponents(CurDeclaration, CurComponents,
                                          /*WhereFoundClauseKind=*/OMPC_map);

    // Save the components and declaration to create the clause. For purposes of
    // the clause creation, any component list that has has base 'this' uses
    // null as base declaration.
    MVLI.VarComponents.resize(MVLI.VarComponents.size() + 1);
    MVLI.VarComponents.back().append(CurComponents.begin(),
                                     CurComponents.end());
    MVLI.VarBaseDeclarations.push_back(isa<MemberExpr>(BE) ? nullptr
                                                           : CurDeclaration);
  }
}

OMPClause *Sema::ActOnOpenMPMapClause(
    ArrayRef<OpenMPMapModifierKind> MapTypeModifiers,
    ArrayRef<SourceLocation> MapTypeModifiersLoc,
    CXXScopeSpec &MapperIdScopeSpec, DeclarationNameInfo &MapperId,
    OpenMPMapClauseKind MapType, bool IsMapTypeImplicit, SourceLocation MapLoc,
    SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
    const OMPVarListLocTy &Locs, bool NoDiagnose,
    ArrayRef<Expr *> UnresolvedMappers) {
  OpenMPMapModifierKind Modifiers[] = {
      OMPC_MAP_MODIFIER_unknown, OMPC_MAP_MODIFIER_unknown,
      OMPC_MAP_MODIFIER_unknown, OMPC_MAP_MODIFIER_unknown,
      OMPC_MAP_MODIFIER_unknown};
  SourceLocation ModifiersLoc[NumberOfOMPMapClauseModifiers];

  // Process map-type-modifiers, flag errors for duplicate modifiers.
  unsigned Count = 0;
  for (unsigned I = 0, E = MapTypeModifiers.size(); I < E; ++I) {
    if (MapTypeModifiers[I] != OMPC_MAP_MODIFIER_unknown &&
        llvm::is_contained(Modifiers, MapTypeModifiers[I])) {
      Diag(MapTypeModifiersLoc[I], diag::err_omp_duplicate_map_type_modifier);
      continue;
    }
    assert(Count < NumberOfOMPMapClauseModifiers &&
           "Modifiers exceed the allowed number of map type modifiers");
    Modifiers[Count] = MapTypeModifiers[I];
    ModifiersLoc[Count] = MapTypeModifiersLoc[I];
    ++Count;
  }

  MappableVarListInfo MVLI(VarList);
  checkMappableExpressionList(*this, DSAStack, OMPC_map, MVLI, Locs.StartLoc,
                              MapperIdScopeSpec, MapperId, UnresolvedMappers,
                              MapType, Modifiers, IsMapTypeImplicit,
                              NoDiagnose);

  // We need to produce a map clause even if we don't have variables so that
  // other diagnostics related with non-existing map clauses are accurate.
  return OMPMapClause::Create(Context, Locs, MVLI.ProcessedVarList,
                              MVLI.VarBaseDeclarations, MVLI.VarComponents,
                              MVLI.UDMapperList, Modifiers, ModifiersLoc,
                              MapperIdScopeSpec.getWithLocInContext(Context),
                              MapperId, MapType, IsMapTypeImplicit, MapLoc);
}

QualType Sema::ActOnOpenMPDeclareReductionType(SourceLocation TyLoc,
                                               TypeResult ParsedType) {
  assert(ParsedType.isUsable());

  QualType ReductionType = GetTypeFromParser(ParsedType.get());
  if (ReductionType.isNull())
    return QualType();

  // [OpenMP 4.0], 2.15 declare reduction Directive, Restrictions, C\C++
  // A type name in a declare reduction directive cannot be a function type, an
  // array type, a reference type, or a type qualified with const, volatile or
  // restrict.
  if (ReductionType.hasQualifiers()) {
    Diag(TyLoc, diag::err_omp_reduction_wrong_type) << 0;
    return QualType();
  }

  if (ReductionType->isFunctionType()) {
    Diag(TyLoc, diag::err_omp_reduction_wrong_type) << 1;
    return QualType();
  }
  if (ReductionType->isReferenceType()) {
    Diag(TyLoc, diag::err_omp_reduction_wrong_type) << 2;
    return QualType();
  }
  if (ReductionType->isArrayType()) {
    Diag(TyLoc, diag::err_omp_reduction_wrong_type) << 3;
    return QualType();
  }
  return ReductionType;
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPDeclareReductionDirectiveStart(
    Scope *S, DeclContext *DC, DeclarationName Name,
    ArrayRef<std::pair<QualType, SourceLocation>> ReductionTypes,
    AccessSpecifier AS, Decl *PrevDeclInScope) {
  SmallVector<Decl *, 8> Decls;
  Decls.reserve(ReductionTypes.size());

  LookupResult Lookup(*this, Name, SourceLocation(), LookupOMPReductionName,
                      forRedeclarationInCurContext());
  // [OpenMP 4.0], 2.15 declare reduction Directive, Restrictions
  // A reduction-identifier may not be re-declared in the current scope for the
  // same type or for a type that is compatible according to the base language
  // rules.
  llvm::DenseMap<QualType, SourceLocation> PreviousRedeclTypes;
  OMPDeclareReductionDecl *PrevDRD = nullptr;
  bool InCompoundScope = true;
  if (S != nullptr) {
    // Find previous declaration with the same name not referenced in other
    // declarations.
    FunctionScopeInfo *ParentFn = getEnclosingFunction();
    InCompoundScope =
        (ParentFn != nullptr) && !ParentFn->CompoundScopes.empty();
    LookupName(Lookup, S);
    FilterLookupForScope(Lookup, DC, S, /*ConsiderLinkage=*/false,
                         /*AllowInlineNamespace=*/false);
    llvm::DenseMap<OMPDeclareReductionDecl *, bool> UsedAsPrevious;
    LookupResult::Filter Filter = Lookup.makeFilter();
    while (Filter.hasNext()) {
      auto *PrevDecl = cast<OMPDeclareReductionDecl>(Filter.next());
      if (InCompoundScope) {
        auto I = UsedAsPrevious.find(PrevDecl);
        if (I == UsedAsPrevious.end())
          UsedAsPrevious[PrevDecl] = false;
        if (OMPDeclareReductionDecl *D = PrevDecl->getPrevDeclInScope())
          UsedAsPrevious[D] = true;
      }
      PreviousRedeclTypes[PrevDecl->getType().getCanonicalType()] =
          PrevDecl->getLocation();
    }
    Filter.done();
    if (InCompoundScope) {
      for (const auto &PrevData : UsedAsPrevious) {
        if (!PrevData.second) {
          PrevDRD = PrevData.first;
          break;
        }
      }
    }
  } else if (PrevDeclInScope != nullptr) {
    auto *PrevDRDInScope = PrevDRD =
        cast<OMPDeclareReductionDecl>(PrevDeclInScope);
    do {
      PreviousRedeclTypes[PrevDRDInScope->getType().getCanonicalType()] =
          PrevDRDInScope->getLocation();
      PrevDRDInScope = PrevDRDInScope->getPrevDeclInScope();
    } while (PrevDRDInScope != nullptr);
  }
  for (const auto &TyData : ReductionTypes) {
    const auto I = PreviousRedeclTypes.find(TyData.first.getCanonicalType());
    bool Invalid = false;
    if (I != PreviousRedeclTypes.end()) {
      Diag(TyData.second, diag::err_omp_declare_reduction_redefinition)
          << TyData.first;
      Diag(I->second, diag::note_previous_definition);
      Invalid = true;
    }
    PreviousRedeclTypes[TyData.first.getCanonicalType()] = TyData.second;
    auto *DRD = OMPDeclareReductionDecl::Create(Context, DC, TyData.second,
                                                Name, TyData.first, PrevDRD);
    DC->addDecl(DRD);
    DRD->setAccess(AS);
    Decls.push_back(DRD);
    if (Invalid)
      DRD->setInvalidDecl();
    else
      PrevDRD = DRD;
  }

  return DeclGroupPtrTy::make(
      DeclGroupRef::Create(Context, Decls.begin(), Decls.size()));
}

void Sema::ActOnOpenMPDeclareReductionCombinerStart(Scope *S, Decl *D) {
  auto *DRD = cast<OMPDeclareReductionDecl>(D);

  // Enter new function scope.
  PushFunctionScope();
  setFunctionHasBranchProtectedScope();
  getCurFunction()->setHasOMPDeclareReductionCombiner();

  if (S != nullptr)
    PushDeclContext(S, DRD);
  else
    CurContext = DRD;

  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  QualType ReductionType = DRD->getType();
  // Create 'T* omp_parm;T omp_in;'. All references to 'omp_in' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_in'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_in;' variable.
  VarDecl *OmpInParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_in");
  // Create 'T* omp_parm;T omp_out;'. All references to 'omp_out' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_out'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_out;' variable.
  VarDecl *OmpOutParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_out");
  if (S != nullptr) {
    PushOnScopeChains(OmpInParm, S);
    PushOnScopeChains(OmpOutParm, S);
  } else {
    DRD->addDecl(OmpInParm);
    DRD->addDecl(OmpOutParm);
  }
  Expr *InE =
      ::buildDeclRefExpr(*this, OmpInParm, ReductionType, D->getLocation());
  Expr *OutE =
      ::buildDeclRefExpr(*this, OmpOutParm, ReductionType, D->getLocation());
  DRD->setCombinerData(InE, OutE);
}

void Sema::ActOnOpenMPDeclareReductionCombinerEnd(Decl *D, Expr *Combiner) {
  auto *DRD = cast<OMPDeclareReductionDecl>(D);
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  PopDeclContext();
  PopFunctionScopeInfo();

  if (Combiner != nullptr)
    DRD->setCombiner(Combiner);
  else
    DRD->setInvalidDecl();
}

VarDecl *Sema::ActOnOpenMPDeclareReductionInitializerStart(Scope *S, Decl *D) {
  auto *DRD = cast<OMPDeclareReductionDecl>(D);

  // Enter new function scope.
  PushFunctionScope();
  setFunctionHasBranchProtectedScope();

  if (S != nullptr)
    PushDeclContext(S, DRD);
  else
    CurContext = DRD;

  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  QualType ReductionType = DRD->getType();
  // Create 'T* omp_parm;T omp_priv;'. All references to 'omp_priv' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_priv'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_priv;' variable.
  VarDecl *OmpPrivParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_priv");
  // Create 'T* omp_parm;T omp_orig;'. All references to 'omp_orig' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_orig'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_orig;' variable.
  VarDecl *OmpOrigParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_orig");
  if (S != nullptr) {
    PushOnScopeChains(OmpPrivParm, S);
    PushOnScopeChains(OmpOrigParm, S);
  } else {
    DRD->addDecl(OmpPrivParm);
    DRD->addDecl(OmpOrigParm);
  }
  Expr *OrigE =
      ::buildDeclRefExpr(*this, OmpOrigParm, ReductionType, D->getLocation());
  Expr *PrivE =
      ::buildDeclRefExpr(*this, OmpPrivParm, ReductionType, D->getLocation());
  DRD->setInitializerData(OrigE, PrivE);
  return OmpPrivParm;
}

void Sema::ActOnOpenMPDeclareReductionInitializerEnd(Decl *D, Expr *Initializer,
                                                     VarDecl *OmpPrivParm) {
  auto *DRD = cast<OMPDeclareReductionDecl>(D);
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  PopDeclContext();
  PopFunctionScopeInfo();

  if (Initializer != nullptr) {
    DRD->setInitializer(Initializer, OMPDeclareReductionDecl::CallInit);
  } else if (OmpPrivParm->hasInit()) {
    DRD->setInitializer(OmpPrivParm->getInit(),
                        OmpPrivParm->isDirectInit()
                            ? OMPDeclareReductionDecl::DirectInit
                            : OMPDeclareReductionDecl::CopyInit);
  } else {
    DRD->setInvalidDecl();
  }
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPDeclareReductionDirectiveEnd(
    Scope *S, DeclGroupPtrTy DeclReductions, bool IsValid) {
  for (Decl *D : DeclReductions.get()) {
    if (IsValid) {
      if (S)
        PushOnScopeChains(cast<OMPDeclareReductionDecl>(D), S,
                          /*AddToContext=*/false);
    } else {
      D->setInvalidDecl();
    }
  }
  return DeclReductions;
}

TypeResult Sema::ActOnOpenMPDeclareMapperVarDecl(Scope *S, Declarator &D) {
  TypeSourceInfo *TInfo = GetTypeForDeclarator(D, S);
  QualType T = TInfo->getType();
  if (D.isInvalidType())
    return true;

  if (getLangOpts().CPlusPlus) {
    // Check that there are no default arguments (C++ only).
    CheckExtraCXXDefaultArguments(D);
  }

  return CreateParsedType(T, TInfo);
}

QualType Sema::ActOnOpenMPDeclareMapperType(SourceLocation TyLoc,
                                            TypeResult ParsedType) {
  assert(ParsedType.isUsable() && "Expect usable parsed mapper type");

  QualType MapperType = GetTypeFromParser(ParsedType.get());
  assert(!MapperType.isNull() && "Expect valid mapper type");

  // [OpenMP 5.0], 2.19.7.3 declare mapper Directive, Restrictions
  //  The type must be of struct, union or class type in C and C++
  if (!MapperType->isStructureOrClassType() && !MapperType->isUnionType()) {
    Diag(TyLoc, diag::err_omp_mapper_wrong_type);
    return QualType();
  }
  return MapperType;
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPDeclareMapperDirective(
    Scope *S, DeclContext *DC, DeclarationName Name, QualType MapperType,
    SourceLocation StartLoc, DeclarationName VN, AccessSpecifier AS,
    Expr *MapperVarRef, ArrayRef<OMPClause *> Clauses, Decl *PrevDeclInScope) {
  LookupResult Lookup(*this, Name, SourceLocation(), LookupOMPMapperName,
                      forRedeclarationInCurContext());
  // [OpenMP 5.0], 2.19.7.3 declare mapper Directive, Restrictions
  //  A mapper-identifier may not be redeclared in the current scope for the
  //  same type or for a type that is compatible according to the base language
  //  rules.
  llvm::DenseMap<QualType, SourceLocation> PreviousRedeclTypes;
  OMPDeclareMapperDecl *PrevDMD = nullptr;
  bool InCompoundScope = true;
  if (S != nullptr) {
    // Find previous declaration with the same name not referenced in other
    // declarations.
    FunctionScopeInfo *ParentFn = getEnclosingFunction();
    InCompoundScope =
        (ParentFn != nullptr) && !ParentFn->CompoundScopes.empty();
    LookupName(Lookup, S);
    FilterLookupForScope(Lookup, DC, S, /*ConsiderLinkage=*/false,
                         /*AllowInlineNamespace=*/false);
    llvm::DenseMap<OMPDeclareMapperDecl *, bool> UsedAsPrevious;
    LookupResult::Filter Filter = Lookup.makeFilter();
    while (Filter.hasNext()) {
      auto *PrevDecl = cast<OMPDeclareMapperDecl>(Filter.next());
      if (InCompoundScope) {
        auto I = UsedAsPrevious.find(PrevDecl);
        if (I == UsedAsPrevious.end())
          UsedAsPrevious[PrevDecl] = false;
        if (OMPDeclareMapperDecl *D = PrevDecl->getPrevDeclInScope())
          UsedAsPrevious[D] = true;
      }
      PreviousRedeclTypes[PrevDecl->getType().getCanonicalType()] =
          PrevDecl->getLocation();
    }
    Filter.done();
    if (InCompoundScope) {
      for (const auto &PrevData : UsedAsPrevious) {
        if (!PrevData.second) {
          PrevDMD = PrevData.first;
          break;
        }
      }
    }
  } else if (PrevDeclInScope) {
    auto *PrevDMDInScope = PrevDMD =
        cast<OMPDeclareMapperDecl>(PrevDeclInScope);
    do {
      PreviousRedeclTypes[PrevDMDInScope->getType().getCanonicalType()] =
          PrevDMDInScope->getLocation();
      PrevDMDInScope = PrevDMDInScope->getPrevDeclInScope();
    } while (PrevDMDInScope != nullptr);
  }
  const auto I = PreviousRedeclTypes.find(MapperType.getCanonicalType());
  bool Invalid = false;
  if (I != PreviousRedeclTypes.end()) {
    Diag(StartLoc, diag::err_omp_declare_mapper_redefinition)
        << MapperType << Name;
    Diag(I->second, diag::note_previous_definition);
    Invalid = true;
  }
  // Build expressions for implicit maps of data members with 'default'
  // mappers.
  SmallVector<OMPClause *, 4> ClausesWithImplicit(Clauses.begin(),
                                                  Clauses.end());
  if (LangOpts.OpenMP >= 50)
    processImplicitMapsWithDefaultMappers(*this, DSAStack, ClausesWithImplicit);
  auto *DMD =
      OMPDeclareMapperDecl::Create(Context, DC, StartLoc, Name, MapperType, VN,
                                   ClausesWithImplicit, PrevDMD);
  if (S)
    PushOnScopeChains(DMD, S);
  else
    DC->addDecl(DMD);
  DMD->setAccess(AS);
  if (Invalid)
    DMD->setInvalidDecl();

  auto *VD = cast<DeclRefExpr>(MapperVarRef)->getDecl();
  VD->setDeclContext(DMD);
  VD->setLexicalDeclContext(DMD);
  DMD->addDecl(VD);
  DMD->setMapperVarRef(MapperVarRef);

  return DeclGroupPtrTy::make(DeclGroupRef(DMD));
}

ExprResult
Sema::ActOnOpenMPDeclareMapperDirectiveVarDecl(Scope *S, QualType MapperType,
                                               SourceLocation StartLoc,
                                               DeclarationName VN) {
  TypeSourceInfo *TInfo =
      Context.getTrivialTypeSourceInfo(MapperType, StartLoc);
  auto *VD = VarDecl::Create(Context, Context.getTranslationUnitDecl(),
                             StartLoc, StartLoc, VN.getAsIdentifierInfo(),
                             MapperType, TInfo, SC_None);
  if (S)
    PushOnScopeChains(VD, S, /*AddToContext=*/false);
  Expr *E = buildDeclRefExpr(*this, VD, MapperType, StartLoc);
  DSAStack->addDeclareMapperVarRef(E);
  return E;
}

bool Sema::isOpenMPDeclareMapperVarDeclAllowed(const VarDecl *VD) const {
  assert(LangOpts.OpenMP && "Expected OpenMP mode.");
  const Expr *Ref = DSAStack->getDeclareMapperVarRef();
  if (const auto *DRE = cast_or_null<DeclRefExpr>(Ref)) {
    if (VD->getCanonicalDecl() == DRE->getDecl()->getCanonicalDecl())
      return true;
    if (VD->isUsableInConstantExpressions(Context))
      return true;
    return false;
  }
  return true;
}

const ValueDecl *Sema::getOpenMPDeclareMapperVarName() const {
  assert(LangOpts.OpenMP && "Expected OpenMP mode.");
  return cast<DeclRefExpr>(DSAStack->getDeclareMapperVarRef())->getDecl();
}

OMPClause *Sema::ActOnOpenMPNumTeamsClause(Expr *NumTeams,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  Expr *ValExpr = NumTeams;
  Stmt *HelperValStmt = nullptr;

  // OpenMP [teams Constrcut, Restrictions]
  // The num_teams expression must evaluate to a positive integer value.
  if (!isNonNegativeIntegerValue(ValExpr, *this, OMPC_num_teams,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
  OpenMPDirectiveKind CaptureRegion =
      getOpenMPCaptureRegionForClause(DKind, OMPC_num_teams, LangOpts.OpenMP);
  if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
    ValExpr = MakeFullExpr(ValExpr).get();
    llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
    ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
    HelperValStmt = buildPreInits(Context, Captures);
  }

  return new (Context) OMPNumTeamsClause(ValExpr, HelperValStmt, CaptureRegion,
                                         StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPThreadLimitClause(Expr *ThreadLimit,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  Expr *ValExpr = ThreadLimit;
  Stmt *HelperValStmt = nullptr;

  // OpenMP [teams Constrcut, Restrictions]
  // The thread_limit expression must evaluate to a positive integer value.
  if (!isNonNegativeIntegerValue(ValExpr, *this, OMPC_thread_limit,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  OpenMPDirectiveKind DKind = DSAStack->getCurrentDirective();
  OpenMPDirectiveKind CaptureRegion = getOpenMPCaptureRegionForClause(
      DKind, OMPC_thread_limit, LangOpts.OpenMP);
  if (CaptureRegion != OMPD_unknown && !CurContext->isDependentContext()) {
    ValExpr = MakeFullExpr(ValExpr).get();
    llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
    ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
    HelperValStmt = buildPreInits(Context, Captures);
  }

  return new (Context) OMPThreadLimitClause(
      ValExpr, HelperValStmt, CaptureRegion, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPPriorityClause(Expr *Priority,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  Expr *ValExpr = Priority;
  Stmt *HelperValStmt = nullptr;
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;

  // OpenMP [2.9.1, task Constrcut]
  // The priority-value is a non-negative numerical scalar expression.
  if (!isNonNegativeIntegerValue(
          ValExpr, *this, OMPC_priority,
          /*StrictlyPositive=*/false, /*BuildCapture=*/true,
          DSAStack->getCurrentDirective(), &CaptureRegion, &HelperValStmt))
    return nullptr;

  return new (Context) OMPPriorityClause(ValExpr, HelperValStmt, CaptureRegion,
                                         StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPGrainsizeClause(Expr *Grainsize,
                                            SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  Expr *ValExpr = Grainsize;
  Stmt *HelperValStmt = nullptr;
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;

  // OpenMP [2.9.2, taskloop Constrcut]
  // The parameter of the grainsize clause must be a positive integer
  // expression.
  if (!isNonNegativeIntegerValue(
          ValExpr, *this, OMPC_grainsize,
          /*StrictlyPositive=*/true, /*BuildCapture=*/true,
          DSAStack->getCurrentDirective(), &CaptureRegion, &HelperValStmt))
    return nullptr;

  return new (Context) OMPGrainsizeClause(ValExpr, HelperValStmt, CaptureRegion,
                                          StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNumTasksClause(Expr *NumTasks,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  Expr *ValExpr = NumTasks;
  Stmt *HelperValStmt = nullptr;
  OpenMPDirectiveKind CaptureRegion = OMPD_unknown;

  // OpenMP [2.9.2, taskloop Constrcut]
  // The parameter of the num_tasks clause must be a positive integer
  // expression.
  if (!isNonNegativeIntegerValue(
          ValExpr, *this, OMPC_num_tasks,
          /*StrictlyPositive=*/true, /*BuildCapture=*/true,
          DSAStack->getCurrentDirective(), &CaptureRegion, &HelperValStmt))
    return nullptr;

  return new (Context) OMPNumTasksClause(ValExpr, HelperValStmt, CaptureRegion,
                                         StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPHintClause(Expr *Hint, SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  // OpenMP [2.13.2, critical construct, Description]
  // ... where hint-expression is an integer constant expression that evaluates
  // to a valid lock hint.
  ExprResult HintExpr = VerifyPositiveIntegerConstantInClause(Hint, OMPC_hint);
  if (HintExpr.isInvalid())
    return nullptr;
  return new (Context)
      OMPHintClause(HintExpr.get(), StartLoc, LParenLoc, EndLoc);
}

/// Tries to find omp_event_handle_t type.
static bool findOMPEventHandleT(Sema &S, SourceLocation Loc,
                                DSAStackTy *Stack) {
  QualType OMPEventHandleT = Stack->getOMPEventHandleT();
  if (!OMPEventHandleT.isNull())
    return true;
  IdentifierInfo *II = &S.PP.getIdentifierTable().get("omp_event_handle_t");
  ParsedType PT = S.getTypeName(*II, Loc, S.getCurScope());
  if (!PT.getAsOpaquePtr() || PT.get().isNull()) {
    S.Diag(Loc, diag::err_omp_implied_type_not_found) << "omp_event_handle_t";
    return false;
  }
  Stack->setOMPEventHandleT(PT.get());
  return true;
}

OMPClause *Sema::ActOnOpenMPDetachClause(Expr *Evt, SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  if (!Evt->isValueDependent() && !Evt->isTypeDependent() &&
      !Evt->isInstantiationDependent() &&
      !Evt->containsUnexpandedParameterPack()) {
    if (!findOMPEventHandleT(*this, Evt->getExprLoc(), DSAStack))
      return nullptr;
    // OpenMP 5.0, 2.10.1 task Construct.
    // event-handle is a variable of the omp_event_handle_t type.
    auto *Ref = dyn_cast<DeclRefExpr>(Evt->IgnoreParenImpCasts());
    if (!Ref) {
      Diag(Evt->getExprLoc(), diag::err_omp_var_expected)
          << "omp_event_handle_t" << 0 << Evt->getSourceRange();
      return nullptr;
    }
    auto *VD = dyn_cast_or_null<VarDecl>(Ref->getDecl());
    if (!VD) {
      Diag(Evt->getExprLoc(), diag::err_omp_var_expected)
          << "omp_event_handle_t" << 0 << Evt->getSourceRange();
      return nullptr;
    }
    if (!Context.hasSameUnqualifiedType(DSAStack->getOMPEventHandleT(),
                                        VD->getType()) ||
        VD->getType().isConstant(Context)) {
      Diag(Evt->getExprLoc(), diag::err_omp_var_expected)
          << "omp_event_handle_t" << 1 << VD->getType()
          << Evt->getSourceRange();
      return nullptr;
    }
    // OpenMP 5.0, 2.10.1 task Construct
    // [detach clause]... The event-handle will be considered as if it was
    // specified on a firstprivate clause.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD, /*FromParent=*/false);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_firstprivate &&
        DVar.RefExpr) {
      Diag(Evt->getExprLoc(), diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(DVar.CKind)
          << getOpenMPClauseName(OMPC_firstprivate);
      reportOriginalDsa(*this, DSAStack, VD, DVar);
      return nullptr;
    }
  }

  return new (Context) OMPDetachClause(Evt, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPDistScheduleClause(
    OpenMPDistScheduleClauseKind Kind, Expr *ChunkSize, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation KindLoc, SourceLocation CommaLoc,
    SourceLocation EndLoc) {
  if (Kind == OMPC_DIST_SCHEDULE_unknown) {
    std::string Values;
    Values += "'";
    Values += getOpenMPSimpleClauseTypeName(OMPC_dist_schedule, 0);
    Values += "'";
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_dist_schedule);
    return nullptr;
  }
  Expr *ValExpr = ChunkSize;
  Stmt *HelperValStmt = nullptr;
  if (ChunkSize) {
    if (!ChunkSize->isValueDependent() && !ChunkSize->isTypeDependent() &&
        !ChunkSize->isInstantiationDependent() &&
        !ChunkSize->containsUnexpandedParameterPack()) {
      SourceLocation ChunkSizeLoc = ChunkSize->getBeginLoc();
      ExprResult Val =
          PerformOpenMPImplicitIntegerConversion(ChunkSizeLoc, ChunkSize);
      if (Val.isInvalid())
        return nullptr;

      ValExpr = Val.get();

      // OpenMP [2.7.1, Restrictions]
      //  chunk_size must be a loop invariant integer expression with a positive
      //  value.
      if (Optional<llvm::APSInt> Result =
              ValExpr->getIntegerConstantExpr(Context)) {
        if (Result->isSigned() && !Result->isStrictlyPositive()) {
          Diag(ChunkSizeLoc, diag::err_omp_negative_expression_in_clause)
              << "dist_schedule" << ChunkSize->getSourceRange();
          return nullptr;
        }
      } else if (getOpenMPCaptureRegionForClause(
                     DSAStack->getCurrentDirective(), OMPC_dist_schedule,
                     LangOpts.OpenMP) != OMPD_unknown &&
                 !CurContext->isDependentContext()) {
        ValExpr = MakeFullExpr(ValExpr).get();
        llvm::MapVector<const Expr *, DeclRefExpr *> Captures;
        ValExpr = tryBuildCapture(*this, ValExpr, Captures).get();
        HelperValStmt = buildPreInits(Context, Captures);
      }
    }
  }

  return new (Context)
      OMPDistScheduleClause(StartLoc, LParenLoc, KindLoc, CommaLoc, EndLoc,
                            Kind, ValExpr, HelperValStmt);
}

OMPClause *Sema::ActOnOpenMPDefaultmapClause(
    OpenMPDefaultmapClauseModifier M, OpenMPDefaultmapClauseKind Kind,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation MLoc,
    SourceLocation KindLoc, SourceLocation EndLoc) {
  if (getLangOpts().OpenMP < 50) {
    if (M != OMPC_DEFAULTMAP_MODIFIER_tofrom ||
        Kind != OMPC_DEFAULTMAP_scalar) {
      std::string Value;
      SourceLocation Loc;
      Value += "'";
      if (M != OMPC_DEFAULTMAP_MODIFIER_tofrom) {
        Value += getOpenMPSimpleClauseTypeName(OMPC_defaultmap,
                                               OMPC_DEFAULTMAP_MODIFIER_tofrom);
        Loc = MLoc;
      } else {
        Value += getOpenMPSimpleClauseTypeName(OMPC_defaultmap,
                                               OMPC_DEFAULTMAP_scalar);
        Loc = KindLoc;
      }
      Value += "'";
      Diag(Loc, diag::err_omp_unexpected_clause_value)
          << Value << getOpenMPClauseName(OMPC_defaultmap);
      return nullptr;
    }
  } else {
    bool isDefaultmapModifier = (M != OMPC_DEFAULTMAP_MODIFIER_unknown);
    bool isDefaultmapKind = (Kind != OMPC_DEFAULTMAP_unknown) ||
                            (LangOpts.OpenMP >= 50 && KindLoc.isInvalid());
    if (!isDefaultmapKind || !isDefaultmapModifier) {
      StringRef KindValue = "'scalar', 'aggregate', 'pointer'";
      if (LangOpts.OpenMP == 50) {
        StringRef ModifierValue = "'alloc', 'from', 'to', 'tofrom', "
                                  "'firstprivate', 'none', 'default'";
        if (!isDefaultmapKind && isDefaultmapModifier) {
          Diag(KindLoc, diag::err_omp_unexpected_clause_value)
              << KindValue << getOpenMPClauseName(OMPC_defaultmap);
        } else if (isDefaultmapKind && !isDefaultmapModifier) {
          Diag(MLoc, diag::err_omp_unexpected_clause_value)
              << ModifierValue << getOpenMPClauseName(OMPC_defaultmap);
        } else {
          Diag(MLoc, diag::err_omp_unexpected_clause_value)
              << ModifierValue << getOpenMPClauseName(OMPC_defaultmap);
          Diag(KindLoc, diag::err_omp_unexpected_clause_value)
              << KindValue << getOpenMPClauseName(OMPC_defaultmap);
        }
      } else {
        StringRef ModifierValue =
            "'alloc', 'from', 'to', 'tofrom', "
            "'firstprivate', 'none', 'default', 'present'";
        if (!isDefaultmapKind && isDefaultmapModifier) {
          Diag(KindLoc, diag::err_omp_unexpected_clause_value)
              << KindValue << getOpenMPClauseName(OMPC_defaultmap);
        } else if (isDefaultmapKind && !isDefaultmapModifier) {
          Diag(MLoc, diag::err_omp_unexpected_clause_value)
              << ModifierValue << getOpenMPClauseName(OMPC_defaultmap);
        } else {
          Diag(MLoc, diag::err_omp_unexpected_clause_value)
              << ModifierValue << getOpenMPClauseName(OMPC_defaultmap);
          Diag(KindLoc, diag::err_omp_unexpected_clause_value)
              << KindValue << getOpenMPClauseName(OMPC_defaultmap);
        }
      }
      return nullptr;
    }

    // OpenMP [5.0, 2.12.5, Restrictions, p. 174]
    //  At most one defaultmap clause for each category can appear on the
    //  directive.
    if (DSAStack->checkDefaultmapCategory(Kind)) {
      Diag(StartLoc, diag::err_omp_one_defaultmap_each_category);
      return nullptr;
    }
  }
  if (Kind == OMPC_DEFAULTMAP_unknown) {
    // Variable category is not specified - mark all categories.
    DSAStack->setDefaultDMAAttr(M, OMPC_DEFAULTMAP_aggregate, StartLoc);
    DSAStack->setDefaultDMAAttr(M, OMPC_DEFAULTMAP_scalar, StartLoc);
    DSAStack->setDefaultDMAAttr(M, OMPC_DEFAULTMAP_pointer, StartLoc);
  } else {
    DSAStack->setDefaultDMAAttr(M, Kind, StartLoc);
  }

  return new (Context)
      OMPDefaultmapClause(StartLoc, LParenLoc, MLoc, KindLoc, EndLoc, Kind, M);
}

bool Sema::ActOnStartOpenMPDeclareTargetContext(
    DeclareTargetContextInfo &DTCI) {
  DeclContext *CurLexicalContext = getCurLexicalContext();
  if (!CurLexicalContext->isFileContext() &&
      !CurLexicalContext->isExternCContext() &&
      !CurLexicalContext->isExternCXXContext() &&
      !isa<CXXRecordDecl>(CurLexicalContext) &&
      !isa<ClassTemplateDecl>(CurLexicalContext) &&
      !isa<ClassTemplatePartialSpecializationDecl>(CurLexicalContext) &&
      !isa<ClassTemplateSpecializationDecl>(CurLexicalContext)) {
    Diag(DTCI.Loc, diag::err_omp_region_not_file_context);
    return false;
  }
  DeclareTargetNesting.push_back(DTCI);
  return true;
}

const Sema::DeclareTargetContextInfo
Sema::ActOnOpenMPEndDeclareTargetDirective() {
  assert(!DeclareTargetNesting.empty() &&
         "check isInOpenMPDeclareTargetContext() first!");
  return DeclareTargetNesting.pop_back_val();
}

void Sema::ActOnFinishedOpenMPDeclareTargetContext(
    DeclareTargetContextInfo &DTCI) {
  for (auto &It : DTCI.ExplicitlyMapped)
    ActOnOpenMPDeclareTargetName(It.first, It.second.Loc, It.second.MT, DTCI);
}

NamedDecl *Sema::lookupOpenMPDeclareTargetName(Scope *CurScope,
                                               CXXScopeSpec &ScopeSpec,
                                               const DeclarationNameInfo &Id) {
  LookupResult Lookup(*this, Id, LookupOrdinaryName);
  LookupParsedName(Lookup, CurScope, &ScopeSpec, true);

  if (Lookup.isAmbiguous())
    return nullptr;
  Lookup.suppressDiagnostics();

  if (!Lookup.isSingleResult()) {
    VarOrFuncDeclFilterCCC CCC(*this);
    if (TypoCorrection Corrected =
            CorrectTypo(Id, LookupOrdinaryName, CurScope, nullptr, CCC,
                        CTK_ErrorRecovery)) {
      diagnoseTypo(Corrected, PDiag(diag::err_undeclared_var_use_suggest)
                                  << Id.getName());
      checkDeclIsAllowedInOpenMPTarget(nullptr, Corrected.getCorrectionDecl());
      return nullptr;
    }

    Diag(Id.getLoc(), diag::err_undeclared_var_use) << Id.getName();
    return nullptr;
  }

  NamedDecl *ND = Lookup.getAsSingle<NamedDecl>();
  if (!isa<VarDecl>(ND) && !isa<FunctionDecl>(ND) &&
      !isa<FunctionTemplateDecl>(ND)) {
    Diag(Id.getLoc(), diag::err_omp_invalid_target_decl) << Id.getName();
    return nullptr;
  }
  return ND;
}

void Sema::ActOnOpenMPDeclareTargetName(NamedDecl *ND, SourceLocation Loc,
                                        OMPDeclareTargetDeclAttr::MapTypeTy MT,
                                        DeclareTargetContextInfo &DTCI) {
  assert((isa<VarDecl>(ND) || isa<FunctionDecl>(ND) ||
          isa<FunctionTemplateDecl>(ND)) &&
         "Expected variable, function or function template.");

  // Diagnose marking after use as it may lead to incorrect diagnosis and
  // codegen.
  if (LangOpts.OpenMP >= 50 &&
      (ND->isUsed(/*CheckUsedAttr=*/false) || ND->isReferenced()))
    Diag(Loc, diag::warn_omp_declare_target_after_first_use);

  // Explicit declare target lists have precedence.
  const unsigned Level = -1;

  auto *VD = cast<ValueDecl>(ND);
  llvm::Optional<OMPDeclareTargetDeclAttr *> ActiveAttr =
      OMPDeclareTargetDeclAttr::getActiveAttr(VD);
  if (ActiveAttr.hasValue() && ActiveAttr.getValue()->getDevType() != DTCI.DT &&
      ActiveAttr.getValue()->getLevel() == Level) {
    Diag(Loc, diag::err_omp_device_type_mismatch)
        << OMPDeclareTargetDeclAttr::ConvertDevTypeTyToStr(DTCI.DT)
        << OMPDeclareTargetDeclAttr::ConvertDevTypeTyToStr(
               ActiveAttr.getValue()->getDevType());
    return;
  }
  if (ActiveAttr.hasValue() && ActiveAttr.getValue()->getMapType() != MT &&
      ActiveAttr.getValue()->getLevel() == Level) {
    Diag(Loc, diag::err_omp_declare_target_to_and_link) << ND;
    return;
  }

  if (ActiveAttr.hasValue() && ActiveAttr.getValue()->getLevel() == Level)
    return;

  Expr *IndirectE = nullptr;
  bool IsIndirect = false;
  if (DTCI.Indirect.hasValue()) {
    IndirectE = DTCI.Indirect.getValue();
    if (!IndirectE)
      IsIndirect = true;
  }
  auto *A = OMPDeclareTargetDeclAttr::CreateImplicit(
      Context, MT, DTCI.DT, IndirectE, IsIndirect, Level,
      SourceRange(Loc, Loc));
  ND->addAttr(A);
  if (ASTMutationListener *ML = Context.getASTMutationListener())
    ML->DeclarationMarkedOpenMPDeclareTarget(ND, A);
  checkDeclIsAllowedInOpenMPTarget(nullptr, ND, Loc);
}

static void checkDeclInTargetContext(SourceLocation SL, SourceRange SR,
                                     Sema &SemaRef, Decl *D) {
  if (!D || !isa<VarDecl>(D))
    return;
  auto *VD = cast<VarDecl>(D);
  Optional<OMPDeclareTargetDeclAttr::MapTypeTy> MapTy =
      OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD);
  if (SemaRef.LangOpts.OpenMP >= 50 &&
      (SemaRef.getCurLambda(/*IgnoreNonLambdaCapturingScope=*/true) ||
       SemaRef.getCurBlock() || SemaRef.getCurCapturedRegion()) &&
      VD->hasGlobalStorage()) {
    if (!MapTy || *MapTy != OMPDeclareTargetDeclAttr::MT_To) {
      // OpenMP 5.0, 2.12.7 declare target Directive, Restrictions
      // If a lambda declaration and definition appears between a
      // declare target directive and the matching end declare target
      // directive, all variables that are captured by the lambda
      // expression must also appear in a to clause.
      SemaRef.Diag(VD->getLocation(),
                   diag::err_omp_lambda_capture_in_declare_target_not_to);
      SemaRef.Diag(SL, diag::note_var_explicitly_captured_here)
          << VD << 0 << SR;
      return;
    }
  }
  if (MapTy.hasValue())
    return;
  SemaRef.Diag(VD->getLocation(), diag::warn_omp_not_in_target_context);
  SemaRef.Diag(SL, diag::note_used_here) << SR;
}

static bool checkValueDeclInTarget(SourceLocation SL, SourceRange SR,
                                   Sema &SemaRef, DSAStackTy *Stack,
                                   ValueDecl *VD) {
  return OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD) ||
         checkTypeMappable(SL, SR, SemaRef, Stack, VD->getType(),
                           /*FullCheck=*/false);
}

void Sema::checkDeclIsAllowedInOpenMPTarget(Expr *E, Decl *D,
                                            SourceLocation IdLoc) {
  if (!D || D->isInvalidDecl())
    return;
  SourceRange SR = E ? E->getSourceRange() : D->getSourceRange();
  SourceLocation SL = E ? E->getBeginLoc() : D->getLocation();
  if (auto *VD = dyn_cast<VarDecl>(D)) {
    // Only global variables can be marked as declare target.
    if (!VD->isFileVarDecl() && !VD->isStaticLocal() &&
        !VD->isStaticDataMember())
      return;
    // 2.10.6: threadprivate variable cannot appear in a declare target
    // directive.
    if (DSAStack->isThreadPrivate(VD)) {
      Diag(SL, diag::err_omp_threadprivate_in_target);
      reportOriginalDsa(*this, DSAStack, VD, DSAStack->getTopDSA(VD, false));
      return;
    }
  }
  if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(D))
    D = FTD->getTemplatedDecl();
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
        OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(FD);
    if (IdLoc.isValid() && Res && *Res == OMPDeclareTargetDeclAttr::MT_Link) {
      Diag(IdLoc, diag::err_omp_function_in_link_clause);
      Diag(FD->getLocation(), diag::note_defined_here) << FD;
      return;
    }
  }
  if (auto *VD = dyn_cast<ValueDecl>(D)) {
    // Problem if any with var declared with incomplete type will be reported
    // as normal, so no need to check it here.
    if ((E || !VD->getType()->isIncompleteType()) &&
        !checkValueDeclInTarget(SL, SR, *this, DSAStack, VD))
      return;
    if (!E && isInOpenMPDeclareTargetContext()) {
      // Checking declaration inside declare target region.
      if (isa<VarDecl>(D) || isa<FunctionDecl>(D) ||
          isa<FunctionTemplateDecl>(D)) {
        llvm::Optional<OMPDeclareTargetDeclAttr *> ActiveAttr =
            OMPDeclareTargetDeclAttr::getActiveAttr(VD);
        unsigned Level = DeclareTargetNesting.size();
        if (ActiveAttr.hasValue() && ActiveAttr.getValue()->getLevel() >= Level)
          return;
        DeclareTargetContextInfo &DTCI = DeclareTargetNesting.back();
        Expr *IndirectE = nullptr;
        bool IsIndirect = false;
        if (DTCI.Indirect.hasValue()) {
          IndirectE = DTCI.Indirect.getValue();
          if (!IndirectE)
            IsIndirect = true;
        }
        auto *A = OMPDeclareTargetDeclAttr::CreateImplicit(
            Context, OMPDeclareTargetDeclAttr::MT_To, DTCI.DT, IndirectE,
            IsIndirect, Level, SourceRange(DTCI.Loc, DTCI.Loc));
        D->addAttr(A);
        if (ASTMutationListener *ML = Context.getASTMutationListener())
          ML->DeclarationMarkedOpenMPDeclareTarget(D, A);
      }
      return;
    }
  }
  if (!E)
    return;
  checkDeclInTargetContext(E->getExprLoc(), E->getSourceRange(), *this, D);
}

OMPClause *Sema::ActOnOpenMPToClause(
    ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
    ArrayRef<SourceLocation> MotionModifiersLoc,
    CXXScopeSpec &MapperIdScopeSpec, DeclarationNameInfo &MapperId,
    SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
    const OMPVarListLocTy &Locs, ArrayRef<Expr *> UnresolvedMappers) {
  OpenMPMotionModifierKind Modifiers[] = {OMPC_MOTION_MODIFIER_unknown,
                                          OMPC_MOTION_MODIFIER_unknown};
  SourceLocation ModifiersLoc[NumberOfOMPMotionModifiers];

  // Process motion-modifiers, flag errors for duplicate modifiers.
  unsigned Count = 0;
  for (unsigned I = 0, E = MotionModifiers.size(); I < E; ++I) {
    if (MotionModifiers[I] != OMPC_MOTION_MODIFIER_unknown &&
        llvm::is_contained(Modifiers, MotionModifiers[I])) {
      Diag(MotionModifiersLoc[I], diag::err_omp_duplicate_motion_modifier);
      continue;
    }
    assert(Count < NumberOfOMPMotionModifiers &&
           "Modifiers exceed the allowed number of motion modifiers");
    Modifiers[Count] = MotionModifiers[I];
    ModifiersLoc[Count] = MotionModifiersLoc[I];
    ++Count;
  }

  MappableVarListInfo MVLI(VarList);
  checkMappableExpressionList(*this, DSAStack, OMPC_to, MVLI, Locs.StartLoc,
                              MapperIdScopeSpec, MapperId, UnresolvedMappers);
  if (MVLI.ProcessedVarList.empty())
    return nullptr;

  return OMPToClause::Create(
      Context, Locs, MVLI.ProcessedVarList, MVLI.VarBaseDeclarations,
      MVLI.VarComponents, MVLI.UDMapperList, Modifiers, ModifiersLoc,
      MapperIdScopeSpec.getWithLocInContext(Context), MapperId);
}

OMPClause *Sema::ActOnOpenMPFromClause(
    ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
    ArrayRef<SourceLocation> MotionModifiersLoc,
    CXXScopeSpec &MapperIdScopeSpec, DeclarationNameInfo &MapperId,
    SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
    const OMPVarListLocTy &Locs, ArrayRef<Expr *> UnresolvedMappers) {
  OpenMPMotionModifierKind Modifiers[] = {OMPC_MOTION_MODIFIER_unknown,
                                          OMPC_MOTION_MODIFIER_unknown};
  SourceLocation ModifiersLoc[NumberOfOMPMotionModifiers];

  // Process motion-modifiers, flag errors for duplicate modifiers.
  unsigned Count = 0;
  for (unsigned I = 0, E = MotionModifiers.size(); I < E; ++I) {
    if (MotionModifiers[I] != OMPC_MOTION_MODIFIER_unknown &&
        llvm::is_contained(Modifiers, MotionModifiers[I])) {
      Diag(MotionModifiersLoc[I], diag::err_omp_duplicate_motion_modifier);
      continue;
    }
    assert(Count < NumberOfOMPMotionModifiers &&
           "Modifiers exceed the allowed number of motion modifiers");
    Modifiers[Count] = MotionModifiers[I];
    ModifiersLoc[Count] = MotionModifiersLoc[I];
    ++Count;
  }

  MappableVarListInfo MVLI(VarList);
  checkMappableExpressionList(*this, DSAStack, OMPC_from, MVLI, Locs.StartLoc,
                              MapperIdScopeSpec, MapperId, UnresolvedMappers);
  if (MVLI.ProcessedVarList.empty())
    return nullptr;

  return OMPFromClause::Create(
      Context, Locs, MVLI.ProcessedVarList, MVLI.VarBaseDeclarations,
      MVLI.VarComponents, MVLI.UDMapperList, Modifiers, ModifiersLoc,
      MapperIdScopeSpec.getWithLocInContext(Context), MapperId);
}

OMPClause *Sema::ActOnOpenMPUseDevicePtrClause(ArrayRef<Expr *> VarList,
                                               const OMPVarListLocTy &Locs) {
  MappableVarListInfo MVLI(VarList);
  SmallVector<Expr *, 8> PrivateCopies;
  SmallVector<Expr *, 8> Inits;

  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP use_device_ptr clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      MVLI.ProcessedVarList.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
      Inits.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    QualType Type = D->getType();
    Type = Type.getNonReferenceType().getUnqualifiedType();

    auto *VD = dyn_cast<VarDecl>(D);

    // Item should be a pointer or reference to pointer.
    if (!Type->isPointerType()) {
      Diag(ELoc, diag::err_omp_usedeviceptr_not_a_pointer)
          << 0 << RefExpr->getSourceRange();
      continue;
    }

    // Build the private variable and the expression that refers to it.
    auto VDPrivate =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr,
                     VD ? cast<DeclRefExpr>(SimpleRefExpr) : nullptr);
    if (VDPrivate->isInvalidDecl())
      continue;

    CurContext->addDecl(VDPrivate);
    DeclRefExpr *VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, RefExpr->getType().getUnqualifiedType(), ELoc);

    // Add temporary variable to initialize the private copy of the pointer.
    VarDecl *VDInit =
        buildVarDecl(*this, RefExpr->getExprLoc(), Type, ".devptr.temp");
    DeclRefExpr *VDInitRefExpr = buildDeclRefExpr(
        *this, VDInit, RefExpr->getType(), RefExpr->getExprLoc());
    AddInitializerToDecl(VDPrivate,
                         DefaultLvalueConversion(VDInitRefExpr).get(),
                         /*DirectInit=*/false);

    // If required, build a capture to implement the privatization initialized
    // with the current list item value.
    DeclRefExpr *Ref = nullptr;
    if (!VD)
      Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/true);
    MVLI.ProcessedVarList.push_back(VD ? RefExpr->IgnoreParens() : Ref);
    PrivateCopies.push_back(VDPrivateRefExpr);
    Inits.push_back(VDInitRefExpr);

    // We need to add a data sharing attribute for this variable to make sure it
    // is correctly captured. A variable that shows up in a use_device_ptr has
    // similar properties of a first private variable.
    DSAStack->addDSA(D, RefExpr->IgnoreParens(), OMPC_firstprivate, Ref);

    // Create a mappable component for the list item. List items in this clause
    // only need a component.
    MVLI.VarBaseDeclarations.push_back(D);
    MVLI.VarComponents.resize(MVLI.VarComponents.size() + 1);
    MVLI.VarComponents.back().emplace_back(SimpleRefExpr, D,
                                           /*IsNonContiguous=*/false);
  }

  if (MVLI.ProcessedVarList.empty())
    return nullptr;

  return OMPUseDevicePtrClause::Create(
      Context, Locs, MVLI.ProcessedVarList, PrivateCopies, Inits,
      MVLI.VarBaseDeclarations, MVLI.VarComponents);
}

OMPClause *Sema::ActOnOpenMPUseDeviceAddrClause(ArrayRef<Expr *> VarList,
                                                const OMPVarListLocTy &Locs) {
  MappableVarListInfo MVLI(VarList);

  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP use_device_addr clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange,
                              /*AllowArraySection=*/true);
    if (Res.second) {
      // It will be analyzed later.
      MVLI.ProcessedVarList.push_back(RefExpr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;
    auto *VD = dyn_cast<VarDecl>(D);

    // If required, build a capture to implement the privatization initialized
    // with the current list item value.
    DeclRefExpr *Ref = nullptr;
    if (!VD)
      Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/true);
    MVLI.ProcessedVarList.push_back(VD ? RefExpr->IgnoreParens() : Ref);

    // We need to add a data sharing attribute for this variable to make sure it
    // is correctly captured. A variable that shows up in a use_device_addr has
    // similar properties of a first private variable.
    DSAStack->addDSA(D, RefExpr->IgnoreParens(), OMPC_firstprivate, Ref);

    // Create a mappable component for the list item. List items in this clause
    // only need a component.
    MVLI.VarBaseDeclarations.push_back(D);
    MVLI.VarComponents.emplace_back();
    Expr *Component = SimpleRefExpr;
    if (VD && (isa<OMPArraySectionExpr>(RefExpr->IgnoreParenImpCasts()) ||
               isa<ArraySubscriptExpr>(RefExpr->IgnoreParenImpCasts())))
      Component = DefaultFunctionArrayLvalueConversion(SimpleRefExpr).get();
    MVLI.VarComponents.back().emplace_back(Component, D,
                                           /*IsNonContiguous=*/false);
  }

  if (MVLI.ProcessedVarList.empty())
    return nullptr;

  return OMPUseDeviceAddrClause::Create(Context, Locs, MVLI.ProcessedVarList,
                                        MVLI.VarBaseDeclarations,
                                        MVLI.VarComponents);
}

OMPClause *Sema::ActOnOpenMPIsDevicePtrClause(ArrayRef<Expr *> VarList,
                                              const OMPVarListLocTy &Locs) {
  MappableVarListInfo MVLI(VarList);
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP is_device_ptr clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      MVLI.ProcessedVarList.push_back(RefExpr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    QualType Type = D->getType();
    // item should be a pointer or array or reference to pointer or array
    if (!Type.getNonReferenceType()->isPointerType() &&
        !Type.getNonReferenceType()->isArrayType()) {
      Diag(ELoc, diag::err_omp_argument_type_isdeviceptr)
          << 0 << RefExpr->getSourceRange();
      continue;
    }

    // Check if the declaration in the clause does not show up in any data
    // sharing attribute.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(D, /*FromParent=*/false);
    if (isOpenMPPrivate(DVar.CKind)) {
      Diag(ELoc, diag::err_omp_variable_in_given_clause_and_dsa)
          << getOpenMPClauseName(DVar.CKind)
          << getOpenMPClauseName(OMPC_is_device_ptr)
          << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
      reportOriginalDsa(*this, DSAStack, D, DVar);
      continue;
    }

    const Expr *ConflictExpr;
    if (DSAStack->checkMappableExprComponentListsForDecl(
            D, /*CurrentRegionOnly=*/true,
            [&ConflictExpr](
                OMPClauseMappableExprCommon::MappableExprComponentListRef R,
                OpenMPClauseKind) -> bool {
              ConflictExpr = R.front().getAssociatedExpression();
              return true;
            })) {
      Diag(ELoc, diag::err_omp_map_shared_storage) << RefExpr->getSourceRange();
      Diag(ConflictExpr->getExprLoc(), diag::note_used_here)
          << ConflictExpr->getSourceRange();
      continue;
    }

    // Store the components in the stack so that they can be used to check
    // against other clauses later on.
    OMPClauseMappableExprCommon::MappableComponent MC(
        SimpleRefExpr, D, /*IsNonContiguous=*/false);
    DSAStack->addMappableExpressionComponents(
        D, MC, /*WhereFoundClauseKind=*/OMPC_is_device_ptr);

    // Record the expression we've just processed.
    MVLI.ProcessedVarList.push_back(SimpleRefExpr);

    // Create a mappable component for the list item. List items in this clause
    // only need a component. We use a null declaration to signal fields in
    // 'this'.
    assert((isa<DeclRefExpr>(SimpleRefExpr) ||
            isa<CXXThisExpr>(cast<MemberExpr>(SimpleRefExpr)->getBase())) &&
           "Unexpected device pointer expression!");
    MVLI.VarBaseDeclarations.push_back(
        isa<DeclRefExpr>(SimpleRefExpr) ? D : nullptr);
    MVLI.VarComponents.resize(MVLI.VarComponents.size() + 1);
    MVLI.VarComponents.back().push_back(MC);
  }

  if (MVLI.ProcessedVarList.empty())
    return nullptr;

  return OMPIsDevicePtrClause::Create(Context, Locs, MVLI.ProcessedVarList,
                                      MVLI.VarBaseDeclarations,
                                      MVLI.VarComponents);
}

OMPClause *Sema::ActOnOpenMPAllocateClause(
    Expr *Allocator, ArrayRef<Expr *> VarList, SourceLocation StartLoc,
    SourceLocation ColonLoc, SourceLocation LParenLoc, SourceLocation EndLoc) {
  if (Allocator) {
    // OpenMP [2.11.4 allocate Clause, Description]
    // allocator is an expression of omp_allocator_handle_t type.
    if (!findOMPAllocatorHandleT(*this, Allocator->getExprLoc(), DSAStack))
      return nullptr;

    ExprResult AllocatorRes = DefaultLvalueConversion(Allocator);
    if (AllocatorRes.isInvalid())
      return nullptr;
    AllocatorRes = PerformImplicitConversion(AllocatorRes.get(),
                                             DSAStack->getOMPAllocatorHandleT(),
                                             Sema::AA_Initializing,
                                             /*AllowExplicit=*/true);
    if (AllocatorRes.isInvalid())
      return nullptr;
    Allocator = AllocatorRes.get();
  } else {
    // OpenMP 5.0, 2.11.4 allocate Clause, Restrictions.
    // allocate clauses that appear on a target construct or on constructs in a
    // target region must specify an allocator expression unless a requires
    // directive with the dynamic_allocators clause is present in the same
    // compilation unit.
    if (LangOpts.OpenMPIsDevice &&
        !DSAStack->hasRequiresDeclWithClause<OMPDynamicAllocatorsClause>())
      targetDiag(StartLoc, diag::err_expected_allocator_expression);
  }
  // Analyze and build list of variables.
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP private clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    auto *VD = dyn_cast<VarDecl>(D);
    DeclRefExpr *Ref = nullptr;
    if (!VD && !CurContext->isDependentContext())
      Ref = buildCapture(*this, D, SimpleRefExpr, /*WithInit=*/false);
    Vars.push_back((VD || CurContext->isDependentContext())
                       ? RefExpr->IgnoreParens()
                       : Ref);
  }

  if (Vars.empty())
    return nullptr;

  if (Allocator)
    DSAStack->addInnerAllocatorExpr(Allocator);
  return OMPAllocateClause::Create(Context, StartLoc, LParenLoc, Allocator,
                                   ColonLoc, EndLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPNontemporalClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP nontemporal clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange);
    if (Res.second)
      // It will be analyzed later.
      Vars.push_back(RefExpr);
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    // OpenMP 5.0, 2.9.3.1 simd Construct, Restrictions.
    // A list-item cannot appear in more than one nontemporal clause.
    if (const Expr *PrevRef =
            DSAStack->addUniqueNontemporal(D, SimpleRefExpr)) {
      Diag(ELoc, diag::err_omp_used_in_clause_twice)
          << 0 << getOpenMPClauseName(OMPC_nontemporal) << ERange;
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(OMPC_nontemporal);
      continue;
    }

    Vars.push_back(RefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OMPNontemporalClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                      Vars);
}

OMPClause *Sema::ActOnOpenMPInclusiveClause(ArrayRef<Expr *> VarList,
                                            SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP nontemporal clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange,
                              /*AllowArraySection=*/true);
    if (Res.second)
      // It will be analyzed later.
      Vars.push_back(RefExpr);
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    const DSAStackTy::DSAVarData DVar =
        DSAStack->getTopDSA(D, /*FromParent=*/true);
    // OpenMP 5.0, 2.9.6, scan Directive, Restrictions.
    // A list item that appears in the inclusive or exclusive clause must appear
    // in a reduction clause with the inscan modifier on the enclosing
    // worksharing-loop, worksharing-loop SIMD, or simd construct.
    if (DVar.CKind != OMPC_reduction || DVar.Modifier != OMPC_REDUCTION_inscan)
      Diag(ELoc, diag::err_omp_inclusive_exclusive_not_reduction)
          << RefExpr->getSourceRange();

    if (DSAStack->getParentDirective() != OMPD_unknown)
      DSAStack->markDeclAsUsedInScanDirective(D);
    Vars.push_back(RefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OMPInclusiveClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPExclusiveClause(ArrayRef<Expr *> VarList,
                                            SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP nontemporal clause.");
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(*this, SimpleRefExpr, ELoc, ERange,
                              /*AllowArraySection=*/true);
    if (Res.second)
      // It will be analyzed later.
      Vars.push_back(RefExpr);
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    OpenMPDirectiveKind ParentDirective = DSAStack->getParentDirective();
    DSAStackTy::DSAVarData DVar;
    if (ParentDirective != OMPD_unknown)
      DVar = DSAStack->getTopDSA(D, /*FromParent=*/true);
    // OpenMP 5.0, 2.9.6, scan Directive, Restrictions.
    // A list item that appears in the inclusive or exclusive clause must appear
    // in a reduction clause with the inscan modifier on the enclosing
    // worksharing-loop, worksharing-loop SIMD, or simd construct.
    if (ParentDirective == OMPD_unknown || DVar.CKind != OMPC_reduction ||
        DVar.Modifier != OMPC_REDUCTION_inscan) {
      Diag(ELoc, diag::err_omp_inclusive_exclusive_not_reduction)
          << RefExpr->getSourceRange();
    } else {
      DSAStack->markDeclAsUsedInScanDirective(D);
    }
    Vars.push_back(RefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OMPExclusiveClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

/// Tries to find omp_alloctrait_t type.
static bool findOMPAlloctraitT(Sema &S, SourceLocation Loc, DSAStackTy *Stack) {
  QualType OMPAlloctraitT = Stack->getOMPAlloctraitT();
  if (!OMPAlloctraitT.isNull())
    return true;
  IdentifierInfo &II = S.PP.getIdentifierTable().get("omp_alloctrait_t");
  ParsedType PT = S.getTypeName(II, Loc, S.getCurScope());
  if (!PT.getAsOpaquePtr() || PT.get().isNull()) {
    S.Diag(Loc, diag::err_omp_implied_type_not_found) << "omp_alloctrait_t";
    return false;
  }
  Stack->setOMPAlloctraitT(PT.get());
  return true;
}

OMPClause *Sema::ActOnOpenMPUsesAllocatorClause(
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc,
    ArrayRef<UsesAllocatorsData> Data) {
  // OpenMP [2.12.5, target Construct]
  // allocator is an identifier of omp_allocator_handle_t type.
  if (!findOMPAllocatorHandleT(*this, StartLoc, DSAStack))
    return nullptr;
  // OpenMP [2.12.5, target Construct]
  // allocator-traits-array is an identifier of const omp_alloctrait_t * type.
  if (llvm::any_of(
          Data,
          [](const UsesAllocatorsData &D) { return D.AllocatorTraits; }) &&
      !findOMPAlloctraitT(*this, StartLoc, DSAStack))
    return nullptr;
  llvm::SmallPtrSet<CanonicalDeclPtr<Decl>, 4> PredefinedAllocators;
  for (int I = 0; I < OMPAllocateDeclAttr::OMPUserDefinedMemAlloc; ++I) {
    auto AllocatorKind = static_cast<OMPAllocateDeclAttr::AllocatorTypeTy>(I);
    StringRef Allocator =
        OMPAllocateDeclAttr::ConvertAllocatorTypeTyToStr(AllocatorKind);
    DeclarationName AllocatorName = &Context.Idents.get(Allocator);
    PredefinedAllocators.insert(LookupSingleName(
        TUScope, AllocatorName, StartLoc, Sema::LookupAnyName));
  }

  SmallVector<OMPUsesAllocatorsClause::Data, 4> NewData;
  for (const UsesAllocatorsData &D : Data) {
    Expr *AllocatorExpr = nullptr;
    // Check allocator expression.
    if (D.Allocator->isTypeDependent()) {
      AllocatorExpr = D.Allocator;
    } else {
      // Traits were specified - need to assign new allocator to the specified
      // allocator, so it must be an lvalue.
      AllocatorExpr = D.Allocator->IgnoreParenImpCasts();
      auto *DRE = dyn_cast<DeclRefExpr>(AllocatorExpr);
      bool IsPredefinedAllocator = false;
      if (DRE)
        IsPredefinedAllocator = PredefinedAllocators.count(DRE->getDecl());
      if (!DRE ||
          !(Context.hasSameUnqualifiedType(
                AllocatorExpr->getType(), DSAStack->getOMPAllocatorHandleT()) ||
            Context.typesAreCompatible(AllocatorExpr->getType(),
                                       DSAStack->getOMPAllocatorHandleT(),
                                       /*CompareUnqualified=*/true)) ||
          (!IsPredefinedAllocator &&
           (AllocatorExpr->getType().isConstant(Context) ||
            !AllocatorExpr->isLValue()))) {
        Diag(D.Allocator->getExprLoc(), diag::err_omp_var_expected)
            << "omp_allocator_handle_t" << (DRE ? 1 : 0)
            << AllocatorExpr->getType() << D.Allocator->getSourceRange();
        continue;
      }
      // OpenMP [2.12.5, target Construct]
      // Predefined allocators appearing in a uses_allocators clause cannot have
      // traits specified.
      if (IsPredefinedAllocator && D.AllocatorTraits) {
        Diag(D.AllocatorTraits->getExprLoc(),
             diag::err_omp_predefined_allocator_with_traits)
            << D.AllocatorTraits->getSourceRange();
        Diag(D.Allocator->getExprLoc(), diag::note_omp_predefined_allocator)
            << cast<NamedDecl>(DRE->getDecl())->getName()
            << D.Allocator->getSourceRange();
        continue;
      }
      // OpenMP [2.12.5, target Construct]
      // Non-predefined allocators appearing in a uses_allocators clause must
      // have traits specified.
      if (!IsPredefinedAllocator && !D.AllocatorTraits) {
        Diag(D.Allocator->getExprLoc(),
             diag::err_omp_nonpredefined_allocator_without_traits);
        continue;
      }
      // No allocator traits - just convert it to rvalue.
      if (!D.AllocatorTraits)
        AllocatorExpr = DefaultLvalueConversion(AllocatorExpr).get();
      DSAStack->addUsesAllocatorsDecl(
          DRE->getDecl(),
          IsPredefinedAllocator
              ? DSAStackTy::UsesAllocatorsDeclKind::PredefinedAllocator
              : DSAStackTy::UsesAllocatorsDeclKind::UserDefinedAllocator);
    }
    Expr *AllocatorTraitsExpr = nullptr;
    if (D.AllocatorTraits) {
      if (D.AllocatorTraits->isTypeDependent()) {
        AllocatorTraitsExpr = D.AllocatorTraits;
      } else {
        // OpenMP [2.12.5, target Construct]
        // Arrays that contain allocator traits that appear in a uses_allocators
        // clause must be constant arrays, have constant values and be defined
        // in the same scope as the construct in which the clause appears.
        AllocatorTraitsExpr = D.AllocatorTraits->IgnoreParenImpCasts();
        // Check that traits expr is a constant array.
        QualType TraitTy;
        if (const ArrayType *Ty =
                AllocatorTraitsExpr->getType()->getAsArrayTypeUnsafe())
          if (const auto *ConstArrayTy = dyn_cast<ConstantArrayType>(Ty))
            TraitTy = ConstArrayTy->getElementType();
        if (TraitTy.isNull() ||
            !(Context.hasSameUnqualifiedType(TraitTy,
                                             DSAStack->getOMPAlloctraitT()) ||
              Context.typesAreCompatible(TraitTy, DSAStack->getOMPAlloctraitT(),
                                         /*CompareUnqualified=*/true))) {
          Diag(D.AllocatorTraits->getExprLoc(),
               diag::err_omp_expected_array_alloctraits)
              << AllocatorTraitsExpr->getType();
          continue;
        }
        // Do not map by default allocator traits if it is a standalone
        // variable.
        if (auto *DRE = dyn_cast<DeclRefExpr>(AllocatorTraitsExpr))
          DSAStack->addUsesAllocatorsDecl(
              DRE->getDecl(),
              DSAStackTy::UsesAllocatorsDeclKind::AllocatorTrait);
      }
    }
    OMPUsesAllocatorsClause::Data &NewD = NewData.emplace_back();
    NewD.Allocator = AllocatorExpr;
    NewD.AllocatorTraits = AllocatorTraitsExpr;
    NewD.LParenLoc = D.LParenLoc;
    NewD.RParenLoc = D.RParenLoc;
  }
  return OMPUsesAllocatorsClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                         NewData);
}

OMPClause *Sema::ActOnOpenMPAffinityClause(
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation EndLoc, Expr *Modifier, ArrayRef<Expr *> Locators) {
  SmallVector<Expr *, 8> Vars;
  for (Expr *RefExpr : Locators) {
    assert(RefExpr && "NULL expr in OpenMP shared clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr) || RefExpr->isTypeDependent()) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    Expr *SimpleExpr = RefExpr->IgnoreParenImpCasts();

    if (!SimpleExpr->isLValue()) {
      Diag(ELoc, diag::err_omp_expected_addressable_lvalue_or_array_item)
          << 1 << 0 << RefExpr->getSourceRange();
      continue;
    }

    ExprResult Res;
    {
      Sema::TentativeAnalysisScope Trap(*this);
      Res = CreateBuiltinUnaryOp(ELoc, UO_AddrOf, SimpleExpr);
    }
    if (!Res.isUsable() && !isa<OMPArraySectionExpr>(SimpleExpr) &&
        !isa<OMPArrayShapingExpr>(SimpleExpr)) {
      Diag(ELoc, diag::err_omp_expected_addressable_lvalue_or_array_item)
          << 1 << 0 << RefExpr->getSourceRange();
      continue;
    }
    Vars.push_back(SimpleExpr);
  }

  return OMPAffinityClause::Create(Context, StartLoc, LParenLoc, ColonLoc,
                                   EndLoc, Modifier, Vars);
}

OMPClause *Sema::ActOnOpenMPBindClause(OpenMPBindClauseKind Kind,
                                       SourceLocation KindLoc,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  if (Kind == OMPC_BIND_unknown) {
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_bind, /*First=*/0,
                                   /*Last=*/unsigned(OMPC_BIND_unknown))
        << getOpenMPClauseName(OMPC_bind);
    return nullptr;
  }

  return OMPBindClause::Create(Context, Kind, KindLoc, StartLoc, LParenLoc,
                               EndLoc);
}
