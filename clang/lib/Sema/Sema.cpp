//===--- Sema.cpp - AST Builder and Semantic Analysis Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which performs semantic analysis and
// builds an AST out of a parse stream.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "TargetAttributesSema.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/APFloat.h"
#include "clang/Sema/CXXFieldCollector.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/ObjCMethodList.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
using namespace clang;
using namespace sema;

FunctionScopeInfo::~FunctionScopeInfo() { }

void FunctionScopeInfo::Clear() {
  HasBranchProtectedScope = false;
  HasBranchIntoScope = false;
  HasIndirectGoto = false;
  
  SwitchStack.clear();
  Returns.clear();
  ErrorTrap.reset();
  PossiblyUnreachableDiags.clear();
}

BlockScopeInfo::~BlockScopeInfo() { }

void Sema::ActOnTranslationUnitScope(Scope *S) {
  TUScope = S;
  PushDeclContext(S, Context.getTranslationUnitDecl());

  VAListTagName = PP.getIdentifierInfo("__va_list_tag");

  if (PP.getLangOptions().ObjC1) {
    // Synthesize "@class Protocol;
    if (Context.getObjCProtoType().isNull()) {
      ObjCInterfaceDecl *ProtocolDecl =
        ObjCInterfaceDecl::Create(Context, CurContext, SourceLocation(),
                                  &Context.Idents.get("Protocol"),
                                  SourceLocation(), true);
      Context.setObjCProtoType(Context.getObjCInterfaceType(ProtocolDecl));
      PushOnScopeChains(ProtocolDecl, TUScope, false);
    }  
  }
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
           bool CompleteTranslationUnit,
           CodeCompleteConsumer *CodeCompleter)
  : TheTargetAttributesSema(0), FPFeatures(pp.getLangOptions()),
    LangOpts(pp.getLangOptions()), PP(pp), Context(ctxt), Consumer(consumer),
    Diags(PP.getDiagnostics()), SourceMgr(PP.getSourceManager()),
    CollectStats(false), ExternalSource(0), CodeCompleter(CodeCompleter),
    CurContext(0), PackContext(0), MSStructPragmaOn(false), VisContext(0),
    ExprNeedsCleanups(0), LateTemplateParser(0), OpaqueParser(0),
    IdResolver(pp.getLangOptions()), CXXTypeInfoDecl(0), MSVCGuidDecl(0),
    GlobalNewDeleteDeclared(false), 
    ObjCShouldCallSuperDealloc(false),
    CompleteTranslationUnit(CompleteTranslationUnit),
    NumSFINAEErrors(0), SuppressAccessChecking(false), 
    AccessCheckingSFINAE(false), InNonInstantiationSFINAEContext(false),
    NonInstantiationEntries(0), ArgumentPackSubstitutionIndex(-1),
    CurrentInstantiationScope(0), TyposCorrected(0),
    AnalysisWarnings(*this)
{
  TUScope = 0;
  LoadedExternalKnownNamespaces = false;
  
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());

  // Tell diagnostics how to render things from the AST library.
  PP.getDiagnostics().SetArgToStringFn(&FormatASTNodeDiagnosticArgument, 
                                       &Context);

  ExprEvalContexts.push_back(
        ExpressionEvaluationContextRecord(PotentiallyEvaluated, 0, false));

  FunctionScopes.push_back(new FunctionScopeInfo(Diags));
}

void Sema::Initialize() {
  // Tell the AST consumer about this Sema object.
  Consumer.Initialize(Context);
  
  // FIXME: Isn't this redundant with the initialization above?
  if (SemaConsumer *SC = dyn_cast<SemaConsumer>(&Consumer))
    SC->InitializeSema(*this);
  
  // Tell the external Sema source about this Sema object.
  if (ExternalSemaSource *ExternalSema
      = dyn_cast_or_null<ExternalSemaSource>(Context.getExternalSource()))
    ExternalSema->InitializeSema(*this);

  // Initialize predefined 128-bit integer types, if needed.
  if (PP.getTargetInfo().getPointerWidth(0) >= 64) {
    // If either of the 128-bit integer types are unavailable to name lookup,
    // define them now.
    DeclarationName Int128 = &Context.Idents.get("__int128_t");
    if (IdentifierResolver::begin(Int128) == IdentifierResolver::end())
      PushOnScopeChains(Context.getInt128Decl(), TUScope);

    DeclarationName UInt128 = &Context.Idents.get("__uint128_t");
    if (IdentifierResolver::begin(UInt128) == IdentifierResolver::end())
      PushOnScopeChains(Context.getUInt128Decl(), TUScope);
  }
  

  // Initialize predefined Objective-C types:
  if (PP.getLangOptions().ObjC1) {
    // If 'SEL' does not yet refer to any declarations, make it refer to the
    // predefined 'SEL'.
    DeclarationName SEL = &Context.Idents.get("SEL");
    if (IdentifierResolver::begin(SEL) == IdentifierResolver::end())
      PushOnScopeChains(Context.getObjCSelDecl(), TUScope);

    // If 'id' does not yet refer to any declarations, make it refer to the
    // predefined 'id'.
    DeclarationName Id = &Context.Idents.get("id");
    if (IdentifierResolver::begin(Id) == IdentifierResolver::end())
      PushOnScopeChains(Context.getObjCIdDecl(), TUScope);
    
    // Create the built-in typedef for 'Class'.
    DeclarationName Class = &Context.Idents.get("Class");
    if (IdentifierResolver::begin(Class) == IdentifierResolver::end())
      PushOnScopeChains(Context.getObjCClassDecl(), TUScope);
  }
}

Sema::~Sema() {
  if (PackContext) FreePackedContext();
  if (VisContext) FreeVisContext();
  delete TheTargetAttributesSema;
  MSStructPragmaOn = false;
  // Kill all the active scopes.
  for (unsigned I = 1, E = FunctionScopes.size(); I != E; ++I)
    delete FunctionScopes[I];
  if (FunctionScopes.size() == 1)
    delete FunctionScopes[0];
  
  // Tell the SemaConsumer to forget about us; we're going out of scope.
  if (SemaConsumer *SC = dyn_cast<SemaConsumer>(&Consumer))
    SC->ForgetSema();

  // Detach from the external Sema source.
  if (ExternalSemaSource *ExternalSema
        = dyn_cast_or_null<ExternalSemaSource>(Context.getExternalSource()))
    ExternalSema->ForgetSema();
}


/// makeUnavailableInSystemHeader - There is an error in the current
/// context.  If we're still in a system header, and we can plausibly
/// make the relevant declaration unavailable instead of erroring, do
/// so and return true.
bool Sema::makeUnavailableInSystemHeader(SourceLocation loc,
                                         StringRef msg) {
  // If we're not in a function, it's an error.
  FunctionDecl *fn = dyn_cast<FunctionDecl>(CurContext);
  if (!fn) return false;

  // If we're in template instantiation, it's an error.
  if (!ActiveTemplateInstantiations.empty())
    return false;
  
  // If that function's not in a system header, it's an error.
  if (!Context.getSourceManager().isInSystemHeader(loc))
    return false;

  // If the function is already unavailable, it's not an error.
  if (fn->hasAttr<UnavailableAttr>()) return true;

  fn->addAttr(new (Context) UnavailableAttr(loc, Context, msg));
  return true;
}

ASTMutationListener *Sema::getASTMutationListener() const {
  return getASTConsumer().GetASTMutationListener();
}

/// \brief Print out statistics about the semantic analysis.
void Sema::PrintStats() const {
  llvm::errs() << "\n*** Semantic Analysis Stats:\n";
  llvm::errs() << NumSFINAEErrors << " SFINAE diagnostics trapped.\n";

  BumpAlloc.PrintStats();
  AnalysisWarnings.PrintStats();
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast.
/// If there is already an implicit cast, merge into the existing one.
/// The result is of the given category.
ExprResult Sema::ImpCastExprToType(Expr *E, QualType Ty,
                                   CastKind Kind, ExprValueKind VK,
                                   const CXXCastPath *BasePath,
                                   CheckedConversionKind CCK) {
  QualType ExprTy = Context.getCanonicalType(E->getType());
  QualType TypeTy = Context.getCanonicalType(Ty);

  if (ExprTy == TypeTy)
    return Owned(E);

  if (getLangOptions().ObjCAutoRefCount)
    CheckObjCARCConversion(SourceRange(), Ty, E, CCK);

  // If this is a derived-to-base cast to a through a virtual base, we
  // need a vtable.
  if (Kind == CK_DerivedToBase && 
      BasePathInvolvesVirtualBase(*BasePath)) {
    QualType T = E->getType();
    if (const PointerType *Pointer = T->getAs<PointerType>())
      T = Pointer->getPointeeType();
    if (const RecordType *RecordTy = T->getAs<RecordType>())
      MarkVTableUsed(E->getLocStart(), 
                     cast<CXXRecordDecl>(RecordTy->getDecl()));
  }

  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(E)) {
    if (ImpCast->getCastKind() == Kind && (!BasePath || BasePath->empty())) {
      ImpCast->setType(Ty);
      ImpCast->setValueKind(VK);
      return Owned(E);
    }
  }

  return Owned(ImplicitCastExpr::Create(Context, Ty, Kind, E, BasePath, VK));
}

/// ScalarTypeToBooleanCastKind - Returns the cast kind corresponding
/// to the conversion from scalar type ScalarTy to the Boolean type.
CastKind Sema::ScalarTypeToBooleanCastKind(QualType ScalarTy) {
  switch (ScalarTy->getScalarTypeKind()) {
  case Type::STK_Bool: return CK_NoOp;
  case Type::STK_Pointer: return CK_PointerToBoolean;
  case Type::STK_MemberPointer: return CK_MemberPointerToBoolean;
  case Type::STK_Integral: return CK_IntegralToBoolean;
  case Type::STK_Floating: return CK_FloatingToBoolean;
  case Type::STK_IntegralComplex: return CK_IntegralComplexToBoolean;
  case Type::STK_FloatingComplex: return CK_FloatingComplexToBoolean;
  }
  return CK_Invalid;
}

ExprValueKind Sema::CastCategory(Expr *E) {
  Expr::Classification Classification = E->Classify(Context);
  return Classification.isRValue() ? VK_RValue :
      (Classification.isLValue() ? VK_LValue : VK_XValue);
}

/// \brief Used to prune the decls of Sema's UnusedFileScopedDecls vector.
static bool ShouldRemoveFromUnused(Sema *SemaRef, const DeclaratorDecl *D) {
  if (D->isUsed())
    return true;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // UnusedFileScopedDecls stores the first declaration.
    // The declaration may have become definition so check again.
    const FunctionDecl *DeclToCheck;
    if (FD->hasBody(DeclToCheck))
      return !SemaRef->ShouldWarnIfUnusedFileScopedDecl(DeclToCheck);

    // Later redecls may add new information resulting in not having to warn,
    // so check again.
    DeclToCheck = FD->getMostRecentDeclaration();
    if (DeclToCheck != FD)
      return !SemaRef->ShouldWarnIfUnusedFileScopedDecl(DeclToCheck);
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    // UnusedFileScopedDecls stores the first declaration.
    // The declaration may have become definition so check again.
    const VarDecl *DeclToCheck = VD->getDefinition(); 
    if (DeclToCheck)
      return !SemaRef->ShouldWarnIfUnusedFileScopedDecl(DeclToCheck);

    // Later redecls may add new information resulting in not having to warn,
    // so check again.
    DeclToCheck = VD->getMostRecentDeclaration();
    if (DeclToCheck != VD)
      return !SemaRef->ShouldWarnIfUnusedFileScopedDecl(DeclToCheck);
  }

  return false;
}

namespace {
  struct UndefinedInternal {
    NamedDecl *decl;
    FullSourceLoc useLoc;

    UndefinedInternal(NamedDecl *decl, FullSourceLoc useLoc)
      : decl(decl), useLoc(useLoc) {}
  };

  bool operator<(const UndefinedInternal &l, const UndefinedInternal &r) {
    return l.useLoc.isBeforeInTranslationUnitThan(r.useLoc);
  }
}

/// checkUndefinedInternals - Check for undefined objects with internal linkage.
static void checkUndefinedInternals(Sema &S) {
  if (S.UndefinedInternals.empty()) return;

  // Collect all the still-undefined entities with internal linkage.
  SmallVector<UndefinedInternal, 16> undefined;
  for (llvm::DenseMap<NamedDecl*,SourceLocation>::iterator
         i = S.UndefinedInternals.begin(), e = S.UndefinedInternals.end();
       i != e; ++i) {
    NamedDecl *decl = i->first;

    // Ignore attributes that have become invalid.
    if (decl->isInvalidDecl()) continue;

    // __attribute__((weakref)) is basically a definition.
    if (decl->hasAttr<WeakRefAttr>()) continue;

    if (FunctionDecl *fn = dyn_cast<FunctionDecl>(decl)) {
      if (fn->isPure() || fn->hasBody())
        continue;
    } else {
      if (cast<VarDecl>(decl)->hasDefinition() != VarDecl::DeclarationOnly)
        continue;
    }

    // We build a FullSourceLoc so that we can sort with array_pod_sort.
    FullSourceLoc loc(i->second, S.Context.getSourceManager());
    undefined.push_back(UndefinedInternal(decl, loc));
  }

  if (undefined.empty()) return;

  // Sort (in order of use site) so that we're not (as) dependent on
  // the iteration order through an llvm::DenseMap.
  llvm::array_pod_sort(undefined.begin(), undefined.end());

  for (SmallVectorImpl<UndefinedInternal>::iterator
         i = undefined.begin(), e = undefined.end(); i != e; ++i) {
    NamedDecl *decl = i->decl;
    S.Diag(decl->getLocation(), diag::warn_undefined_internal)
      << isa<VarDecl>(decl) << decl;
    S.Diag(i->useLoc, diag::note_used_here);
  }
}

void Sema::LoadExternalWeakUndeclaredIdentifiers() {
  if (!ExternalSource)
    return;
  
  SmallVector<std::pair<IdentifierInfo *, WeakInfo>, 4> WeakIDs;
  ExternalSource->ReadWeakUndeclaredIdentifiers(WeakIDs);
  for (unsigned I = 0, N = WeakIDs.size(); I != N; ++I) {
    llvm::DenseMap<IdentifierInfo*,WeakInfo>::iterator Pos
      = WeakUndeclaredIdentifiers.find(WeakIDs[I].first);
    if (Pos != WeakUndeclaredIdentifiers.end())
      continue;
    
    WeakUndeclaredIdentifiers.insert(WeakIDs[I]);
  }
}

/// ActOnEndOfTranslationUnit - This is called at the very end of the
/// translation unit when EOF is reached and all but the top-level scope is
/// popped.
void Sema::ActOnEndOfTranslationUnit() {
  // At PCH writing, implicit instantiations and VTable handling info are
  // stored and performed when the PCH is included.
  if (CompleteTranslationUnit) {
    // If any dynamic classes have their key function defined within
    // this translation unit, then those vtables are considered "used" and must
    // be emitted.
    for (DynamicClassesType::iterator I = DynamicClasses.begin(ExternalSource),
                                      E = DynamicClasses.end();
         I != E; ++I) {
      assert(!(*I)->isDependentType() &&
             "Should not see dependent types here!");
      if (const CXXMethodDecl *KeyFunction = Context.getKeyFunction(*I)) {
        const FunctionDecl *Definition = 0;
        if (KeyFunction->hasBody(Definition))
          MarkVTableUsed(Definition->getLocation(), *I, true);
      }
    }

    // If DefinedUsedVTables ends up marking any virtual member functions it
    // might lead to more pending template instantiations, which we then need
    // to instantiate.
    DefineUsedVTables();

    // C++: Perform implicit template instantiations.
    //
    // FIXME: When we perform these implicit instantiations, we do not
    // carefully keep track of the point of instantiation (C++ [temp.point]).
    // This means that name lookup that occurs within the template
    // instantiation will always happen at the end of the translation unit,
    // so it will find some names that should not be found. Although this is
    // common behavior for C++ compilers, it is technically wrong. In the
    // future, we either need to be able to filter the results of name lookup
    // or we need to perform template instantiations earlier.
    PerformPendingInstantiations();
  }
  
  // Remove file scoped decls that turned out to be used.
  UnusedFileScopedDecls.erase(std::remove_if(UnusedFileScopedDecls.begin(0, 
                                                                         true),
                                             UnusedFileScopedDecls.end(),
                              std::bind1st(std::ptr_fun(ShouldRemoveFromUnused),
                                           this)),
                              UnusedFileScopedDecls.end());

  if (!CompleteTranslationUnit) {
    TUScope = 0;
    return;
  }

  // Check for #pragma weak identifiers that were never declared
  // FIXME: This will cause diagnostics to be emitted in a non-determinstic
  // order!  Iterating over a densemap like this is bad.
  LoadExternalWeakUndeclaredIdentifiers();
  for (llvm::DenseMap<IdentifierInfo*,WeakInfo>::iterator
       I = WeakUndeclaredIdentifiers.begin(),
       E = WeakUndeclaredIdentifiers.end(); I != E; ++I) {
    if (I->second.getUsed()) continue;

    Diag(I->second.getLocation(), diag::warn_weak_identifier_undeclared)
      << I->first;
  }

  // C99 6.9.2p2:
  //   A declaration of an identifier for an object that has file
  //   scope without an initializer, and without a storage-class
  //   specifier or with the storage-class specifier static,
  //   constitutes a tentative definition. If a translation unit
  //   contains one or more tentative definitions for an identifier,
  //   and the translation unit contains no external definition for
  //   that identifier, then the behavior is exactly as if the
  //   translation unit contains a file scope declaration of that
  //   identifier, with the composite type as of the end of the
  //   translation unit, with an initializer equal to 0.
  llvm::SmallSet<VarDecl *, 32> Seen;
  for (TentativeDefinitionsType::iterator 
            T = TentativeDefinitions.begin(ExternalSource),
         TEnd = TentativeDefinitions.end();
       T != TEnd; ++T) 
  {
    VarDecl *VD = (*T)->getActingDefinition();

    // If the tentative definition was completed, getActingDefinition() returns
    // null. If we've already seen this variable before, insert()'s second
    // return value is false.
    if (VD == 0 || VD->isInvalidDecl() || !Seen.insert(VD))
      continue;

    if (const IncompleteArrayType *ArrayT
        = Context.getAsIncompleteArrayType(VD->getType())) {
      if (RequireCompleteType(VD->getLocation(),
                              ArrayT->getElementType(),
                              diag::err_tentative_def_incomplete_type_arr)) {
        VD->setInvalidDecl();
        continue;
      }

      // Set the length of the array to 1 (C99 6.9.2p5).
      Diag(VD->getLocation(), diag::warn_tentative_incomplete_array);
      llvm::APInt One(Context.getTypeSize(Context.getSizeType()), true);
      QualType T = Context.getConstantArrayType(ArrayT->getElementType(),
                                                One, ArrayType::Normal, 0);
      VD->setType(T);
    } else if (RequireCompleteType(VD->getLocation(), VD->getType(),
                                   diag::err_tentative_def_incomplete_type))
      VD->setInvalidDecl();

    // Notify the consumer that we've completed a tentative definition.
    if (!VD->isInvalidDecl())
      Consumer.CompleteTentativeDefinition(VD);

  }

  if (LangOpts.CPlusPlus0x &&
      Diags.getDiagnosticLevel(diag::warn_delegating_ctor_cycle,
                               SourceLocation())
        != Diagnostic::Ignored)
    CheckDelegatingCtorCycles();

  // If there were errors, disable 'unused' warnings since they will mostly be
  // noise.
  if (!Diags.hasErrorOccurred()) {
    // Output warning for unused file scoped decls.
    for (UnusedFileScopedDeclsType::iterator
           I = UnusedFileScopedDecls.begin(ExternalSource),
           E = UnusedFileScopedDecls.end(); I != E; ++I) {
      if (ShouldRemoveFromUnused(this, *I))
        continue;
      
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(*I)) {
        const FunctionDecl *DiagD;
        if (!FD->hasBody(DiagD))
          DiagD = FD;
        if (DiagD->isDeleted())
          continue; // Deleted functions are supposed to be unused.
        if (DiagD->isReferenced()) {
          if (isa<CXXMethodDecl>(DiagD))
            Diag(DiagD->getLocation(), diag::warn_unneeded_member_function)
                  << DiagD->getDeclName();
          else
            Diag(DiagD->getLocation(), diag::warn_unneeded_internal_decl)
                  << /*function*/0 << DiagD->getDeclName();
        } else {
          Diag(DiagD->getLocation(),
               isa<CXXMethodDecl>(DiagD) ? diag::warn_unused_member_function
                                         : diag::warn_unused_function)
                << DiagD->getDeclName();
        }
      } else {
        const VarDecl *DiagD = cast<VarDecl>(*I)->getDefinition();
        if (!DiagD)
          DiagD = cast<VarDecl>(*I);
        if (DiagD->isReferenced()) {
          Diag(DiagD->getLocation(), diag::warn_unneeded_internal_decl)
                << /*variable*/1 << DiagD->getDeclName();
        } else {
          Diag(DiagD->getLocation(), diag::warn_unused_variable)
                << DiagD->getDeclName();
        }
      }
    }

    checkUndefinedInternals(*this);
  }

  // Check we've noticed that we're no longer parsing the initializer for every
  // variable. If we miss cases, then at best we have a performance issue and
  // at worst a rejects-valid bug.
  assert(ParsingInitForAutoVars.empty() &&
         "Didn't unmark var as having its initializer parsed");

  TUScope = 0;
}


//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

DeclContext *Sema::getFunctionLevelDeclContext() {
  DeclContext *DC = CurContext;

  while (isa<BlockDecl>(DC) || isa<EnumDecl>(DC))
    DC = DC->getParent();

  return DC;
}

/// getCurFunctionDecl - If inside of a function body, this returns a pointer
/// to the function decl for the function being parsed.  If we're currently
/// in a 'block', this returns the containing context.
FunctionDecl *Sema::getCurFunctionDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  return dyn_cast<FunctionDecl>(DC);
}

ObjCMethodDecl *Sema::getCurMethodDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  return dyn_cast<ObjCMethodDecl>(DC);
}

NamedDecl *Sema::getCurFunctionOrMethodDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  if (isa<ObjCMethodDecl>(DC) || isa<FunctionDecl>(DC))
    return cast<NamedDecl>(DC);
  return 0;
}

Sema::SemaDiagnosticBuilder::~SemaDiagnosticBuilder() {
  if (!isActive())
    return;
  
  if (llvm::Optional<TemplateDeductionInfo*> Info = SemaRef.isSFINAEContext()) {
    switch (DiagnosticIDs::getDiagnosticSFINAEResponse(getDiagID())) {
    case DiagnosticIDs::SFINAE_Report:
      // Fall through; we'll report the diagnostic below.
      break;
      
    case DiagnosticIDs::SFINAE_AccessControl:
      // Per C++ Core Issue 1170, access control is part of SFINAE.
      // Additionally, the AccessCheckingSFINAE flag can be used to temporary
      // make access control a part of SFINAE for the purposes of checking
      // type traits.
      if (!SemaRef.AccessCheckingSFINAE &&
          !SemaRef.getLangOptions().CPlusPlus0x)
        break;
        
    case DiagnosticIDs::SFINAE_SubstitutionFailure:
      // Count this failure so that we know that template argument deduction
      // has failed.
      ++SemaRef.NumSFINAEErrors;
      SemaRef.Diags.setLastDiagnosticIgnored();
      SemaRef.Diags.Clear();
      Clear();
      return;
      
    case DiagnosticIDs::SFINAE_Suppress:
      // Make a copy of this suppressed diagnostic and store it with the
      // template-deduction information;
      FlushCounts();
      DiagnosticInfo DiagInfo(&SemaRef.Diags);
        
      if (*Info)
        (*Info)->addSuppressedDiagnostic(DiagInfo.getLocation(),
                        PartialDiagnostic(DiagInfo,
                                          SemaRef.Context.getDiagAllocator()));
        
      // Suppress this diagnostic.        
      SemaRef.Diags.setLastDiagnosticIgnored();
      SemaRef.Diags.Clear();
      Clear();
      return;
    }
  }
  
  // Emit the diagnostic.
  if (!this->Emit())
    return;

  // If this is not a note, and we're in a template instantiation
  // that is different from the last template instantiation where
  // we emitted an error, print a template instantiation
  // backtrace.
  if (!DiagnosticIDs::isBuiltinNote(DiagID) &&
      !SemaRef.ActiveTemplateInstantiations.empty() &&
      SemaRef.ActiveTemplateInstantiations.back()
        != SemaRef.LastTemplateInstantiationErrorContext) {
    SemaRef.PrintInstantiationStack();
    SemaRef.LastTemplateInstantiationErrorContext
      = SemaRef.ActiveTemplateInstantiations.back();
  }
}

Sema::SemaDiagnosticBuilder Sema::Diag(SourceLocation Loc, unsigned DiagID) {
  DiagnosticBuilder DB = Diags.Report(Loc, DiagID);
  return SemaDiagnosticBuilder(DB, *this, DiagID);
}

Sema::SemaDiagnosticBuilder
Sema::Diag(SourceLocation Loc, const PartialDiagnostic& PD) {
  SemaDiagnosticBuilder Builder(Diag(Loc, PD.getDiagID()));
  PD.Emit(Builder);

  return Builder;
}

/// \brief Looks through the macro-expansion chain for the given
/// location, looking for a macro expansion with the given name.
/// If one is found, returns true and sets the location to that
/// expansion loc.
bool Sema::findMacroSpelling(SourceLocation &locref, StringRef name) {
  SourceLocation loc = locref;
  if (!loc.isMacroID()) return false;

  // There's no good way right now to look at the intermediate
  // expansions, so just jump to the expansion location.
  loc = getSourceManager().getExpansionLoc(loc);

  // If that's written with the name, stop here.
  SmallVector<char, 16> buffer;
  if (getPreprocessor().getSpelling(loc, buffer) == name) {
    locref = loc;
    return true;
  }
  return false;
}

/// \brief Determines the active Scope associated with the given declaration
/// context.
///
/// This routine maps a declaration context to the active Scope object that
/// represents that declaration context in the parser. It is typically used
/// from "scope-less" code (e.g., template instantiation, lazy creation of
/// declarations) that injects a name for name-lookup purposes and, therefore,
/// must update the Scope.
///
/// \returns The scope corresponding to the given declaraion context, or NULL
/// if no such scope is open.
Scope *Sema::getScopeForContext(DeclContext *Ctx) {
  
  if (!Ctx)
    return 0;
  
  Ctx = Ctx->getPrimaryContext();
  for (Scope *S = getCurScope(); S; S = S->getParent()) {
    // Ignore scopes that cannot have declarations. This is important for
    // out-of-line definitions of static class members.
    if (S->getFlags() & (Scope::DeclScope | Scope::TemplateParamScope))
      if (DeclContext *Entity = static_cast<DeclContext *> (S->getEntity()))
        if (Ctx == Entity->getPrimaryContext())
          return S;
  }
  
  return 0;
}

/// \brief Enter a new function scope
void Sema::PushFunctionScope() {
  if (FunctionScopes.size() == 1) {
    // Use the "top" function scope rather than having to allocate
    // memory for a new scope.
    FunctionScopes.back()->Clear();
    FunctionScopes.push_back(FunctionScopes.back());
    return;
  }
  
  FunctionScopes.push_back(new FunctionScopeInfo(getDiagnostics()));
}

void Sema::PushBlockScope(Scope *BlockScope, BlockDecl *Block) {
  FunctionScopes.push_back(new BlockScopeInfo(getDiagnostics(),
                                              BlockScope, Block));
}

void Sema::PopFunctionOrBlockScope(const AnalysisBasedWarnings::Policy *WP,
                                   const Decl *D, const BlockExpr *blkExpr) {
  FunctionScopeInfo *Scope = FunctionScopes.pop_back_val();  
  assert(!FunctionScopes.empty() && "mismatched push/pop!");
  
  // Issue any analysis-based warnings.
  if (WP && D)
    AnalysisWarnings.IssueWarnings(*WP, Scope, D, blkExpr);
  else {
    for (SmallVectorImpl<sema::PossiblyUnreachableDiag>::iterator
         i = Scope->PossiblyUnreachableDiags.begin(),
         e = Scope->PossiblyUnreachableDiags.end();
         i != e; ++i) {
      const sema::PossiblyUnreachableDiag &D = *i;
      Diag(D.Loc, D.PD);
    }
  }

  if (FunctionScopes.back() != Scope) {
    delete Scope;
  }
}

/// \brief Determine whether any errors occurred within this function/method/
/// block.
bool Sema::hasAnyUnrecoverableErrorsInThisFunction() const {
  return getCurFunction()->ErrorTrap.hasUnrecoverableErrorOccurred();
}

BlockScopeInfo *Sema::getCurBlock() {
  if (FunctionScopes.empty())
    return 0;
  
  return dyn_cast<BlockScopeInfo>(FunctionScopes.back());  
}

// Pin this vtable to this file.
ExternalSemaSource::~ExternalSemaSource() {}

std::pair<ObjCMethodList, ObjCMethodList>
ExternalSemaSource::ReadMethodPool(Selector Sel) {
  return std::pair<ObjCMethodList, ObjCMethodList>();
}

void ExternalSemaSource::ReadKnownNamespaces(
                           SmallVectorImpl<NamespaceDecl *> &Namespaces) {  
}

void PrettyDeclStackTraceEntry::print(raw_ostream &OS) const {
  SourceLocation Loc = this->Loc;
  if (!Loc.isValid() && TheDecl) Loc = TheDecl->getLocation();
  if (Loc.isValid()) {
    Loc.print(OS, S.getSourceManager());
    OS << ": ";
  }
  OS << Message;

  if (TheDecl && isa<NamedDecl>(TheDecl)) {
    std::string Name = cast<NamedDecl>(TheDecl)->getNameAsString();
    if (!Name.empty())
      OS << " '" << Name << '\'';
  }

  OS << '\n';
}

/// \brief Figure out if an expression could be turned into a call.
///
/// Use this when trying to recover from an error where the programmer may have
/// written just the name of a function instead of actually calling it.
///
/// \param E - The expression to examine.
/// \param ZeroArgCallReturnTy - If the expression can be turned into a call
///  with no arguments, this parameter is set to the type returned by such a
///  call; otherwise, it is set to an empty QualType.
/// \param NonTemplateOverloads - If the expression is an overloaded function
///  name, this parameter is populated with the decls of the various overloads.
bool Sema::isExprCallable(const Expr &E, QualType &ZeroArgCallReturnTy,
                          UnresolvedSetImpl &NonTemplateOverloads) {
  ZeroArgCallReturnTy = QualType();
  NonTemplateOverloads.clear();
  if (const OverloadExpr *Overloads = dyn_cast<OverloadExpr>(&E)) {
    for (OverloadExpr::decls_iterator it = Overloads->decls_begin(),
         DeclsEnd = Overloads->decls_end(); it != DeclsEnd; ++it) {
      // Our overload set may include TemplateDecls, which we'll ignore for our
      // present purpose.
      if (const FunctionDecl *OverloadDecl = dyn_cast<FunctionDecl>(*it)) {
        NonTemplateOverloads.addDecl(*it);
        if (OverloadDecl->getMinRequiredArguments() == 0)
          ZeroArgCallReturnTy = OverloadDecl->getResultType();
      }
    }
    return true;
  }

  if (const DeclRefExpr *DeclRef = dyn_cast<DeclRefExpr>(&E)) {
    if (const FunctionDecl *Fun = dyn_cast<FunctionDecl>(DeclRef->getDecl())) {
      if (Fun->getMinRequiredArguments() == 0)
        ZeroArgCallReturnTy = Fun->getResultType();
      return true;
    }
  }

  // We don't have an expression that's convenient to get a FunctionDecl from,
  // but we can at least check if the type is "function of 0 arguments".
  QualType ExprTy = E.getType();
  const FunctionType *FunTy = NULL;
  QualType PointeeTy = ExprTy->getPointeeType();
  if (!PointeeTy.isNull())
    FunTy = PointeeTy->getAs<FunctionType>();
  if (!FunTy)
    FunTy = ExprTy->getAs<FunctionType>();
  if (!FunTy && ExprTy == Context.BoundMemberTy) {
    // Look for the bound-member type.  If it's still overloaded, give up,
    // although we probably should have fallen into the OverloadExpr case above
    // if we actually have an overloaded bound member.
    QualType BoundMemberTy = Expr::findBoundMemberType(&E);
    if (!BoundMemberTy.isNull())
      FunTy = BoundMemberTy->castAs<FunctionType>();
  }

  if (const FunctionProtoType *FPT =
      dyn_cast_or_null<FunctionProtoType>(FunTy)) {
    if (FPT->getNumArgs() == 0)
      ZeroArgCallReturnTy = FunTy->getResultType();
    return true;
  }
  return false;
}

/// \brief Give notes for a set of overloads.
///
/// A companion to isExprCallable. In cases when the name that the programmer
/// wrote was an overloaded function, we may be able to make some guesses about
/// plausible overloads based on their return types; such guesses can be handed
/// off to this method to be emitted as notes.
///
/// \param Overloads - The overloads to note.
/// \param FinalNoteLoc - If we've suppressed printing some overloads due to
///  -fshow-overloads=best, this is the location to attach to the note about too
///  many candidates. Typically this will be the location of the original
///  ill-formed expression.
void Sema::NoteOverloads(const UnresolvedSetImpl &Overloads,
                         const SourceLocation FinalNoteLoc) {
  int ShownOverloads = 0;
  int SuppressedOverloads = 0;
  for (UnresolvedSetImpl::iterator It = Overloads.begin(),
       DeclsEnd = Overloads.end(); It != DeclsEnd; ++It) {
    // FIXME: Magic number for max shown overloads stolen from
    // OverloadCandidateSet::NoteCandidates.
    if (ShownOverloads >= 4 &&
        Diags.getShowOverloads() == Diagnostic::Ovl_Best) {
      ++SuppressedOverloads;
      continue;
    }
    Diag(cast<FunctionDecl>(*It)->getSourceRange().getBegin(),
         diag::note_member_ref_possible_intended_overload);
    ++ShownOverloads;
  }
  if (SuppressedOverloads)
    Diag(FinalNoteLoc, diag::note_ovl_too_many_candidates)
        << SuppressedOverloads;
}
