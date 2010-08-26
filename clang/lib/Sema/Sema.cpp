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
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
using namespace clang;
using namespace sema;

FunctionScopeInfo::~FunctionScopeInfo() { }

void FunctionScopeInfo::Clear(unsigned NumErrors) {
  HasBranchProtectedScope = false;
  HasBranchIntoScope = false;
  HasIndirectGoto = false;
  
  LabelMap.clear();
  SwitchStack.clear();
  Returns.clear();
  NumErrorsAtStartOfFunction = NumErrors;
}

BlockScopeInfo::~BlockScopeInfo() { }

void Sema::ActOnTranslationUnitScope(Scope *S) {
  TUScope = S;
  PushDeclContext(S, Context.getTranslationUnitDecl());

  VAListTagName = PP.getIdentifierInfo("__va_list_tag");

  if (!Context.isInt128Installed() && // May be set by ASTReader.
      PP.getTargetInfo().getPointerWidth(0) >= 64) {
    TypeSourceInfo *TInfo;

    // Install [u]int128_t for 64-bit targets.
    TInfo = Context.getTrivialTypeSourceInfo(Context.Int128Ty);
    PushOnScopeChains(TypedefDecl::Create(Context, CurContext,
                                          SourceLocation(),
                                          &Context.Idents.get("__int128_t"),
                                          TInfo), TUScope);

    TInfo = Context.getTrivialTypeSourceInfo(Context.UnsignedInt128Ty);
    PushOnScopeChains(TypedefDecl::Create(Context, CurContext,
                                          SourceLocation(),
                                          &Context.Idents.get("__uint128_t"),
                                          TInfo), TUScope);
    Context.setInt128Installed();
  }


  if (!PP.getLangOptions().ObjC1) return;

  // Built-in ObjC types may already be set by ASTReader (hence isNull checks).
  if (Context.getObjCSelType().isNull()) {
    // Create the built-in typedef for 'SEL'.
    QualType SelT = Context.getPointerType(Context.ObjCBuiltinSelTy);
    TypeSourceInfo *SelInfo = Context.getTrivialTypeSourceInfo(SelT);
    TypedefDecl *SelTypedef
      = TypedefDecl::Create(Context, CurContext, SourceLocation(),
                            &Context.Idents.get("SEL"), SelInfo);
    PushOnScopeChains(SelTypedef, TUScope);
    Context.setObjCSelType(Context.getTypeDeclType(SelTypedef));
    Context.ObjCSelRedefinitionType = Context.getObjCSelType();
  }

  // Synthesize "@class Protocol;
  if (Context.getObjCProtoType().isNull()) {
    ObjCInterfaceDecl *ProtocolDecl =
      ObjCInterfaceDecl::Create(Context, CurContext, SourceLocation(),
                                &Context.Idents.get("Protocol"),
                                SourceLocation(), true);
    Context.setObjCProtoType(Context.getObjCInterfaceType(ProtocolDecl));
    PushOnScopeChains(ProtocolDecl, TUScope, false);
  }
  // Create the built-in typedef for 'id'.
  if (Context.getObjCIdType().isNull()) {
    QualType T = Context.getObjCObjectType(Context.ObjCBuiltinIdTy, 0, 0);
    T = Context.getObjCObjectPointerType(T);
    TypeSourceInfo *IdInfo = Context.getTrivialTypeSourceInfo(T);
    TypedefDecl *IdTypedef
      = TypedefDecl::Create(Context, CurContext, SourceLocation(),
                            &Context.Idents.get("id"), IdInfo);
    PushOnScopeChains(IdTypedef, TUScope);
    Context.setObjCIdType(Context.getTypeDeclType(IdTypedef));
    Context.ObjCIdRedefinitionType = Context.getObjCIdType();
  }
  // Create the built-in typedef for 'Class'.
  if (Context.getObjCClassType().isNull()) {
    QualType T = Context.getObjCObjectType(Context.ObjCBuiltinClassTy, 0, 0);
    T = Context.getObjCObjectPointerType(T);
    TypeSourceInfo *ClassInfo = Context.getTrivialTypeSourceInfo(T);
    TypedefDecl *ClassTypedef
      = TypedefDecl::Create(Context, CurContext, SourceLocation(),
                            &Context.Idents.get("Class"), ClassInfo);
    PushOnScopeChains(ClassTypedef, TUScope);
    Context.setObjCClassType(Context.getTypeDeclType(ClassTypedef));
    Context.ObjCClassRedefinitionType = Context.getObjCClassType();
  }  
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
           bool CompleteTranslationUnit,
           CodeCompleteConsumer *CodeCompleter)
  : TheTargetAttributesSema(0),
    LangOpts(pp.getLangOptions()), PP(pp), Context(ctxt), Consumer(consumer),
    Diags(PP.getDiagnostics()), SourceMgr(PP.getSourceManager()),
    ExternalSource(0), CodeCompleter(CodeCompleter), CurContext(0), 
    PackContext(0), VisContext(0), ParsingDeclDepth(0),
    IdResolver(pp.getLangOptions()), GlobalNewDeleteDeclared(false), 
    CompleteTranslationUnit(CompleteTranslationUnit),
    NumSFINAEErrors(0), SuppressAccessChecking(false),
    NonInstantiationEntries(0), CurrentInstantiationScope(0), TyposCorrected(0),
    AnalysisWarnings(*this)
{
  TUScope = 0;
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());

  // Tell diagnostics how to render things from the AST library.
  PP.getDiagnostics().SetArgToStringFn(&FormatASTNodeDiagnosticArgument, 
                                       &Context);

  ExprEvalContexts.push_back(
                  ExpressionEvaluationContextRecord(PotentiallyEvaluated, 0));  

  FunctionScopes.push_back(new FunctionScopeInfo(Diags.getNumErrors()));
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
}

Sema::~Sema() {
  if (PackContext) FreePackedContext();
  if (VisContext) FreeVisContext();
  delete TheTargetAttributesSema;

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

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast.
/// If there is already an implicit cast, merge into the existing one.
/// The result is of the given category.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Ty,
                             CastKind Kind, ExprValueKind VK,
                             const CXXCastPath *BasePath) {
  QualType ExprTy = Context.getCanonicalType(Expr->getType());
  QualType TypeTy = Context.getCanonicalType(Ty);

  if (ExprTy == TypeTy)
    return;

  if (Expr->getType()->isPointerType() && Ty->isPointerType()) {
    QualType ExprBaseType = cast<PointerType>(ExprTy)->getPointeeType();
    QualType BaseType = cast<PointerType>(TypeTy)->getPointeeType();
    if (ExprBaseType.getAddressSpace() != BaseType.getAddressSpace()) {
      Diag(Expr->getExprLoc(), diag::err_implicit_pointer_address_space_cast)
        << Expr->getSourceRange();
    }
  }

  // If this is a derived-to-base cast to a through a virtual base, we
  // need a vtable.
  if (Kind == CK_DerivedToBase && 
      BasePathInvolvesVirtualBase(*BasePath)) {
    QualType T = Expr->getType();
    if (const PointerType *Pointer = T->getAs<PointerType>())
      T = Pointer->getPointeeType();
    if (const RecordType *RecordTy = T->getAs<RecordType>())
      MarkVTableUsed(Expr->getLocStart(), 
                     cast<CXXRecordDecl>(RecordTy->getDecl()));
  }

  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(Expr)) {
    if (ImpCast->getCastKind() == Kind && (!BasePath || BasePath->empty())) {
      ImpCast->setType(Ty);
      ImpCast->setValueKind(VK);
      return;
    }
  }

  Expr = ImplicitCastExpr::Create(Context, Ty, Kind, Expr, BasePath, VK);
}

ExprValueKind Sema::CastCategory(Expr *E) {
  Expr::Classification Classification = E->Classify(Context);
  return Classification.isRValue() ? VK_RValue :
      (Classification.isLValue() ? VK_LValue : VK_XValue);
}

void Sema::DeleteExpr(ExprTy *E) {
}
void Sema::DeleteStmt(StmtTy *S) {
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

/// ActOnEndOfTranslationUnit - This is called at the very end of the
/// translation unit when EOF is reached and all but the top-level scope is
/// popped.
void Sema::ActOnEndOfTranslationUnit() {
  // At PCH writing, implicit instantiations and VTable handling info are
  // stored and performed when the PCH is included.
  if (CompleteTranslationUnit)
    while (1) {
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

      /// If DefinedUsedVTables ends up marking any virtual member
      /// functions it might lead to more pending template
      /// instantiations, which is why we need to loop here.
      if (!DefineUsedVTables())
        break;
    }
  
  // Remove file scoped decls that turned out to be used.
  UnusedFileScopedDecls.erase(std::remove_if(UnusedFileScopedDecls.begin(),
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
  for (unsigned i = 0, e = TentativeDefinitions.size(); i != e; ++i) {
    VarDecl *VD = TentativeDefinitions[i]->getActingDefinition();

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
  
  // Output warning for unused file scoped decls.
  for (llvm::SmallVectorImpl<const DeclaratorDecl*>::iterator
         I = UnusedFileScopedDecls.begin(),
         E = UnusedFileScopedDecls.end(); I != E; ++I) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(*I)) {
      const FunctionDecl *DiagD;
      if (!FD->hasBody(DiagD))
        DiagD = FD;
      Diag(DiagD->getLocation(),
           isa<CXXMethodDecl>(DiagD) ? diag::warn_unused_member_function
                                     : diag::warn_unused_function)
            << DiagD->getDeclName();
    } else {
      const VarDecl *DiagD = cast<VarDecl>(*I)->getDefinition();
      if (!DiagD)
        DiagD = cast<VarDecl>(*I);
      Diag(DiagD->getLocation(), diag::warn_unused_variable)
            << DiagD->getDeclName();
    }
  }

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
  if (!this->Emit())
    return;

  // If this is not a note, and we're in a template instantiation
  // that is different from the last template instantiation where
  // we emitted an error, print a template instantiation
  // backtrace.
  if (!SemaRef.Diags.isBuiltinNote(DiagID) &&
      !SemaRef.ActiveTemplateInstantiations.empty() &&
      SemaRef.ActiveTemplateInstantiations.back()
        != SemaRef.LastTemplateInstantiationErrorContext) {
    SemaRef.PrintInstantiationStack();
    SemaRef.LastTemplateInstantiationErrorContext
      = SemaRef.ActiveTemplateInstantiations.back();
  }
}

Sema::SemaDiagnosticBuilder Sema::Diag(SourceLocation Loc, unsigned DiagID) {
  if (isSFINAEContext()) {
    switch (Diagnostic::getDiagnosticSFINAEResponse(DiagID)) {
    case Diagnostic::SFINAE_Report:
      // Fall through; we'll report the diagnostic below.
      break;

    case Diagnostic::SFINAE_SubstitutionFailure:
      // Count this failure so that we know that template argument deduction
      // has failed.
      ++NumSFINAEErrors;
      // Fall through
        
    case Diagnostic::SFINAE_Suppress:
      // Suppress this diagnostic.
      Diags.setLastDiagnosticIgnored();
      return SemaDiagnosticBuilder(*this);
    }
  }
  
  DiagnosticBuilder DB = Diags.Report(FullSourceLoc(Loc, SourceMgr), DiagID);
  return SemaDiagnosticBuilder(DB, *this, DiagID);
}

Sema::SemaDiagnosticBuilder
Sema::Diag(SourceLocation Loc, const PartialDiagnostic& PD) {
  SemaDiagnosticBuilder Builder(Diag(Loc, PD.getDiagID()));
  PD.Emit(Builder);

  return Builder;
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
    FunctionScopes.back()->Clear(getDiagnostics().getNumErrors());
    FunctionScopes.push_back(FunctionScopes.back());
    return;
  }
  
  FunctionScopes.push_back(
                      new FunctionScopeInfo(getDiagnostics().getNumErrors()));
}

void Sema::PushBlockScope(Scope *BlockScope, BlockDecl *Block) {
  FunctionScopes.push_back(new BlockScopeInfo(getDiagnostics().getNumErrors(),
                                              BlockScope, Block));
}

void Sema::PopFunctionOrBlockScope() {
  FunctionScopeInfo *Scope = FunctionScopes.pop_back_val();
  assert(!FunctionScopes.empty() && "mismatched push/pop!");
  if (FunctionScopes.back() != Scope)
    delete Scope;
}

/// \brief Determine whether any errors occurred within this function/method/
/// block.
bool Sema::hasAnyErrorsInThisFunction() const {
  return getCurFunction()->NumErrorsAtStartOfFunction
             != getDiagnostics().getNumErrors();
}

BlockScopeInfo *Sema::getCurBlock() {
  if (FunctionScopes.empty())
    return 0;
  
  return dyn_cast<BlockScopeInfo>(FunctionScopes.back());  
}

// Pin this vtable to this file.
ExternalSemaSource::~ExternalSemaSource() {}

void PrettyDeclStackTraceEntry::print(llvm::raw_ostream &OS) const {
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
