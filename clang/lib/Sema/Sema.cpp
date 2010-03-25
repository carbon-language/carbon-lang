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

#include "Sema.h"
#include "TargetAttributesSema.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/APFloat.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
using namespace clang;

FunctionScopeInfo::~FunctionScopeInfo() { }

void FunctionScopeInfo::Clear(unsigned NumErrors) {
  NeedsScopeChecking = false;
  LabelMap.clear();
  SwitchStack.clear();
  NumErrorsAtStartOfFunction = NumErrors;
}

BlockScopeInfo::~BlockScopeInfo() { }

static inline RecordDecl *CreateStructDecl(ASTContext &C, const char *Name) {
  if (C.getLangOptions().CPlusPlus)
    return CXXRecordDecl::Create(C, TagDecl::TK_struct,
                                 C.getTranslationUnitDecl(),
                                 SourceLocation(), &C.Idents.get(Name));

  return RecordDecl::Create(C, TagDecl::TK_struct,
                            C.getTranslationUnitDecl(),
                            SourceLocation(), &C.Idents.get(Name));
}

void Sema::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;
  PushDeclContext(S, Context.getTranslationUnitDecl());

  if (PP.getTargetInfo().getPointerWidth(0) >= 64) {
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
  }


  if (!PP.getLangOptions().ObjC1) return;

  // Built-in ObjC types may already be set by PCHReader (hence isNull checks).
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
    QualType IdT = Context.getObjCObjectPointerType(Context.ObjCBuiltinIdTy);
    TypeSourceInfo *IdInfo = Context.getTrivialTypeSourceInfo(IdT);
    TypedefDecl *IdTypedef
      = TypedefDecl::Create(Context, CurContext, SourceLocation(),
                            &Context.Idents.get("id"), IdInfo);
    PushOnScopeChains(IdTypedef, TUScope);
    Context.setObjCIdType(Context.getTypeDeclType(IdTypedef));
    Context.ObjCIdRedefinitionType = Context.getObjCIdType();
  }
  // Create the built-in typedef for 'Class'.
  if (Context.getObjCClassType().isNull()) {
    QualType ClassType
      = Context.getObjCObjectPointerType(Context.ObjCBuiltinClassTy);
    TypeSourceInfo *ClassInfo = Context.getTrivialTypeSourceInfo(ClassType);
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
    PackContext(0), TopFunctionScope(0), ParsingDeclDepth(0),
    IdResolver(pp.getLangOptions()), StdNamespace(0), StdBadAlloc(0),
    GlobalNewDeleteDeclared(false), 
    CompleteTranslationUnit(CompleteTranslationUnit),
    NumSFINAEErrors(0), NonInstantiationEntries(0), 
    CurrentInstantiationScope(0), TyposCorrected(0),
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
}

Sema::~Sema() {
  if (PackContext) FreePackedContext();
  delete TheTargetAttributesSema;
  while (!FunctionScopes.empty())
    PopFunctionOrBlockScope();
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast.
/// If there is already an implicit cast, merge into the existing one.
/// If isLvalue, the result of the cast is an lvalue.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Ty,
                             CastExpr::CastKind Kind, bool isLvalue) {
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

  CheckImplicitConversion(Expr, Ty);

  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(Expr)) {
    if (ImpCast->getCastKind() == Kind) {
      ImpCast->setType(Ty);
      ImpCast->setLvalueCast(isLvalue);
      return;
    }
  }

  Expr = new (Context) ImplicitCastExpr(Ty, Kind, Expr, isLvalue);
}

void Sema::DeleteExpr(ExprTy *E) {
  if (E) static_cast<Expr*>(E)->Destroy(Context);
}
void Sema::DeleteStmt(StmtTy *S) {
  if (S) static_cast<Stmt*>(S)->Destroy(Context);
}

/// ActOnEndOfTranslationUnit - This is called at the very end of the
/// translation unit when EOF is reached and all but the top-level scope is
/// popped.
void Sema::ActOnEndOfTranslationUnit() {
  
  // Remove functions that turned out to be used.
  UnusedStaticFuncs.erase(std::remove_if(UnusedStaticFuncs.begin(), 
                                         UnusedStaticFuncs.end(), 
                                         std::mem_fun(&FunctionDecl::isUsed)), 
                          UnusedStaticFuncs.end());
  
  while (1) {
    // C++: Perform implicit template instantiations.
    //
    // FIXME: When we perform these implicit instantiations, we do not carefully
    // keep track of the point of instantiation (C++ [temp.point]). This means
    // that name lookup that occurs within the template instantiation will
    // always happen at the end of the translation unit, so it will find
    // some names that should not be found. Although this is common behavior
    // for C++ compilers, it is technically wrong. In the future, we either need
    // to be able to filter the results of name lookup or we need to perform
    // template instantiations earlier.
    PerformPendingImplicitInstantiations();
    
    /// If ProcessPendingClassesWithUnmarkedVirtualMembers ends up marking 
    /// any virtual member functions it might lead to more pending template
    /// instantiations, which is why we need to loop here.
    if (!ProcessPendingClassesWithUnmarkedVirtualMembers())
      break;
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

  if (!CompleteTranslationUnit)
    return;

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
  
  // Output warning for unused functions.
  for (std::vector<FunctionDecl*>::iterator
       F = UnusedStaticFuncs.begin(),
       FEnd = UnusedStaticFuncs.end();
       F != FEnd;
       ++F)
    Diag((*F)->getLocation(), diag::warn_unused_function) << (*F)->getDeclName();
  
}


//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

DeclContext *Sema::getFunctionLevelDeclContext() {
  DeclContext *DC = CurContext;

  while (isa<BlockDecl>(DC))
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


/// \brief Enter a new function scope
void Sema::PushFunctionScope() {
  if (FunctionScopes.empty()) {
    // Use the "top" function scope rather than having to allocate memory for
    // a new scope.
    TopFunctionScope.Clear(getDiagnostics().getNumErrors());
    FunctionScopes.push_back(&TopFunctionScope);
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
  if (FunctionScopes.back() != &TopFunctionScope)
    delete FunctionScopes.back();
  else
    TopFunctionScope.Clear(getDiagnostics().getNumErrors());
  
  FunctionScopes.pop_back();
}

/// \brief Determine whether any errors occurred within this function/method/
/// block.
bool Sema::hasAnyErrorsInThisFunction() const {
  unsigned NumErrors = TopFunctionScope.NumErrorsAtStartOfFunction;
  if (!FunctionScopes.empty())
    NumErrors = FunctionScopes.back()->NumErrorsAtStartOfFunction;
  return NumErrors != getDiagnostics().getNumErrors();
}

BlockScopeInfo *Sema::getCurBlock() {
  if (FunctionScopes.empty())
    return 0;
  
  return dyn_cast<BlockScopeInfo>(FunctionScopes.back());  
}
