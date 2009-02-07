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
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
using namespace clang;

/// ConvertQualTypeToStringFn - This function is used to pretty print the 
/// specified QualType as a string in diagnostics.
static void ConvertArgToStringFn(Diagnostic::ArgumentKind Kind, intptr_t Val,
                                      const char *Modifier, unsigned ModLen,
                                      const char *Argument, unsigned ArgLen,
                                      llvm::SmallVectorImpl<char> &Output) {
  
  std::string S;
  if (Kind == Diagnostic::ak_qualtype) {
    QualType Ty(QualType::getFromOpaquePtr(reinterpret_cast<void*>(Val)));

    // FIXME: Playing with std::string is really slow.
    S = Ty.getAsString();

    assert(ModLen == 0 && ArgLen == 0 &&
           "Invalid modifier for QualType argument");
    
  } else if (Kind == Diagnostic::ak_declarationname) {
   
    DeclarationName N = DeclarationName::getFromOpaqueInteger(Val);
    S = N.getAsString();
    
    if (ModLen == 9 && !memcmp(Modifier, "objcclass", 9) && ArgLen == 0)
      S = '+' + S;
    else if (ModLen == 12 && !memcmp(Modifier, "objcinstance", 12) && ArgLen==0)
      S = '-' + S;
    else
      assert(ModLen == 0 && ArgLen == 0 &&
             "Invalid modifier for DeclarationName argument");
  } else {
    assert(Kind == Diagnostic::ak_nameddecl);
    if (ModLen == 1 && Modifier[0] == 'q' && ArgLen == 0)
      S = reinterpret_cast<NamedDecl*>(Val)->getQualifiedNameAsString();
    else { 
      assert(ModLen == 0 && ArgLen == 0 &&
           "Invalid modifier for NamedDecl* argument");
      S = reinterpret_cast<NamedDecl*>(Val)->getNameAsString();
    }
  }
  Output.append(S.begin(), S.end());
}


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
  if (!PP.getLangOptions().ObjC1) return;
  
  // Synthesize "typedef struct objc_selector *SEL;"
  RecordDecl *SelTag = CreateStructDecl(Context, "objc_selector");
  PushOnScopeChains(SelTag, TUScope);
  
  QualType SelT = Context.getPointerType(Context.getTagDeclType(SelTag));
  TypedefDecl *SelTypedef = TypedefDecl::Create(Context, CurContext,
                                                SourceLocation(),
                                                &Context.Idents.get("SEL"),
                                                SelT);
  PushOnScopeChains(SelTypedef, TUScope);
  Context.setObjCSelType(SelTypedef);

  // FIXME: Make sure these don't leak!
  RecordDecl *ClassTag = CreateStructDecl(Context, "objc_class");
  QualType ClassT = Context.getPointerType(Context.getTagDeclType(ClassTag));
  TypedefDecl *ClassTypedef = 
    TypedefDecl::Create(Context, CurContext, SourceLocation(),
                        &Context.Idents.get("Class"), ClassT);
  PushOnScopeChains(ClassTag, TUScope);
  PushOnScopeChains(ClassTypedef, TUScope);
  Context.setObjCClassType(ClassTypedef);
  // Synthesize "@class Protocol;
  ObjCInterfaceDecl *ProtocolDecl =
    ObjCInterfaceDecl::Create(Context, CurContext, SourceLocation(),
                              &Context.Idents.get("Protocol"), 
                              SourceLocation(), true);
  Context.setObjCProtoType(Context.getObjCInterfaceType(ProtocolDecl));
  PushOnScopeChains(ProtocolDecl, TUScope);
  
  // Synthesize "typedef struct objc_object { Class isa; } *id;"
  RecordDecl *ObjectTag = CreateStructDecl(Context, "objc_object");

  QualType ObjT = Context.getPointerType(Context.getTagDeclType(ObjectTag));
  PushOnScopeChains(ObjectTag, TUScope);
  TypedefDecl *IdTypedef = TypedefDecl::Create(Context, CurContext,
                                               SourceLocation(),
                                               &Context.Idents.get("id"),
                                               ObjT);
  PushOnScopeChains(IdTypedef, TUScope);
  Context.setObjCIdType(IdTypedef);
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer)
  : LangOpts(pp.getLangOptions()), PP(pp), Context(ctxt), Consumer(consumer),
    Diags(PP.getDiagnostics()),
    SourceMgr(PP.getSourceManager()), CurContext(0), PreDeclaratorDC(0),
    CurBlock(0), PackContext(0), IdResolver(pp.getLangOptions()),
    GlobalNewDeleteDeclared(false) {
  
  // Get IdentifierInfo objects for known functions for which we
  // do extra checking.  
  IdentifierTable &IT = PP.getIdentifierTable();  

  KnownFunctionIDs[id_printf]        = &IT.get("printf");
  KnownFunctionIDs[id_fprintf]       = &IT.get("fprintf");
  KnownFunctionIDs[id_sprintf]       = &IT.get("sprintf");
  KnownFunctionIDs[id_sprintf_chk]   = &IT.get("__builtin___sprintf_chk");
  KnownFunctionIDs[id_snprintf]      = &IT.get("snprintf");
  KnownFunctionIDs[id_snprintf_chk]  = &IT.get("__builtin___snprintf_chk");
  KnownFunctionIDs[id_asprintf]      = &IT.get("asprintf");
  KnownFunctionIDs[id_NSLog]         = &IT.get("NSLog");
  KnownFunctionIDs[id_vsnprintf]     = &IT.get("vsnprintf");
  KnownFunctionIDs[id_vasprintf]     = &IT.get("vasprintf");
  KnownFunctionIDs[id_vfprintf]      = &IT.get("vfprintf");
  KnownFunctionIDs[id_vsprintf]      = &IT.get("vsprintf");
  KnownFunctionIDs[id_vsprintf_chk]  = &IT.get("__builtin___vsprintf_chk");
  KnownFunctionIDs[id_vsnprintf]     = &IT.get("vsnprintf");
  KnownFunctionIDs[id_vsnprintf_chk] = &IT.get("__builtin___vsnprintf_chk");
  KnownFunctionIDs[id_vprintf]       = &IT.get("vprintf");

  StdNamespace = 0;
  TUScope = 0;
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());
      
  // Tell diagnostics how to render things from the AST library.
  PP.getDiagnostics().SetArgToStringFn(ConvertArgToStringFn);
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast. 
/// If there is already an implicit cast, merge into the existing one.
/// If isLvalue, the result of the cast is an lvalue.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Ty, bool isLvalue) {
  QualType ExprTy = Context.getCanonicalType(Expr->getType());
  QualType TypeTy = Context.getCanonicalType(Ty);
  
  if (ExprTy == TypeTy)
    return;
  
  if (Expr->getType().getTypePtr()->isPointerType() &&
      Ty.getTypePtr()->isPointerType()) {
    QualType ExprBaseType = 
      cast<PointerType>(ExprTy.getUnqualifiedType())->getPointeeType();
    QualType BaseType =
      cast<PointerType>(TypeTy.getUnqualifiedType())->getPointeeType();
    if (ExprBaseType.getAddressSpace() != BaseType.getAddressSpace()) {
      Diag(Expr->getExprLoc(), diag::err_implicit_pointer_address_space_cast)
        << Expr->getSourceRange();
    }
  }
  
  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(Expr)) {
    ImpCast->setType(Ty);
    ImpCast->setLvalueCast(isLvalue);
  } else 
    Expr = new (Context) ImplicitCastExpr(Ty, Expr, isLvalue);
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

}


//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

/// getCurFunctionDecl - If inside of a function body, this returns a pointer
/// to the function decl for the function being parsed.  If we're currently
/// in a 'block', this returns the containing context.
FunctionDecl *Sema::getCurFunctionDecl() {
  DeclContext *DC = CurContext;
  while (isa<BlockDecl>(DC))
    DC = DC->getParent();
  return dyn_cast<FunctionDecl>(DC);
}

ObjCMethodDecl *Sema::getCurMethodDecl() {
  DeclContext *DC = CurContext;
  while (isa<BlockDecl>(DC))
    DC = DC->getParent();
  return dyn_cast<ObjCMethodDecl>(DC);
}

NamedDecl *Sema::getCurFunctionOrMethodDecl() {
  DeclContext *DC = CurContext;
  while (isa<BlockDecl>(DC))
    DC = DC->getParent();
  if (isa<ObjCMethodDecl>(DC) || isa<FunctionDecl>(DC))
    return cast<NamedDecl>(DC);
  return 0;
}

