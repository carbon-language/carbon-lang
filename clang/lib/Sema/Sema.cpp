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
#include "clang/Basic/Diagnostic.h"

using namespace clang;

static inline RecordDecl *CreateStructDecl(ASTContext &C, const char *Name)
{
  if (C.getLangOptions().CPlusPlus)
    return CXXRecordDecl::Create(C, TagDecl::TK_struct, 
                                 C.getTranslationUnitDecl(),
                                 SourceLocation(), &C.Idents.get(Name));
  else
    return RecordDecl::Create(C, TagDecl::TK_struct, 
                              C.getTranslationUnitDecl(),
                              SourceLocation(), &C.Idents.get(Name));
}

void Sema::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;
  CurContext = Context.getTranslationUnitDecl();
  if (!PP.getLangOptions().ObjC1) return;
  
  // Synthesize "typedef struct objc_selector *SEL;"
  RecordDecl *SelTag = CreateStructDecl(Context, "objc_selector");
  PushOnScopeChains(SelTag, TUScope);
  
  QualType SelT = Context.getPointerType(Context.getTagDeclType(SelTag));
  TypedefDecl *SelTypedef = TypedefDecl::Create(Context, CurContext,
                                                SourceLocation(),
                                                &Context.Idents.get("SEL"),
                                                SelT, 0);
  PushOnScopeChains(SelTypedef, TUScope);
  Context.setObjCSelType(SelTypedef);

  // FIXME: Make sure these don't leak!
  RecordDecl *ClassTag = CreateStructDecl(Context, "objc_class");
  QualType ClassT = Context.getPointerType(Context.getTagDeclType(ClassTag));
  TypedefDecl *ClassTypedef = 
    TypedefDecl::Create(Context, CurContext, SourceLocation(),
                        &Context.Idents.get("Class"), ClassT, 0);
  PushOnScopeChains(ClassTag, TUScope);
  PushOnScopeChains(ClassTypedef, TUScope);
  Context.setObjCClassType(ClassTypedef);
  // Synthesize "@class Protocol;
  ObjCInterfaceDecl *ProtocolDecl =
    ObjCInterfaceDecl::Create(Context, SourceLocation(),
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
                                               ObjT, 0);
  PushOnScopeChains(IdTypedef, TUScope);
  Context.setObjCIdType(IdTypedef);
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer)
  : PP(pp), Context(ctxt), Consumer(consumer), CurContext(0), CurBlock(0),
    IdResolver(pp.getLangOptions()) {
  
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

  SuperID = &IT.get("super");

  // ObjC builtin typedef names.
  Ident_id = &IT.get("id");
  Ident_Class = &IT.get("Class");
  Ident_SEL = &IT.get("SEL");
  Ident_Protocol = &IT.get("Protocol");

  TUScope = 0;
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast. 
/// If there is already an implicit cast, merge into the existing one.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Ty) {
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
      Diag(Expr->getExprLoc(), diag::err_implicit_pointer_address_space_cast,
           Expr->getSourceRange());
    }
  }
  
  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(Expr))
    ImpCast->setType(Ty);
  else 
    Expr = new ImplicitCastExpr(Ty, Expr);
}

void Sema::DeleteExpr(ExprTy *E) {
  delete static_cast<Expr*>(E);
}
void Sema::DeleteStmt(StmtTy *S) {
  delete static_cast<Stmt*>(S);
}

/// ActOnEndOfTranslationUnit - This is called at the very end of the
/// translation unit when EOF is reached and all but the top-level scope is
/// popped.
void Sema::ActOnEndOfTranslationUnit() {

}


//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

bool Sema::Diag(SourceLocation Loc, unsigned DiagID) {
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg) {
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID,  &Msg, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2) {
  std::string MsgArr[] = { Msg1, Msg2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID,  MsgArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const SourceRange& Range) {
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, 0, 0, &Range,1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                const SourceRange& Range) {
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, &Msg, 1, &Range,1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, const SourceRange& Range) {
  std::string MsgArr[] = { Msg1, Msg2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, MsgArr, 2, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1, 
                const std::string &Msg2, const std::string &Msg3, 
                const SourceRange& R1) {
  std::string MsgArr[] = { Msg1, Msg2, Msg3 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, MsgArr, 3, &R1, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID,
                const SourceRange& R1, const SourceRange& R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, 0, 0, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                const SourceRange& R1, const SourceRange& R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID,  &Msg, 1, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Range, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, const SourceRange& R1,
                const SourceRange& R2) {
  std::string MsgArr[] = { Msg1, Msg2 };
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Range),DiagID, MsgArr,2,RangeArr, 2);
  return true;
}

const LangOptions &Sema::getLangOptions() const {
  return PP.getLangOptions();
}

ObjCMethodDecl *Sema::getCurMethodDecl() {
  return dyn_cast<ObjCMethodDecl>(CurContext);
}
