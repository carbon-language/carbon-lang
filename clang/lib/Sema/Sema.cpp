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
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Parse/Scope.h"

using namespace clang;

bool Sema::isBuiltinObjCType(TypedefDecl *TD) {
  const char *typeName = TD->getIdentifier()->getName();
  return strcmp(typeName, "id") == 0 || strcmp(typeName, "Class") == 0 ||
         strcmp(typeName, "SEL") == 0 || strcmp(typeName, "Protocol") == 0;
}

void Sema::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;
  CurContext = Context.getTranslationUnitDecl();
  if (!PP.getLangOptions().ObjC1) return;
  
  // Synthesize "typedef struct objc_selector *SEL;"
  RecordDecl *SelTag = RecordDecl::Create(Context, TagDecl::TK_struct,
                                          CurContext,
                                          SourceLocation(), 
                                          &Context.Idents.get("objc_selector"),
                                          0);
  PushOnScopeChains(SelTag, TUScope);
  
  QualType SelT = Context.getPointerType(Context.getTagDeclType(SelTag));
  TypedefDecl *SelTypedef = TypedefDecl::Create(Context, CurContext,
                                                SourceLocation(),
                                                &Context.Idents.get("SEL"),
                                                SelT, 0);
  PushOnScopeChains(SelTypedef, TUScope);
  Context.setObjCSelType(SelTypedef);

  // FIXME: Make sure these don't leak!
  RecordDecl *ClassTag = RecordDecl::Create(Context, TagDecl::TK_struct,
                                            CurContext,
                                            SourceLocation(),
                                            &Context.Idents.get("objc_class"),
                                            0);
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
  RecordDecl *ObjectTag = 
    RecordDecl::Create(Context, TagDecl::TK_struct, CurContext,
                       SourceLocation(),
                       &Context.Idents.get("objc_object"), 0);
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
  : PP(pp), Context(ctxt), Consumer(consumer), CurContext(0) {
  
  // Get IdentifierInfo objects for known functions for which we
  // do extra checking.  
  IdentifierTable &IT = PP.getIdentifierTable();  

  KnownFunctionIDs[id_printf]    = &IT.get("printf");
  KnownFunctionIDs[id_fprintf]   = &IT.get("fprintf");
  KnownFunctionIDs[id_sprintf]   = &IT.get("sprintf");
  KnownFunctionIDs[id_snprintf]  = &IT.get("snprintf");
  KnownFunctionIDs[id_asprintf]  = &IT.get("asprintf");
  KnownFunctionIDs[id_NSLog]     = &IT.get("NSLog");
  KnownFunctionIDs[id_vsnprintf] = &IT.get("vsnprintf");
  KnownFunctionIDs[id_vasprintf] = &IT.get("vasprintf");
  KnownFunctionIDs[id_vfprintf]  = &IT.get("vfprintf");
  KnownFunctionIDs[id_vsprintf]  = &IT.get("vsprintf");
  KnownFunctionIDs[id_vprintf]   = &IT.get("vprintf");

  TUScope = 0;
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast. 
/// If there is already an implicit cast, merge into the existing one.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Type) {
  if (Context.getCanonicalType(Expr->getType()) ==
        Context.getCanonicalType(Type)) return;
  
  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(Expr))
    ImpCast->setType(Type);
  else 
    Expr = new ImplicitCastExpr(Type, Expr);
}



void Sema::DeleteExpr(ExprTy *E) {
  delete static_cast<Expr*>(E);
}
void Sema::DeleteStmt(StmtTy *S) {
  delete static_cast<Stmt*>(S);
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

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, SourceRange Range) {
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, 0, 0, &Range,1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                SourceRange Range) {
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, &Msg, 1, &Range,1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, SourceRange Range) {
  std::string MsgArr[] = { Msg1, Msg2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, MsgArr, 2, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1, 
                const std::string &Msg2, const std::string &Msg3, 
                SourceRange R1) {
  std::string MsgArr[] = { Msg1, Msg2, Msg3 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, MsgArr, 3, &R1, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID,
                SourceRange R1, SourceRange R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID, 0, 0, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                SourceRange R1, SourceRange R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Loc), DiagID,  &Msg, 1, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Range, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, SourceRange R1, SourceRange R2) {
  std::string MsgArr[] = { Msg1, Msg2 };
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(PP.getFullLoc(Range),DiagID, MsgArr,2,RangeArr, 2);
  return true;
}

const LangOptions &Sema::getLangOptions() const {
  return PP.getLangOptions();
}
