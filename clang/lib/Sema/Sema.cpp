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

bool Sema::isObjCObjectPointerType(QualType type) const {
  if (!type->isPointerType() && !type->isObjCQualifiedIdType())
    return false;
  if (type == Context.getObjCIdType() || type == Context.getObjCClassType() ||
      type->isObjCQualifiedIdType())
    return true;
  
  if (type->isPointerType()) {
    PointerType *pointerType = static_cast<PointerType*>(type.getTypePtr());
    type = pointerType->getPointeeType();
  }
  return (type->isObjCInterfaceType() || type->isObjCQualifiedIdType());
}

void Sema::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;
  CurContext = Context.getTranslationUnitDecl();
  if (!PP.getLangOptions().ObjC1) return;
  
  TypedefType *t;
  
  // Add the built-in ObjC types.
  t = cast<TypedefType>(Context.getObjCIdType().getTypePtr());
  PushOnScopeChains(t->getDecl(), TUScope);
  t = cast<TypedefType>(Context.getObjCClassType().getTypePtr());
  PushOnScopeChains(t->getDecl(), TUScope);
  ObjCInterfaceType *it = cast<ObjCInterfaceType>(Context.getObjCProtoType());
  ObjCInterfaceDecl *IDecl = it->getDecl();
  PushOnScopeChains(IDecl, TUScope);
  
  // Synthesize "typedef struct objc_selector *SEL;"
  RecordDecl *SelTag = RecordDecl::Create(Context, TagDecl::TK_struct, CurContext,
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
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer)
  : PP(pp), Context(ctxt), Consumer(consumer), 
    CurFunctionDecl(0), CurMethodDecl(0), CurContext(0) {
  
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

  // FIXME: Move this initialization up to Sema::ActOnTranslationUnitScope()
  // and make sure the decls get inserted into TUScope!
  // FIXME: And make sure they don't leak!
  if (PP.getLangOptions().ObjC1) {
    TranslationUnitDecl *TUDecl = Context.getTranslationUnitDecl();

    // Synthesize "typedef struct objc_class *Class;"
    RecordDecl *ClassTag = RecordDecl::Create(Context, TagDecl::TK_struct,
                                              TUDecl,
                                              SourceLocation(),
                                              &IT.get("objc_class"), 0);
    QualType ClassT = Context.getPointerType(Context.getTagDeclType(ClassTag));
    TypedefDecl *ClassTypedef = 
      TypedefDecl::Create(Context, TUDecl, SourceLocation(),
                          &Context.Idents.get("Class"), ClassT, 0);
    Context.setObjCClassType(ClassTypedef);
    
    // Synthesize "@class Protocol;
    ObjCInterfaceDecl *ProtocolDecl =
      ObjCInterfaceDecl::Create(Context, SourceLocation(), 0, 
                                &Context.Idents.get("Protocol"), 
                                SourceLocation(), true);
    Context.setObjCProtoType(Context.getObjCInterfaceType(ProtocolDecl));
    
    // Synthesize "typedef struct objc_object { Class isa; } *id;"
    RecordDecl *ObjectTag = 
      RecordDecl::Create(Context, TagDecl::TK_struct, TUDecl,
                         SourceLocation(),
                         &IT.get("objc_object"), 0);
    FieldDecl *IsaDecl = FieldDecl::Create(Context, SourceLocation(),
                                           &Context.Idents.get("isa"),
                                           Context.getObjCClassType());
    ObjectTag->defineBody(&IsaDecl, 1);
    QualType ObjT = Context.getPointerType(Context.getTagDeclType(ObjectTag));
    TypedefDecl *IdTypedef = TypedefDecl::Create(Context, TUDecl,
                                                 SourceLocation(),
                                                 &Context.Idents.get("id"),
                                                 ObjT, 0);
    Context.setObjCIdType(IdTypedef);
  }
  TUScope = 0;
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast. 
/// If there is already an implicit cast, merge into the existing one.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Type) {
  if (Expr->getType().getCanonicalType() == Type.getCanonicalType()) return;
  
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
