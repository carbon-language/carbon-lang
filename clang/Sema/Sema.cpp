//===--- Sema.cpp - AST Builder and Semantic Analysis Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

bool Sema::isBuiltinObjcType(TypedefDecl *TD) {
  const char *typeName = TD->getIdentifier()->getName();
  return strcmp(typeName, "id") == 0 || strcmp(typeName, "Class") == 0 ||
         strcmp(typeName, "SEL") == 0;
}

void Sema::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;
  if (PP.getLangOptions().ObjC1) {
    TypedefType *t;
    
    // Add the built-in ObjC types.
    t = dyn_cast<TypedefType>(Context.getObjcIdType().getTypePtr());
    t->getDecl()->getIdentifier()->setFETokenInfo(t->getDecl());
    TUScope->AddDecl(t->getDecl());
    t = dyn_cast<TypedefType>(Context.getObjcClassType().getTypePtr());
    t->getDecl()->getIdentifier()->setFETokenInfo(t->getDecl());
    TUScope->AddDecl(t->getDecl());
    t = dyn_cast<TypedefType>(Context.getObjcSelType().getTypePtr());
    t->getDecl()->getIdentifier()->setFETokenInfo(t->getDecl());
    TUScope->AddDecl(t->getDecl());
  }
}

/// FIXME: remove this.
/// GetObjcProtoType - See comments for Sema::GetObjcIdType above; replace "id"
/// with "Protocol".
QualType Sema::GetObjcProtoType(SourceLocation Loc) {
  assert(TUScope && "GetObjcProtoType(): Top-level scope is null");
  if (Context.getObjcProtoType().isNull()) {
    IdentifierInfo *ProtoIdent = &Context.Idents.get("Protocol");
    ScopedDecl *ProtoDecl = LookupScopedDecl(ProtoIdent, Decl::IDNS_Ordinary, 
                                           SourceLocation(), TUScope);
    TypedefDecl *ObjcProtoTypedef = dyn_cast_or_null<TypedefDecl>(ProtoDecl);
    if (!ObjcProtoTypedef) {
      Diag(Loc, diag::err_missing_proto_definition);
      return QualType();
    }
    Context.setObjcProtoType(ObjcProtoTypedef);
  }
  return Context.getObjcProtoType();
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt)
  : PP(pp), Context(ctxt), CurFunctionDecl(0), CurMethodDecl(0) {
  
  // Get IdentifierInfo objects for known functions for which we
  // do extra checking.  
  IdentifierTable& IT = PP.getIdentifierTable();  

  KnownFunctionIDs[ id_printf  ] = &IT.get("printf");
  KnownFunctionIDs[ id_fprintf ] = &IT.get("fprintf");
  KnownFunctionIDs[ id_sprintf ] = &IT.get("sprintf");
  KnownFunctionIDs[ id_snprintf ] = &IT.get("snprintf");
  KnownFunctionIDs[ id_asprintf ] = &IT.get("asprintf");
  KnownFunctionIDs[ id_vsnprintf ] = &IT.get("vsnprintf");
  KnownFunctionIDs[ id_vasprintf ] = &IT.get("vasprintf");
  KnownFunctionIDs[ id_vfprintf ] = &IT.get("vfprintf");
  KnownFunctionIDs[ id_vsprintf ] = &IT.get("vsprintf");
  KnownFunctionIDs[ id_vprintf ] = &IT.get("vprintf");

  if (PP.getLangOptions().ObjC1) {
    // Synthesize "typedef struct objc_class *Class;"
    RecordDecl *ClassTag = new RecordDecl(Decl::Struct, SourceLocation(), 
                                          &IT.get("objc_class"), 0);
    QualType ClassT = Context.getPointerType(Context.getTagDeclType(ClassTag));
    TypedefDecl *ClassTypedef = new TypedefDecl(SourceLocation(),
                                                &Context.Idents.get("Class"),
                                                ClassT, 0);
    Context.setObjcClassType(ClassTypedef);
    
    // Synthesize "typedef struct objc_object { Class isa; } *id;"
    RecordDecl *ObjectTag = new RecordDecl(Decl::Struct, SourceLocation(), 
                                          &IT.get("objc_object"), 0);
    FieldDecl *IsaDecl = new FieldDecl(SourceLocation(), 0, 
                                       Context.getObjcClassType());
    ObjectTag->defineBody(&IsaDecl, 1);
    QualType ObjT = Context.getPointerType(Context.getTagDeclType(ObjectTag));
    TypedefDecl *IdTypedef = new TypedefDecl(SourceLocation(),
                                             &Context.Idents.get("id"),
                                             ObjT, 0);
    Context.setObjcIdType(IdTypedef);
    
    // Synthesize "typedef struct objc_selector *SEL;"
    RecordDecl *SelTag = new RecordDecl(Decl::Struct, SourceLocation(), 
                                          &IT.get("objc_selector"), 0);
    QualType SelT = Context.getPointerType(Context.getTagDeclType(SelTag));
    TypedefDecl *SelTypedef = new TypedefDecl(SourceLocation(),
                                              &Context.Idents.get("SEL"),
                                              SelT, 0);
    Context.setObjcSelType(SelTypedef);
  }
  TUScope = 0;
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
  PP.getDiagnostics().Report(Loc, DiagID);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg) {
  PP.getDiagnostics().Report(Loc, DiagID, &Msg, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2) {
  std::string MsgArr[] = { Msg1, Msg2 };
  PP.getDiagnostics().Report(Loc, DiagID, MsgArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, SourceRange Range) {
  PP.getDiagnostics().Report(Loc, DiagID, 0, 0, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                SourceRange Range) {
  PP.getDiagnostics().Report(Loc, DiagID, &Msg, 1, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, SourceRange Range) {
  std::string MsgArr[] = { Msg1, Msg2 };
  PP.getDiagnostics().Report(Loc, DiagID, MsgArr, 2, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID,
                SourceRange R1, SourceRange R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(Loc, DiagID, 0, 0, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                SourceRange R1, SourceRange R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(Loc, DiagID, &Msg, 1, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Range, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, SourceRange R1, SourceRange R2) {
  std::string MsgArr[] = { Msg1, Msg2 };
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(Range, DiagID, MsgArr, 2, RangeArr, 2);
  return true;
}

const LangOptions &Sema::getLangOptions() const {
  return PP.getLangOptions();
}
