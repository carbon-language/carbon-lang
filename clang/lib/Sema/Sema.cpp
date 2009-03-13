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
                                 llvm::SmallVectorImpl<char> &Output,
                                 void *Cookie) {
  ASTContext &Context = *static_cast<ASTContext*>(Cookie);
  
  std::string S;
  if (Kind == Diagnostic::ak_qualtype) {
    assert(ModLen == 0 && ArgLen == 0 &&
           "Invalid modifier for QualType argument");

    QualType Ty(QualType::getFromOpaquePtr(reinterpret_cast<void*>(Val)));

    // FIXME: Playing with std::string is really slow.
    S = Ty.getAsString();
    
    // If this is a sugared type (like a typedef, typeof, etc), then unwrap one
    // level of the sugar so that the type is more obvious to the user.
    QualType DesugaredTy = Ty->getDesugaredType();
    DesugaredTy.setCVRQualifiers(DesugaredTy.getCVRQualifiers() |
                                 Ty.getCVRQualifiers());

    if (Ty != DesugaredTy &&
        // If the desugared type is a vector type, we don't want to expand it,
        // it will turn into an attribute mess. People want their "vec4".
        !isa<VectorType>(DesugaredTy) &&
      
        // Don't desugar magic Objective-C types.
        Ty.getUnqualifiedType() != Context.getObjCIdType() &&
        Ty.getUnqualifiedType() != Context.getObjCSelType() &&
        Ty.getUnqualifiedType() != Context.getObjCProtoType() &&
        Ty.getUnqualifiedType() != Context.getObjCClassType() &&
        
        // Not va_list.
        Ty.getUnqualifiedType() != Context.getBuiltinVaListType()) {
      S = "'"+S+"' (aka '";
      S += DesugaredTy.getAsString();
      S += "')";
      Output.append(S.begin(), S.end());
      return;
    }
    
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
  
  Output.push_back('\'');
  Output.append(S.begin(), S.end());
  Output.push_back('\'');
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

  KnownFunctionIDs[id_NSLog]         = &IT.get("NSLog");
  KnownFunctionIDs[id_NSLogv]         = &IT.get("NSLogv");
  KnownFunctionIDs[id_asprintf]      = &IT.get("asprintf");
  KnownFunctionIDs[id_vasprintf]     = &IT.get("vasprintf");

  StdNamespace = 0;
  TUScope = 0;
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());
      
  // Tell diagnostics how to render things from the AST library.
  PP.getDiagnostics().SetArgToStringFn(ConvertArgToStringFn, &Context);
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
  if (!getLangOptions().CPlusPlus) {
    // Note: we traverse the scope's list of declarations rather than
    // the DeclContext's list, because we only want to see the most
    // recent declaration of each identifier.
    for (Scope::decl_iterator I = TUScope->decl_begin(), 
                           IEnd = TUScope->decl_end();
         I != IEnd; ++I) {
      Decl *D = static_cast<Decl *>(*I);
      if (D->isInvalidDecl())
        continue;

      if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
        if (VD->isTentativeDefinition(Context)) {
          if (const IncompleteArrayType *ArrayT 
                = Context.getAsIncompleteArrayType(VD->getType())) {
            if (RequireCompleteType(VD->getLocation(), 
                                    ArrayT->getElementType(),
                                 diag::err_tentative_def_incomplete_type_arr))
              VD->setInvalidDecl();
            else {
              // Set the length of the array to 1 (C99 6.9.2p5).
              llvm::APSInt One(Context.getTypeSize(Context.getSizeType()), 
                               true);
              QualType T 
                = Context.getConstantArrayType(ArrayT->getElementType(),
                                               One, ArrayType::Normal, 0);
              VD->setType(T);
            }
          } else if (RequireCompleteType(VD->getLocation(), VD->getType(), 
                                    diag::err_tentative_def_incomplete_type))
            VD->setInvalidDecl();
        }
      }
    }
  }
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

