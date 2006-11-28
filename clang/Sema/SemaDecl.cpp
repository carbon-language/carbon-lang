//===--- SemaDecl.cpp - Semantic Analysis for Declarations ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for declarations.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
using namespace llvm;
using namespace clang;


Sema::DeclTy *Sema::isTypeName(const IdentifierInfo &II, Scope *S) const {
  return dyn_cast_or_null<TypeDecl>(II.getFETokenInfo<Decl>());
}

void Sema::PopScope(SourceLocation Loc, Scope *S) {
  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    IdentifierInfo &II = *static_cast<IdentifierInfo*>(*I);
    Decl *D = II.getFETokenInfo<Decl>();
    assert(D && "This decl didn't get pushed??");
    
    II.setFETokenInfo(D->getNext());
    
    // This will have to be revisited for C++: there we want to nest stuff in
    // namespace decls etc.  Even for C, we might want a top-level translation
    // unit decl or something.
    if (!CurFunctionDecl)
      continue;

    // Chain this decl to the containing function, it now owns the memory for
    // the decl.
    D->setNext(CurFunctionDecl->getDeclChain());
    CurFunctionDecl->setDeclChain(D);
  }
}

/// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
/// no declarator (e.g. "struct foo;") is parsed.
Sema::DeclTy *Sema::ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
  // TODO: emit error on 'int;' or 'const enum foo;'.
  // TODO: emit error on 'typedef int;'
  // if (!DS.isMissingDeclaratorOk()) Diag(...);
  
  // TODO: Register 'struct foo;' with the type system as an opaque struct.
  
  // TODO: Check that we don't already have 'union foo;' or something else
  // that conflicts.
  return 0;
}

Action::DeclTy *
Sema::ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init, 
                      DeclTy *LastInGroup) {
  IdentifierInfo *II = D.getIdentifier();
  Decl *PrevDecl = 0;
  
  if (II) {
    PrevDecl = II->getFETokenInfo<Decl>();
    
    // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  }
  
  Decl *New;
  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    New = ParseTypedefDecl(S, D, PrevDecl);
  } else if (D.isFunctionDeclarator())
    New = new FunctionDecl(II, GetTypeForDeclarator(D, S), PrevDecl);
  else
    New = new VarDecl(II, GetTypeForDeclarator(D, S), PrevDecl);
  
  if (!New) return 0;
  
  
  // If this has an identifier, add it to the scope stack.
  if (II) {
    // If PrevDecl includes conflicting name here, emit a diagnostic.
    II->setFETokenInfo(New);
    S->AddDecl(II);
  }
  
  // If this is a top-level decl that is chained to some other (e.g. int A,B,C;)
  // remember this in the LastInGroupList list.
  if (LastInGroup && S->getParent() == 0)
    LastInGroupList.push_back((Decl*)LastInGroup);
  
  return New;
}



Sema::DeclTy *Sema::ParseStartOfFunctionDef(Scope *S, Declarator &D
                                            /* TODO: FORMAL ARG INFO.*/) {
  assert(CurFunctionDecl == 0 && "Function parsing confused");
  
  FunctionDecl *FD = static_cast<FunctionDecl*>(ParseDeclarator(S, D, 0, 0));
  CurFunctionDecl = FD;
  return FD;
}

Sema::DeclTy *Sema::ParseFunctionDefBody(DeclTy *D, StmtTy *Body) {
  FunctionDecl *FD = static_cast<FunctionDecl*>(D);
  FD->setBody((Stmt*)Body);
  
  assert(FD == CurFunctionDecl && "Function parsing confused");
  CurFunctionDecl = 0;
  return FD;
}


/// ImplicitlyDefineFunction - An undeclared identifier was used in a function
/// call, forming a call to an implicitly defined function (per C99 6.5.1p2).
Decl *Sema::ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                     Scope *S) {
  if (getLangOptions().C99)  // Extension in C99.
    Diag(Loc, diag::ext_implicit_function_decl, II.getName());
  else  // Legal in C90, but warn about it.
    Diag(Loc, diag::warn_implicit_function_decl, II.getName());
  
  // FIXME: handle stuff like:
  // void foo() { extern float X(); }
  // void bar() { X(); }  <-- implicit decl for X in another scope.

  // Set a Declarator for the implicit definition: int foo();
  const char *Dummy;
  DeclSpec DS;
  bool Error = DS.SetTypeSpecType(DeclSpec::TST_int, Loc, Dummy);
  assert(!Error && "Error setting up implicit decl!");
  Declarator D(DS, Declarator::BlockContext);
  D.AddTypeInfo(DeclaratorTypeInfo::getFunction(false, false, true, Loc));
  D.SetIdentifier(&II, Loc);
  
  Decl *Result = static_cast<Decl*>(ParseDeclarator(S, D, 0, 0));
  
  // Visit this implicit declaration like any other top-level form.
  LastInGroupList.push_back(Result);
  return Result;
}


Decl *Sema::ParseTypedefDecl(Scope *S, Declarator &D, Decl *PrevDecl) {
  assert(D.getIdentifier() && "Wrong callback for declspec withotu declarator");
  
  TypeRef T = GetTypeForDeclarator(D, S);
  if (T.isNull()) return 0;
  
  return new TypedefDecl(D.getIdentifier(), T, PrevDecl);
}

