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
#include "clang/AST/ASTContext.h"
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
    Decl *D = static_cast<Decl*>(*I);
    assert(D && "This decl didn't get pushed??");
    IdentifierInfo *II = D->getIdentifier();
    if (!II) continue;
    
    // Unlink this decl from the identifier.  Because the scope contains decls
    // in an unordered collection, and because we have multiple identifier
    // namespaces (e.g. tag, normal, label),the decl may not be the first entry.
    if (II->getFETokenInfo<Decl>() == D) {
      // Normal case, no multiple decls in different namespaces.
      II->setFETokenInfo(D->getNext());
    } else {
      // Scan ahead.  There are only three namespaces in C, so this loop can
      // never execute more than 3 times.
      Decl *SomeDecl = II->getFETokenInfo<Decl>();
      while (SomeDecl->getNext() != D) {
        SomeDecl = SomeDecl->getNext();
        assert(SomeDecl && "Didn't find this decl on its identifier's chain!");
      }
      SomeDecl->setNext(D->getNext());
    }
    
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

/// LookupScopedDecl - Look up the inner-most declaration in the specified
/// namespace.
static Decl *LookupScopedDecl(IdentifierInfo *II, Decl::IdentifierNamespace NS){
  if (II == 0) return 0;
  
  // Scan up the scope chain looking for a decl that matches this identifier
  // that is in the appropriate namespace.  This search should not take long, as
  // shadowing of names is uncommon, and deep shadowing is extremely uncommon.
  for (Decl *D = II->getFETokenInfo<Decl>(); D; D = D->getNext())
    if (D->getIdentifierNamespace() == NS)
      return D;
  return 0;
}


Action::DeclTy *
Sema::ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init, 
                      DeclTy *LastInGroup) {
  IdentifierInfo *II = D.getIdentifier();
  
  if (Decl *PrevDecl = LookupScopedDecl(II, Decl::IDNS_Ordinary)) {
    // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
    if (S->isDeclScope(PrevDecl)) {
      // TODO: This is totally simplistic.  It should handle merging functions
      // together etc, merging extern int X; int X; ...
      Diag(D.getIdentifierLoc(), diag::err_redefinition, II->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
  }
  
  Decl *New;
  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef)
    New = ParseTypedefDecl(S, D);
  else if (D.isFunctionDeclarator())
    New = new FunctionDecl(D.getIdentifierLoc(), II, GetTypeForDeclarator(D,S));
  else
    New = new VarDecl(D.getIdentifierLoc(), II, GetTypeForDeclarator(D, S));
  
  if (!New) return 0;
  
  
  // If this has an identifier, add it to the scope stack.
  if (II) {
    New->setNext(II->getFETokenInfo<Decl>());
    II->setFETokenInfo(New);
    S->AddDecl(New);
  }
  
  // If this is a top-level decl that is chained to some other (e.g. int A,B,C;)
  // remember this in the LastInGroupList list.
  if (LastInGroup && S->getParent() == 0)
    LastInGroupList.push_back((Decl*)LastInGroup);
  
  return New;
}

VarDecl *
Sema::ParseParamDeclarator(DeclaratorChunk &FTI, unsigned ArgNo,
                           Scope *FnScope) {
  const DeclaratorChunk::ParamInfo &PI = FTI.Fun.ArgInfo[ArgNo];

  IdentifierInfo *II = PI.Ident;
  if (Decl *PrevDecl = LookupScopedDecl(II, Decl::IDNS_Ordinary)) {
    
    // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  }
  
  VarDecl *New = new VarDecl(PI.IdentLoc, II, static_cast<Type*>(PI.TypeInfo));
  
  // If this has an identifier, add it to the scope stack.
  if (II) {
    New->setNext(II->getFETokenInfo<Decl>());
    II->setFETokenInfo(New);
    FnScope->AddDecl(New);
  }

  return New;
}
  

Sema::DeclTy *Sema::ParseStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
  assert(CurFunctionDecl == 0 && "Function parsing confused");
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;
  
  // Verify 6.9.1p6: 'every identifier in the identifier list shall be declared'
  // for a K&R function.
  if (!FTI.hasPrototype) {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
      if (FTI.ArgInfo[i].TypeInfo == 0) {
        Diag(FTI.ArgInfo[i].IdentLoc, diag::err_param_not_declared,
             FTI.ArgInfo[i].Ident->getName());
        // Implicitly declare the argument as type 'int' for lack of a better
        // type.
        FTI.ArgInfo[i].TypeInfo = Context.IntTy.getAsOpaquePtr();
      }
    }
   
    // Since this is a function definition, act as though we have information
    // about the arguments.
    FTI.hasPrototype = true;
  } else {
    // FIXME: Diagnose arguments without names in C.
    
  }
  
  Scope *GlobalScope = FnBodyScope->getParent();
  
  FunctionDecl *FD =
    static_cast<FunctionDecl*>(ParseDeclarator(GlobalScope, D, 0, 0));
  CurFunctionDecl = FD;
  
  // Create Decl objects for each parameter, adding them to the FunctionDecl.
  SmallVector<VarDecl*, 16> Params;
  
  // Check for C99 6.7.5.3p10 - foo(void) is a non-varargs function that takes
  // no arguments, not a function that takes a single void argument.
  if (FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
      FTI.ArgInfo[0].TypeInfo == Context.VoidTy.getAsOpaquePtr()) {
    // empty arg list, don't push any params.
  } else {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
      Params.push_back(ParseParamDeclarator(D.getTypeObject(0), i,FnBodyScope));
  }
  
  FD->setParams(&Params[0], Params.size());
  
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
  D.AddTypeInfo(DeclaratorChunk::getFunction(false, false, 0, 0, Loc));
  D.SetIdentifier(&II, Loc);
  
  Decl *Result = static_cast<Decl*>(ParseDeclarator(S, D, 0, 0));
  
  // Visit this implicit declaration like any other top-level form.
  LastInGroupList.push_back(Result);
  return Result;
}


Decl *Sema::ParseTypedefDecl(Scope *S, Declarator &D) {
  assert(D.getIdentifier() && "Wrong callback for declspec withotu declarator");
  
  TypeRef T = GetTypeForDeclarator(D, S);
  if (T.isNull()) return 0;
  
  // Scope manipulation handled by caller.
  return new TypedefDecl(D.getIdentifierLoc(), D.getIdentifier(), T);
}


/// ParseStructUnionTag - This is invoked when we see 'struct foo' or
/// 'struct {'.  In the former case, Name will be non-null.  In the later case,
/// Name will be null.  isUnion indicates whether this is a union or struct tag.
/// isUse indicates whether this is a use of a preexisting struct tag, or if it
/// is a definition or declaration of a new one.
Sema::DeclTy *Sema::ParseStructUnionTag(Scope *S, bool isUnion, bool isUse,
                                        SourceLocation KWLoc, 
                                        IdentifierInfo *Name,
                                        SourceLocation NameLoc) {
  // If this is a use of an existing tag, it must have a name.
  assert((isUse || Name != 0) && "Nameless record must have a name!");
  
  // If this is a named struct, check to see if there was a previous forward
  // declaration or definition.
  if (Decl *PrevDecl = LookupScopedDecl(Name, Decl::IDNS_Tag)) {
    
    // If this is a use of a previous tag, or if the tag is already declared in
    // the same scope (so that the definition/declaration completes or
    // rementions the tag), reuse the decl.
    if (isUse || S->isDeclScope(PrevDecl)) {
      
      
    }
    
    // TODO: verify it's struct/union, etc.
    
    
    
  }
  
  // If there is an identifier, use the location of the identifier as the
  // location of the decl, otherwise use the location of the struct/union
  // keyword.
  SourceLocation Loc = NameLoc.isValid() ? NameLoc : KWLoc;
  
  // Otherwise, if this is the first time we've seen this tag, create the decl.
  Decl *New = new RecordDecl(isUnion ? Decl::Union : Decl::Struct, Loc, Name);
  
  // If this has an identifier, add it to the scope stack.
  if (Name) {
    New->setNext(Name->getFETokenInfo<Decl>());
    Name->setFETokenInfo(New);
    S->AddDecl(New);
  }
  
  return New;
}
