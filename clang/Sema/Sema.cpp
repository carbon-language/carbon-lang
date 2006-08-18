//===--- Builder.cpp - AST Builder Implementation -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which builds an AST out of a parse
// stream.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Action.h"
#include "clang/Parse/Scope.h"
#include "clang/AST/Decl.h"
#include "clang/Lex/IdentifierTable.h"
#include "llvm/Support/Visibility.h"
using namespace llvm;
using namespace clang;

/// ASTBuilder
namespace {
class VISIBILITY_HIDDEN ASTBuilder : public Action {
public:
  //===--------------------------------------------------------------------===//
  // Symbol table tracking callbacks.
  //
  virtual bool isTypedefName(const IdentifierInfo &II, Scope *S) const;
  virtual void ParseDeclarator(SourceLocation Loc, Scope *S, Declarator &D,
                               ExprTy *Init);
  virtual void PopScope(SourceLocation Loc, Scope *S);
};
} // end anonymous namespace


//===----------------------------------------------------------------------===//
// Symbol table tracking callbacks.
//===----------------------------------------------------------------------===//

bool ASTBuilder::isTypedefName(const IdentifierInfo &II, Scope *S) const {
  Decl *D = II.getFETokenInfo<Decl>();
  return D != 0 && D->getDeclSpecs().StorageClassSpec == DeclSpec::SCS_typedef;
}

void ASTBuilder::ParseDeclarator(SourceLocation Loc, Scope *S, Declarator &D,
                                 ExprTy *Init) {
  IdentifierInfo *II = D.getIdentifier();
  Decl *PrevDecl = II ? II->getFETokenInfo<Decl>() : 0;

  Decl *New = new Decl(II, D.getDeclSpec(), Loc, PrevDecl);
  
  // If this has an identifier, add it to the scope stack.
  if (II) {
    // If PrevDecl includes conflicting name here, emit a diagnostic.
    II->setFETokenInfo(New);
    S->AddDecl(II);
  }
}

void ASTBuilder::PopScope(SourceLocation Loc, Scope *S) {
  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    IdentifierInfo &II = *static_cast<IdentifierInfo*>(*I);
    Decl *D = II.getFETokenInfo<Decl>();
    assert(D && "This decl didn't get pushed??");
    
    Decl *Next = D->getNext();

    // FIXME: Push the decl on the parent function list if in a function.
    delete D;
    
    II.setFETokenInfo(Next);
  }
}


/// Interface to the Builder.cpp file.
///
Action *CreateASTBuilderActions() {
  return new ASTBuilder();
}



