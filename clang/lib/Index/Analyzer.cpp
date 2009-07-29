//===--- Analyzer.cpp - Analysis for indexing information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Analyzer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/Analyzer.h"
#include "clang/Index/Entity.h"
#include "clang/Index/TranslationUnit.h"
#include "clang/Index/Handlers.h"
#include "clang/Index/ASTLocation.h"
#include "clang/Index/DeclReferenceMap.h"
#include "clang/Index/IndexProvider.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace idx;

namespace  {

//===----------------------------------------------------------------------===//
// DeclEntityAnalyzer Implementation
//===----------------------------------------------------------------------===//

class VISIBILITY_HIDDEN DeclEntityAnalyzer : public TranslationUnitHandler {
  Entity Ent;
  TULocationHandler &TULocHandler;
  
public:
  DeclEntityAnalyzer(Entity ent, TULocationHandler &handler)
    : Ent(ent), TULocHandler(handler) { }
  
  virtual void Handle(TranslationUnit *TU) {
    assert(TU && "Passed null translation unit");

    Decl *D = Ent.getDecl(TU->getASTContext());
    assert(D && "Couldn't resolve Entity");

    for (Decl::redecl_iterator I = D->redecls_begin(),
                               E = D->redecls_end(); I != E; ++I)
      TULocHandler.Handle(TULocation(TU, ASTLocation(*I)));
  }
};

//===----------------------------------------------------------------------===//
// RefEntityAnalyzer Implementation
//===----------------------------------------------------------------------===//

class VISIBILITY_HIDDEN RefEntityAnalyzer : public TranslationUnitHandler {
  Entity Ent;
  TULocationHandler &TULocHandler;
  
public:
  RefEntityAnalyzer(Entity ent, TULocationHandler &handler)
    : Ent(ent), TULocHandler(handler) { }
  
  virtual void Handle(TranslationUnit *TU) {
    assert(TU && "Passed null translation unit");

    Decl *D = Ent.getDecl(TU->getASTContext());
    assert(D && "Couldn't resolve Entity");
    NamedDecl *ND = dyn_cast<NamedDecl>(D);
    if (!ND)
      return;

    DeclReferenceMap &RefMap = TU->getDeclReferenceMap();
    for (DeclReferenceMap::astlocation_iterator
           I = RefMap.refs_begin(ND), E = RefMap.refs_end(ND); I != E; ++I)
      TULocHandler.Handle(TULocation(TU, ASTLocation(*I)));
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Analyzer Implementation
//===----------------------------------------------------------------------===//

void Analyzer::FindDeclarations(Decl *D, TULocationHandler &Handler) {
  assert(D && "Passed null declaration");
  Entity Ent = Entity::get(D, Prog);
  if (Ent.isInvalid())
    return;
  
  DeclEntityAnalyzer DEA(Ent, Handler);
  Idxer.GetTranslationUnitsFor(Ent, DEA);
}

void Analyzer::FindReferences(Decl *D, TULocationHandler &Handler) {
  assert(D && "Passed null declaration");
  Entity Ent = Entity::get(D, Prog);
  if (Ent.isInvalid())
    return;
  
  RefEntityAnalyzer REA(Ent, Handler);
  Idxer.GetTranslationUnitsFor(Ent, REA);
}
