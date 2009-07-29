//===--- Indexer.cpp - IndexProvider implementation -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  IndexProvider implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/Indexer.h"
#include "clang/Index/Program.h"
#include "clang/Index/Entity.h"
#include "clang/Index/Handlers.h"
#include "clang/Index/TranslationUnit.h"
#include "clang/AST/DeclBase.h"
using namespace clang;
using namespace idx;

namespace {

class EntityIndexer : public EntityHandler {
  TranslationUnit *TU;
  Indexer::MapTy &Map;
  
public:
  EntityIndexer(TranslationUnit *tu, Indexer::MapTy &map) : TU(tu), Map(map) { }

  virtual void Handle(Entity Ent) {
    if (Ent.isInternalToTU())
      return;
    Map[Ent].insert(TU);
  }
};

} // anonymous namespace

void Indexer::IndexAST(TranslationUnit *TU) {
  assert(TU && "Passed null TranslationUnit");
  CtxTUMap[&TU->getASTContext()] = TU;
  EntityIndexer Idx(TU, Map);
  Prog.FindEntities(TU->getASTContext(), Idx);
}

void Indexer::GetTranslationUnitsFor(Entity Ent,
                                     TranslationUnitHandler &Handler) {
  assert(Ent.isValid() && "Expected valid Entity");

  if (Ent.isInternalToTU()) {
    Decl *D = Ent.getInternalDecl();
    CtxTUMapTy::iterator I = CtxTUMap.find(&D->getASTContext());
    if (I != CtxTUMap.end())
      Handler.Handle(I->second);
    return;
  }

  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return;
  
  TUSetTy &Set = I->second;
  for (TUSetTy::iterator I = Set.begin(), E = Set.end(); I != E; ++I)
    Handler.Handle(*I);
}
