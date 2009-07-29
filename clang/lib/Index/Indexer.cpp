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
using namespace clang;
using namespace idx;

namespace {

class EntityIndexer : public EntityHandler {
  TranslationUnit *TU;
  Indexer::MapTy &Map;
  
public:
  EntityIndexer(TranslationUnit *tu, Indexer::MapTy &map) : TU(tu), Map(map) { }

  virtual void HandleEntity(Entity Ent) {
    if (Ent.isInternalToTU())
      return;
    Map[Ent].insert(TU);
  }
};

} // anonymous namespace

void Indexer::IndexAST(TranslationUnit *TU) {
  EntityIndexer Idx(TU, Map);
  Prog.FindEntities(TU->getASTContext(), &Idx);
}

void Indexer::GetTranslationUnitsFor(Entity Ent,
                                     TranslationUnitHandler *Handler) {
  assert(Ent.isValid() && "Expected valid Entity");
  assert(!Ent.isInternalToTU() &&
         "Expected an Entity visible outside of its translation unit");

  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return;
  
  TUSetTy &Set = I->second;
  for (TUSetTy::iterator I = Set.begin(), E = Set.end(); I != E; ++I)
    Handler->Handle(*I);
}

static Indexer::TUSetTy EmptySet;

Indexer::translation_unit_iterator
Indexer::translation_units_begin(Entity Ent) const {
  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return EmptySet.begin();
  
   return I->second.begin();
}

Indexer::translation_unit_iterator
Indexer::translation_units_end(Entity Ent) const {
  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return EmptySet.end();

  return I->second.end();
}

bool Indexer::translation_units_empty(Entity Ent) const {
  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return true;

  return I->second.begin() == I->second.end();
}
