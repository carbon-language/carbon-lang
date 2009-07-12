//===--- IndexProvider.cpp - Map of entities to translation units ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSaE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Maps Entities to TranslationUnits
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexProvider.h"
#include "clang/Index/Program.h"
#include "clang/Index/EntityHandler.h"
#include "clang/Index/TranslationUnit.h"
using namespace clang;
using namespace idx;

class IndexProvider::Indexer : public EntityHandler {
  TranslationUnit *TU;
  MapTy &Map;
  
public:
  Indexer(TranslationUnit *tu, MapTy &map) : TU(tu), Map(map) { }

  virtual void HandleEntity(Entity *Ent) {
    MapTy::iterator I = Map.find(Ent);
    if (I != Map.end()) {
      I->second.insert(TU);
      return;
    }
    
    Map[Ent].insert(TU);
  }
};

void IndexProvider::IndexAST(TranslationUnit *TU) {
  Indexer Idx(TU, Map);
  Prog.FindEntities(TU->getASTContext(), &Idx);
}

IndexProvider::translation_unit_iterator
IndexProvider::translation_units_begin(Entity *Ent) const {
  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return translation_unit_iterator(0);
  
  return I->second.begin();
}

IndexProvider::translation_unit_iterator
IndexProvider::translation_units_end(Entity *Ent) const {
  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return translation_unit_iterator(0);
  
  return I->second.end();
}

bool IndexProvider::translation_units_empty(Entity *Ent) const {
  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return true;

  return I->second.begin() == I->second.end();
}
