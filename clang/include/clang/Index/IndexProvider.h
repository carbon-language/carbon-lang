//===--- IndexProvider.h - Map of entities to translation units -*- C++ -*-===//
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

#ifndef LLVM_CLANG_INDEX_INDEXPROVIDER_H
#define LLVM_CLANG_INDEX_INDEXPROVIDER_H

#include "llvm/ADT/SmallPtrSet.h"
#include <map>

namespace clang {

namespace idx {
  class Program;
  class Entity;
  class TranslationUnit;

/// \brief Maps Entities to TranslationUnits.
class IndexProvider {
  typedef llvm::SmallPtrSet<TranslationUnit *, 4> TUSetTy;
  typedef std::map<Entity *, TUSetTy> MapTy;
  class Indexer;

public:
  explicit IndexProvider(Program &prog) : Prog(prog) { }

  Program &getProgram() const { return Prog; }

  /// \brief Find all Entities and map them to the given translation unit.
  void IndexAST(TranslationUnit *TU);

  typedef TUSetTy::iterator translation_unit_iterator;

  translation_unit_iterator translation_units_begin(Entity *Ent) const;
  translation_unit_iterator translation_units_end(Entity *Ent) const;
  bool translation_units_empty(Entity *Ent) const;
  
private:
  Program &Prog;
  mutable MapTy Map;
};

} // namespace idx

} // namespace clang

#endif
