//===--- Indexer.h - IndexProvider implementation ---------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_INDEX_INDEXER_H
#define LLVM_CLANG_INDEX_INDEXER_H

#include "clang/Index/IndexProvider.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>

namespace clang {

namespace idx {
  class Program;
  class TranslationUnit;

/// \brief Maps information to TranslationUnits.
class Indexer : public IndexProvider {
public:
  typedef llvm::SmallPtrSet<TranslationUnit *, 4> TUSetTy;
  typedef std::map<Entity, TUSetTy> MapTy;

  explicit Indexer(Program &prog) : Prog(prog) { }

  Program &getProgram() const { return Prog; }

  /// \brief Find all Entities and map them to the given translation unit.
  void IndexAST(TranslationUnit *TU);

  virtual void GetTranslationUnitsFor(Entity Ent,
                                      TranslationUnitHandler &Handler);

  typedef TUSetTy::iterator translation_unit_iterator;

  translation_unit_iterator translation_units_begin(Entity Ent) const;
  translation_unit_iterator translation_units_end(Entity Ent) const;
  bool translation_units_empty(Entity Ent) const;

private:
  Program &Prog;
  MapTy Map;
  CtxTUMapTy CtxTUMap;
};

} // namespace idx

} // namespace clang

#endif
