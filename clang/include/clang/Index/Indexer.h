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
#include "clang/Index/Entity.h"
#include "clang/Index/GlobalSelector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include <map>

namespace clang {
  class ASTContext;

namespace idx {
  class Program;
  class TranslationUnit;

/// \brief Maps information to TranslationUnits.
class Indexer : public IndexProvider {
public:
  typedef llvm::SmallPtrSet<TranslationUnit *, 4> TUSetTy;
  typedef llvm::DenseMap<ASTContext *, TranslationUnit *> CtxTUMapTy;
  typedef std::map<Entity, TUSetTy> MapTy;
  typedef std::map<GlobalSelector, TUSetTy> SelMapTy;

  explicit Indexer(Program &prog) :
    Prog(prog) { }

  Program &getProgram() const { return Prog; }

  /// \brief Find all Entities and map them to the given translation unit.
  void IndexAST(TranslationUnit *TU);

  virtual void GetTranslationUnitsFor(Entity Ent,
                                      TranslationUnitHandler &Handler);
  virtual void GetTranslationUnitsFor(GlobalSelector Sel,
                                      TranslationUnitHandler &Handler);

private:
  Program &Prog;

  MapTy Map;
  CtxTUMapTy CtxTUMap;
  SelMapTy SelMap;
};

} // namespace idx

} // namespace clang

#endif
