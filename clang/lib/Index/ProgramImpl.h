//===--- ProgramImpl.h - Internal Program implementation---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Internal implementation for the Program class
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_PROGRAMIMPL_H
#define LLVM_CLANG_INDEX_PROGRAMIMPL_H

#include "EntityImpl.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"

namespace clang {

namespace idx {
  class EntityListener;

class ProgramImpl {
public:
  typedef llvm::FoldingSet<EntityImpl> EntitySetTy;

private:
  EntitySetTy Entities;
  llvm::BumpPtrAllocator BumpAlloc;

  IdentifierTable Identifiers;
  SelectorTable Selectors;

  ProgramImpl(const ProgramImpl&); // do not implement
  ProgramImpl &operator=(const ProgramImpl &); // do not implement

public:
  ProgramImpl() : Identifiers(LangOptions()) { }

  EntitySetTy &getEntities() { return Entities; }
  IdentifierTable &getIdents() { return Identifiers; }
  SelectorTable &getSelectors() { return Selectors; }

  void *Allocate(unsigned Size, unsigned Align = 8) {
    return BumpAlloc.Allocate(Size, Align);
  }
};

} // namespace idx

} // namespace clang

#endif
