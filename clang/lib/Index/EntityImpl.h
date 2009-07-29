//===--- EntityImpl.h - Internal Entity implementation---------*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Internal implementation for the Entity class
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_ENTITYIMPL_H
#define LLVM_CLANG_INDEX_ENTITYIMPL_H

#include "clang/Index/Entity.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringSet.h"

namespace clang {
  class IdentifierInfo;

namespace idx {
  class ProgramImpl;

class EntityImpl : public llvm::FoldingSetNode {
  Entity Parent;
  IdentifierInfo *Id;

  /// \brief Identifier namespace.
  unsigned IdNS;

public:
  EntityImpl(Entity parent, IdentifierInfo *id, unsigned idNS)
    : Parent(parent), Id(id), IdNS(idNS) { }

  /// \brief Find the Decl that can be referred to by this entity.
  Decl *getDecl(ASTContext &AST);

  /// \brief Get an Entity associated with the given Decl.
  /// \returns Null if an Entity cannot refer to this Decl.
  static Entity get(Decl *D, ProgramImpl &Prog);
  
  std::string getPrintableName();

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, Parent, Id, IdNS);
  }
  static void Profile(llvm::FoldingSetNodeID &ID, Entity Parent,
                      IdentifierInfo *Id, unsigned IdNS) {
    ID.AddPointer(Parent.getAsOpaquePtr());
    ID.AddPointer(Id);
    ID.AddInteger(IdNS);
  }
};

} // namespace idx

} // namespace clang

#endif
