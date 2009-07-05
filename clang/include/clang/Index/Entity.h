//===--- Entity.h - Cross-translation-unit "token" for decls ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Entity is a ASTContext-independent way to refer to declarations that are
//  visible across translation units.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_ENTITY_H
#define LLVM_CLANG_INDEX_ENTITY_H

#include "llvm/ADT/FoldingSet.h"

namespace clang {
  class ASTContext;
  class Decl;

namespace idx {
  class Program;

/// \brief A ASTContext-independent way to refer to declarations that are
/// visible across translation units.
///
/// Entity is basically the link for declarations that are semantically the same
/// in multiple ASTContexts. A client will convert a Decl into an Entity and
/// later use that Entity to find the "same" Decl into another ASTContext.
///
/// An Entity may only refer to declarations that can be visible by multiple
/// translation units, e.g. a static function cannot have an Entity associated
/// with it.
///
/// Entities are uniqued so pointer equality can be used (note that the same
/// Program object should be used when getting Entities).
///
class Entity : public llvm::FoldingSetNode {
public:
  /// \brief Find the Decl that can be referred to by this entity.
  Decl *getDecl(ASTContext &AST);

  /// \brief Get an Entity associated with the given Decl.
  /// \returns Null if an Entity cannot refer to this Decl.
  static Entity *get(Decl *D, Program &Prog);

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, Parent, Id);
  }
  static void Profile(llvm::FoldingSetNodeID &ID, Entity *Parent, void *Id) {
    ID.AddPointer(Parent);
    ID.AddPointer(Id);
  }
  
private:
  Entity *Parent;
  void *Id;
  
  Entity(Entity *parent, void *id) : Parent(parent), Id(id) { }
  friend class EntityGetter;
};
  
} // namespace idx

} // namespace clang

#endif
