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

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
  class ASTContext;
  class Decl;

namespace idx {
  class Program;
  class EntityImpl;

/// \brief A ASTContext-independent way to refer to declarations.
///
/// Entity is basically the link for declarations that are semantically the same
/// in multiple ASTContexts. A client will convert a Decl into an Entity and
/// later use that Entity to find the "same" Decl into another ASTContext.
/// Declarations that are semantically the same and visible across translation
/// units will be associated with the same Entity.
///
/// An Entity may also refer to declarations that cannot be visible across
/// translation units, e.g. static functions with the same name in multiple
/// translation units will be associated with different Entities.
///
/// Entities can be checked for equality but note that the same Program object
/// should be used when getting Entities.
///
class Entity {
  /// \brief Stores the Decl directly if it is not visible outside of its own
  /// translation unit, otherwise it stores the associated EntityImpl.
  llvm::PointerUnion<Decl *, EntityImpl *> Val;

  explicit Entity(Decl *D);
  explicit Entity(EntityImpl *impl) : Val(impl) { }
  friend class EntityGetter;

public:
  Entity() { }

  /// \brief Find the Decl that can be referred to by this entity.
  Decl *getDecl(ASTContext &AST) const;

  /// \brief If this Entity represents a declaration that is internal to its
  /// translation unit, getInternalDecl() returns it.
  Decl *getInternalDecl() const {
    assert(isInternalToTU() && "This Entity is not internal!");
    return Val.get<Decl *>();
  }

  /// \brief Get a printable name for debugging purpose.
  std::string getPrintableName() const;

  /// \brief Get an Entity associated with the given Decl.
  /// \returns invalid Entity if an Entity cannot refer to this Decl.
  static Entity get(Decl *D, Program &Prog);

  /// \brief Get an Entity associated with a name in the global namespace.
  static Entity get(StringRef Name, Program &Prog);

  /// \brief true if the Entity is not visible outside the trasnlation unit.
  bool isInternalToTU() const {
    assert(isValid() && "This Entity is not valid!");
    return Val.is<Decl *>();
  }

  bool isValid() const { return !Val.isNull(); }
  bool isInvalid() const { return !isValid(); }

  void *getAsOpaquePtr() const { return Val.getOpaqueValue(); }
  static Entity getFromOpaquePtr(void *Ptr) {
    Entity Ent;
    Ent.Val = llvm::PointerUnion<Decl *, EntityImpl *>::getFromOpaqueValue(Ptr);
    return Ent;
  }

  friend bool operator==(const Entity &LHS, const Entity &RHS) {
    return LHS.getAsOpaquePtr() == RHS.getAsOpaquePtr();
  }

  // For use in a std::map.
  friend bool operator < (const Entity &LHS, const Entity &RHS) {
    return LHS.getAsOpaquePtr() < RHS.getAsOpaquePtr();
  }

  // For use in DenseMap/DenseSet.
  static Entity getEmptyMarker() {
    Entity Ent;
    Ent.Val =
      llvm::PointerUnion<Decl *, EntityImpl *>::getFromOpaqueValue((void*)-1);
    return Ent;
  }
  static Entity getTombstoneMarker() {
    Entity Ent;
    Ent.Val =
      llvm::PointerUnion<Decl *, EntityImpl *>::getFromOpaqueValue((void*)-2);
    return Ent;
  }
};

} // namespace idx

} // namespace clang

namespace llvm {
/// Define DenseMapInfo so that Entities can be used as keys in DenseMap and
/// DenseSets.
template<>
struct DenseMapInfo<clang::idx::Entity> {
  static inline clang::idx::Entity getEmptyKey() {
    return clang::idx::Entity::getEmptyMarker();
  }

  static inline clang::idx::Entity getTombstoneKey() {
    return clang::idx::Entity::getTombstoneMarker();
  }

  static unsigned getHashValue(clang::idx::Entity);

  static inline bool
  isEqual(clang::idx::Entity LHS, clang::idx::Entity RHS) {
    return LHS == RHS;
  }
};
  
template <>
struct isPodLike<clang::idx::Entity> { static const bool value = true; };

}  // end namespace llvm

#endif
