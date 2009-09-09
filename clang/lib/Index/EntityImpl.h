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
#include "clang/AST/DeclarationName.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringSet.h"

namespace clang {

namespace idx {
  class ProgramImpl;

class EntityImpl : public llvm::FoldingSetNode {
  Entity Parent;
  DeclarationName Name;

  /// \brief Identifier namespace.
  unsigned IdNS;

  /// \brief If Name is a selector, this keeps track whether it's for an
  /// instance method.
  bool IsObjCInstanceMethod;

public:
  EntityImpl(Entity parent, DeclarationName name, unsigned idNS,
             bool isObjCInstanceMethod)
    : Parent(parent), Name(name), IdNS(idNS),
      IsObjCInstanceMethod(isObjCInstanceMethod) { }

  /// \brief Find the Decl that can be referred to by this entity.
  Decl *getDecl(ASTContext &AST);

  /// \brief Get an Entity associated with the given Decl.
  /// \returns Null if an Entity cannot refer to this Decl.
  static Entity get(Decl *D, Program &Prog, ProgramImpl &ProgImpl);

  std::string getPrintableName();

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, Parent, Name, IdNS, IsObjCInstanceMethod);
  }
  static void Profile(llvm::FoldingSetNodeID &ID, Entity Parent,
                      DeclarationName Name, unsigned IdNS,
                      bool isObjCInstanceMethod) {
    ID.AddPointer(Parent.getAsOpaquePtr());
    ID.AddPointer(Name.getAsOpaquePtr());
    ID.AddInteger(IdNS);
    ID.AddBoolean(isObjCInstanceMethod);
  }
};

} // namespace idx

} // namespace clang

#endif
