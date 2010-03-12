//===--- Entity.cpp - Cross-translation-unit "token" for decls ------------===//
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

#include "EntityImpl.h"
#include "ProgramImpl.h"
#include "clang/Index/Program.h"
#include "clang/Index/GlobalSelector.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
using namespace clang;
using namespace idx;

// FIXME: Entity is really really basic currently, mostly written to work
// on variables and functions. Should support types and other decls eventually..


//===----------------------------------------------------------------------===//
// EntityGetter
//===----------------------------------------------------------------------===//

namespace clang {
namespace idx {

/// \brief Gets the Entity associated with a Decl.
class EntityGetter : public DeclVisitor<EntityGetter, Entity> {
  Program &Prog;
  ProgramImpl &ProgImpl;

public:
  EntityGetter(Program &prog, ProgramImpl &progImpl)
    : Prog(prog), ProgImpl(progImpl) { }

  Entity VisitNamedDecl(NamedDecl *D);
  Entity VisitVarDecl(VarDecl *D);
  Entity VisitFunctionDecl(FunctionDecl *D);
};

}
}

Entity EntityGetter::VisitNamedDecl(NamedDecl *D) {
  Entity Parent;
  if (!D->getDeclContext()->isTranslationUnit()) {
    Parent = Visit(cast<Decl>(D->getDeclContext()));
    // FIXME: Anonymous structs ?
    if (Parent.isInvalid())
      return Entity();
  }
  if (Parent.isValid() && Parent.isInternalToTU())
    return Entity(D);

  // FIXME: Only works for DeclarationNames that are identifiers and selectors.
  // Treats other DeclarationNames as internal Decls for now..

  DeclarationName LocalName = D->getDeclName();
  if (!LocalName)
    return Entity(D);

  DeclarationName GlobName;

  if (IdentifierInfo *II = LocalName.getAsIdentifierInfo()) {
    IdentifierInfo *GlobII = &ProgImpl.getIdents().get(II->getName());
    GlobName = DeclarationName(GlobII);
  } else {
    Selector LocalSel = LocalName.getObjCSelector();

    // Treats other DeclarationNames as internal Decls for now..
    if (LocalSel.isNull())
      return Entity(D);

    Selector GlobSel =
        (uintptr_t)GlobalSelector::get(LocalSel, Prog).getAsOpaquePtr();
    GlobName = DeclarationName(GlobSel);
  }

  assert(GlobName);

  unsigned IdNS = D->getIdentifierNamespace();

  ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D);
  bool isObjCInstanceMethod = MD && MD->isInstanceMethod();

  llvm::FoldingSetNodeID ID;
  EntityImpl::Profile(ID, Parent, GlobName, IdNS, isObjCInstanceMethod);

  ProgramImpl::EntitySetTy &Entities = ProgImpl.getEntities();
  void *InsertPos = 0;
  if (EntityImpl *Ent = Entities.FindNodeOrInsertPos(ID, InsertPos))
    return Entity(Ent);

  void *Buf = ProgImpl.Allocate(sizeof(EntityImpl));
  EntityImpl *New =
      new (Buf) EntityImpl(Parent, GlobName, IdNS, isObjCInstanceMethod);
  Entities.InsertNode(New, InsertPos);

  return Entity(New);
}

Entity EntityGetter::VisitVarDecl(VarDecl *D) {
  // If it's static it cannot be referred to by another translation unit.
  if (D->getStorageClass() == VarDecl::Static)
    return Entity(D);

  return VisitNamedDecl(D);
}

Entity EntityGetter::VisitFunctionDecl(FunctionDecl *D) {
  // If it's static it cannot be refered to by another translation unit.
  if (D->getStorageClass() == FunctionDecl::Static)
    return Entity(D);

  return VisitNamedDecl(D);
}

//===----------------------------------------------------------------------===//
// EntityImpl Implementation
//===----------------------------------------------------------------------===//

Decl *EntityImpl::getDecl(ASTContext &AST) {
  DeclContext *DC =
    Parent.isInvalid() ? AST.getTranslationUnitDecl()
                       : cast<DeclContext>(Parent.getDecl(AST));
  if (!DC)
    return 0; // Couldn't get the parent context.

  DeclarationName LocalName;

  if (IdentifierInfo *GlobII = Name.getAsIdentifierInfo()) {
    IdentifierInfo &II = AST.Idents.get(GlobII->getName());
    LocalName = DeclarationName(&II);
  } else {
    Selector GlobSel = Name.getObjCSelector();
    assert(!GlobSel.isNull() && "A not handled yet declaration name");
    GlobalSelector GSel =
        GlobalSelector::getFromOpaquePtr(GlobSel.getAsOpaquePtr());
    LocalName = GSel.getSelector(AST);
  }

  assert(LocalName);

  DeclContext::lookup_result Res = DC->lookup(LocalName);
  for (DeclContext::lookup_iterator I = Res.first, E = Res.second; I!=E; ++I) {
    Decl *D = *I;
    if (D->getIdentifierNamespace() == IdNS) {
      if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
        if (MD->isInstanceMethod() == IsObjCInstanceMethod)
          return MD;
      } else
        return D;
    }
  }

  return 0; // Failed to find a decl using this Entity.
}

/// \brief Get an Entity associated with the given Decl.
/// \returns Null if an Entity cannot refer to this Decl.
Entity EntityImpl::get(Decl *D, Program &Prog, ProgramImpl &ProgImpl) {
  assert(D && "Passed null Decl");
  return EntityGetter(Prog, ProgImpl).Visit(D);
}

std::string EntityImpl::getPrintableName() {
  return Name.getAsString();
}

//===----------------------------------------------------------------------===//
// Entity Implementation
//===----------------------------------------------------------------------===//

Entity::Entity(Decl *D) : Val(D->getCanonicalDecl()) { }

/// \brief Find the Decl that can be referred to by this entity.
Decl *Entity::getDecl(ASTContext &AST) const {
  if (isInvalid())
    return 0;

  if (Decl *D = Val.dyn_cast<Decl *>())
    // Check that the passed AST is actually the one that this Decl belongs to.
    return (&D->getASTContext() == &AST) ? D : 0;

  return Val.get<EntityImpl *>()->getDecl(AST);
}

std::string Entity::getPrintableName() const {
  if (isInvalid())
    return "<< Invalid >>";

  if (Decl *D = Val.dyn_cast<Decl *>()) {
    if (NamedDecl *ND = dyn_cast<NamedDecl>(D))
      return ND->getNameAsString();
    else
      return std::string();
  }

  return Val.get<EntityImpl *>()->getPrintableName();
}

/// \brief Get an Entity associated with the given Decl.
/// \returns Null if an Entity cannot refer to this Decl.
Entity Entity::get(Decl *D, Program &Prog) {
  if (D == 0)
    return Entity();
  ProgramImpl &ProgImpl = *static_cast<ProgramImpl*>(Prog.Impl);
  return EntityImpl::get(D, Prog, ProgImpl);
}

unsigned
llvm::DenseMapInfo<Entity>::getHashValue(Entity E) {
  return DenseMapInfo<void*>::getHashValue(E.getAsOpaquePtr());
}
