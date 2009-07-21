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
  ProgramImpl &Prog;

public:
  EntityGetter(ProgramImpl &prog) : Prog(prog) { }
  
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

  // FIXME: Only works for DeclarationNames that are identifiers.

  DeclarationName Name = D->getDeclName();

  if (!Name.isIdentifier())
    return Entity();

  IdentifierInfo *II = Name.getAsIdentifierInfo();
  if (!II)
      return Entity();

  EntityImpl::IdEntryTy *Id =
      &Prog.getIdents().GetOrCreateValue(II->getName(),
                                         II->getName() + II->getLength());
  unsigned IdNS = D->getIdentifierNamespace();

  llvm::FoldingSetNodeID ID;
  EntityImpl::Profile(ID, Parent, Id, IdNS);

  ProgramImpl::EntitySetTy &Entities = Prog.getEntities();
  void *InsertPos = 0;
  if (EntityImpl *Ent = Entities.FindNodeOrInsertPos(ID, InsertPos))
    return Entity(Ent);

  void *Buf = Prog.Allocate(sizeof(EntityImpl));
  EntityImpl *New = new (Buf) EntityImpl(Parent, Id, IdNS);
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
    
  IdentifierInfo &II = AST.Idents.get(Id->getKeyData());
  
  DeclContext::lookup_result Res = DC->lookup(DeclarationName(&II));
  for (DeclContext::lookup_iterator I = Res.first, E = Res.second; I!=E; ++I) {
    if ((*I)->getIdentifierNamespace() == IdNS)
      return *I;
  }

  return 0; // Failed to find a decl using this Entity.
}

/// \brief Get an Entity associated with the given Decl.
/// \returns Null if an Entity cannot refer to this Decl.
Entity EntityImpl::get(Decl *D, ProgramImpl &Prog) {
  assert(D && "Passed null Decl");
  return EntityGetter(Prog).Visit(D);
}

//===----------------------------------------------------------------------===//
// Entity Implementation
//===----------------------------------------------------------------------===//

/// \brief Find the Decl that can be referred to by this entity.
Decl *Entity::getDecl(ASTContext &AST) {
  if (isInvalid())
    return 0;
  
  if (Decl *D = Val.dyn_cast<Decl *>())
    // Check that the passed AST is actually the one that this Decl belongs to.
    return (&D->getASTContext() == &AST) ? D : 0;
  
  return Val.get<EntityImpl *>()->getDecl(AST);
}

std::string Entity::getPrintableName(ASTContext &Ctx) {
  if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(getDecl(Ctx))) {
    return ND->getNameAsString();
  }
  return std::string();
}

/// \brief Get an Entity associated with the given Decl.
/// \returns Null if an Entity cannot refer to this Decl.
Entity Entity::get(Decl *D, Program &Prog) {
  if (D == 0)
    return Entity();
  ProgramImpl &ProgImpl = *static_cast<ProgramImpl*>(Prog.Impl);
  return EntityImpl::get(D, ProgImpl);
}

unsigned 
llvm::DenseMapInfo<Entity>::getHashValue(Entity E) {
  return DenseMapInfo<void*>::getHashValue(E.getAsOpaquePtr());
}
