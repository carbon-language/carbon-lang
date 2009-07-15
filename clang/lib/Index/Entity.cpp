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

#include "clang/Index/Entity.h"
#include "clang/Index/Program.h"
#include "ProgramImpl.h"
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
class EntityGetter : public DeclVisitor<EntityGetter, Entity *> {
  ProgramImpl &Prog;
  
public:
  EntityGetter(ProgramImpl &prog) : Prog(prog) { }
  
  Entity *get(Entity *Parent, DeclarationName Name);
  
  Entity *VisitNamedDecl(NamedDecl *D);
  Entity *VisitVarDecl(VarDecl *D);
  Entity *VisitFunctionDecl(FunctionDecl *D); 
};

}
}

Entity *EntityGetter::get(Entity *Parent, DeclarationName Name) {
  // FIXME: Only works for DeclarationNames that are identifiers.

  if (!Name.isIdentifier())
    return 0;

  IdentifierInfo *II = Name.getAsIdentifierInfo();
  ProgramImpl::IdEntryTy *Id =
      &Prog.getIdents().GetOrCreateValue(II->getName(),
                                         II->getName() + II->getLength());
  
  llvm::FoldingSetNodeID ID;
  Entity::Profile(ID, Parent, Id);
  
  ProgramImpl::EntitySetTy &Entities = Prog.getEntities();
  void *InsertPos = 0;
  if (Entity *Ent = Entities.FindNodeOrInsertPos(ID, InsertPos))
    return Ent;
  
  void *Buf = Prog.Allocate(sizeof(Entity));
  Entity *New = new (Buf) Entity(Parent, Id);
  Entities.InsertNode(New, InsertPos);
  return New;
}

Entity *EntityGetter::VisitNamedDecl(NamedDecl *D) {
  // FIXME: Function declarations that are inside functions ?
  if (!D->getDeclContext()->isFileContext())
    return 0;

  Entity *Parent = Visit(cast<Decl>(D->getDeclContext()));
  return get(Parent, D->getDeclName());
}

Entity *EntityGetter::VisitVarDecl(VarDecl *D) {
  // If it's static it cannot be referred to by another translation unit.
  if (D->getStorageClass() == VarDecl::Static)
    return 0;
  
  return VisitNamedDecl(D);
}

Entity *EntityGetter::VisitFunctionDecl(FunctionDecl *D) {
  // If it's static it cannot be refered to by another translation unit.
  if (D->getStorageClass() == FunctionDecl::Static)
    return 0;
  
  return VisitNamedDecl(D);
}

//===----------------------------------------------------------------------===//
// Entity Implementation
//===----------------------------------------------------------------------===//

/// \brief Find the Decl that can be referred to by this entity.
Decl *Entity::getDecl(ASTContext &AST) {
  DeclContext *DC =
    Parent == 0 ? AST.getTranslationUnitDecl()
                : cast<DeclContext>(Parent->getDecl(AST));
  if (!DC)
    return 0; // Couldn't get the parent context.
    
  ProgramImpl::IdEntryTy *Entry = static_cast<ProgramImpl::IdEntryTy *>(Id);
  IdentifierInfo &II = AST.Idents.get(Entry->getKeyData());
  
  DeclContext::lookup_result Res = DC->lookup(DeclarationName(&II));
  for (DeclContext::lookup_iterator I = Res.first, E = Res.second; I!=E; ++I) {
    if (!isa<TagDecl>(*I))
      return *I;
  }

  return 0; // Failed to find a decl using this Entity.
}

const char *Entity::getName(ASTContext &Ctx) {
  if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(getDecl(Ctx))) {
    return ND->getNameAsCString();
  }
  return 0;
}

/// \brief Get an Entity associated with the given Decl.
/// \returns Null if an Entity cannot refer to this Decl.
Entity *Entity::get(Decl *D, Program &Prog) {
  assert(D && "Passed null Decl");
  ProgramImpl &Impl = *static_cast<ProgramImpl*>(Prog.Impl);
  return EntityGetter(Impl).Visit(D);
}
