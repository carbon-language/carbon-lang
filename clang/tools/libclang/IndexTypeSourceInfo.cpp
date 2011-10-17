//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"

#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;
using namespace cxindex;

namespace {

class TypeIndexer : public RecursiveASTVisitor<TypeIndexer> {
  IndexingContext &IndexCtx;
  const NamedDecl *Parent;
  const DeclContext *ParentDC;

public:
  TypeIndexer(IndexingContext &indexCtx, const NamedDecl *parent,
              const DeclContext *DC)
    : IndexCtx(indexCtx), Parent(parent), ParentDC(DC) { }
  
  bool shouldWalkTypesOfTypeLocs() const { return false; }

  bool VisitTypedefTypeLoc(TypedefTypeLoc TL) {
    IndexCtx.handleReference(TL.getTypedefNameDecl(), TL.getNameLoc(),
                             Parent, ParentDC);
    return true;
  }

  bool VisitTagTypeLoc(TagTypeLoc TL) {
    TagDecl *D = TL.getDecl();

    if (TL.isDefinition()) {
      IndexCtx.indexTagDecl(D);
      return true;
    }

    if (D->getLocation() == TL.getNameLoc())
      IndexCtx.handleTagDecl(D);
    else
      IndexCtx.handleReference(D, TL.getNameLoc(),
                               Parent, ParentDC);
    return true;
  }

  bool VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
    IndexCtx.handleReference(TL.getIFaceDecl(), TL.getNameLoc(),
                             Parent, ParentDC);
    return true;
  }

  bool VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
    for (unsigned i = 0, e = TL.getNumProtocols(); i != e; ++i) {
      IndexCtx.handleReference(TL.getProtocol(i), TL.getProtocolLoc(i),
                               Parent, ParentDC);
    }
    return true;
  }
};

} // anonymous namespace

void IndexingContext::indexTypeSourceInfo(TypeSourceInfo *TInfo,
                                          const NamedDecl *Parent,
                                          const DeclContext *DC) {
  if (!TInfo || TInfo->getTypeLoc().isNull())
    return;
  
  if (DC == 0)
    DC = Parent->getDeclContext();
  indexTypeLoc(TInfo->getTypeLoc(), Parent, DC);
}

void IndexingContext::indexTypeLoc(TypeLoc TL,
                                   const NamedDecl *Parent,
                                   const DeclContext *DC) {
  TypeIndexer(*this, Parent, DC).TraverseTypeLoc(TL);
}

void IndexingContext::indexTagDecl(const TagDecl *D) {
  handleTagDecl(D);
  if (D->isThisDeclarationADefinition()) {
    invokeStartedTagTypeDefinition(D);
    indexDeclContext(D);
    invokeEndedContainer(D);
  }
}
