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
#include "clang/Analysis/Support/SaveAndRestore.h"

using namespace clang;
using namespace cxindex;

namespace {

class BodyIndexer : public RecursiveASTVisitor<BodyIndexer> {
  IndexingContext &IndexCtx;
  const DeclContext *ParentDC;
  bool InPseudoObject;

  typedef RecursiveASTVisitor<BodyIndexer> base;
public:
  BodyIndexer(IndexingContext &indexCtx, const DeclContext *DC)
    : IndexCtx(indexCtx), ParentDC(DC), InPseudoObject(false) { }
  
  bool shouldWalkTypesOfTypeLocs() const { return false; }

  bool TraverseTypeLoc(TypeLoc TL) {
    IndexCtx.indexTypeLoc(TL, 0, ParentDC);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    IndexCtx.handleReference(E->getDecl(), E->getLocation(), 0, ParentDC, E);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *E) {
    IndexCtx.handleReference(E->getMemberDecl(), E->getMemberLoc(), 0, ParentDC,
                             E);
    return true;
  }

  bool VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
    IndexCtx.handleReference(E->getDecl(), E->getLocation(), 0, ParentDC, E);
    return true;
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (TypeSourceInfo *Cls = E->getClassReceiverTypeInfo())
      IndexCtx.indexTypeSourceInfo(Cls, 0, ParentDC);

    if (ObjCMethodDecl *MD = E->getMethodDecl())
      IndexCtx.handleReference(MD, E->getSelectorStartLoc(), 0, ParentDC, E,
                               InPseudoObject ? CXIdxEntityRef_Implicit
                                              : CXIdxEntityRef_Direct);
    return true;
  }

  bool VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
    if (E->isImplicitProperty()) {
      if (ObjCMethodDecl *MD = E->getImplicitPropertyGetter())
        IndexCtx.handleReference(MD, E->getLocation(), 0, ParentDC, E,
                                 CXIdxEntityRef_Implicit);
      if (ObjCMethodDecl *MD = E->getImplicitPropertySetter())
        IndexCtx.handleReference(MD, E->getLocation(), 0, ParentDC, E,
                                 CXIdxEntityRef_Implicit);
    } else {
      IndexCtx.handleReference(E->getExplicitProperty(), E->getLocation(), 0,
                               ParentDC, E);
    }
    return true;
  }

  bool TraversePseudoObjectExpr(PseudoObjectExpr *E) {
    SaveAndRestore<bool> InPseudo(InPseudoObject, true);
    return base::TraversePseudoObjectExpr(E);
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *E) {
    IndexCtx.handleReference(E->getConstructor(), E->getLocation(), 0,
                             ParentDC, E);
    return true;
  }
};

} // anonymous namespace

void IndexingContext::indexBody(const Stmt *S, const DeclContext *DC) {
  BodyIndexer(*this, DC).TraverseStmt(const_cast<Stmt*>(S));
}
