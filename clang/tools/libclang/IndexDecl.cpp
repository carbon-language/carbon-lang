//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"

#include "clang/AST/DeclVisitor.h"

using namespace clang;
using namespace cxindex;

namespace {

class IndexingDeclVisitor : public DeclVisitor<IndexingDeclVisitor, bool> {
  IndexingContext &IndexCtx;

public:
  explicit IndexingDeclVisitor(IndexingContext &indexCtx)
    : IndexCtx(indexCtx) { }

  bool VisitFunctionDecl(FunctionDecl *D) {
    IndexCtx.handleFunction(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.invokeStartedStatementBody(D, D);
        IndexCtx.indexBody(Body, D);
        IndexCtx.invokeEndedContainer(D);
      }
    }
    return true;
  }

  bool VisitVarDecl(VarDecl *D) {
    IndexCtx.handleVar(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitFieldDecl(FieldDecl *D) {
    IndexCtx.handleField(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }
  
  bool VisitEnumConstantDecl(EnumConstantDecl *D) {
    IndexCtx.handleEnumerator(D);
    return true;
  }

  bool VisitTypedefDecl(TypedefDecl *D) {
    IndexCtx.handleTypedef(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitTagDecl(TagDecl *D) {
    // Non-free standing tags are handled in indexTypeSourceInfo.
    if (D->isFreeStanding())
      IndexCtx.indexTagDecl(D);
    return true;
  }

  bool VisitObjCClassDecl(ObjCClassDecl *D) {
    ObjCClassDecl::ObjCClassRef *Ref = D->getForwardDecl();
    if (Ref->getInterface()->getLocation() == Ref->getLocation()) {
      IndexCtx.handleObjCInterface(Ref->getInterface());
    } else {
      IndexCtx.handleReference(Ref->getInterface(),
                               Ref->getLocation(),
                               0,
                               Ref->getInterface()->getDeclContext());
    }
    return true;
  }

  bool VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
    ObjCForwardProtocolDecl::protocol_loc_iterator LI = D->protocol_loc_begin();
    for (ObjCForwardProtocolDecl::protocol_iterator
           I = D->protocol_begin(), E = D->protocol_end(); I != E; ++I, ++LI) {
      SourceLocation Loc = *LI;
      ObjCProtocolDecl *PD = *I;

      if (PD->getLocation() == Loc) {
        IndexCtx.handleObjCProtocol(PD);
      } else {
        IndexCtx.handleReference(PD, Loc, 0, PD->getDeclContext());
      }
    }
    return true;
  }

  bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
    // Only definitions are handled here.
    if (D->isForwardDecl())
      return true;

    if (!D->isInitiallyForwardDecl())
      IndexCtx.handleObjCInterface(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.invokeStartedObjCContainer(D);
    IndexCtx.defineObjCInterface(D);
    IndexCtx.indexDeclContext(D);
    IndexCtx.invokeEndedContainer(D);
    return true;
  }

  bool VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
    // Only definitions are handled here.
    if (D->isForwardDecl())
      return true;

    if (!D->isInitiallyForwardDecl())
      IndexCtx.handleObjCProtocol(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.invokeStartedObjCContainer(D);
    IndexCtx.indexDeclContext(D);
    IndexCtx.invokeEndedContainer(D);
    return true;
  }

  bool VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
    ObjCInterfaceDecl *Class = D->getClassInterface();
    if (Class->isImplicitInterfaceDecl())
      IndexCtx.handleObjCInterface(Class);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.invokeStartedObjCContainer(D);
    IndexCtx.indexDeclContext(D);
    IndexCtx.invokeEndedContainer(D);
    return true;
  }

  bool VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
    if (!D->IsClassExtension())
      IndexCtx.handleObjCCategory(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.invokeStartedObjCContainer(D);
    IndexCtx.indexDeclContext(D);
    IndexCtx.invokeEndedContainer(D);
    return true;
  }

  bool VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.invokeStartedObjCContainer(D);
    IndexCtx.indexDeclContext(D);
    IndexCtx.invokeEndedContainer(D);
    return true;
  }

  bool VisitObjCMethodDecl(ObjCMethodDecl *D) {
    IndexCtx.handleObjCMethod(D);
    IndexCtx.indexTypeSourceInfo(D->getResultTypeSourceInfo(), D);
    for (ObjCMethodDecl::param_iterator
           I = D->param_begin(), E = D->param_end(); I != E; ++I)
      IndexCtx.indexTypeSourceInfo((*I)->getTypeSourceInfo(), D);

    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.invokeStartedStatementBody(D, D);
        IndexCtx.indexBody(Body, D);
        IndexCtx.invokeEndedContainer(D);
      }
    }
    return true;
  }

  bool VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
    IndexCtx.handleObjCProperty(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }
};

} // anonymous namespace

void IndexingContext::indexDecl(const Decl *D) {
  bool Handled = IndexingDeclVisitor(*this).Visit(const_cast<Decl*>(D));
  if (!Handled && isa<DeclContext>(D))
    indexDeclContext(cast<DeclContext>(D));
}

void IndexingContext::indexDeclContext(const DeclContext *DC) {
  for (DeclContext::decl_iterator
         I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I) {
    indexDecl(*I);
  }
}

void IndexingContext::indexDeclGroupRef(DeclGroupRef DG) {
  for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I) {
    Decl *D = *I;
    if (isNotFromSourceFile(D->getLocation()))
      return;

    if (isa<ObjCMethodDecl>(D))
      continue; // Wait for the objc container.

    indexDecl(D);
  }
}

void IndexingContext::indexTUDeclsInObjCContainer() {
  for (unsigned i = 0, e = TUDeclsInObjCContainer.size(); i != e; ++i)
    indexDeclGroupRef(TUDeclsInObjCContainer[i]);
  TUDeclsInObjCContainer.clear();
}
