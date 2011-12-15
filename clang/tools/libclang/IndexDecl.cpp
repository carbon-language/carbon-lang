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

  void handleDeclarator(DeclaratorDecl *D, const NamedDecl *Parent = 0) {
    if (!Parent) Parent = D;
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), Parent);
    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), Parent);
  }

  bool VisitFunctionDecl(FunctionDecl *D) {
    IndexCtx.handleFunction(D);
    handleDeclarator(D);
    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, D);
      }
    }
    return true;
  }

  bool VisitVarDecl(VarDecl *D) {
    IndexCtx.handleVar(D);
    handleDeclarator(D);
    IndexCtx.indexBody(D->getInit(), D);
    return true;
  }

  bool VisitFieldDecl(FieldDecl *D) {
    IndexCtx.handleField(D);
    handleDeclarator(D);
    if (D->isBitField())
      IndexCtx.indexBody(D->getBitWidth(), D);
    else if (D->hasInClassInitializer())
      IndexCtx.indexBody(D->getInClassInitializer(), D);
    return true;
  }
  
  bool VisitEnumConstantDecl(EnumConstantDecl *D) {
    IndexCtx.handleEnumerator(D);
    IndexCtx.indexBody(D->getInitExpr(), D);
    return true;
  }

  bool VisitTypedefDecl(TypedefNameDecl *D) {
    IndexCtx.handleTypedefName(D);
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
    IndexCtx.handleObjCClass(D);
    return true;
  }

  bool VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
    ObjCForwardProtocolDecl::protocol_loc_iterator LI = D->protocol_loc_begin();
    for (ObjCForwardProtocolDecl::protocol_iterator
           I = D->protocol_begin(), E = D->protocol_end(); I != E; ++I, ++LI) {
      SourceLocation Loc = *LI;
      ObjCProtocolDecl *PD = *I;

      bool isRedeclaration = PD->getLocation() != Loc;
      IndexCtx.handleObjCForwardProtocol(PD, Loc, isRedeclaration);
    }
    return true;
  }

  bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
    // Forward decls are handled at VisitObjCClassDecl.
    if (!D->isThisDeclarationADefinition())
      return true;

    IndexCtx.handleObjCInterface(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
    // Forward decls are handled at VisitObjCForwardProtocolDecl.
    if (D->isForwardDecl())
      return true;

    IndexCtx.handleObjCProtocol(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
    const ObjCInterfaceDecl *Class = D->getClassInterface();
    if (!Class)
      return true;

    if (Class->isImplicitInterfaceDecl())
      IndexCtx.handleObjCInterface(Class);

    IndexCtx.handleObjCImplementation(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
    IndexCtx.handleObjCCategory(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
    const ObjCCategoryDecl *Cat = D->getCategoryDecl();
    if (!Cat)
      return true;

    IndexCtx.handleObjCCategoryImpl(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCMethodDecl(ObjCMethodDecl *D) {
    IndexCtx.handleObjCMethod(D);
    IndexCtx.indexTypeSourceInfo(D->getResultTypeSourceInfo(), D);
    for (ObjCMethodDecl::param_iterator
           I = D->param_begin(), E = D->param_end(); I != E; ++I)
      handleDeclarator(*I, D);

    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, D);
      }
    }
    return true;
  }

  bool VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
    IndexCtx.handleObjCProperty(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
    ObjCPropertyDecl *PD = D->getPropertyDecl();
    IndexCtx.handleSynthesizedObjCProperty(D);

    if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
      return true;
    assert(D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize);
    
    if (ObjCIvarDecl *IvarD = D->getPropertyIvarDecl()) {
      if (!IvarD->getSynthesize())
        IndexCtx.handleReference(IvarD, D->getPropertyIvarDeclLoc(), 0,
                                 D->getDeclContext());
    }

    if (ObjCMethodDecl *MD = PD->getGetterMethodDecl()) {
      if (MD->isSynthesized())
        IndexCtx.handleSynthesizedObjCMethod(MD, D->getLocation());
    }
    if (ObjCMethodDecl *MD = PD->getSetterMethodDecl()) {
      if (MD->isSynthesized())
        IndexCtx.handleSynthesizedObjCMethod(MD, D->getLocation());
    }
    return true;
  }

  bool VisitNamespaceDecl(NamespaceDecl *D) {
    IndexCtx.handleNamespace(D);
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitClassTemplateDecl(ClassTemplateDecl *D) {
    IndexCtx.handleClassTemplate(D);
    if (D->isThisDeclarationADefinition())
      IndexCtx.indexDeclContext(D->getTemplatedDecl());
    return true;
  }

  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
    IndexCtx.handleFunctionTemplate(D);
    FunctionDecl *FD = D->getTemplatedDecl();
    handleDeclarator(FD, D);
    if (FD->isThisDeclarationADefinition()) {
      const Stmt *Body = FD->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, FD);
      }
    }
    return true;
  }

  bool VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
    IndexCtx.handleTypeAliasTemplate(D);
    IndexCtx.indexTypeSourceInfo(D->getTemplatedDecl()->getTypeSourceInfo(), D);
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

void IndexingContext::indexTopLevelDecl(Decl *D) {
  if (isNotFromSourceFile(D->getLocation()))
    return;

  if (isa<ObjCMethodDecl>(D))
    return; // Wait for the objc container.

  indexDecl(D);
}

void IndexingContext::indexDeclGroupRef(DeclGroupRef DG) {
  for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I)
    indexTopLevelDecl(*I);
}

void IndexingContext::indexTUDeclsInObjCContainer() {
  for (unsigned i = 0, e = TUDeclsInObjCContainer.size(); i != e; ++i)
    indexDeclGroupRef(TUDeclsInObjCContainer[i]);
  TUDeclsInObjCContainer.clear();
}
