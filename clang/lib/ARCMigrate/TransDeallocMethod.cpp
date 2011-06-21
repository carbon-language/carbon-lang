//===--- TransDeallocMethod.cpp - Tranformations to ARC mode --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"

using namespace clang;
using namespace arcmt;
using namespace trans;
using llvm::StringRef;

void trans::removeDeallocMethod(MigrationPass &pass) {
  ASTContext &Ctx = pass.Ctx;
  TransformActions &TA = pass.TA;
  DeclContext *DC = Ctx.getTranslationUnitDecl();
  ObjCMethodDecl *DeallocMethodDecl = 0;
  IdentifierInfo *II = &Ctx.Idents.get("dealloc");

  for (DeclContext::decl_iterator
         I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I) {
    Decl *D = *I;
    if (ObjCImplementationDecl *IMD = dyn_cast<ObjCImplementationDecl>(D)) {
      DeallocMethodDecl = 0;
      for (ObjCImplementationDecl::instmeth_iterator
             I = IMD->instmeth_begin(), E = IMD->instmeth_end();
          I != E; ++I) {
        ObjCMethodDecl *OMD = *I;
        if (OMD->isInstanceMethod() &&
            OMD->getSelector() == Ctx.Selectors.getSelector(0, &II)) {
          DeallocMethodDecl = OMD;
          break;
        }
      }
      if (DeallocMethodDecl && 
          DeallocMethodDecl->getCompoundBody()->body_empty()) {
        Transaction Trans(TA);
        TA.remove(DeallocMethodDecl->getSourceRange());
      }
    }
  }
}
