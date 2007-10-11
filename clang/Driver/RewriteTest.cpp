//===--- RewriteTest.cpp - Playground for the code rewriter ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hacks and fun related to the code rewriter.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"

using namespace clang;


namespace {
  class ASTViewer : public ASTConsumer {
    SourceManager *SM;
  public:
    void Initialize(ASTContext &Context, unsigned MainFileID) {
      SM = &Context.SourceMgr;
    }
    
    virtual void HandleTopLevelDecl(Decl *D);
  };
}

ASTConsumer *clang::CreateCodeRewriterTest() { return new ASTViewer(); }




void ASTViewer::HandleTopLevelDecl(Decl *D) {
  if (NamedDecl *ND = dyn_cast<NamedDecl>(D))
    if (ND->getName())
      printf("%s\n", ND->getName());
  
}
