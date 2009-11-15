//===- PrintFunctionNames.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Example clang-cc plugin which simply prints the names of all the top-level
// decls in the input file.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/AST.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {

class PrintFunctionsConsumer : public ASTConsumer {
public:
  virtual void HandleTopLevelDecl(DeclGroupRef DG) {
    for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
      const Decl *D = *i;
      if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
        llvm::errs() << "top-level-decl: \"" << ND->getNameAsString() << "\"\n";
    }
  }
}; 

class PrintFunctionNamesAction : public ASTFrontendAction {
protected:
  ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) {
    return new PrintFunctionsConsumer();
  }
};

}

FrontendPluginRegistry::Add<PrintFunctionNamesAction>
X("print-fns", "print function names");
