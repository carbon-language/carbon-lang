//===--- clang-wpa.cpp - clang whole program analyzer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool reads a sequence of precompiled AST files, and do various
// cross translation unit analyses.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CallGraph.h"

#include "clang/Basic/FileManager.h"
#include "clang/Index/TranslationUnit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace idx;

static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input AST files>"));

// FIXME: this duplicates the one in index-test.cpp.
class TUnit : public TranslationUnit {
public:
  TUnit(ASTUnit *ast, const std::string &filename)
    : AST(ast), Filename(filename) {}
  ASTContext &getASTContext() { return AST->getASTContext(); }
  llvm::OwningPtr<ASTUnit> AST;
  std::string Filename;
};

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "clang-wpa");
  FileManager FileMgr;
  std::vector<TUnit*> TUnits;

  if (InputFilenames.empty())
    return 0;

  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];

    std::string ErrMsg;
    llvm::OwningPtr<ASTUnit> AST;

    AST.reset(ASTUnit::LoadFromPCHFile(InFile, FileMgr, &ErrMsg));

    if (!AST) {
      llvm::errs() << "[" << InFile << "] error: " << ErrMsg << '\n';
      return 1;
    }

    TUnit *TU = new TUnit(AST.take(), InFile);
    TUnits.push_back(TU);
  }

  llvm::OwningPtr<CallGraph> CG;
  CG.reset(new CallGraph());

  for (unsigned i = 0, e = TUnits.size(); i != e; ++i)
    CG->addTU(*(TUnits[i]->AST));

  CG->ViewCallGraph();
}
