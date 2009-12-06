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

#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Index/CallGraph.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace idx;

static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input AST files>"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "clang-wpa");
  FileManager FileMgr;
  std::vector<ASTUnit*> ASTUnits;

  if (InputFilenames.empty())
    return 0;

  DiagnosticOptions DiagOpts;
  llvm::OwningPtr<Diagnostic> Diags(
    CompilerInstance::createDiagnostics(DiagOpts, argc, argv));

  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];
    llvm::OwningPtr<ASTUnit> AST(ASTUnit::LoadFromPCHFile(InFile, *Diags));
    if (!AST)
      return 1;

    ASTUnits.push_back(AST.take());
  }

  llvm::OwningPtr<CallGraph> CG;
  CG.reset(new CallGraph());

  for (unsigned i = 0, e = ASTUnits.size(); i != e; ++i)
    CG->addTU(ASTUnits[i]->getASTContext());

  CG->ViewCallGraph();
}
