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
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
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

  TextDiagnosticBuffer DiagClient;
  Diagnostic Diags(&DiagClient);

  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];

    std::string ErrMsg;
    llvm::OwningPtr<ASTUnit> AST;

    AST.reset(ASTUnit::LoadFromPCHFile(InFile, Diags, FileMgr, &ErrMsg));

    if (!AST) {
      llvm::errs() << "[" << InFile << "] error: " << ErrMsg << '\n';
      return 1;
    }

    ASTUnits.push_back(AST.take());
  }

  llvm::OwningPtr<CallGraph> CG;
  CG.reset(new CallGraph());

  for (unsigned i = 0, e = ASTUnits.size(); i != e; ++i)
    CG->addTU(*ASTUnits[i]);

  CG->ViewCallGraph();
}
