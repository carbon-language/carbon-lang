//===--- index-test.cpp - Indexing test bed -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This utility may be invoked in the following manner:
//   index-test --help                - Output help info.
//   index-test [options]             - Read from stdin.
//   index-test [options] file        - Read from "file".
//   index-test [options] file1 file2 - Read these files.
//
//  Files must be AST files.
//
//===----------------------------------------------------------------------===//
//
//   -Wfatal-errors
//   -ftabstop=width
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/Utils.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
using namespace clang;


static llvm::cl::list<ParsedSourceLocation>
PointAtLocation("point-at", llvm::cl::Optional,
                 llvm::cl::value_desc("source-location"),
   llvm::cl::desc("Point at the given source location of the first AST file"));

static llvm::cl::opt<bool>
DisableFree("disable-free",
           llvm::cl::desc("Disable freeing of memory on exit"),
           llvm::cl::init(false));

static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input AST files>"));

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                     "LLVM 'Clang' Indexing Test Bed: http://clang.llvm.org\n");
  
  FileManager FileMgr;
  
  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");

  // FIXME: Only the first AST file is used for now.

  const std::string &InFile = InputFilenames[0];
  
  std::string ErrMsg;
  llvm::OwningPtr<ASTUnit> AST;

  AST.reset(ASTUnit::LoadFromPCHFile(InFile, FileMgr, &ErrMsg));
  if (!AST) {
    llvm::errs() << "[" << InFile << "] Error: " << ErrMsg << '\n';
    return 1;
  }

  struct ASTPoint {
    Decl *D;
    Stmt *Node;
    ASTPoint() : D(0), Node(0) {}
  };
  
  ASTPoint Point;

  if (!PointAtLocation.empty()) {
    const std::string &Filename = PointAtLocation[0].FileName;
    const FileEntry *File = FileMgr.getFile(Filename);
    if (File == 0) {
      llvm::errs() << "File '" << Filename << "' does not exist\n";
      return 1;
    }
    unsigned Line = PointAtLocation[0].Line;
    unsigned Col = PointAtLocation[0].Column;

    SourceLocation Loc = AST->getSourceManager().getLocation(File, Line, Col);
    if (Loc.isInvalid()) {
      llvm::errs() << "[" << InFile << "] Error: " <<
        "Couldn't resolve source location (invalid location)\n";
      return 1;
    }
    
    llvm::tie(Point.D, Point.Node) =
      ResolveLocationInAST(AST->getASTContext(), Loc);
    if (Point.D == 0) {
      llvm::errs() << "[" << InFile << "] Error: " <<
        "Couldn't resolve source location (no declaration found)\n";
      return 1;
    }
  }
  
  if (Point.D) {
    if (PointAtLocation.empty()) {
      llvm::errs() << "'-print-point-info' should be used together "
                      "with '-point-at'\n";
      return 1;
    }
    
    llvm::raw_ostream &OS = llvm::outs();
    assert(Point.D && "If no node was found we should have exited with error");
    OS << "Declaration node at point: " << Point.D->getDeclKindName() << " ";
    if (NamedDecl *ND = dyn_cast<NamedDecl>(Point.D))
      OS << ND->getNameAsString();
    OS << "\n";
    
    if (Point.Node) {
      OS << "Statement node at point: " << Point.Node->getStmtClassName()
         << " ";
      Point.Node->printPretty(OS, AST->getASTContext());
      OS << "\n";
    }
  }

  if (DisableFree)
    AST.take();

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();
  
  return 0;
}
