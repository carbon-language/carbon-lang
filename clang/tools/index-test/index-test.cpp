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
//   -point-at  [file:column:line]
//       Point at a declaration/statement/expression. If no other operation is
//       specified, prints some info about it.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/Utils.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ASTNode.h"
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

  ASTNode Node;

  if (!PointAtLocation.empty()) {
    const std::string &Filename = PointAtLocation[0].FileName;
    const FileEntry *File = FileMgr.getFile(Filename);

    // Safety check. Using an out-of-date AST file will only lead to crashes
    // or incorrect results.
    // FIXME: Check all the source files that make up the AST file.
    const FileEntry *ASTFile = FileMgr.getFile(InFile);
    if (File->getModificationTime() > ASTFile->getModificationTime()) {
      llvm::errs() << "[" << InFile << "] Error: " <<
        "Pointing at a source file which was modified after creating "
        "the AST file\n";
      return 1;
    }

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
    
    Node = ResolveLocationInAST(AST->getASTContext(), Loc);
    if (Node.isInvalid()) {
      llvm::errs() << "[" << InFile << "] Error: " <<
        "Couldn't resolve source location (no declaration found)\n";
      return 1;
    }
  }
  
  if (Node.isValid()) {
    llvm::raw_ostream &OS = llvm::outs();
    OS << "Declaration node at point: " << Node.getDecl()->getDeclKindName()
       << " ";
    if (NamedDecl *ND = dyn_cast<NamedDecl>(Node.getDecl()))
      OS << ND->getNameAsString();
    OS << "\n";
    
    if (const char *Comment =
          AST->getASTContext().getCommentForDecl(Node.getDecl()))
      OS << "Comment associated with this declaration:\n" << Comment << "\n";
        
    if (Node.getStmt()) {
      OS << "Statement node at point: " << Node.getStmt()->getStmtClassName()
         << " ";
      Node.getStmt()->printPretty(OS, AST->getASTContext(), 0,
                         PrintingPolicy(AST->getASTContext().getLangOptions()));
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
