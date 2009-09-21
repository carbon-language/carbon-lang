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
//   -point-at  [file:line:column]
//       Point at a declaration/statement/expression. If no other operation is
//       specified, prints some info about it.
//
//   -print-refs
//       Print ASTLocations that reference the -point-at node
//
//   -print-defs
//       Print ASTLocations that define the -point-at node
//
//   -print-decls
//       Print ASTLocations that declare the -point-at node
//
//===----------------------------------------------------------------------===//

#include "clang/Index/Program.h"
#include "clang/Index/Indexer.h"
#include "clang/Index/Entity.h"
#include "clang/Index/TranslationUnit.h"
#include "clang/Index/ASTLocation.h"
#include "clang/Index/DeclReferenceMap.h"
#include "clang/Index/SelectorMap.h"
#include "clang/Index/Handlers.h"
#include "clang/Index/Analyzer.h"
#include "clang/Index/Utils.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
using namespace clang;
using namespace idx;

class TUnit : public TranslationUnit {
public:
  TUnit(ASTUnit *ast, const std::string &filename)
    : AST(ast), Filename(filename),
      DeclRefMap(ast->getASTContext()),
      SelMap(ast->getASTContext()) { }

  virtual ASTContext &getASTContext() { return AST->getASTContext(); }
  virtual DeclReferenceMap &getDeclReferenceMap() { return DeclRefMap; }
  virtual SelectorMap &getSelectorMap() { return SelMap; }

  llvm::OwningPtr<ASTUnit> AST;
  std::string Filename;
  DeclReferenceMap DeclRefMap;
  SelectorMap SelMap;
};

static llvm::cl::list<ParsedSourceLocation>
PointAtLocation("point-at", llvm::cl::Optional,
                 llvm::cl::value_desc("source-location"),
   llvm::cl::desc("Point at the given source location of the first AST file"));

enum ProgActions {
  PrintPoint,     // Just print the point-at node
  PrintRefs,      // Print references of the point-at node
  PrintDefs,      // Print definitions of the point-at node
  PrintDecls      // Print declarations of the point-at node
};

static llvm::cl::opt<ProgActions>
ProgAction(
        llvm::cl::desc("Choose action to perform on the pointed-at AST node:"),
        llvm::cl::ZeroOrMore,
           llvm::cl::init(PrintPoint),
           llvm::cl::values(
             clEnumValN(PrintRefs, "print-refs",
                        "Print references"),
             clEnumValN(PrintDefs, "print-defs",
                        "Print definitions"),
             clEnumValN(PrintDecls, "print-decls",
                        "Print declarations"),
             clEnumValEnd));

static llvm::cl::opt<bool>
DisableFree("disable-free",
           llvm::cl::desc("Disable freeing of memory on exit"),
           llvm::cl::init(false));

static bool HadErrors = false;

static void ProcessObjCMessage(ObjCMessageExpr *Msg, Indexer &Idxer) {
  llvm::raw_ostream &OS = llvm::outs();
  typedef Storing<TULocationHandler> ResultsTy;
  ResultsTy Results;

  Analyzer Analyz(Idxer.getProgram(), Idxer);

  switch (ProgAction) {
  default: assert(0);
  case PrintRefs:
    llvm::errs() << "Error: Cannot -print-refs on a ObjC message expression\n";
    HadErrors = true;
    return;

  case PrintDecls: {
    Analyz.FindObjCMethods(Msg, Results);
    for (ResultsTy::iterator
           I = Results.begin(), E = Results.end(); I != E; ++I)
      I->print(OS);
    break;
  }

  case PrintDefs: {
    Analyz.FindObjCMethods(Msg, Results);
    for (ResultsTy::iterator
           I = Results.begin(), E = Results.end(); I != E; ++I) {
      const ObjCMethodDecl *D = cast<ObjCMethodDecl>(I->getDecl());
      if (D->isThisDeclarationADefinition())
        I->print(OS);
    }
    break;
  }

  }
}

static void ProcessASTLocation(ASTLocation ASTLoc, Indexer &Idxer) {
  assert(ASTLoc.isValid());

  if (ObjCMessageExpr *Msg =
        dyn_cast_or_null<ObjCMessageExpr>(ASTLoc.getStmt()))
    return ProcessObjCMessage(Msg, Idxer);

  Decl *D = ASTLoc.getReferencedDecl();
  if (D == 0) {
    llvm::errs() << "Error: Couldn't get referenced Decl for the ASTLocation\n";
    HadErrors = true;
    return;
  }

  llvm::raw_ostream &OS = llvm::outs();
  typedef Storing<TULocationHandler> ResultsTy;
  ResultsTy Results;

  Analyzer Analyz(Idxer.getProgram(), Idxer);

  switch (ProgAction) {
  default: assert(0);
  case PrintRefs: {
    Analyz.FindReferences(D, Results);
    for (ResultsTy::iterator
           I = Results.begin(), E = Results.end(); I != E; ++I)
      I->print(OS);
    break;
  }

  case PrintDecls: {
    Analyz.FindDeclarations(D, Results);
    for (ResultsTy::iterator
           I = Results.begin(), E = Results.end(); I != E; ++I)
      I->print(OS);
    break;
  }

  case PrintDefs: {
    Analyz.FindDeclarations(D, Results);
    for (ResultsTy::iterator
           I = Results.begin(), E = Results.end(); I != E; ++I) {
      const Decl *D = I->getDecl();
      bool isDef = false;
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
        isDef = FD->isThisDeclarationADefinition();
      else if (const VarDecl *VD = dyn_cast<VarDecl>(D))
        isDef = VD->getInit() != 0;
      else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
        isDef = MD->isThisDeclarationADefinition();

      if (isDef)
        I->print(OS);
    }
    break;
  }

  }
}

static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input AST files>"));

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                     "LLVM 'Clang' Indexing Test Bed: http://clang.llvm.org\n");

  Program Prog;
  Indexer Idxer(Prog);
  llvm::SmallVector<TUnit*, 4> TUnits;

  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");

  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];

    std::string ErrMsg;
    llvm::OwningPtr<ASTUnit> AST;

    AST.reset(ASTUnit::LoadFromPCHFile(InFile, Idxer.getDiagnostics(),
                                       Idxer.getFileManager(), &ErrMsg));
    if (!AST) {
      llvm::errs() << "[" << InFile << "] Error: " << ErrMsg << '\n';
      return 1;
    }

    TUnit *TU = new TUnit(AST.take(), InFile);
    TUnits.push_back(TU);

    Idxer.IndexAST(TU);
  }

  ASTLocation ASTLoc;
  const std::string &FirstFile = TUnits[0]->Filename;
  ASTUnit *FirstAST = TUnits[0]->AST.get();

  if (!PointAtLocation.empty()) {
    const std::string &Filename = PointAtLocation[0].FileName;
    const FileEntry *File = Idxer.getFileManager().getFile(Filename);
    if (File == 0) {
      llvm::errs() << "File '" << Filename << "' does not exist\n";
      return 1;
    }

    // Safety check. Using an out-of-date AST file will only lead to crashes
    // or incorrect results.
    // FIXME: Check all the source files that make up the AST file.
    const FileEntry *ASTFile = Idxer.getFileManager().getFile(FirstFile);
    if (File->getModificationTime() > ASTFile->getModificationTime()) {
      llvm::errs() << "[" << FirstFile << "] Error: " <<
        "Pointing at a source file which was modified after creating "
        "the AST file\n";
      return 1;
    }

    unsigned Line = PointAtLocation[0].Line;
    unsigned Col = PointAtLocation[0].Column;

    SourceLocation Loc =
      FirstAST->getSourceManager().getLocation(File, Line, Col);
    if (Loc.isInvalid()) {
      llvm::errs() << "[" << FirstFile << "] Error: " <<
        "Couldn't resolve source location (invalid location)\n";
      return 1;
    }

    ASTLoc = ResolveLocationInAST(FirstAST->getASTContext(), Loc);
    if (ASTLoc.isInvalid()) {
      llvm::errs() << "[" << FirstFile << "] Error: " <<
        "Couldn't resolve source location (no declaration found)\n";
      return 1;
    }
  }

  if (ASTLoc.isValid()) {
    if (ProgAction == PrintPoint) {
      llvm::raw_ostream &OS = llvm::outs();
      ASTLoc.print(OS);
      if (const char *Comment =
            FirstAST->getASTContext().getCommentForDecl(ASTLoc.getDecl()))
        OS << "Comment associated with this declaration:\n" << Comment << "\n";
    } else {
      ProcessASTLocation(ASTLoc, Idxer);
    }
  }

  if (HadErrors)
    return 1;

  if (!DisableFree) {
    for (int i=0, e=TUnits.size(); i != e; ++i)
      delete TUnits[i];
  }

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return 0;
}
