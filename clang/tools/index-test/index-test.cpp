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
#include "clang/Index/IndexProvider.h"
#include "clang/Index/Entity.h"
#include "clang/Index/TranslationUnit.h"
#include "clang/Index/ASTLocation.h"
#include "clang/Index/DeclReferenceMap.h"
#include "clang/Index/Utils.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
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
    : AST(ast), Filename(filename) { }
  
  virtual ASTContext &getASTContext() { return AST->getASTContext(); }
  
  llvm::OwningPtr<ASTUnit> AST;
  std::string Filename;
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

static void ProcessDecl(Decl *D) {
  assert(D);
  llvm::raw_ostream &OS = llvm::outs();
  
  switch (ProgAction) {
  default: assert(0);
  case PrintRefs: {
    NamedDecl *ND = dyn_cast<NamedDecl>(D);
    if (!ND)
      return;

    DeclReferenceMap RefMap(ND->getASTContext());
    for (DeclReferenceMap::astlocation_iterator
           I = RefMap.refs_begin(ND), E = RefMap.refs_end(ND); I != E; ++I)
      I->print(OS);
    break;
  }
  
  case PrintDefs: {
    const Decl *DefD = 0;
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      const FunctionDecl *DFD = 0;
      FD->getBody(DFD);
      DefD = DFD;
    } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
      const VarDecl *DVD = 0;
      VD->getDefinition(DVD);
      DefD = DVD;
    } 

    if (DefD)
      ASTLocation(DefD).print(OS);
    break;    
  }
  
  case PrintDecls :
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      for (FunctionDecl::redecl_iterator I = FD->redecls_begin(),
                                         E = FD->redecls_end(); I != E; ++I)
        ASTLocation(*I).print(OS);
    } else if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
      for (VarDecl::redecl_iterator I = VD->redecls_begin(),
                                    E = VD->redecls_end(); I != E; ++I)
        ASTLocation(*I).print(OS);
    } else
      ASTLocation(D).print(OS);
    break;
    
  }
}

static Decl *getDeclFromExpr(Stmt *E) {
  if (DeclRefExpr *RefExpr = dyn_cast<DeclRefExpr>(E))
    return RefExpr->getDecl();
  if (MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (CallExpr *CE = dyn_cast<CallExpr>(E))
    return getDeclFromExpr(CE->getCallee());
  if (CastExpr *CE = dyn_cast<CastExpr>(E))
    return getDeclFromExpr(CE->getSubExpr());
  
  return 0;
}

static void ProcessASTLocation(ASTLocation ASTLoc, IndexProvider &IdxProvider) {
  assert(ASTLoc.isValid());

  Decl *D = 0;
  if (ASTLoc.isStmt())
    D = getDeclFromExpr(ASTLoc.getStmt());
  else
    D = ASTLoc.getDecl();
  
  if (D == 0) {
    llvm::errs() << "Error: Couldn't get a Decl out of the ASTLocation";
    HadErrors = true;
    return;
  }

  Entity *Ent = Entity::get(D, IdxProvider.getProgram());
  // If there is no Entity associated with this Decl, it means that it's not
  // visible to other translation units.
  if (!Ent)
    return ProcessDecl(D);

  // Find the "same" Decl in other translation units and print information.
  for (IndexProvider::translation_unit_iterator
         I = IdxProvider.translation_units_begin(Ent),
         E = IdxProvider.translation_units_end(Ent); I != E; ++I) {
    TUnit *TU = static_cast<TUnit*>(*I);
    Decl *OtherD = Ent->getDecl(TU->getASTContext());
    assert(OtherD && "Couldn't resolve Entity");
    ProcessDecl(OtherD);
  }
}

static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input AST files>"));

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                     "LLVM 'Clang' Indexing Test Bed: http://clang.llvm.org\n");
  
  FileManager FileMgr;

  Program Prog;
  IndexProvider IdxProvider(Prog);
  llvm::SmallVector<TUnit*, 4> TUnits;
  
  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");

  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];
    
    std::string ErrMsg;
    llvm::OwningPtr<ASTUnit> AST;

    AST.reset(ASTUnit::LoadFromPCHFile(InFile, FileMgr, &ErrMsg));
    if (!AST) {
      llvm::errs() << "[" << InFile << "] Error: " << ErrMsg << '\n';
      return 1;
    }

    TUnit *TU = new TUnit(AST.take(), InFile);
    TUnits.push_back(TU);
    
    IdxProvider.IndexAST(TU);
  }

  ASTLocation ASTLoc;
  const std::string &FirstFile = TUnits[0]->Filename;
  ASTUnit *FirstAST = TUnits[0]->AST.get();

  if (!PointAtLocation.empty()) {
    const std::string &Filename = PointAtLocation[0].FileName;
    const FileEntry *File = FileMgr.getFile(Filename);

    // Safety check. Using an out-of-date AST file will only lead to crashes
    // or incorrect results.
    // FIXME: Check all the source files that make up the AST file.
    const FileEntry *ASTFile = FileMgr.getFile(FirstFile);
    if (File->getModificationTime() > ASTFile->getModificationTime()) {
      llvm::errs() << "[" << FirstFile << "] Error: " <<
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
      ProcessASTLocation(ASTLoc, IdxProvider);
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
