//===- tools/clang-ast-dump/ClangASTDump.cpp - Clang AST Dump tool --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-ast-dump tool that dumps specified parts
//  of an AST of a number of translation units.
//
//  Run with '-help' for details.
//
//  This tool uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommandLineClangTool.h"
#include "clang/Tooling/Tooling.h"

using namespace clang::tooling;
using namespace llvm;

cl::opt<std::string> FilterString(
  "f",
  cl::desc("Filter string"),
  cl::Optional);

cl::opt<bool> ListAll(
  "l",
  cl::desc("List all filterable nodes"),
  cl::init(false),
  cl::Optional);

static const char *MoreHelpText =
    "-f <filter-string> can be used to dump only AST declaration nodes having\n"
    "\ta certain substring in a qualified name.\n"
    "\n"
    "-l \tlists qualified names of all filterable declaration nodes.\n"
    "\n";

namespace {

using namespace clang;

class SelectiveDumpVisitor :
     public RecursiveASTVisitor<SelectiveDumpVisitor> {
  typedef RecursiveASTVisitor<SelectiveDumpVisitor> base;
public:
  SelectiveDumpVisitor(const std::string &FilterString, bool ListAll)
      : FilterString(FilterString), ListAll(ListAll) {}

  ASTConsumer* newASTConsumer() {
    return new DumpConsumer(this);
  }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

  void Run(TranslationUnitDecl *D) {
    if (ListAll) {
      llvm::outs().changeColor(llvm::raw_ostream::BLUE) <<
          "Listing all filterable nodes:\n";
      llvm::outs().resetColor();
      TraverseDecl(D);
      return;
    }

    if (FilterString.empty()) {
      llvm::outs().changeColor(llvm::raw_ostream::BLUE) <<
          "Dumping translation unit:\n";
      llvm::outs().resetColor();
      D->dumpXML(llvm::outs());
      return;
    }

    TraverseDecl(D);
  }

  bool TraverseDecl(Decl *D) {
    if (ListAll) {
      std::string Name = getName(D);
      if (!Name.empty())
        llvm::outs() << Name << "\n";
      return base::TraverseDecl(D);
    }

    if (filterMatches(D)) {
      llvm::outs().changeColor(llvm::raw_ostream::BLUE) <<
          "Dumping " << getName(D) << ":\n";
      llvm::outs().resetColor();
      D->dumpXML(llvm::outs());
      return true;
    }
    return base::TraverseDecl(D);
  }

private:
  std::string getName(Decl *D) {
    if (isa<NamedDecl>(D))
      return cast<NamedDecl>(D)->getQualifiedNameAsString();
    return "";
  }
  bool filterMatches(Decl *D) {
    return getName(D).find(FilterString) != std::string::npos;
  }

  class DumpConsumer : public ASTConsumer {
  public:
    DumpConsumer(SelectiveDumpVisitor *Visitor) : Visitor(Visitor) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {
      Visitor->Context = &Context;
      Visitor->Run(Context.getTranslationUnitDecl());
    }

  private:
    SelectiveDumpVisitor *Visitor;
  };

  ASTContext *Context;
  std::string FilterString;
  bool ListAll;
};

} // namespace

int main(int argc, const char **argv) {
  CommandLineClangTool Tool;
  cl::extrahelp MoreHelp(MoreHelpText);
  Tool.initialize(argc, argv);
  SelectiveDumpVisitor Dumper(FilterString, ListAll);
  return Tool.run(newFrontendActionFactory(&Dumper));
}
