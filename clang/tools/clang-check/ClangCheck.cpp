//===- tools/clang-check/ClangCheck.cpp - Clang check tool ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-check tool that runs the
//  clang::SyntaxOnlyAction over a number of translation units.
//
//  This tool uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommandLineClangTool.h"
#include "clang/Tooling/Tooling.h"

using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static const char *MoreHelpText =
    "\tFor example, to run clang-check on all files in a subtree of the\n"
    "\tsource tree, use:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check\n"
    "\n"
    "\tor using a specific build path:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check -p build/path\n"
    "\n"
    "\tNote, that path/in/subtree and current directory should follow the\n"
    "\trules described above.\n"
    "\n";

namespace {
class ActionFactory {
public:
  ActionFactory()
    : Options(createDriverOptTable()),
      ASTDump(
        "ast-dump",
        cl::desc(Options->getOptionHelpText(options::OPT_ast_dump))),
      ASTList(
        "ast-list",
        cl::desc(Options->getOptionHelpText(options::OPT_ast_list))),
      ASTPrint(
        "ast-print",
        cl::desc(Options->getOptionHelpText(options::OPT_ast_print))),
      ASTDumpFilter(
        "ast-dump-filter",
        cl::desc(Options->getOptionHelpText(options::OPT_ast_dump_filter))) {}

  clang::ASTConsumer *newASTConsumer() {
    if (ASTList)
      return clang::CreateASTDeclNodeLister();
    if (ASTDump)
      return clang::CreateASTDumper(ASTDumpFilter);
    if (ASTPrint)
      return clang::CreateASTPrinter(&llvm::outs(), ASTDumpFilter);
    return new clang::ASTConsumer();
  }
private:
  OwningPtr<OptTable> Options;
  cl::opt<bool> ASTDump;
  cl::opt<bool> ASTList;
  cl::opt<bool> ASTPrint;
  cl::opt<std::string> ASTDumpFilter;
};
}

int main(int argc, const char **argv) {
  ActionFactory Factory;
  CommandLineClangTool Tool;
  cl::extrahelp MoreHelp(MoreHelpText);
  Tool.initialize(argc, argv);
  return Tool.run(newFrontendActionFactory(&Factory));
}
