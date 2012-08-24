//===--- tools/clang-check/ClangCheck.cpp - Clang check tool --------------===//
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

#include "clang/AST/ASTConsumer.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp(
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
    "\n"
);

static OwningPtr<OptTable> Options(createDriverOptTable());
static cl::opt<bool> ASTDump(
    "ast-dump",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_dump)));
static cl::opt<bool> ASTList(
    "ast-list",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_list)));
static cl::opt<bool> ASTPrint(
    "ast-print",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_print)));
static cl::opt<std::string> ASTDumpFilter(
    "ast-dump-filter",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_dump_filter)));

namespace {
class ActionFactory {
public:
  clang::ASTConsumer *newASTConsumer() {
    if (ASTList)
      return clang::CreateASTDeclNodeLister();
    if (ASTDump)
      return clang::CreateASTDumper(ASTDumpFilter);
    if (ASTPrint)
      return clang::CreateASTPrinter(&llvm::outs(), ASTDumpFilter);
    return new clang::ASTConsumer();
  }
};
}

int main(int argc, const char **argv) {
  ActionFactory Factory;
  CommonOptionsParser OptionsParser(argc, argv);
  ClangTool Tool(OptionsParser.GetCompilations(),
                 OptionsParser.GetSourcePathList());
  return Tool.run(newFrontendActionFactory(&Factory));
}
