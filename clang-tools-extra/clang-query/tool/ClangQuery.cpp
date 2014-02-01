//===---- ClangQuery.cpp - clang-query tool -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool is for interactive exploration of the Clang AST using AST matchers.
// It currently allows the user to enter a matcher at an interactive prompt and
// view the resulting bindings as diagnostics, AST pretty prints or AST dumps.
// Example session:
//
// $ cat foo.c
// void foo(void) {}
// $ clang-query foo.c --
// clang-query> match functionDecl()
//
// Match #1:
//
// foo.c:1:1: note: "root" binds here
// void foo(void) {}
// ^~~~~~~~~~~~~~~~~
// 1 match.
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QueryParser.h"
#include "QuerySession.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include <fstream>
#include <string>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::ast_matchers::dynamic;
using namespace clang::query;
using namespace clang::tooling;
using namespace llvm;

static cl::opt<std::string> BuildPath("b", cl::desc("Specify build path"),
                                      cl::value_desc("<path>"));

static cl::list<std::string> Commands("c", cl::desc("Specify command to run"),
                                      cl::value_desc("<command>"));

static cl::list<std::string> CommandFiles("f",
                                          cl::desc("Read commands from file"),
                                          cl::value_desc("<file>"));

static cl::list<std::string> SourcePaths(cl::Positional,
                                         cl::desc("<source0> [... <sourceN>]"),
                                         cl::OneOrMore);

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  cl::ParseCommandLineOptions(argc, argv);

  if (!Commands.empty() && !CommandFiles.empty()) {
    llvm::errs() << argv[0] << ": cannot specify both -c and -f\n";
    return 1;
  }

  llvm::OwningPtr<CompilationDatabase> Compilations(
        FixedCompilationDatabase::loadFromCommandLine(argc, argv));
  if (!Compilations) {  // Couldn't find a compilation DB from the command line
    std::string ErrorMessage;
    Compilations.reset(
      !BuildPath.empty() ?
        CompilationDatabase::autoDetectFromDirectory(BuildPath, ErrorMessage) :
        CompilationDatabase::autoDetectFromSource(SourcePaths[0], ErrorMessage)
      );

    // Still no compilation DB? - bail.
    if (!Compilations)
      llvm::report_fatal_error(ErrorMessage);
  }

  ClangTool Tool(*Compilations, SourcePaths);
  std::vector<ASTUnit *> ASTs;
  if (Tool.buildASTs(ASTs) != 0)
    return 1;

  QuerySession QS(ASTs);

  if (!Commands.empty()) {
    for (cl::list<std::string>::iterator I = Commands.begin(),
                                         E = Commands.end();
         I != E; ++I) {
      QueryRef Q = QueryParser::parse(I->c_str());
      if (!Q->run(llvm::outs(), QS))
        return 1;
    }
  } else if (!CommandFiles.empty()) {
    for (cl::list<std::string>::iterator I = CommandFiles.begin(),
                                         E = CommandFiles.end();
         I != E; ++I) {
      std::ifstream Input(I->c_str());
      if (!Input.is_open()) {
        llvm::errs() << argv[0] << ": cannot open " << *I << "\n";
        return 1;
      }
      while (Input.good()) {
        std::string Line;
        std::getline(Input, Line);

        QueryRef Q = QueryParser::parse(Line.c_str());
        if (!Q->run(llvm::outs(), QS))
          return 1;
      }
    }
  } else {
    LineEditor LE("clang-query");
    LE.setListCompleter(QueryParser::complete);
    while (llvm::Optional<std::string> Line = LE.readLine()) {
      QueryRef Q = QueryParser::parse(*Line);
      Q->run(llvm::outs(), QS);
    }
  }

  llvm::DeleteContainerPointers(ASTs);

  return 0;
}
