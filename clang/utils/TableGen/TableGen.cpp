//===- TableGen.cpp - Top-Level TableGen implementation for Clang ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for Clang's TableGen.
//
//===----------------------------------------------------------------------===//

#include "ClangASTNodesEmitter.h"
#include "ClangAttrEmitter.h"
#include "ClangDiagnosticsEmitter.h"
#include "ClangSACheckersEmitter.h"
#include "NeonEmitter.h"
#include "OptParserEmitter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenAction.h"

using namespace llvm;

enum ActionType {
  GenClangAttrClasses,
  GenClangAttrImpl,
  GenClangAttrList,
  GenClangAttrPCHRead,
  GenClangAttrPCHWrite,
  GenClangAttrSpellingList,
  GenClangAttrLateParsedList,
  GenClangDiagsDefs,
  GenClangDiagGroups,
  GenClangDiagsIndexName,
  GenClangDeclNodes,
  GenClangStmtNodes,
  GenClangSACheckers,
  GenOptParserDefs, GenOptParserImpl,
  GenArmNeon,
  GenArmNeonSema,
  GenArmNeonTest
};

namespace {
  cl::opt<ActionType>
  Action(cl::desc("Action to perform:"),
         cl::values(clEnumValN(GenOptParserDefs, "gen-opt-parser-defs",
                               "Generate option definitions"),
                    clEnumValN(GenOptParserImpl, "gen-opt-parser-impl",
                               "Generate option parser implementation"),
                    clEnumValN(GenClangAttrClasses, "gen-clang-attr-classes",
                               "Generate clang attribute clases"),
                    clEnumValN(GenClangAttrImpl, "gen-clang-attr-impl",
                               "Generate clang attribute implementations"),
                    clEnumValN(GenClangAttrList, "gen-clang-attr-list",
                               "Generate a clang attribute list"),
                    clEnumValN(GenClangAttrPCHRead, "gen-clang-attr-pch-read",
                               "Generate clang PCH attribute reader"),
                    clEnumValN(GenClangAttrPCHWrite, "gen-clang-attr-pch-write",
                               "Generate clang PCH attribute writer"),
                    clEnumValN(GenClangAttrSpellingList,
                               "gen-clang-attr-spelling-list",
                               "Generate a clang attribute spelling list"),
                    clEnumValN(GenClangAttrLateParsedList,
                               "gen-clang-attr-late-parsed-list",
                               "Generate a clang attribute LateParsed list"),
                    clEnumValN(GenClangDiagsDefs, "gen-clang-diags-defs",
                               "Generate Clang diagnostics definitions"),
                    clEnumValN(GenClangDiagGroups, "gen-clang-diag-groups",
                               "Generate Clang diagnostic groups"),
                    clEnumValN(GenClangDiagsIndexName,
                               "gen-clang-diags-index-name",
                               "Generate Clang diagnostic name index"),
                    clEnumValN(GenClangDeclNodes, "gen-clang-decl-nodes",
                               "Generate Clang AST declaration nodes"),
                    clEnumValN(GenClangStmtNodes, "gen-clang-stmt-nodes",
                               "Generate Clang AST statement nodes"),
                    clEnumValN(GenClangSACheckers, "gen-clang-sa-checkers",
                               "Generate Clang Static Analyzer checkers"),
                    clEnumValN(GenArmNeon, "gen-arm-neon",
                               "Generate arm_neon.h for clang"),
                    clEnumValN(GenArmNeonSema, "gen-arm-neon-sema",
                               "Generate ARM NEON sema support for clang"),
                    clEnumValN(GenArmNeonTest, "gen-arm-neon-test",
                               "Generate ARM NEON tests for clang"),
                    clEnumValEnd));

  cl::opt<std::string>
  ClangComponent("clang-component",
                 cl::desc("Only use warnings from specified component"),
                 cl::value_desc("component"), cl::Hidden);
}

class ClangTableGenAction : public TableGenAction {
public:
  bool operator()(raw_ostream &OS, RecordKeeper &Records) {
    switch (Action) {
    case GenClangAttrClasses:
      ClangAttrClassEmitter(Records).run(OS);
      break;
    case GenClangAttrImpl:
      ClangAttrImplEmitter(Records).run(OS);
      break;
    case GenClangAttrList:
      ClangAttrListEmitter(Records).run(OS);
      break;
    case GenClangAttrPCHRead:
      ClangAttrPCHReadEmitter(Records).run(OS);
      break;
    case GenClangAttrPCHWrite:
      ClangAttrPCHWriteEmitter(Records).run(OS);
      break;
    case GenClangAttrSpellingList:
      ClangAttrSpellingListEmitter(Records).run(OS);
      break;
    case GenClangAttrLateParsedList:
      ClangAttrLateParsedListEmitter(Records).run(OS);
      break;
    case GenClangDiagsDefs:
      ClangDiagsDefsEmitter(Records, ClangComponent).run(OS);
      break;
    case GenClangDiagGroups:
      ClangDiagGroupsEmitter(Records).run(OS);
      break;
    case GenClangDiagsIndexName:
      ClangDiagsIndexNameEmitter(Records).run(OS);
      break;
    case GenClangDeclNodes:
      ClangASTNodesEmitter(Records, "Decl", "Decl").run(OS);
      ClangDeclContextEmitter(Records).run(OS);
      break;
    case GenClangStmtNodes:
      ClangASTNodesEmitter(Records, "Stmt", "").run(OS);
      break;
    case GenClangSACheckers:
      ClangSACheckersEmitter(Records).run(OS);
      break;
    case GenOptParserDefs:
      OptParserEmitter(Records, true).run(OS);
      break;
    case GenOptParserImpl:
      OptParserEmitter(Records, false).run(OS);
      break;
    case GenArmNeon:
      NeonEmitter(Records).run(OS);
      break;
    case GenArmNeonSema:
      NeonEmitter(Records).runHeader(OS);
      break;
    case GenArmNeonTest:
      NeonEmitter(Records).runTests(OS);
      break;
    default:
      assert(1 && "Invalid Action");
      return true;
    }

    return false;
  }
};

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  ClangTableGenAction Action;
  return TableGenMain(argv[0], Action);
}
