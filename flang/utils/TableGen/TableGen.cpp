//===- TableGen.cpp - Top-Level TableGen implementation for Flang ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for Flang's TableGen.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h" // Declares all backends.
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace flang;

enum ActionType {
  GenFlangDiagsDefs,
  GenFlangDiagGroups,
  GenFlangDiagsIndexName,
  GenDiagDocs,
  GenOptDocs
};

namespace {
cl::opt<ActionType> Action(
    cl::desc("Action to perform:"),
    cl::values(
        clEnumValN(GenFlangDiagsDefs, "gen-flang-diags-defs",
                   "Generate Flang diagnostics definitions"),
        clEnumValN(GenFlangDiagGroups, "gen-flang-diag-groups",
                   "Generate Flang diagnostic groups"),
        clEnumValN(GenFlangDiagsIndexName, "gen-flang-diags-index-name",
                   "Generate Flang diagnostic name index"),
        clEnumValN(GenDiagDocs, "gen-diag-docs",
                   "Generate diagnostic documentation"),
        clEnumValN(GenOptDocs, "gen-opt-docs", "Generate option documentation")));

cl::opt<std::string>
FlangComponent("flang-component",
               cl::desc("Only use warnings from specified component"),
               cl::value_desc("component"), cl::Hidden);

bool FlangTableGenMain(raw_ostream &OS, RecordKeeper &Records) {
  switch (Action) {
  case GenFlangDiagsDefs:
    EmitFlangDiagsDefs(Records, OS, FlangComponent);
    break;
  case GenFlangDiagGroups:
    EmitFlangDiagGroups(Records, OS);
    break;
  case GenFlangDiagsIndexName:
    EmitFlangDiagsIndexName(Records, OS);
    break;
  case GenDiagDocs:
    EmitFlangDiagDocs(Records, OS);
    break;
  case GenOptDocs:
    EmitFlangOptDocs(Records, OS);
    break;
  }

  return false;
}
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;

  return TableGenMain(argv[0], &FlangTableGenMain);
}

#ifdef __has_feature
#if __has_feature(address_sanitizer)
#include <sanitizer/lsan_interface.h>
// Disable LeakSanitizer for this binary as it has too many leaks that are not
// very interesting to fix. See compiler-rt/include/sanitizer/lsan_interface.h .
int __lsan_is_turned_off() { return 1; }
#endif  // __has_feature(address_sanitizer)
#endif  // defined(__has_feature)
