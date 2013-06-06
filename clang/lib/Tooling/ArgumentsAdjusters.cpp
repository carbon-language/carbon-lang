//===--- ArgumentsAdjusters.cpp - Command line arguments adjuster ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of classes which implement ArgumentsAdjuster
// interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace tooling {

void ArgumentsAdjuster::anchor() {
}

/// Add -fsyntax-only option to the commnand line arguments.
CommandLineArguments
ClangSyntaxOnlyAdjuster::Adjust(const CommandLineArguments &Args) {
  CommandLineArguments AdjustedArgs;
  for (size_t i = 0, e = Args.size(); i != e; ++i) {
    StringRef Arg = Args[i];
    // FIXME: Remove options that generate output.
    if (!Arg.startswith("-fcolor-diagnostics") &&
        !Arg.startswith("-fdiagnostics-color"))
      AdjustedArgs.push_back(Args[i]);
  }
  AdjustedArgs.push_back("-fsyntax-only");
  return AdjustedArgs;
}

CommandLineArguments
ClangStripOutputAdjuster::Adjust(const CommandLineArguments &Args) {
  CommandLineArguments AdjustedArgs;
  for (size_t i = 0, e = Args.size(); i < e; ++i) {
    StringRef Arg = Args[i];
    if(!Arg.startswith("-o"))
      AdjustedArgs.push_back(Args[i]);

    if(Arg == "-o") {
      // Output is specified as -o foo. Skip the next argument also.
      ++i;
    }
    // Else, the output is specified as -ofoo. Just do nothing.
  }
  return AdjustedArgs;
}

} // end namespace tooling
} // end namespace clang

