//===-- ClangReplaceMain.cpp - Main file for clang-replace tool -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the main function for the clang-replace tool.
///
//===----------------------------------------------------------------------===//

#include "ApplyReplacements.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace clang;
using namespace clang::replace;

static cl::opt<std::string> Directory(cl::Positional, cl::Required,
                                      cl::desc("<Search Root Directory>"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());

  TUReplacements TUs;

  error_code ErrorCode =
      collectReplacementsFromDirectory(Directory, TUs, Diagnostics);

  if (ErrorCode) {
    errs() << "Trouble iterating over directory '" << Directory
           << "': " << ErrorCode.message() << "\n";
    return false;
  }

  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);

  FileToReplacementsMap GroupedReplacements;
  if (!mergeAndDeduplicate(TUs, GroupedReplacements, SM))
    return 1;

  if (!applyReplacements(GroupedReplacements, SM))
    return 1;
}
