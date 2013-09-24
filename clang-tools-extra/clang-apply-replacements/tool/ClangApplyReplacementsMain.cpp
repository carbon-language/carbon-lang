//===-- ClangApplyReplacementsMain.cpp - Main file for the tool -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the main function for the
/// clang-apply-replacements tool.
///
//===----------------------------------------------------------------------===//

#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace clang;
using namespace clang::replace;

static cl::opt<std::string> Directory(cl::Positional, cl::Required,
                                      cl::desc("<Search Root Directory>"));

static cl::opt<bool> RemoveTUReplacementFiles(
    "remove-change-desc-files",
    cl::desc("Remove the change description files regardless of successful\n"
             "merging/replacing."),
    cl::init(false));

// Update this list of options to show in -help as new options are added.
// Should add even those options marked as 'Hidden'. Any option not listed
// here will get marked 'ReallyHidden' so they don't appear in any -help text.
const char *OptionsToShow[] = { "help", "version", "remove-change-desc-files" };

// Helper object to remove the TUReplacement files (triggered by
// "remove-change-desc-files" command line option) when exiting current scope.
class ScopedFileRemover {
public:
  ScopedFileRemover(const TUReplacementFiles &Files,
                    clang::DiagnosticsEngine &Diagnostics)
      : TURFiles(Files), Diag(Diagnostics) {}

  ~ScopedFileRemover() {
    deleteReplacementFiles(TURFiles, Diag);
  }

private:
  const TUReplacementFiles &TURFiles;
  clang::DiagnosticsEngine &Diag;
};

void printVersion() {
  outs() << "clang-apply-replacements version " CLANG_VERSION_STRING << "\n";
}

int main(int argc, char **argv) {
  // Only include our options in -help output.
  StringMap<cl::Option*> OptMap;
  cl::getRegisteredOptions(OptMap);
  const char **EndOpts = OptionsToShow + array_lengthof(OptionsToShow);
  for (StringMap<cl::Option *>::iterator I = OptMap.begin(), E = OptMap.end();
       I != E; ++I) {
    if (std::find(OptionsToShow, EndOpts, I->getKey()) == EndOpts)
      I->getValue()->setHiddenFlag(cl::ReallyHidden);
  }

  cl::SetVersionPrinter(&printVersion);
  cl::ParseCommandLineOptions(argc, argv);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());

  TUReplacements TUs;
  TUReplacementFiles TURFiles;

  error_code ErrorCode =
      collectReplacementsFromDirectory(Directory, TUs, TURFiles, Diagnostics);

  if (ErrorCode) {
    errs() << "Trouble iterating over directory '" << Directory
           << "': " << ErrorCode.message() << "\n";
    return false;
  }

  // Remove the TUReplacementFiles (triggered by "remove-change-desc-files"
  // command line option) when exiting main().
  OwningPtr<ScopedFileRemover> Remover;
  if (RemoveTUReplacementFiles)
    Remover.reset(new ScopedFileRemover(TURFiles, Diagnostics));

  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);

  FileToReplacementsMap GroupedReplacements;
  if (!mergeAndDeduplicate(TUs, GroupedReplacements, SM))
    return 1;

  Rewriter DestRewriter(SM, LangOptions());
  if (!applyReplacements(GroupedReplacements, DestRewriter)) {
    errs() << "Failed to apply all replacements. No changes made.\n";
    return 1;
  }

  if (!writeFiles(DestRewriter))
    return 1;

  return 0;
}
