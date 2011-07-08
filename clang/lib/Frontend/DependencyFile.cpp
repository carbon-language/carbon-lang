//===--- DependencyFile.cpp - Generate dependency file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code generates dependency files.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
class DependencyFileCallback : public PPCallbacks {
  std::vector<std::string> Files;
  llvm::StringSet<> FilesSet;
  const Preprocessor *PP;
  std::vector<std::string> Targets;
  llvm::raw_ostream *OS;
  bool IncludeSystemHeaders;
  bool PhonyTarget;
private:
  bool FileMatchesDepCriteria(const char *Filename,
                              SrcMgr::CharacteristicKind FileType);
  void OutputDependencyFile();

public:
  DependencyFileCallback(const Preprocessor *_PP,
                         llvm::raw_ostream *_OS,
                         const DependencyOutputOptions &Opts)
    : PP(_PP), Targets(Opts.Targets), OS(_OS),
      IncludeSystemHeaders(Opts.IncludeSystemHeaders),
      PhonyTarget(Opts.UsePhonyTargets) {}

  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType);

  virtual void EndOfMainFile() {
    OutputDependencyFile();
    delete OS;
    OS = 0;
  }
};
}

void clang::AttachDependencyFileGen(Preprocessor &PP,
                                    const DependencyOutputOptions &Opts) {
  if (Opts.Targets.empty()) {
    PP.getDiagnostics().Report(diag::err_fe_dependency_file_requires_MT);
    return;
  }

  std::string Err;
  llvm::raw_ostream *OS(new llvm::raw_fd_ostream(Opts.OutputFile.c_str(), Err));
  if (!Err.empty()) {
    PP.getDiagnostics().Report(diag::err_fe_error_opening)
      << Opts.OutputFile << Err;
    return;
  }

  PP.addPPCallbacks(new DependencyFileCallback(&PP, OS, Opts));
}

/// FileMatchesDepCriteria - Determine whether the given Filename should be
/// considered as a dependency.
bool DependencyFileCallback::FileMatchesDepCriteria(const char *Filename,
                                          SrcMgr::CharacteristicKind FileType) {
  if (strcmp("<built-in>", Filename) == 0)
    return false;

  if (IncludeSystemHeaders)
    return true;

  return FileType == SrcMgr::C_User;
}

void DependencyFileCallback::FileChanged(SourceLocation Loc,
                                         FileChangeReason Reason,
                                         SrcMgr::CharacteristicKind FileType) {
  if (Reason != PPCallbacks::EnterFile)
    return;

  // Dependency generation really does want to go all the way to the
  // file entry for a source location to find out what is depended on.
  // We do not want #line markers to affect dependency generation!
  SourceManager &SM = PP->getSourceManager();

  const FileEntry *FE =
    SM.getFileEntryForID(SM.getFileID(SM.getInstantiationLoc(Loc)));
  if (FE == 0) return;

  llvm::StringRef Filename = FE->getName();
  if (!FileMatchesDepCriteria(Filename.data(), FileType))
    return;

  // Remove leading "./" (or ".//" or "././" etc.)
  while (Filename.size() > 2 && Filename[0] == '.' &&
         llvm::sys::path::is_separator(Filename[1])) {
    Filename = Filename.substr(1);
    while (llvm::sys::path::is_separator(Filename[0]))
      Filename = Filename.substr(1);
  }
    

  if (FilesSet.insert(Filename))
    Files.push_back(Filename);
}

/// PrintFilename - GCC escapes spaces, but apparently not ' or " or other
/// scary characters.
static void PrintFilename(llvm::raw_ostream &OS, llvm::StringRef Filename) {
  for (unsigned i = 0, e = Filename.size(); i != e; ++i) {
    if (Filename[i] == ' ')
      OS << '\\';
    OS << Filename[i];
  }
}

void DependencyFileCallback::OutputDependencyFile() {
  // Write out the dependency targets, trying to avoid overly long
  // lines when possible. We try our best to emit exactly the same
  // dependency file as GCC (4.2), assuming the included files are the
  // same.
  const unsigned MaxColumns = 75;
  unsigned Columns = 0;

  for (std::vector<std::string>::iterator
         I = Targets.begin(), E = Targets.end(); I != E; ++I) {
    unsigned N = I->length();
    if (Columns == 0) {
      Columns += N;
    } else if (Columns + N + 2 > MaxColumns) {
      Columns = N + 2;
      *OS << " \\\n  ";
    } else {
      Columns += N + 1;
      *OS << ' ';
    }
    // Targets already quoted as needed.
    *OS << *I;
  }

  *OS << ':';
  Columns += 1;

  // Now add each dependency in the order it was seen, but avoiding
  // duplicates.
  for (std::vector<std::string>::iterator I = Files.begin(),
         E = Files.end(); I != E; ++I) {
    // Start a new line if this would exceed the column limit. Make
    // sure to leave space for a trailing " \" in case we need to
    // break the line on the next iteration.
    unsigned N = I->length();
    if (Columns + (N + 1) + 2 > MaxColumns) {
      *OS << " \\\n ";
      Columns = 2;
    }
    *OS << ' ';
    PrintFilename(*OS, *I);
    Columns += N + 1;
  }
  *OS << '\n';

  // Create phony targets if requested.
  if (PhonyTarget && !Files.empty()) {
    // Skip the first entry, this is always the input file itself.
    for (std::vector<std::string>::iterator I = Files.begin() + 1,
           E = Files.end(); I != E; ++I) {
      *OS << '\n';
      PrintFilename(*OS, *I);
      *OS << ":\n";
    }
  }
}

