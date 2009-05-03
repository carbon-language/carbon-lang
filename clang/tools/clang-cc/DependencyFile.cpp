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

#include "clang-cc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/System/Path.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace clang;

namespace {
class VISIBILITY_HIDDEN DependencyFileCallback : public PPCallbacks {
  std::vector<std::string> Files;
  llvm::StringSet<> FilesSet;
  const Preprocessor *PP;
  std::vector<std::string> Targets;
  llvm::raw_ostream *OS;

private:
  bool FileMatchesDepCriteria(const char *Filename,
                              SrcMgr::CharacteristicKind FileType);
  void OutputDependencyFile();

public:
  DependencyFileCallback(const Preprocessor *_PP, 
                         llvm::raw_ostream *_OS, 
                         const std::vector<std::string> &_Targets)
    : PP(_PP), Targets(_Targets), OS(_OS) {
  }

  ~DependencyFileCallback() {
    OutputDependencyFile();
    OS->flush();
    delete OS;
  }

  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType);
};
}

//===----------------------------------------------------------------------===//
// Dependency file options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<std::string>
DependencyFile("dependency-file",
               llvm::cl::desc("Filename (or -) to write dependency output to"));

static llvm::cl::opt<bool>
DependenciesIncludeSystemHeaders("sys-header-deps",
                 llvm::cl::desc("Include system headers in dependency output"));

static llvm::cl::list<std::string>
DependencyTargets("MT",
         llvm::cl::desc("Specify target for dependency"));

// FIXME: Implement feature
static llvm::cl::opt<bool>
PhonyDependencyTarget("MP",
            llvm::cl::desc("Create phony target for each dependency "
                           "(other than main file)"));

bool clang::CreateDependencyFileGen(Preprocessor *PP,
                                    std::string &ErrStr) {
  ErrStr = "";
  if (DependencyFile.empty())
    return false;

  if (DependencyTargets.empty()) {
    ErrStr = "-dependency-file requires at least one -MT option\n";
    return false;
  }

  std::string ErrMsg;
  llvm::raw_ostream *OS;
  if (DependencyFile == "-") {
    OS = new llvm::raw_stdout_ostream();
  } else {
    OS = new llvm::raw_fd_ostream(DependencyFile.c_str(), false, ErrStr);
    if (!ErrMsg.empty()) {
      ErrStr = "unable to open dependency file: " + ErrMsg;
      return false;
    }
  }

  DependencyFileCallback *PPDep = 
    new DependencyFileCallback(PP, OS, DependencyTargets);
  PP->setPPCallbacks(PPDep);
  return true;
}

/// FileMatchesDepCriteria - Determine whether the given Filename should be
/// considered as a dependency.
bool DependencyFileCallback::FileMatchesDepCriteria(const char *Filename,
                                          SrcMgr::CharacteristicKind FileType) {
  if (strcmp("<built-in>", Filename) == 0)
    return false;

  if (DependenciesIncludeSystemHeaders)
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
  
  const char *Filename = FE->getName();
  if (!FileMatchesDepCriteria(Filename, FileType))
    return;

  // Remove leading "./"
  if (Filename[0] == '.' && Filename[1] == '/')
    Filename = &Filename[2];

  if (FilesSet.insert(Filename))
    Files.push_back(Filename);
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
      *OS << *I;
    } else if (Columns + N + 2 > MaxColumns) {
      Columns = N + 2;
      *OS << " \\\n  " << *I;
    } else {
      Columns += N + 1;
      *OS << ' ' << *I;
    }
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
    *OS << ' ' << *I;
    Columns += N + 1;
  }
  *OS << '\n';

  // Create phony targets if requested.
  if (PhonyDependencyTarget) {
    // Skip the first entry, this is always the input file itself.
    for (std::vector<std::string>::iterator I = Files.begin() + 1,
           E = Files.end(); I != E; ++I) {
      *OS << '\n';
      *OS << *I << ":\n";
    }
  }
}

