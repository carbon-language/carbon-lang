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

#include "clang.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/System/Path.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <string>

using namespace clang;

namespace {
class VISIBILITY_HIDDEN DependencyFileCallback : public PPCallbacks {
  std::vector<std::string> Files;
  llvm::StringSet<> FilesSet;
  const Preprocessor *PP;
  std::ofstream OS;
  const std::string &InputFile;
  std::string Target;

private:
  bool FileMatchesDepCriteria(const char *Filename,
                              SrcMgr::CharacteristicKind FileType);
  void OutputDependencyFile();

public:
  DependencyFileCallback(const Preprocessor *PP, 
                         const std::string &InputFile,
                         const std::string &DepFile,
                         const std::string &Target,
                         const char  *&ErrStr);
  ~DependencyFileCallback();
  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType);
};
}

static const char *DependencyFileExt = "d";
static const char *ObjectFileExt = "o";

//===----------------------------------------------------------------------===//
// Dependency file options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<bool>
GenerateDependencyFile("MD",
             llvm::cl::desc("Generate dependency for main source file "
                            "(system headers included)"));

static llvm::cl::opt<bool>
GenerateDependencyFileNoSysHeaders("MMD",
             llvm::cl::desc("Generate dependency for main source file "
                            "(no system headers)"));

static llvm::cl::opt<std::string>
DependencyOutputFile("MF",
           llvm::cl::desc("Specify dependency output file"));

static llvm::cl::opt<std::string>
DependencyTarget("MT",
         llvm::cl::desc("Specify target for dependency"));

// FIXME: Implement feature
static llvm::cl::opt<bool>
PhonyDependencyTarget("MP",
            llvm::cl::desc("Create phony target for each dependency "
                           "(other than main file)"));

bool clang::CreateDependencyFileGen(Preprocessor *PP,
                                    std::string &OutputFile,
                                    const std::string &InputFile,
                                    const char  *&ErrStr) {
  assert(!InputFile.empty() && "No file given");
  
  ErrStr = NULL;

  if (!GenerateDependencyFile && !GenerateDependencyFileNoSysHeaders) {
    if (!DependencyOutputFile.empty() || !DependencyTarget.empty() || 
        PhonyDependencyTarget)
      ErrStr = "Error: to generate dependencies you must specify -MD or -MMD\n";
    return false;
  }
  
  // Handle conflicting options
  if (GenerateDependencyFileNoSysHeaders)
    GenerateDependencyFile = false;

  // Determine name of dependency output filename
  llvm::sys::Path DepFile;
  if (!DependencyOutputFile.empty())
    DepFile = DependencyOutputFile;
  else if (!OutputFile.empty()) {
    DepFile = OutputFile;
    DepFile.eraseSuffix();
    DepFile.appendSuffix(DependencyFileExt);
  }
  else {
    DepFile = InputFile;
    DepFile.eraseSuffix();
    DepFile.appendSuffix(DependencyFileExt);
  }

  // Determine name of target
  std::string Target;
  if (!DependencyTarget.empty())
    Target = DependencyTarget;
  else if (!OutputFile.empty()) {
    llvm::sys::Path TargetPath(OutputFile);
    TargetPath.eraseSuffix();
    TargetPath.appendSuffix(ObjectFileExt);
    Target = TargetPath.toString();
  }
  else {
    llvm::sys::Path TargetPath(InputFile);
    TargetPath.eraseSuffix();
    TargetPath.appendSuffix(ObjectFileExt);
    Target = TargetPath.toString();
  }

  DependencyFileCallback *PPDep = 
    new DependencyFileCallback(PP, InputFile, DepFile.toString(),
                               Target, ErrStr);
  if (ErrStr){
    delete PPDep;
    return false;
  }
  else {
    PP->setPPCallbacks(PPDep);
    return true;
  }
}

/// FileMatchesDepCriteria - Determine whether the given Filename should be
/// considered as a dependency.
bool DependencyFileCallback::FileMatchesDepCriteria(const char *Filename,
                                          SrcMgr::CharacteristicKind FileType) {
  if (strcmp(InputFile.c_str(), Filename) != 0 &&
      strcmp("<predefines>", Filename) != 0) {
      if (GenerateDependencyFileNoSysHeaders)
        return FileType == SrcMgr::C_User;
      else
        return true;
  }

  return false;
}

void DependencyFileCallback::FileChanged(SourceLocation Loc,
                                         FileChangeReason Reason,
                                         SrcMgr::CharacteristicKind FileType) {
  if (Reason != PPCallbacks::EnterFile)
    return;

  const char *Filename = PP->getSourceManager().getSourceName(Loc);
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
  
  OS << Target << ":";
  unsigned Columns = Target.length() + 1;
  
  // Now add each dependency in the order it was seen, but avoiding
  // duplicates.
  for (std::vector<std::string>::iterator I = Files.begin(),
         E = Files.end(); I != E; ++I) {
    // Start a new line if this would exceed the column limit. Make
    // sure to leave space for a trailing " \" in case we need to
    // break the line on the next iteration.
    unsigned N = I->length();
    if (Columns + (N + 1) + 2 > MaxColumns) {
      OS << " \\\n ";
      Columns = 2;
    }
    OS << " " << *I;
    Columns += N + 1;
  }
  OS << "\n";

  // Create phony targets if requested.
  if (PhonyDependencyTarget) {
    // Skip the first entry, this is always the input file itself.
    for (std::vector<std::string>::iterator I = Files.begin() + 1,
           E = Files.end(); I != E; ++I) {
      OS << "\n";
      OS << *I << ":\n";
    }
  }
}

DependencyFileCallback::DependencyFileCallback(const Preprocessor *PP,
                                               const std::string &InputFile,
                                               const std::string &DepFile,
                                               const std::string &Target,
                                               const char  *&ErrStr)
  : PP(PP), InputFile(InputFile), Target(Target) {

  OS.open(DepFile.c_str());
  if (OS.fail())
    ErrStr = "Could not open dependency output file\n";
  else
    ErrStr = NULL;

  Files.push_back(InputFile);
}

DependencyFileCallback::~DependencyFileCallback() {
  if ((!GenerateDependencyFile && !GenerateDependencyFileNoSysHeaders) || 
      OS.fail())
    return;
  
  OutputDependencyFile();
  OS.close();
}

