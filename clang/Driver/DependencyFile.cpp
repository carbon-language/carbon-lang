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
#include <fstream>
#include <string>

using namespace clang;

namespace {
class VISIBILITY_HIDDEN DependencyFileCallback : public PPCallbacks {
  llvm::StringSet<> Files;
  const Preprocessor *PP;
  std::ofstream OS;
  const std::string &InputFile;
  std::string Target;

private:
  bool FileMatchesDepCriteria(const char *Filename,
                              SrcMgr::Characteristic_t FileType);
  void OutputDependencyFile();

public:
  DependencyFileCallback(const Preprocessor *PP, 
                         const std::string &InputFile,
                         const std::string &DepFile,
                         const std::string &Target,
                         const char  *&ErrStr);
  ~DependencyFileCallback();
  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::Characteristic_t FileType);
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
                                            SrcMgr::Characteristic_t FileType) {
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
                                         SrcMgr::Characteristic_t FileType) {
  if (Reason != PPCallbacks::EnterFile)
    return;

  const char *Filename = PP->getSourceManager().getSourceName(Loc);
  if (!FileMatchesDepCriteria(Filename, FileType))
    return;

  // Remove leading "./"
  if(Filename[0] == '.' && Filename[1] == '/')
    Filename = &Filename[2];

  Files.insert(Filename);
}

void DependencyFileCallback::OutputDependencyFile() {
  std::string Output;
  // Add "target: mainfile"
  Output += Target;
  Output += ": ";
  Output += InputFile;

  // Now add each dependency
  for (llvm::StringSet<>::iterator I = Files.begin(),
       E = Files.end(); I != E; ++I) {
      // FIXME: Wrap lines
      Output += " ";
      Output += I->getKeyData();
  }

  OS << Output << "\n";
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
}

DependencyFileCallback::~DependencyFileCallback() {
  if ((!GenerateDependencyFile && !GenerateDependencyFileNoSysHeaders) || 
      OS.fail())
    return;
  
  OutputDependencyFile();
  OS.close();
}

