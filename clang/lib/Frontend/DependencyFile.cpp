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
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
struct DepCollectorPPCallbacks : public PPCallbacks {
  DependencyCollector &DepCollector;
  SourceManager &SM;
  DepCollectorPPCallbacks(DependencyCollector &L, SourceManager &SM)
      : DepCollector(L), SM(SM) { }

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    if (Reason != PPCallbacks::EnterFile)
      return;

    // Dependency generation really does want to go all the way to the
    // file entry for a source location to find out what is depended on.
    // We do not want #line markers to affect dependency generation!
    const FileEntry *FE =
        SM.getFileEntryForID(SM.getFileID(SM.getExpansionLoc(Loc)));
    if (!FE)
      return;

    StringRef Filename = FE->getName();

    // Remove leading "./" (or ".//" or "././" etc.)
    while (Filename.size() > 2 && Filename[0] == '.' &&
           llvm::sys::path::is_separator(Filename[1])) {
      Filename = Filename.substr(1);
      while (llvm::sys::path::is_separator(Filename[0]))
        Filename = Filename.substr(1);
    }

    DepCollector.maybeAddDependency(Filename, /*FromModule*/false,
                                   FileType != SrcMgr::C_User,
                                   /*IsModuleFile*/false, /*IsMissing*/false);
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported) override {
    if (!File)
      DepCollector.maybeAddDependency(FileName, /*FromModule*/false,
                                     /*IsSystem*/false, /*IsModuleFile*/false,
                                     /*IsMissing*/true);
    // Files that actually exist are handled by FileChanged.
  }

  void EndOfMainFile() override {
    DepCollector.finishedMainFile();
  }
};

struct DepCollectorASTListener : public ASTReaderListener {
  DependencyCollector &DepCollector;
  DepCollectorASTListener(DependencyCollector &L) : DepCollector(L) { }
  bool needsInputFileVisitation() override { return true; }
  bool needsSystemInputFileVisitation() override {
    return DepCollector.needSystemDependencies();
  }
  void visitModuleFile(StringRef Filename) override {
    DepCollector.maybeAddDependency(Filename, /*FromModule*/true,
                                   /*IsSystem*/false, /*IsModuleFile*/true,
                                   /*IsMissing*/false);
  }
  bool visitInputFile(StringRef Filename, bool IsSystem,
                      bool IsOverridden) override {
    if (IsOverridden)
      return true;

    DepCollector.maybeAddDependency(Filename, /*FromModule*/true, IsSystem,
                                   /*IsModuleFile*/false, /*IsMissing*/false);
    return true;
  }
};
} // end anonymous namespace

void DependencyCollector::maybeAddDependency(StringRef Filename, bool FromModule,
                                            bool IsSystem, bool IsModuleFile,
                                            bool IsMissing) {
  if (Seen.insert(Filename) &&
      sawDependency(Filename, FromModule, IsSystem, IsModuleFile, IsMissing))
    Dependencies.push_back(Filename);
}

static bool isSpecialFilename(StringRef Filename) {
  return llvm::StringSwitch<bool>(Filename)
      .Case("<built-in>", true)
      .Case("<stdin>", true)
      .Default(false);
}

bool DependencyCollector::sawDependency(StringRef Filename, bool FromModule,
                                       bool IsSystem, bool IsModuleFile,
                                       bool IsMissing) {
  return !isSpecialFilename(Filename) &&
         (needSystemDependencies() || !IsSystem);
}

DependencyCollector::~DependencyCollector() { }
void DependencyCollector::attachToPreprocessor(Preprocessor &PP) {
  PP.addPPCallbacks(
      llvm::make_unique<DepCollectorPPCallbacks>(*this, PP.getSourceManager()));
}
void DependencyCollector::attachToASTReader(ASTReader &R) {
  R.addListener(llvm::make_unique<DepCollectorASTListener>(*this));
}

namespace {
/// Private implementation for DependencyFileGenerator
class DFGImpl : public PPCallbacks {
  std::vector<std::string> Files;
  llvm::StringSet<> FilesSet;
  const Preprocessor *PP;
  std::string OutputFile;
  std::vector<std::string> Targets;
  bool IncludeSystemHeaders;
  bool PhonyTarget;
  bool AddMissingHeaderDeps;
  bool SeenMissingHeader;
  bool IncludeModuleFiles;
private:
  bool FileMatchesDepCriteria(const char *Filename,
                              SrcMgr::CharacteristicKind FileType);
  void OutputDependencyFile();

public:
  DFGImpl(const Preprocessor *_PP, const DependencyOutputOptions &Opts)
    : PP(_PP), OutputFile(Opts.OutputFile), Targets(Opts.Targets),
      IncludeSystemHeaders(Opts.IncludeSystemHeaders),
      PhonyTarget(Opts.UsePhonyTargets),
      AddMissingHeaderDeps(Opts.AddMissingHeaderDeps),
      SeenMissingHeader(false),
      IncludeModuleFiles(Opts.IncludeModuleFiles) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported) override;

  void EndOfMainFile() override {
    OutputDependencyFile();
  }

  void AddFilename(StringRef Filename);
  bool includeSystemHeaders() const { return IncludeSystemHeaders; }
  bool includeModuleFiles() const { return IncludeModuleFiles; }
};

class DFGASTReaderListener : public ASTReaderListener {
  DFGImpl &Parent;
public:
  DFGASTReaderListener(DFGImpl &Parent)
  : Parent(Parent) { }
  bool needsInputFileVisitation() override { return true; }
  bool needsSystemInputFileVisitation() override {
    return Parent.includeSystemHeaders();
  }
  void visitModuleFile(StringRef Filename) override;
  bool visitInputFile(StringRef Filename, bool isSystem,
                      bool isOverridden) override;
};
}

DependencyFileGenerator::DependencyFileGenerator(void *Impl)
: Impl(Impl) { }

DependencyFileGenerator *DependencyFileGenerator::CreateAndAttachToPreprocessor(
    clang::Preprocessor &PP, const clang::DependencyOutputOptions &Opts) {

  if (Opts.Targets.empty()) {
    PP.getDiagnostics().Report(diag::err_fe_dependency_file_requires_MT);
    return nullptr;
  }

  // Disable the "file not found" diagnostic if the -MG option was given.
  if (Opts.AddMissingHeaderDeps)
    PP.SetSuppressIncludeNotFoundError(true);

  DFGImpl *Callback = new DFGImpl(&PP, Opts);
  PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(Callback));
  return new DependencyFileGenerator(Callback);
}

void DependencyFileGenerator::AttachToASTReader(ASTReader &R) {
  DFGImpl *I = reinterpret_cast<DFGImpl *>(Impl);
  assert(I && "missing implementation");
  R.addListener(llvm::make_unique<DFGASTReaderListener>(*I));
}

/// FileMatchesDepCriteria - Determine whether the given Filename should be
/// considered as a dependency.
bool DFGImpl::FileMatchesDepCriteria(const char *Filename,
                                     SrcMgr::CharacteristicKind FileType) {
  if (isSpecialFilename(Filename))
    return false;

  if (IncludeSystemHeaders)
    return true;

  return FileType == SrcMgr::C_User;
}

void DFGImpl::FileChanged(SourceLocation Loc,
                          FileChangeReason Reason,
                          SrcMgr::CharacteristicKind FileType,
                          FileID PrevFID) {
  if (Reason != PPCallbacks::EnterFile)
    return;

  // Dependency generation really does want to go all the way to the
  // file entry for a source location to find out what is depended on.
  // We do not want #line markers to affect dependency generation!
  SourceManager &SM = PP->getSourceManager();

  const FileEntry *FE =
    SM.getFileEntryForID(SM.getFileID(SM.getExpansionLoc(Loc)));
  if (!FE) return;

  StringRef Filename = FE->getName();
  if (!FileMatchesDepCriteria(Filename.data(), FileType))
    return;

  // Remove leading "./" (or ".//" or "././" etc.)
  while (Filename.size() > 2 && Filename[0] == '.' &&
         llvm::sys::path::is_separator(Filename[1])) {
    Filename = Filename.substr(1);
    while (llvm::sys::path::is_separator(Filename[0]))
      Filename = Filename.substr(1);
  }
    
  AddFilename(Filename);
}

void DFGImpl::InclusionDirective(SourceLocation HashLoc,
                                 const Token &IncludeTok,
                                 StringRef FileName,
                                 bool IsAngled,
                                 CharSourceRange FilenameRange,
                                 const FileEntry *File,
                                 StringRef SearchPath,
                                 StringRef RelativePath,
                                 const Module *Imported) {
  if (!File) {
    if (AddMissingHeaderDeps)
      AddFilename(FileName);
    else
      SeenMissingHeader = true;
  }
}

void DFGImpl::AddFilename(StringRef Filename) {
  if (FilesSet.insert(Filename))
    Files.push_back(Filename);
}

/// PrintFilename - GCC escapes spaces, # and $, but apparently not ' or " or
/// other scary characters.
static void PrintFilename(raw_ostream &OS, StringRef Filename) {
  for (unsigned i = 0, e = Filename.size(); i != e; ++i) {
    if (Filename[i] == ' ' || Filename[i] == '#')
      OS << '\\';
    else if (Filename[i] == '$') // $ is escaped by $$.
      OS << '$';
    OS << Filename[i];
  }
}

void DFGImpl::OutputDependencyFile() {
  if (SeenMissingHeader) {
    llvm::sys::fs::remove(OutputFile);
    return;
  }

  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFile, EC, llvm::sys::fs::F_Text);
  if (EC) {
    PP->getDiagnostics().Report(diag::err_fe_error_opening) << OutputFile
                                                            << EC.message();
    return;
  }

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
      OS << " \\\n  ";
    } else {
      Columns += N + 1;
      OS << ' ';
    }
    // Targets already quoted as needed.
    OS << *I;
  }

  OS << ':';
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
      OS << " \\\n ";
      Columns = 2;
    }
    OS << ' ';
    PrintFilename(OS, *I);
    Columns += N + 1;
  }
  OS << '\n';

  // Create phony targets if requested.
  if (PhonyTarget && !Files.empty()) {
    // Skip the first entry, this is always the input file itself.
    for (std::vector<std::string>::iterator I = Files.begin() + 1,
           E = Files.end(); I != E; ++I) {
      OS << '\n';
      PrintFilename(OS, *I);
      OS << ":\n";
    }
  }
}

bool DFGASTReaderListener::visitInputFile(llvm::StringRef Filename,
                                          bool IsSystem, bool IsOverridden) {
  assert(!IsSystem || needsSystemInputFileVisitation());
  if (IsOverridden)
    return true;

  Parent.AddFilename(Filename);
  return true;
}

void DFGASTReaderListener::visitModuleFile(llvm::StringRef Filename) {
  if (Parent.includeModuleFiles())
    Parent.AddFilename(Filename);
}
