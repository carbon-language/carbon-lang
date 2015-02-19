//===--- extra/modularize/ModularizeUtilities.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class for loading and validating a module map or
// header list by checking that all headers in the corresponding directories
// are accounted for.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "CoverageChecker.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "ModularizeUtilities.h"

using namespace clang;
using namespace llvm;
using namespace Modularize;

// Subclass TargetOptions so we can construct it inline with
// the minimal option, the triple.
class ModuleMapTargetOptions : public clang::TargetOptions {
public:
  ModuleMapTargetOptions() { Triple = llvm::sys::getDefaultTargetTriple(); }
};

// ModularizeUtilities class implementation.

// Constructor.
ModularizeUtilities::ModularizeUtilities(std::vector<std::string> &InputPaths,
                                         llvm::StringRef Prefix)
  : InputFilePaths(InputPaths),
    HeaderPrefix(Prefix),
    HasModuleMap(false),
    // Init clang stuff needed for loading the module map and preprocessing.
    LangOpts(new LangOptions()), DiagIDs(new DiagnosticIDs()),
    DiagnosticOpts(new DiagnosticOptions()),
    DC(llvm::errs(), DiagnosticOpts.get()),
    Diagnostics(
    new DiagnosticsEngine(DiagIDs, DiagnosticOpts.get(), &DC, false)),
    TargetOpts(new ModuleMapTargetOptions()),
    Target(TargetInfo::CreateTargetInfo(*Diagnostics, TargetOpts)),
    FileMgr(new FileManager(FileSystemOpts)),
    SourceMgr(new SourceManager(*Diagnostics, *FileMgr, false)),
    HeaderSearchOpts(new HeaderSearchOptions()),
    HeaderInfo(new HeaderSearch(HeaderSearchOpts, *SourceMgr, *Diagnostics,
    *LangOpts, Target.get())) {
}

// Create instance of ModularizeUtilities, to simplify setting up
// subordinate objects.
ModularizeUtilities *ModularizeUtilities::createModularizeUtilities(
    std::vector<std::string> &InputPaths, llvm::StringRef Prefix) {

  return new ModularizeUtilities(InputPaths, Prefix);
}

// Load all header lists and dependencies.
std::error_code ModularizeUtilities::loadAllHeaderListsAndDependencies() {
  typedef std::vector<std::string>::iterator Iter;
  // For each input file.
  for (Iter I = InputFilePaths.begin(), E = InputFilePaths.end(); I != E; ++I) {
    llvm::StringRef InputPath = *I;
    // If it's a module map.
    if (InputPath.endswith(".modulemap")) {
      // Load the module map.
      if (std::error_code EC = loadModuleMap(InputPath))
        return EC;
    }
    else {
      // Else we assume it's a header list and load it.
      if (std::error_code EC = loadSingleHeaderListsAndDependencies(InputPath)) {
        errs() << "modularize: error: Unable to get header list '" << InputPath
          << "': " << EC.message() << '\n';
        return EC;
      }
    }
  }
  return std::error_code();
}

// Do coverage checks.
// For each loaded module map, do header coverage check.
// Starting from the directory of the module.map file,
// Find all header files, optionally looking only at files
// covered by the include path options, and compare against
// the headers referenced by the module.map file.
// Display warnings for unaccounted-for header files.
// Returns 0 if there were no errors or warnings, 1 if there
// were warnings, 2 if any other problem, such as a bad
// module map path argument was specified.
std::error_code ModularizeUtilities::doCoverageCheck(
    std::vector<std::string> &IncludePaths,
    llvm::ArrayRef<std::string> CommandLine) {
  int ModuleMapCount = ModuleMaps.size();
  int ModuleMapIndex;
  std::error_code EC;
  for (ModuleMapIndex = 0; ModuleMapIndex < ModuleMapCount; ++ModuleMapIndex) {
    std::unique_ptr<clang::ModuleMap> &ModMap = ModuleMaps[ModuleMapIndex];
    CoverageChecker *Checker = CoverageChecker::createCoverageChecker(
      InputFilePaths[ModuleMapIndex], IncludePaths, CommandLine, ModMap.get());
    std::error_code LocalEC = Checker->doChecks();
    if (LocalEC.value() > 0)
      EC = LocalEC;
  }
  return EC;
}

// Load single header list and dependencies.
std::error_code ModularizeUtilities::loadSingleHeaderListsAndDependencies(
    llvm::StringRef InputPath) {

  // By default, use the path component of the list file name.
  SmallString<256> HeaderDirectory(InputPath);
  llvm::sys::path::remove_filename(HeaderDirectory);
  SmallString<256> CurrentDirectory;
  llvm::sys::fs::current_path(CurrentDirectory);

  // Get the prefix if we have one.
  if (HeaderPrefix.size() != 0)
    HeaderDirectory = HeaderPrefix;

  // Read the header list file into a buffer.
  ErrorOr<std::unique_ptr<MemoryBuffer>> listBuffer =
    MemoryBuffer::getFile(InputPath);
  if (std::error_code EC = listBuffer.getError())
    return EC;

  // Parse the header list into strings.
  SmallVector<StringRef, 32> Strings;
  listBuffer.get()->getBuffer().split(Strings, "\n", -1, false);

  // Collect the header file names from the string list.
  for (SmallVectorImpl<StringRef>::iterator I = Strings.begin(),
    E = Strings.end();
    I != E; ++I) {
    StringRef Line = I->trim();
    // Ignore comments and empty lines.
    if (Line.empty() || (Line[0] == '#'))
      continue;
    std::pair<StringRef, StringRef> TargetAndDependents = Line.split(':');
    SmallString<256> HeaderFileName;
    // Prepend header file name prefix if it's not absolute.
    if (llvm::sys::path::is_absolute(TargetAndDependents.first))
      llvm::sys::path::native(TargetAndDependents.first, HeaderFileName);
    else {
      if (HeaderDirectory.size() != 0)
        HeaderFileName = HeaderDirectory;
      else
        HeaderFileName = CurrentDirectory;
      llvm::sys::path::append(HeaderFileName, TargetAndDependents.first);
      llvm::sys::path::native(HeaderFileName);
    }
    // Handle optional dependencies.
    DependentsVector Dependents;
    SmallVector<StringRef, 4> DependentsList;
    TargetAndDependents.second.split(DependentsList, " ", -1, false);
    int Count = DependentsList.size();
    for (int Index = 0; Index < Count; ++Index) {
      SmallString<256> Dependent;
      if (llvm::sys::path::is_absolute(DependentsList[Index]))
        Dependent = DependentsList[Index];
      else {
        if (HeaderDirectory.size() != 0)
          Dependent = HeaderDirectory;
        else
          Dependent = CurrentDirectory;
        llvm::sys::path::append(Dependent, DependentsList[Index]);
      }
      llvm::sys::path::native(Dependent);
      Dependents.push_back(getCanonicalPath(Dependent.str()));
    }
    // Get canonical form.
    HeaderFileName = getCanonicalPath(HeaderFileName);
    // Save the resulting header file path and dependencies.
    HeaderFileNames.push_back(HeaderFileName.str());
    Dependencies[HeaderFileName.str()] = Dependents;
  }
  return std::error_code();
}

// Load single module map and extract header file list.
std::error_code ModularizeUtilities::loadModuleMap(
    llvm::StringRef InputPath) {
  // Get file entry for module.modulemap file.
  const FileEntry *ModuleMapEntry =
    SourceMgr->getFileManager().getFile(InputPath);

  // return error if not found.
  if (!ModuleMapEntry) {
    llvm::errs() << "error: File \"" << InputPath << "\" not found.\n";
    return std::error_code(1, std::generic_category());
  }

  // Because the module map parser uses a ForwardingDiagnosticConsumer,
  // which doesn't forward the BeginSourceFile call, we do it explicitly here.
  DC.BeginSourceFile(*LangOpts, nullptr);

  // Figure out the home directory for the module map file.
  const DirectoryEntry *Dir = ModuleMapEntry->getDir();
  StringRef DirName(Dir->getName());
  if (llvm::sys::path::filename(DirName) == "Modules") {
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.endswith(".framework"))
      Dir = FileMgr->getDirectory(DirName);
    // FIXME: This assert can fail if there's a race between the above check
    // and the removal of the directory.
    assert(Dir && "parent must exist");
  }

  std::unique_ptr<ModuleMap> ModMap;
  ModMap.reset(new ModuleMap(*SourceMgr, *Diagnostics, *LangOpts,
    Target.get(), *HeaderInfo));

  // Parse module.modulemap file into module map.
  if (ModMap->parseModuleMapFile(ModuleMapEntry, false, Dir)) {
    return std::error_code(1, std::generic_category());
  }

  // Do matching end call.
  DC.EndSourceFile();

  if (!collectModuleMapHeaders(ModMap.get()))
    return std::error_code(1, std::generic_category());

  // Save module map.
  ModuleMaps.push_back(std::move(ModMap));

  // Indicate we are using module maps.
  HasModuleMap = true;

  return std::error_code();
}

// Collect module map headers.
// Walks the modules and collects referenced headers into
// HeaderFileNames.
bool ModularizeUtilities::collectModuleMapHeaders(clang::ModuleMap *ModMap) {
  for (ModuleMap::module_iterator I = ModMap->module_begin(),
    E = ModMap->module_end();
    I != E; ++I) {
    if (!collectModuleHeaders(*I->second))
      return false;
  }
  return true;
}

// Collect referenced headers from one module.
// Collects the headers referenced in the given module into
// HeaderFileNames.
bool ModularizeUtilities::collectModuleHeaders(const Module &Mod) {

  // Ignore explicit modules because they often have dependencies
  // we can't know.
  if (Mod.IsExplicit)
    return true;

  // Treat headers in umbrella directory as dependencies.
  DependentsVector UmbrellaDependents;

  // Recursively do submodules.
  for (Module::submodule_const_iterator MI = Mod.submodule_begin(),
      MIEnd = Mod.submodule_end();
      MI != MIEnd; ++MI)
    collectModuleHeaders(**MI);

  if (const FileEntry *UmbrellaHeader = Mod.getUmbrellaHeader()) {
    std::string HeaderPath = getCanonicalPath(UmbrellaHeader->getName());
    // Collect umbrella header.
    HeaderFileNames.push_back(HeaderPath);

    // FUTURE: When needed, umbrella header header collection goes here.
  }
  else if (const DirectoryEntry *UmbrellaDir = Mod.getUmbrellaDir()) {
    // If there normal headers, assume these are umbrellas and skip collection.
    if (Mod.Headers->size() == 0) {
      // Collect headers in umbrella directory.
      if (!collectUmbrellaHeaders(UmbrellaDir->getName(), UmbrellaDependents))
        return false;
    }
  }

  // We ignore HK_Private, HK_Textual, HK_PrivateTextual, and HK_Excluded,
  // assuming they are marked as such either because of unsuitability for
  // modules or because they are meant to be included by another header,
  // and thus should be ignored by modularize.

  int NormalHeaderCount = Mod.Headers[clang::Module::HK_Normal].size();

  for (int Index = 0; Index < NormalHeaderCount; ++Index) {
    DependentsVector NormalDependents;
    // Collect normal header.
    const clang::Module::Header &Header(
      Mod.Headers[clang::Module::HK_Normal][Index]);
    std::string HeaderPath = getCanonicalPath(Header.Entry->getName());
    HeaderFileNames.push_back(HeaderPath);
  }

  return true;
}

// Collect headers from an umbrella directory.
bool ModularizeUtilities::collectUmbrellaHeaders(StringRef UmbrellaDirName,
  DependentsVector &Dependents) {
  // Initialize directory name.
  SmallString<256> Directory(UmbrellaDirName);
  // Walk the directory.
  std::error_code EC;
  llvm::sys::fs::file_status Status;
  for (llvm::sys::fs::directory_iterator I(Directory.str(), EC), E; I != E;
    I.increment(EC)) {
    if (EC)
      return false;
    std::string File(I->path());
    I->status(Status);
    llvm::sys::fs::file_type Type = Status.type();
    // If the file is a directory, ignore the name and recurse.
    if (Type == llvm::sys::fs::file_type::directory_file) {
      if (!collectUmbrellaHeaders(File, Dependents))
        return false;
      continue;
    }
    // If the file does not have a common header extension, ignore it.
    if (!isHeader(File))
      continue;
    // Save header name.
    std::string HeaderPath = getCanonicalPath(File);
    Dependents.push_back(HeaderPath);
  }
  return true;
}

// Convert header path to canonical form.
// The canonical form is basically just use forward slashes, and remove "./".
// \param FilePath The file path, relative to the module map directory.
// \returns The file path in canonical form.
std::string ModularizeUtilities::getCanonicalPath(StringRef FilePath) {
  std::string Tmp(FilePath);
  std::replace(Tmp.begin(), Tmp.end(), '\\', '/');
  StringRef Tmp2(Tmp);
  if (Tmp2.startswith("./"))
    Tmp = Tmp2.substr(2);
  return Tmp;
}

// Check for header file extension.
// If the file extension is .h, .inc, or missing, it's
// assumed to be a header.
// \param FileName The file name.  Must not be a directory.
// \returns true if it has a header extension or no extension.
bool ModularizeUtilities::isHeader(StringRef FileName) {
  StringRef Extension = llvm::sys::path::extension(FileName);
  if (Extension.size() == 0)
    return false;
  if (Extension.equals_lower(".h"))
    return true;
  if (Extension.equals_lower(".inc"))
    return true;
  return false;
}

// Get directory path component from file path.
// \returns the component of the given path, which will be
// relative if the given path is relative, absolute if the
// given path is absolute, or "." if the path has no leading
// path component.
std::string ModularizeUtilities::getDirectoryFromPath(StringRef Path) {
  SmallString<256> Directory(Path);
  sys::path::remove_filename(Directory);
  if (Directory.size() == 0)
    return ".";
  return Directory.str();
}
