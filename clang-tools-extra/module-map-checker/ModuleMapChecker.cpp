//===--- extra/module-map-checker/ModuleMapChecker.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool that validates a module map by checking that
// all headers in the corresponding directories are accounted for.
//
// Usage:   module-map-checker [(module-map-checker options)]
//            (module-map-file) [(front end options)]
//
// Options:
//
//    -I(include path)      Look at headers only in this directory tree.
//                          Must be a path relative to the module.map file.
//                          There can be multiple -I options, for when the
//                          module map covers multiple directories, and
//                          excludes higher or sibling directories not
//                          specified. If this option is omitted, the
//                          directory containing the module-map-file is
//                          the root of the header tree to be searched for
//                          headers.
//
//    -dump-module-map      Dump the module map object during the check.
//                          This displays the modules and headers.
//
//    (front end options)   In the case of use of an umbrella header, this can
//                          be used to pass options to the compiler front end
//                          preprocessor, such as -D or -I options.
//
// This program uses the Clang ModuleMap class to read and parse the module
// map file.  Starting at the module map file directory, or just the include
// paths, if specified, it will collect the names of all the files it
// considers headers (no extension, .h, or .inc--if you need more, modify the
// isHeader function).  It then compares the headers against those referenced
// in the module map, either explicitly named, or implicitly named via an
// umbrella directory or umbrella file, as parsed by the ModuleMap object.
// If headers are found which are not referenced or covered by an umbrella
// directory or file, warning messages will be produced, and this program
// will return an error code of 1.  Other errors result in an error code of 2.
// If no problems are found, an error code of 0 is returned.
//
// Note that in the case of umbrella headers, this tool invokes the compiler
// to preprocess the file, and uses a callback to collect the header files
// included by the umbrella header or any of its nested includes.  If any
// front end options are needed for these compiler invocations, these
// can be included on the command line after the module map file argument.
//
// Warning message have the form:
//
//  warning: module.map does not account for file: Level3A.h
//
// Note that for the case of the module map referencing a file that does
// not exist, the module map parser in Clang will (at the time of this
// writing) display an error message.
//
// Potential problems with this program:
//
// 1. Might need a better header matching mechanism, or extensions to the
//    canonical file format used.
//
// 2. It might need to support additional header file extensions.
//
// Future directions:
//
// 1. Add an option to fix the problems found, writing a new module map.
//    Include an extra option to add unaccounted-for headers as excluded.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "ModuleMapChecker.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace clang::tooling;
namespace cl = llvm::cl;
namespace sys = llvm::sys;

// Option for include paths.
static cl::list<std::string>
IncludePaths("I", cl::desc("Include path."
                           " Must be relative to module.map file."),
             cl::ZeroOrMore, cl::value_desc("path"));

// Option for dumping the parsed module map.
static cl::opt<bool>
DumpModuleMap("dump-module-map", cl::init(false),
              cl::desc("Dump the parsed module map information."));

// Option for module.map path.
static cl::opt<std::string>
ModuleMapPath(cl::Positional, cl::init("module.map"),
              cl::desc("<The module.map file path."
                       " Uses module.map in current directory if omitted.>"));

// Collect all other arguments, which will be passed to the front end.
static cl::list<std::string>
CC1Arguments(cl::ConsumeAfter, cl::desc("<arguments to be passed to front end "
                                        "for parsing umbrella headers>..."));

int main(int Argc, const char **Argv) {

  // Parse command line.
  cl::ParseCommandLineOptions(Argc, Argv, "module-map-checker.\n");

  // Create checker object.
  std::unique_ptr<ModuleMapChecker> Checker(
      ModuleMapChecker::createModuleMapChecker(ModuleMapPath, IncludePaths,
                                               DumpModuleMap, CC1Arguments));

  // Do the checks.  The return value is the program return code,
  // 0 for okay, 1 for module map warnings produced, 2 for any other error.
  std::error_code ReturnCode = Checker->doChecks();

  if (ReturnCode == std::error_code(1, std::generic_category()))
    return 1; // Module map warnings were issued.
  else if (ReturnCode == std::error_code(2, std::generic_category()))
    return 2; // Some other error occurred.
  else
    return 0; // No errors or warnings.
}

// Preprocessor callbacks.
// We basically just collect include files.
class ModuleMapCheckerCallbacks : public PPCallbacks {
public:
  ModuleMapCheckerCallbacks(ModuleMapChecker &Checker) : Checker(Checker) {}
  ~ModuleMapCheckerCallbacks() {}

  // Include directive callback.
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported) {
    Checker.collectUmbrellaHeaderHeader(File->getName());
  }

private:
  ModuleMapChecker &Checker;
};

// Frontend action stuff:

// Consumer is responsible for setting up the callbacks.
class ModuleMapCheckerConsumer : public ASTConsumer {
public:
  ModuleMapCheckerConsumer(ModuleMapChecker &Checker, Preprocessor &PP) {
    // PP takes ownership.
    PP.addPPCallbacks(llvm::make_unique<ModuleMapCheckerCallbacks>(Checker));
  }
};

class ModuleMapCheckerAction : public SyntaxOnlyAction {
public:
  ModuleMapCheckerAction(ModuleMapChecker &Checker) : Checker(Checker) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<ModuleMapCheckerConsumer>(Checker,
                                                       CI.getPreprocessor());
  }

private:
  ModuleMapChecker &Checker;
};

class ModuleMapCheckerFrontendActionFactory : public FrontendActionFactory {
public:
  ModuleMapCheckerFrontendActionFactory(ModuleMapChecker &Checker)
      : Checker(Checker) {}

  virtual ModuleMapCheckerAction *create() {
    return new ModuleMapCheckerAction(Checker);
  }

private:
  ModuleMapChecker &Checker;
};

// ModuleMapChecker class implementation.

// Constructor.
ModuleMapChecker::ModuleMapChecker(StringRef ModuleMapPath,
                                   std::vector<std::string> &IncludePaths,
                                   bool DumpModuleMap,
                                   ArrayRef<std::string> CommandLine)
    : ModuleMapPath(ModuleMapPath), IncludePaths(IncludePaths),
      DumpModuleMap(DumpModuleMap), CommandLine(CommandLine),
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
                                  *LangOpts, Target.get())),
      ModMap(new ModuleMap(*SourceMgr, *Diagnostics, *LangOpts, Target.get(),
                           *HeaderInfo)) {}

// Create instance of ModuleMapChecker, to simplify setting up
// subordinate objects.
ModuleMapChecker *ModuleMapChecker::createModuleMapChecker(
    StringRef ModuleMapPath, std::vector<std::string> &IncludePaths,
    bool DumpModuleMap, ArrayRef<std::string> CommandLine) {

  return new ModuleMapChecker(ModuleMapPath, IncludePaths, DumpModuleMap,
                              CommandLine);
}

// Do checks.
// Starting from the directory of the module.map file,
// Find all header files, optionally looking only at files
// covered by the include path options, and compare against
// the headers referenced by the module.map file.
// Display warnings for unaccounted-for header files.
// Returns error_code of 0 if there were no errors or warnings, 1 if there
//   were warnings, 2 if any other problem, such as if a bad
//   module map path argument was specified.
std::error_code ModuleMapChecker::doChecks() {
  std::error_code returnValue;

  // Load the module map.
  if (!loadModuleMap())
    return std::error_code(2, std::generic_category());

  // Collect the headers referenced in the modules.
  collectModuleHeaders();

  // Collect the file system headers.
  if (!collectFileSystemHeaders())
    return std::error_code(2, std::generic_category());

  // Do the checks.  These save the problematic file names.
  findUnaccountedForHeaders();

  // Check for warnings.
  if (UnaccountedForHeaders.size())
    returnValue = std::error_code(1, std::generic_category());

  // Dump module map if requested.
  if (DumpModuleMap) {
    llvm::errs() << "\nDump of module map:\n\n";
    ModMap->dump();
  }

  return returnValue;
}

// The following functions are called by doChecks.

// Load module map.
// Returns true if module.map file loaded successfully.
bool ModuleMapChecker::loadModuleMap() {
  // Get file entry for module.map file.
  const FileEntry *ModuleMapEntry =
      SourceMgr->getFileManager().getFile(ModuleMapPath);

  // return error if not found.
  if (!ModuleMapEntry) {
    llvm::errs() << "error: File \"" << ModuleMapPath << "\" not found.\n";
    return false;
  }

  // Because the module map parser uses a ForwardingDiagnosticConsumer,
  // which doesn't forward the BeginSourceFile call, we do it explicitly here.
  DC.BeginSourceFile(*LangOpts, nullptr);

  // Parse module.map file into module map.
  if (ModMap->parseModuleMapFile(ModuleMapEntry, false))
    return false;

  // Do matching end call.
  DC.EndSourceFile();

  return true;
}

// Collect module headers.
// Walks the modules and collects referenced headers into
// ModuleMapHeadersSet.
void ModuleMapChecker::collectModuleHeaders() {
  for (ModuleMap::module_iterator I = ModMap->module_begin(),
                                  E = ModMap->module_end();
       I != E; ++I) {
    collectModuleHeaders(*I->second);
  }
}

// Collect referenced headers from one module.
// Collects the headers referenced in the given module into
// ModuleMapHeadersSet.
// FIXME: Doesn't collect files from umbrella header.
bool ModuleMapChecker::collectModuleHeaders(const Module &Mod) {

  if (const FileEntry *UmbrellaHeader = Mod.getUmbrellaHeader()) {
    // Collect umbrella header.
    ModuleMapHeadersSet.insert(getCanonicalPath(UmbrellaHeader->getName()));
    // Preprocess umbrella header and collect the headers it references.
    if (!collectUmbrellaHeaderHeaders(UmbrellaHeader->getName()))
      return false;
  } else if (const DirectoryEntry *UmbrellaDir = Mod.getUmbrellaDir()) {
    // Collect headers in umbrella directory.
    if (!collectUmbrellaHeaders(UmbrellaDir->getName()))
      return false;
  }

  for (unsigned I = 0, N = Mod.NormalHeaders.size(); I != N; ++I) {
    ModuleMapHeadersSet.insert(
        getCanonicalPath(Mod.NormalHeaders[I]->getName()));
  }

  for (unsigned I = 0, N = Mod.ExcludedHeaders.size(); I != N; ++I) {
    ModuleMapHeadersSet.insert(
        getCanonicalPath(Mod.ExcludedHeaders[I]->getName()));
  }

  for (unsigned I = 0, N = Mod.PrivateHeaders.size(); I != N; ++I) {
    ModuleMapHeadersSet.insert(
        getCanonicalPath(Mod.PrivateHeaders[I]->getName()));
  }

  for (Module::submodule_const_iterator MI = Mod.submodule_begin(),
                                        MIEnd = Mod.submodule_end();
       MI != MIEnd; ++MI)
    collectModuleHeaders(**MI);

  return true;
}

// Collect headers from an umbrella directory.
bool ModuleMapChecker::collectUmbrellaHeaders(StringRef UmbrellaDirName) {
  // Initialize directory name.
  SmallString<256> Directory(ModuleMapDirectory);
  if (UmbrellaDirName.size())
    sys::path::append(Directory, UmbrellaDirName);
  if (Directory.size() == 0)
    Directory = ".";
  // Walk the directory.
  std::error_code EC;
  sys::fs::file_status Status;
  for (sys::fs::directory_iterator I(Directory.str(), EC), E; I != E;
       I.increment(EC)) {
    if (EC)
      return false;
    std::string File(I->path());
    I->status(Status);
    sys::fs::file_type Type = Status.type();
    // If the file is a directory, ignore the name.
    if (Type == sys::fs::file_type::directory_file)
      continue;
    // If the file does not have a common header extension, ignore it.
    if (!isHeader(File))
      continue;
    // Save header name.
    ModuleMapHeadersSet.insert(getCanonicalPath(File));
  }
  return true;
}

// Collect headers rferenced from an umbrella file.
bool
ModuleMapChecker::collectUmbrellaHeaderHeaders(StringRef UmbrellaHeaderName) {

  SmallString<256> PathBuf(ModuleMapDirectory);

  // If directory is empty, it's the current directory.
  if (ModuleMapDirectory.length() == 0)
    sys::fs::current_path(PathBuf);

  // Create the compilation database.
  std::unique_ptr<CompilationDatabase> Compilations;
  Compilations.reset(new FixedCompilationDatabase(Twine(PathBuf), CommandLine));

  std::vector<std::string> HeaderPath;
  HeaderPath.push_back(UmbrellaHeaderName);

  // Create the tool and run the compilation.
  ClangTool Tool(*Compilations, HeaderPath);
  int HadErrors = Tool.run(new ModuleMapCheckerFrontendActionFactory(*this));

  // If we had errors, exit early.
  return HadErrors ? false : true;
}

// Called from ModuleMapCheckerCallbacks to track a header included
// from an umbrella header.
void ModuleMapChecker::collectUmbrellaHeaderHeader(StringRef HeaderName) {

  SmallString<256> PathBuf(ModuleMapDirectory);
  // If directory is empty, it's the current directory.
  if (ModuleMapDirectory.length() == 0)
    sys::fs::current_path(PathBuf);
  // HeaderName will have an absolute path, so if it's the module map
  // directory, we remove it, also skipping trailing separator.
  if (HeaderName.startswith(PathBuf))
    HeaderName = HeaderName.substr(PathBuf.size() + 1);
  // Save header name.
  ModuleMapHeadersSet.insert(getCanonicalPath(HeaderName));
}

// Collect file system header files.
// This function scans the file system for header files,
// starting at the directory of the module.map file,
// optionally filtering out all but the files covered by
// the include path options.
// Returns true if no errors.
bool ModuleMapChecker::collectFileSystemHeaders() {

  // Get directory containing the module.map file.
  // Might be relative to current directory, absolute, or empty.
  ModuleMapDirectory = getDirectoryFromPath(ModuleMapPath);

  // If no include paths specified, we do the whole tree starting
  // at the module.map directory.
  if (IncludePaths.size() == 0) {
    if (!collectFileSystemHeaders(StringRef("")))
      return false;
  } else {
    // Otherwise we only look at the sub-trees specified by the
    // include paths.
    for (std::vector<std::string>::const_iterator I = IncludePaths.begin(),
                                                  E = IncludePaths.end();
         I != E; ++I) {
      if (!collectFileSystemHeaders(*I))
        return false;
    }
  }

  // Sort it, because different file systems might order the file differently.
  std::sort(FileSystemHeaders.begin(), FileSystemHeaders.end());

  return true;
}

// Collect file system header files from the given path.
// This function scans the file system for header files,
// starting at the given directory, which is assumed to be
// relative to the directory of the module.map file.
// \returns True if no errors.
bool ModuleMapChecker::collectFileSystemHeaders(StringRef IncludePath) {

  // Initialize directory name.
  SmallString<256> Directory(ModuleMapDirectory);
  if (IncludePath.size())
    sys::path::append(Directory, IncludePath);
  if (Directory.size() == 0)
    Directory = ".";
  if (IncludePath.startswith("/") || IncludePath.startswith("\\") ||
      ((IncludePath.size() >= 2) && (IncludePath[1] == ':'))) {
    llvm::errs() << "error: Include path \"" << IncludePath
                 << "\" is not relative to the module map file.\n";
    return false;
  }

  // Recursively walk the directory tree.
  std::error_code EC;
  sys::fs::file_status Status;
  int Count = 0;
  for (sys::fs::recursive_directory_iterator I(Directory.str(), EC), E; I != E;
       I.increment(EC)) {
    if (EC)
      return false;
    std::string file(I->path());
    I->status(Status);
    sys::fs::file_type type = Status.type();
    // If the file is a directory, ignore the name (but still recurses).
    if (type == sys::fs::file_type::directory_file)
      continue;
    // If the file does not have a common header extension, ignore it.
    if (!isHeader(file))
      continue;
    // Save header name.
    FileSystemHeaders.push_back(getCanonicalPath(file));
    Count++;
  }
  if (Count == 0) {
    llvm::errs() << "warning: No headers found in include path: \""
                 << IncludePath << "\"\n";
  }
  return true;
}

// Find headers unaccounted-for in module map.
// This function compares the list of collected header files
// against those referenced in the module map.  Display
// warnings for unaccounted-for header files.
// Save unaccounted-for file list for possible.
// fixing action.
// FIXME: There probably needs to be some canonalization
// of file names so that header path can be correctly
// matched.  Also, a map could be used for the headers
// referenced in the module, but
void ModuleMapChecker::findUnaccountedForHeaders() {
  // Walk over file system headers.
  for (std::vector<std::string>::const_iterator I = FileSystemHeaders.begin(),
                                                E = FileSystemHeaders.end();
       I != E; ++I) {
    // Look for header in module map.
    if (ModuleMapHeadersSet.insert(*I)) {
      UnaccountedForHeaders.push_back(*I);
      llvm::errs() << "warning: " << ModuleMapPath
                   << " does not account for file: " << *I << "\n";
    }
  }
}

// Utility functions.

// Get directory path component from file path.
// \returns the component of the given path, which will be
// relative if the given path is relative, absolute if the
// given path is absolute, or "." if the path has no leading
// path component.
std::string ModuleMapChecker::getDirectoryFromPath(StringRef Path) {
  SmallString<256> Directory(Path);
  sys::path::remove_filename(Directory);
  if (Directory.size() == 0)
    return ".";
  return Directory.str();
}

// Convert header path to canonical form.
// The canonical form is basically just use forward slashes, and remove "./".
// \param FilePath The file path, relative to the module map directory.
// \returns The file path in canonical form.
std::string ModuleMapChecker::getCanonicalPath(StringRef FilePath) {
  std::string Tmp(FilePath);
  std::replace(Tmp.begin(), Tmp.end(), '\\', '/');
  StringRef Result(Tmp);
  if (Result.startswith("./"))
    Result = Result.substr(2);
  return Result;
}

// Check for header file extension.
// If the file extension is .h, .inc, or missing, it's
// assumed to be a header.
// \param FileName The file name.  Must not be a directory.
// \returns true if it has a header extension or no extension.
bool ModuleMapChecker::isHeader(StringRef FileName) {
  StringRef Extension = sys::path::extension(FileName);
  if (Extension.size() == 0)
    return false;
  if (Extension.equals_lower(".h"))
    return true;
  if (Extension.equals_lower(".inc"))
    return true;
  return false;
}
