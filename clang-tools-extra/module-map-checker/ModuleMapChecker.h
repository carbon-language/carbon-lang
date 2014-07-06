//===-- ModuleMapChecker.h - Common defs for module-map-checker -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//
///
/// \file
/// \brief Common definitions for ModuleMapChecker.
///
//===--------------------------------------------------------------------===//

#ifndef MODULEMAPCHECKER_H
#define MODULEMAPCHECKER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Host.h"
#include <string>
#include <vector>

/// Subclass TargetOptions so we can construct it inline with
/// the minimal option, the triple.
class ModuleMapTargetOptions : public clang::TargetOptions {
public:
  ModuleMapTargetOptions() { Triple = llvm::sys::getDefaultTargetTriple(); }
};

/// Module map checker class.
/// This is the heart of the checker.
/// The doChecks function does the main work.
/// The data members store the options and internally collected data.
class ModuleMapChecker {
  // Checker arguments.

  /// The module.map file path. Can be relative or absolute.
  llvm::StringRef ModuleMapPath;
  /// The include paths to check for files.
  /// (Note that other directories above these paths are ignored.
  /// To expect all files to be accounted for from the module.map
  /// file directory on down, leave this empty.)
  std::vector<std::string> IncludePaths;
  /// Flag to dump the module map information during check.
  bool DumpModuleMap;
  /// The remaining arguments, to be passed to the front end.
  llvm::ArrayRef<std::string> CommandLine;

  // Supporting objects.

  /// Options controlling the language variant.
  std::shared_ptr<clang::LangOptions> LangOpts;
  /// Diagnostic IDs.
  const llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs;
  /// Options controlling the diagnostic engine.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagnosticOpts;
  /// Diagnostic consumer.
  clang::TextDiagnosticPrinter DC;
  /// Diagnostic engine.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> Diagnostics;
  /// Options controlling the target.
  std::shared_ptr<clang::TargetOptions> TargetOpts;
  /// Target information.
  llvm::IntrusiveRefCntPtr<clang::TargetInfo> Target;
  /// Options controlling the file system manager.
  clang::FileSystemOptions FileSystemOpts;
  /// File system manager.
  llvm::IntrusiveRefCntPtr<clang::FileManager> FileMgr;
  /// Source manager.
  llvm::IntrusiveRefCntPtr<clang::SourceManager> SourceMgr;
  /// Options controlling the \#include directive.
  llvm::IntrusiveRefCntPtr<clang::HeaderSearchOptions> HeaderSearchOpts;
  /// Header search manager.
  std::unique_ptr<clang::HeaderSearch> HeaderInfo;
  /// The module map.
  std::unique_ptr<clang::ModuleMap> ModMap;

  // Internal data.

  /// Directory containing the module map.
  /// Might be relative to the current directory, or absolute.
  std::string ModuleMapDirectory;
  /// Set of all the headers found in the module map.
  llvm::StringSet<llvm::MallocAllocator> ModuleMapHeadersSet;
  /// All the headers found in the file system starting at the
  /// module map, or the union of those from the include paths.
  std::vector<std::string> FileSystemHeaders;
  /// Headers found in file system, but not in module map.
  std::vector<std::string> UnaccountedForHeaders;

public:
  /// Constructor.
  /// You can use the static createModuleMapChecker to create an instance
  /// of this object.
  /// \param ModuleMapPath The module.map file path.
  ///   Can be relative or absolute.
  /// \param IncludePaths The include paths to check for files.
  ///   (Note that other directories above these paths are ignored.
  ///   To expect all files to be accounted for from the module.map
  ///   file directory on down, leave this empty.)
  /// \param DumpModuleMap Flag to dump the module map information
  ///   during check.
  ModuleMapChecker(llvm::StringRef ModuleMapPath,
                   std::vector<std::string> &IncludePaths, bool DumpModuleMap,
                   llvm::ArrayRef<std::string> CommandLine);

  /// Create instance of ModuleMapChecker.
  /// \param ModuleMapPath The module.map file path.
  ///   Can be relative or absolute.
  /// \param IncludePaths The include paths to check for files.
  ///   (Note that other directories above these paths are ignored.
  ///   To expect all files to be accounted for from the module.map
  ///   file directory on down, leave this empty.)
  /// \param DumpModuleMap Flag to dump the module map information
  ///   during check.
  /// \returns Initialized ModuleMapChecker object.
  static ModuleMapChecker *createModuleMapChecker(
      llvm::StringRef ModuleMapPath, std::vector<std::string> &IncludePaths,
      bool DumpModuleMap, llvm::ArrayRef<std::string> CommandLine);

  /// Do checks.
  /// Starting from the directory of the module.map file,
  /// Find all header files, optionally looking only at files
  /// covered by the include path options, and compare against
  /// the headers referenced by the module.map file.
  /// Display warnings for unaccounted-for header files.
  /// \returns 0 if there were no errors or warnings, 1 if there
  ///   were warnings, 2 if any other problem, such as a bad
  ///   module map path argument was specified.
  std::error_code doChecks();

  // The following functions are called by doChecks.

  /// Load module map.
  /// \returns True if module.map file loaded successfully.
  bool loadModuleMap();

  /// Collect module headers.
  /// Walks the modules and collects referenced headers into
  /// ModuleMapHeadersSet.
  void collectModuleHeaders();

  /// Collect referenced headers from one module.
  /// Collects the headers referenced in the given module into
  /// ModuleMapHeadersSet.
  /// \param Mod The module reference.
  /// \return True if no errors.
  bool collectModuleHeaders(const clang::Module &Mod);

  /// Collect headers from an umbrella directory.
  /// \param UmbrellaDirName The umbrella directory name.
  /// \return True if no errors.
  bool collectUmbrellaHeaders(llvm::StringRef UmbrellaDirName);

  /// Collect headers rferenced from an umbrella file.
  /// \param UmbrellaHeaderName The umbrella file path.
  /// \return True if no errors.
  bool collectUmbrellaHeaderHeaders(llvm::StringRef UmbrellaHeaderName);

  /// Called from ModuleMapCheckerCallbacks to track a header included
  /// from an umbrella header.
  /// \param HeaderName The header file path.
  void collectUmbrellaHeaderHeader(llvm::StringRef HeaderName);

  /// Collect file system header files.
  /// This function scans the file system for header files,
  /// starting at the directory of the module.map file,
  /// optionally filtering out all but the files covered by
  /// the include path options.
  /// \returns True if no errors.
  bool collectFileSystemHeaders();

  /// Collect file system header files from the given path.
  /// This function scans the file system for header files,
  /// starting at the given directory, which is assumed to be
  /// relative to the directory of the module.map file.
  /// \returns True if no errors.
  bool collectFileSystemHeaders(llvm::StringRef IncludePath);

  /// Find headers unaccounted-for in module map.
  /// This function compares the list of collected header files
  /// against those referenced in the module map.  Display
  /// warnings for unaccounted-for header files.
  /// Save unaccounted-for file list for possible.
  /// fixing action.
  void findUnaccountedForHeaders();

  // Utility functions.

  /// Get directory path component from file path.
  /// \returns the component of the given path, which will be
  /// relative if the given path is relative, absolute if the
  /// given path is absolute, or "." if the path has no leading
  /// path component.
  std::string getDirectoryFromPath(llvm::StringRef Path);

  /// Convert header path to canonical form.
  /// The canonical form is basically just use forward slashes,
  /// and remove "./".
  /// \param FilePath The file path.
  /// \returns The file path in canonical form.
  std::string getCanonicalPath(llvm::StringRef FilePath);

  /// Check for header file extension.
  /// If the file extension is .h, .inc, or missing, it's
  /// assumed to be a header.
  /// \param FileName The file name.  Must not be a directory.
  /// \returns true if it has a header extension or no extension.
  bool isHeader(llvm::StringRef FileName);
};

#endif // MODULEMAPCHECKER_H
