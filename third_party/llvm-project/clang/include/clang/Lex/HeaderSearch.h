//===- HeaderSearch.h - Resolve Header File Locations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HeaderSearch interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_HEADERSEARCH_H
#define LLVM_CLANG_LEX_HEADERSEARCH_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/HeaderMap.h"
#include "clang/Lex/ModuleMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

class Triple;

} // namespace llvm

namespace clang {

class DiagnosticsEngine;
class DirectoryEntry;
class ExternalPreprocessorSource;
class FileEntry;
class FileManager;
class HeaderSearch;
class HeaderSearchOptions;
class IdentifierInfo;
class LangOptions;
class Module;
class Preprocessor;
class TargetInfo;

/// The preprocessor keeps track of this information for each
/// file that is \#included.
struct HeaderFileInfo {
  // TODO: Whether the file was imported is not a property of the file itself.
  // It's a preprocessor state, move it there.
  /// True if this is a \#import'd file.
  unsigned isImport : 1;

  /// True if this is a \#pragma once file.
  unsigned isPragmaOnce : 1;

  /// Keep track of whether this is a system header, and if so,
  /// whether it is C++ clean or not.  This can be set by the include paths or
  /// by \#pragma gcc system_header.  This is an instance of
  /// SrcMgr::CharacteristicKind.
  unsigned DirInfo : 3;

  /// Whether this header file info was supplied by an external source,
  /// and has not changed since.
  unsigned External : 1;

  /// Whether this header is part of a module.
  unsigned isModuleHeader : 1;

  /// Whether this header is part of the module that we are building.
  unsigned isCompilingModuleHeader : 1;

  /// Whether this structure is considered to already have been
  /// "resolved", meaning that it was loaded from the external source.
  unsigned Resolved : 1;

  /// Whether this is a header inside a framework that is currently
  /// being built.
  ///
  /// When a framework is being built, the headers have not yet been placed
  /// into the appropriate framework subdirectories, and therefore are
  /// provided via a header map. This bit indicates when this is one of
  /// those framework headers.
  unsigned IndexHeaderMapHeader : 1;

  /// Whether this file has been looked up as a header.
  unsigned IsValid : 1;

  /// The ID number of the controlling macro.
  ///
  /// This ID number will be non-zero when there is a controlling
  /// macro whose IdentifierInfo may not yet have been loaded from
  /// external storage.
  unsigned ControllingMacroID = 0;

  /// If this file has a \#ifndef XXX (or equivalent) guard that
  /// protects the entire contents of the file, this is the identifier
  /// for the macro that controls whether or not it has any effect.
  ///
  /// Note: Most clients should use getControllingMacro() to access
  /// the controlling macro of this header, since
  /// getControllingMacro() is able to load a controlling macro from
  /// external storage.
  const IdentifierInfo *ControllingMacro = nullptr;

  /// If this header came from a framework include, this is the name
  /// of the framework.
  StringRef Framework;

  /// List of aliases that this header is known as.
  /// Most headers should only have at most one alias, but a handful
  /// have two.
  llvm::SetVector<llvm::SmallString<32>,
                  llvm::SmallVector<llvm::SmallString<32>, 2>,
                  llvm::SmallSet<llvm::SmallString<32>, 2>>
      Aliases;

  HeaderFileInfo()
      : isImport(false), isPragmaOnce(false), DirInfo(SrcMgr::C_User),
        External(false), isModuleHeader(false), isCompilingModuleHeader(false),
        Resolved(false), IndexHeaderMapHeader(false), IsValid(false)  {}

  /// Retrieve the controlling macro for this header file, if
  /// any.
  const IdentifierInfo *
  getControllingMacro(ExternalPreprocessorSource *External);
};

/// An external source of header file information, which may supply
/// information about header files already included.
class ExternalHeaderFileInfoSource {
public:
  virtual ~ExternalHeaderFileInfoSource();

  /// Retrieve the header file information for the given file entry.
  ///
  /// \returns Header file information for the given file entry, with the
  /// \c External bit set. If the file entry is not known, return a
  /// default-constructed \c HeaderFileInfo.
  virtual HeaderFileInfo GetHeaderFileInfo(const FileEntry *FE) = 0;
};

/// This structure is used to record entries in our framework cache.
struct FrameworkCacheEntry {
  /// The directory entry which should be used for the cached framework.
  const DirectoryEntry *Directory;

  /// Whether this framework has been "user-specified" to be treated as if it
  /// were a system framework (even if it was found outside a system framework
  /// directory).
  bool IsUserSpecifiedSystemFramework;
};

namespace detail {
template <bool Const, typename T>
using Qualified = std::conditional_t<Const, const T, T>;

/// Forward iterator over the search directories of \c HeaderSearch.
template <bool IsConst>
struct SearchDirIteratorImpl
    : llvm::iterator_facade_base<SearchDirIteratorImpl<IsConst>,
                                 std::forward_iterator_tag,
                                 Qualified<IsConst, DirectoryLookup>> {
  /// Const -> non-const iterator conversion.
  template <typename Enable = std::enable_if<IsConst, bool>>
  SearchDirIteratorImpl(const SearchDirIteratorImpl<false> &Other)
      : HS(Other.HS), Idx(Other.Idx) {}

  SearchDirIteratorImpl(const SearchDirIteratorImpl &) = default;

  SearchDirIteratorImpl &operator=(const SearchDirIteratorImpl &) = default;

  bool operator==(const SearchDirIteratorImpl &RHS) const {
    return HS == RHS.HS && Idx == RHS.Idx;
  }

  SearchDirIteratorImpl &operator++() {
    assert(*this && "Invalid iterator.");
    ++Idx;
    return *this;
  }

  Qualified<IsConst, DirectoryLookup> &operator*() const {
    assert(*this && "Invalid iterator.");
    return HS->SearchDirs[Idx];
  }

  /// Creates an invalid iterator.
  SearchDirIteratorImpl(std::nullptr_t) : HS(nullptr), Idx(0) {}

  /// Checks whether the iterator is valid.
  explicit operator bool() const { return HS != nullptr; }

private:
  /// The parent \c HeaderSearch. This is \c nullptr for invalid iterator.
  Qualified<IsConst, HeaderSearch> *HS;

  /// The index of the current element.
  size_t Idx;

  /// The constructor that creates a valid iterator.
  SearchDirIteratorImpl(Qualified<IsConst, HeaderSearch> &HS, size_t Idx)
      : HS(&HS), Idx(Idx) {}

  /// Only HeaderSearch is allowed to instantiate valid iterators.
  friend HeaderSearch;

  /// Enables const -> non-const conversion.
  friend SearchDirIteratorImpl<!IsConst>;
};
} // namespace detail

using ConstSearchDirIterator = detail::SearchDirIteratorImpl<true>;
using SearchDirIterator = detail::SearchDirIteratorImpl<false>;

using ConstSearchDirRange = llvm::iterator_range<ConstSearchDirIterator>;
using SearchDirRange = llvm::iterator_range<SearchDirIterator>;

/// Encapsulates the information needed to find the file referenced
/// by a \#include or \#include_next, (sub-)framework lookup, etc.
class HeaderSearch {
  friend class DirectoryLookup;

  friend ConstSearchDirIterator;
  friend SearchDirIterator;

  /// Header-search options used to initialize this header search.
  std::shared_ptr<HeaderSearchOptions> HSOpts;

  /// Mapping from SearchDir to HeaderSearchOptions::UserEntries indices.
  llvm::DenseMap<unsigned, unsigned> SearchDirToHSEntry;

  DiagnosticsEngine &Diags;
  FileManager &FileMgr;

  /// \#include search path information.  Requests for \#include "x" search the
  /// directory of the \#including file first, then each directory in SearchDirs
  /// consecutively. Requests for <x> search the current dir first, then each
  /// directory in SearchDirs, starting at AngledDirIdx, consecutively.  If
  /// NoCurDirSearch is true, then the check for the file in the current
  /// directory is suppressed.
  std::vector<DirectoryLookup> SearchDirs;
  /// Whether the DirectoryLookup at the corresponding index in SearchDirs has
  /// been successfully used to lookup a file.
  std::vector<bool> SearchDirsUsage;
  unsigned AngledDirIdx = 0;
  unsigned SystemDirIdx = 0;
  bool NoCurDirSearch = false;

  /// \#include prefixes for which the 'system header' property is
  /// overridden.
  ///
  /// For a \#include "x" or \#include \<x> directive, the last string in this
  /// list which is a prefix of 'x' determines whether the file is treated as
  /// a system header.
  std::vector<std::pair<std::string, bool>> SystemHeaderPrefixes;

  /// The hash used for module cache paths.
  std::string ModuleHash;

  /// The path to the module cache.
  std::string ModuleCachePath;

  /// All of the preprocessor-specific data about files that are
  /// included, indexed by the FileEntry's UID.
  mutable std::vector<HeaderFileInfo> FileInfo;

  /// Keeps track of each lookup performed by LookupFile.
  struct LookupFileCacheInfo {
    /// Starting search directory iterator that the cached search was performed
    /// from. If there is a hit and this value doesn't match the current query,
    /// the cache has to be ignored.
    ConstSearchDirIterator StartIt = nullptr;

    /// The search directory iterator that satisfied the query.
    ConstSearchDirIterator HitIt = nullptr;

    /// This is non-null if the original filename was mapped to a framework
    /// include via a headermap.
    const char *MappedName = nullptr;

    /// Default constructor -- Initialize all members with zero.
    LookupFileCacheInfo() = default;

    void reset(ConstSearchDirIterator NewStartIt) {
      StartIt = NewStartIt;
      MappedName = nullptr;
    }
  };
  llvm::StringMap<LookupFileCacheInfo, llvm::BumpPtrAllocator> LookupFileCache;

  /// Collection mapping a framework or subframework
  /// name like "Carbon" to the Carbon.framework directory.
  llvm::StringMap<FrameworkCacheEntry, llvm::BumpPtrAllocator> FrameworkMap;

  /// Maps include file names (including the quotes or
  /// angle brackets) to other include file names.  This is used to support the
  /// include_alias pragma for Microsoft compatibility.
  using IncludeAliasMap =
      llvm::StringMap<std::string, llvm::BumpPtrAllocator>;
  std::unique_ptr<IncludeAliasMap> IncludeAliases;

  /// This is a mapping from FileEntry -> HeaderMap, uniquing headermaps.
  std::vector<std::pair<const FileEntry *, std::unique_ptr<HeaderMap>>> HeaderMaps;

  /// The mapping between modules and headers.
  mutable ModuleMap ModMap;

  /// Describes whether a given directory has a module map in it.
  llvm::DenseMap<const DirectoryEntry *, bool> DirectoryHasModuleMap;

  /// Set of module map files we've already loaded, and a flag indicating
  /// whether they were valid or not.
  llvm::DenseMap<const FileEntry *, bool> LoadedModuleMaps;

  /// Uniqued set of framework names, which is used to track which
  /// headers were included as framework headers.
  llvm::StringSet<llvm::BumpPtrAllocator> FrameworkNames;

  /// Entity used to resolve the identifier IDs of controlling
  /// macros into IdentifierInfo pointers, and keep the identifire up to date,
  /// as needed.
  ExternalPreprocessorSource *ExternalLookup = nullptr;

  /// Entity used to look up stored header file information.
  ExternalHeaderFileInfoSource *ExternalSource = nullptr;

public:
  HeaderSearch(std::shared_ptr<HeaderSearchOptions> HSOpts,
               SourceManager &SourceMgr, DiagnosticsEngine &Diags,
               const LangOptions &LangOpts, const TargetInfo *Target);
  HeaderSearch(const HeaderSearch &) = delete;
  HeaderSearch &operator=(const HeaderSearch &) = delete;

  /// Retrieve the header-search options with which this header search
  /// was initialized.
  HeaderSearchOptions &getHeaderSearchOpts() const { return *HSOpts; }

  FileManager &getFileMgr() const { return FileMgr; }

  DiagnosticsEngine &getDiags() const { return Diags; }

  /// Interface for setting the file search paths.
  void SetSearchPaths(std::vector<DirectoryLookup> dirs, unsigned angledDirIdx,
                      unsigned systemDirIdx, bool noCurDirSearch,
                      llvm::DenseMap<unsigned, unsigned> searchDirToHSEntry);

  /// Add an additional search path.
  void AddSearchPath(const DirectoryLookup &dir, bool isAngled);

  /// Add an additional system search path.
  void AddSystemSearchPath(const DirectoryLookup &dir) {
    SearchDirs.push_back(dir);
    SearchDirsUsage.push_back(false);
  }

  /// Set the list of system header prefixes.
  void SetSystemHeaderPrefixes(ArrayRef<std::pair<std::string, bool>> P) {
    SystemHeaderPrefixes.assign(P.begin(), P.end());
  }

  /// Checks whether the map exists or not.
  bool HasIncludeAliasMap() const { return (bool)IncludeAliases; }

  /// Map the source include name to the dest include name.
  ///
  /// The Source should include the angle brackets or quotes, the dest
  /// should not.  This allows for distinction between <> and "" headers.
  void AddIncludeAlias(StringRef Source, StringRef Dest) {
    if (!IncludeAliases)
      IncludeAliases.reset(new IncludeAliasMap);
    (*IncludeAliases)[Source] = std::string(Dest);
  }

  /// Maps one header file name to a different header
  /// file name, for use with the include_alias pragma.  Note that the source
  /// file name should include the angle brackets or quotes.  Returns StringRef
  /// as null if the header cannot be mapped.
  StringRef MapHeaderToIncludeAlias(StringRef Source) {
    assert(IncludeAliases && "Trying to map headers when there's no map");

    // Do any filename replacements before anything else
    IncludeAliasMap::const_iterator Iter = IncludeAliases->find(Source);
    if (Iter != IncludeAliases->end())
      return Iter->second;
    return {};
  }

  /// Set the hash to use for module cache paths.
  void setModuleHash(StringRef Hash) { ModuleHash = std::string(Hash); }

  /// Set the path to the module cache.
  void setModuleCachePath(StringRef CachePath) {
    ModuleCachePath = std::string(CachePath);
  }

  /// Retrieve the module hash.
  StringRef getModuleHash() const { return ModuleHash; }

  /// Retrieve the path to the module cache.
  StringRef getModuleCachePath() const { return ModuleCachePath; }

  /// Consider modules when including files from this directory.
  void setDirectoryHasModuleMap(const DirectoryEntry* Dir) {
    DirectoryHasModuleMap[Dir] = true;
  }

  /// Forget everything we know about headers so far.
  void ClearFileInfo() {
    FileInfo.clear();
  }

  void SetExternalLookup(ExternalPreprocessorSource *EPS) {
    ExternalLookup = EPS;
  }

  ExternalPreprocessorSource *getExternalLookup() const {
    return ExternalLookup;
  }

  /// Set the external source of header information.
  void SetExternalSource(ExternalHeaderFileInfoSource *ES) {
    ExternalSource = ES;
  }

  /// Set the target information for the header search, if not
  /// already known.
  void setTarget(const TargetInfo &Target);

  /// Given a "foo" or \<foo> reference, look up the indicated file,
  /// return null on failure.
  ///
  /// \returns If successful, this returns 'UsedDir', the DirectoryLookup member
  /// the file was found in, or null if not applicable.
  ///
  /// \param IncludeLoc Used for diagnostics if valid.
  ///
  /// \param isAngled indicates whether the file reference is a <> reference.
  ///
  /// \param CurDir If non-null, the file was found in the specified directory
  /// search location.  This is used to implement \#include_next.
  ///
  /// \param Includers Indicates where the \#including file(s) are, in case
  /// relative searches are needed. In reverse order of inclusion.
  ///
  /// \param SearchPath If non-null, will be set to the search path relative
  /// to which the file was found. If the include path is absolute, SearchPath
  /// will be set to an empty string.
  ///
  /// \param RelativePath If non-null, will be set to the path relative to
  /// SearchPath at which the file was found. This only differs from the
  /// Filename for framework includes.
  ///
  /// \param SuggestedModule If non-null, and the file found is semantically
  /// part of a known module, this will be set to the module that should
  /// be imported instead of preprocessing/parsing the file found.
  ///
  /// \param IsMapped If non-null, and the search involved header maps, set to
  /// true.
  ///
  /// \param IsFrameworkFound If non-null, will be set to true if a framework is
  /// found in any of searched SearchDirs. Will be set to false if a framework
  /// is found only through header maps. Doesn't guarantee the requested file is
  /// found.
  Optional<FileEntryRef> LookupFile(
      StringRef Filename, SourceLocation IncludeLoc, bool isAngled,
      ConstSearchDirIterator FromDir, ConstSearchDirIterator *CurDir,
      ArrayRef<std::pair<const FileEntry *, const DirectoryEntry *>> Includers,
      SmallVectorImpl<char> *SearchPath, SmallVectorImpl<char> *RelativePath,
      Module *RequestingModule, ModuleMap::KnownHeader *SuggestedModule,
      bool *IsMapped, bool *IsFrameworkFound, bool SkipCache = false,
      bool BuildSystemModule = false);

  /// Look up a subframework for the specified \#include file.
  ///
  /// For example, if \#include'ing <HIToolbox/HIToolbox.h> from
  /// within ".../Carbon.framework/Headers/Carbon.h", check to see if
  /// HIToolbox is a subframework within Carbon.framework.  If so, return
  /// the FileEntry for the designated file, otherwise return null.
  Optional<FileEntryRef> LookupSubframeworkHeader(
      StringRef Filename, const FileEntry *ContextFileEnt,
      SmallVectorImpl<char> *SearchPath, SmallVectorImpl<char> *RelativePath,
      Module *RequestingModule, ModuleMap::KnownHeader *SuggestedModule);

  /// Look up the specified framework name in our framework cache.
  /// \returns The DirectoryEntry it is in if we know, null otherwise.
  FrameworkCacheEntry &LookupFrameworkCache(StringRef FWName) {
    return FrameworkMap[FWName];
  }

  /// Mark the specified file as a target of a \#include,
  /// \#include_next, or \#import directive.
  ///
  /// \return false if \#including the file will have no effect or true
  /// if we should include it.
  bool ShouldEnterIncludeFile(Preprocessor &PP, const FileEntry *File,
                              bool isImport, bool ModulesEnabled, Module *M,
                              bool &IsFirstIncludeOfFile);

  /// Return whether the specified file is a normal header,
  /// a system header, or a C++ friendly system header.
  SrcMgr::CharacteristicKind getFileDirFlavor(const FileEntry *File) {
    return (SrcMgr::CharacteristicKind)getFileInfo(File).DirInfo;
  }

  /// Mark the specified file as a "once only" file due to
  /// \#pragma once.
  void MarkFileIncludeOnce(const FileEntry *File) {
    HeaderFileInfo &FI = getFileInfo(File);
    FI.isPragmaOnce = true;
  }

  /// Mark the specified file as a system header, e.g. due to
  /// \#pragma GCC system_header.
  void MarkFileSystemHeader(const FileEntry *File) {
    getFileInfo(File).DirInfo = SrcMgr::C_System;
  }

  void AddFileAlias(const FileEntry *File, StringRef Alias) {
    getFileInfo(File).Aliases.insert(Alias);
  }

  /// Mark the specified file as part of a module.
  void MarkFileModuleHeader(const FileEntry *FE,
                            ModuleMap::ModuleHeaderRole Role,
                            bool isCompilingModuleHeader);

  /// Mark the specified file as having a controlling macro.
  ///
  /// This is used by the multiple-include optimization to eliminate
  /// no-op \#includes.
  void SetFileControllingMacro(const FileEntry *File,
                               const IdentifierInfo *ControllingMacro) {
    getFileInfo(File).ControllingMacro = ControllingMacro;
  }

  /// Determine whether this file is intended to be safe from
  /// multiple inclusions, e.g., it has \#pragma once or a controlling
  /// macro.
  ///
  /// This routine does not consider the effect of \#import
  bool isFileMultipleIncludeGuarded(const FileEntry *File);

  /// Determine whether the given file is known to have ever been \#imported.
  bool hasFileBeenImported(const FileEntry *File) {
    const HeaderFileInfo *FI = getExistingFileInfo(File);
    return FI && FI->isImport;
  }

  /// Determine which HeaderSearchOptions::UserEntries have been successfully
  /// used so far and mark their index with 'true' in the resulting bit vector.
  std::vector<bool> computeUserEntryUsage() const;

  /// This method returns a HeaderMap for the specified
  /// FileEntry, uniquing them through the 'HeaderMaps' datastructure.
  const HeaderMap *CreateHeaderMap(const FileEntry *FE);

  /// Get filenames for all registered header maps.
  void getHeaderMapFileNames(SmallVectorImpl<std::string> &Names) const;

  /// Retrieve the name of the cached module file that should be used
  /// to load the given module.
  ///
  /// \param Module The module whose module file name will be returned.
  ///
  /// \returns The name of the module file that corresponds to this module,
  /// or an empty string if this module does not correspond to any module file.
  std::string getCachedModuleFileName(Module *Module);

  /// Retrieve the name of the prebuilt module file that should be used
  /// to load a module with the given name.
  ///
  /// \param ModuleName The module whose module file name will be returned.
  ///
  /// \param FileMapOnly If true, then only look in the explicit module name
  //  to file name map and skip the directory search.
  ///
  /// \returns The name of the module file that corresponds to this module,
  /// or an empty string if this module does not correspond to any module file.
  std::string getPrebuiltModuleFileName(StringRef ModuleName,
                                        bool FileMapOnly = false);

  /// Retrieve the name of the prebuilt module file that should be used
  /// to load the given module.
  ///
  /// \param Module The module whose module file name will be returned.
  ///
  /// \returns The name of the module file that corresponds to this module,
  /// or an empty string if this module does not correspond to any module file.
  std::string getPrebuiltImplicitModuleFileName(Module *Module);

  /// Retrieve the name of the (to-be-)cached module file that should
  /// be used to load a module with the given name.
  ///
  /// \param ModuleName The module whose module file name will be returned.
  ///
  /// \param ModuleMapPath A path that when combined with \c ModuleName
  /// uniquely identifies this module. See Module::ModuleMap.
  ///
  /// \returns The name of the module file that corresponds to this module,
  /// or an empty string if this module does not correspond to any module file.
  std::string getCachedModuleFileName(StringRef ModuleName,
                                      StringRef ModuleMapPath);

  /// Lookup a module Search for a module with the given name.
  ///
  /// \param ModuleName The name of the module we're looking for.
  ///
  /// \param ImportLoc Location of the module include/import.
  ///
  /// \param AllowSearch Whether we are allowed to search in the various
  /// search directories to produce a module definition. If not, this lookup
  /// will only return an already-known module.
  ///
  /// \param AllowExtraModuleMapSearch Whether we allow to search modulemaps
  /// in subdirectories.
  ///
  /// \returns The module with the given name.
  Module *lookupModule(StringRef ModuleName,
                       SourceLocation ImportLoc = SourceLocation(),
                       bool AllowSearch = true,
                       bool AllowExtraModuleMapSearch = false);

  /// Try to find a module map file in the given directory, returning
  /// \c nullptr if none is found.
  const FileEntry *lookupModuleMapFile(const DirectoryEntry *Dir,
                                       bool IsFramework);

  /// Determine whether there is a module map that may map the header
  /// with the given file name to a (sub)module.
  /// Always returns false if modules are disabled.
  ///
  /// \param Filename The name of the file.
  ///
  /// \param Root The "root" directory, at which we should stop looking for
  /// module maps.
  ///
  /// \param IsSystem Whether the directories we're looking at are system
  /// header directories.
  bool hasModuleMap(StringRef Filename, const DirectoryEntry *Root,
                    bool IsSystem);

  /// Retrieve the module that corresponds to the given file, if any.
  ///
  /// \param File The header that we wish to map to a module.
  /// \param AllowTextual Whether we want to find textual headers too.
  ModuleMap::KnownHeader findModuleForHeader(const FileEntry *File,
                                             bool AllowTextual = false) const;

  /// Retrieve all the modules corresponding to the given file.
  ///
  /// \ref findModuleForHeader should typically be used instead of this.
  ArrayRef<ModuleMap::KnownHeader>
  findAllModulesForHeader(const FileEntry *File) const;

  /// Read the contents of the given module map file.
  ///
  /// \param File The module map file.
  /// \param IsSystem Whether this file is in a system header directory.
  /// \param ID If the module map file is already mapped (perhaps as part of
  ///        processing a preprocessed module), the ID of the file.
  /// \param Offset [inout] An offset within ID to start parsing. On exit,
  ///        filled by the end of the parsed contents (either EOF or the
  ///        location of an end-of-module-map pragma).
  /// \param OriginalModuleMapFile The original path to the module map file,
  ///        used to resolve paths within the module (this is required when
  ///        building the module from preprocessed source).
  /// \returns true if an error occurred, false otherwise.
  bool loadModuleMapFile(const FileEntry *File, bool IsSystem,
                         FileID ID = FileID(), unsigned *Offset = nullptr,
                         StringRef OriginalModuleMapFile = StringRef());

  /// Collect the set of all known, top-level modules.
  ///
  /// \param Modules Will be filled with the set of known, top-level modules.
  void collectAllModules(SmallVectorImpl<Module *> &Modules);

  /// Load all known, top-level system modules.
  void loadTopLevelSystemModules();

private:
  /// Lookup a module with the given module name and search-name.
  ///
  /// \param ModuleName The name of the module we're looking for.
  ///
  /// \param SearchName The "search-name" to derive filesystem paths from
  /// when looking for the module map; this is usually equal to ModuleName,
  /// but for compatibility with some buggy frameworks, additional attempts
  /// may be made to find the module under a related-but-different search-name.
  ///
  /// \param ImportLoc Location of the module include/import.
  ///
  /// \param AllowExtraModuleMapSearch Whether we allow to search modulemaps
  /// in subdirectories.
  ///
  /// \returns The module named ModuleName.
  Module *lookupModule(StringRef ModuleName, StringRef SearchName,
                       SourceLocation ImportLoc,
                       bool AllowExtraModuleMapSearch = false);

  /// Retrieve the name of the (to-be-)cached module file that should
  /// be used to load a module with the given name.
  ///
  /// \param ModuleName The module whose module file name will be returned.
  ///
  /// \param ModuleMapPath A path that when combined with \c ModuleName
  /// uniquely identifies this module. See Module::ModuleMap.
  ///
  /// \param CachePath A path to the module cache.
  ///
  /// \returns The name of the module file that corresponds to this module,
  /// or an empty string if this module does not correspond to any module file.
  std::string getCachedModuleFileNameImpl(StringRef ModuleName,
                                          StringRef ModuleMapPath,
                                          StringRef CachePath);

  /// Retrieve a module with the given name, which may be part of the
  /// given framework.
  ///
  /// \param Name The name of the module to retrieve.
  ///
  /// \param Dir The framework directory (e.g., ModuleName.framework).
  ///
  /// \param IsSystem Whether the framework directory is part of the system
  /// frameworks.
  ///
  /// \returns The module, if found; otherwise, null.
  Module *loadFrameworkModule(StringRef Name,
                              const DirectoryEntry *Dir,
                              bool IsSystem);

  /// Load all of the module maps within the immediate subdirectories
  /// of the given search directory.
  void loadSubdirectoryModuleMaps(DirectoryLookup &SearchDir);

  /// Find and suggest a usable module for the given file.
  ///
  /// \return \c true if the file can be used, \c false if we are not permitted to
  ///         find this file due to requirements from \p RequestingModule.
  bool findUsableModuleForHeader(const FileEntry *File,
                                 const DirectoryEntry *Root,
                                 Module *RequestingModule,
                                 ModuleMap::KnownHeader *SuggestedModule,
                                 bool IsSystemHeaderDir);

  /// Find and suggest a usable module for the given file, which is part of
  /// the specified framework.
  ///
  /// \return \c true if the file can be used, \c false if we are not permitted to
  ///         find this file due to requirements from \p RequestingModule.
  bool findUsableModuleForFrameworkHeader(
      const FileEntry *File, StringRef FrameworkName, Module *RequestingModule,
      ModuleMap::KnownHeader *SuggestedModule, bool IsSystemFramework);

  /// Look up the file with the specified name and determine its owning
  /// module.
  Optional<FileEntryRef>
  getFileAndSuggestModule(StringRef FileName, SourceLocation IncludeLoc,
                          const DirectoryEntry *Dir, bool IsSystemHeaderDir,
                          Module *RequestingModule,
                          ModuleMap::KnownHeader *SuggestedModule);

  /// Cache the result of a successful lookup at the given include location
  /// using the search path at \c HitIt.
  void cacheLookupSuccess(LookupFileCacheInfo &CacheLookup,
                          ConstSearchDirIterator HitIt,
                          SourceLocation IncludeLoc);

  /// Note that a lookup at the given include location was successful using the
  /// search path at index `HitIdx`.
  void noteLookupUsage(unsigned HitIdx, SourceLocation IncludeLoc);

public:
  /// Retrieve the module map.
  ModuleMap &getModuleMap() { return ModMap; }

  /// Retrieve the module map.
  const ModuleMap &getModuleMap() const { return ModMap; }

  unsigned header_file_size() const { return FileInfo.size(); }

  /// Return the HeaderFileInfo structure for the specified FileEntry,
  /// in preparation for updating it in some way.
  HeaderFileInfo &getFileInfo(const FileEntry *FE);

  /// Return the HeaderFileInfo structure for the specified FileEntry,
  /// if it has ever been filled in.
  /// \param WantExternal Whether the caller wants purely-external header file
  ///        info (where \p External is true).
  const HeaderFileInfo *getExistingFileInfo(const FileEntry *FE,
                                            bool WantExternal = true) const;

  SearchDirIterator search_dir_begin() { return {*this, 0}; }
  SearchDirIterator search_dir_end() { return {*this, SearchDirs.size()}; }
  SearchDirRange search_dir_range() {
    return {search_dir_begin(), search_dir_end()};
  }

  ConstSearchDirIterator search_dir_begin() const { return quoted_dir_begin(); }
  ConstSearchDirIterator search_dir_end() const { return system_dir_end(); }
  ConstSearchDirRange search_dir_range() const {
    return {search_dir_begin(), search_dir_end()};
  }

  unsigned search_dir_size() const { return SearchDirs.size(); }

  ConstSearchDirIterator quoted_dir_begin() const { return {*this, 0}; }
  ConstSearchDirIterator quoted_dir_end() const { return angled_dir_begin(); }

  ConstSearchDirIterator angled_dir_begin() const {
    return {*this, AngledDirIdx};
  }
  ConstSearchDirIterator angled_dir_end() const { return system_dir_begin(); }

  ConstSearchDirIterator system_dir_begin() const {
    return {*this, SystemDirIdx};
  }
  ConstSearchDirIterator system_dir_end() const {
    return {*this, SearchDirs.size()};
  }

  /// Get the index of the given search directory.
  unsigned searchDirIdx(const DirectoryLookup &DL) const;

  /// Retrieve a uniqued framework name.
  StringRef getUniqueFrameworkName(StringRef Framework);

  /// Suggest a path by which the specified file could be found, for use in
  /// diagnostics to suggest a #include. Returned path will only contain forward
  /// slashes as separators. MainFile is the absolute path of the file that we
  /// are generating the diagnostics for. It will try to shorten the path using
  /// MainFile location, if none of the include search directories were prefix
  /// of File.
  ///
  /// \param IsSystem If non-null, filled in to indicate whether the suggested
  ///        path is relative to a system header directory.
  std::string suggestPathToFileForDiagnostics(const FileEntry *File,
                                              llvm::StringRef MainFile,
                                              bool *IsSystem = nullptr);

  /// Suggest a path by which the specified file could be found, for use in
  /// diagnostics to suggest a #include. Returned path will only contain forward
  /// slashes as separators. MainFile is the absolute path of the file that we
  /// are generating the diagnostics for. It will try to shorten the path using
  /// MainFile location, if none of the include search directories were prefix
  /// of File.
  ///
  /// \param WorkingDir If non-empty, this will be prepended to search directory
  /// paths that are relative.
  std::string suggestPathToFileForDiagnostics(llvm::StringRef File,
                                              llvm::StringRef WorkingDir,
                                              llvm::StringRef MainFile,
                                              bool *IsSystem = nullptr);

  void PrintStats();

  size_t getTotalMemory() const;

private:
  /// Describes what happened when we tried to load a module map file.
  enum LoadModuleMapResult {
    /// The module map file had already been loaded.
    LMM_AlreadyLoaded,

    /// The module map file was loaded by this invocation.
    LMM_NewlyLoaded,

    /// There is was directory with the given name.
    LMM_NoDirectory,

    /// There was either no module map file or the module map file was
    /// invalid.
    LMM_InvalidModuleMap
  };

  LoadModuleMapResult loadModuleMapFileImpl(const FileEntry *File,
                                            bool IsSystem,
                                            const DirectoryEntry *Dir,
                                            FileID ID = FileID(),
                                            unsigned *Offset = nullptr);

  /// Try to load the module map file in the given directory.
  ///
  /// \param DirName The name of the directory where we will look for a module
  /// map file.
  /// \param IsSystem Whether this is a system header directory.
  /// \param IsFramework Whether this is a framework directory.
  ///
  /// \returns The result of attempting to load the module map file from the
  /// named directory.
  LoadModuleMapResult loadModuleMapFile(StringRef DirName, bool IsSystem,
                                        bool IsFramework);

  /// Try to load the module map file in the given directory.
  ///
  /// \param Dir The directory where we will look for a module map file.
  /// \param IsSystem Whether this is a system header directory.
  /// \param IsFramework Whether this is a framework directory.
  ///
  /// \returns The result of attempting to load the module map file from the
  /// named directory.
  LoadModuleMapResult loadModuleMapFile(const DirectoryEntry *Dir,
                                        bool IsSystem, bool IsFramework);
};

/// Apply the header search options to get given HeaderSearch object.
void ApplyHeaderSearchOptions(HeaderSearch &HS,
                              const HeaderSearchOptions &HSOpts,
                              const LangOptions &Lang,
                              const llvm::Triple &triple);

} // namespace clang

#endif // LLVM_CLANG_LEX_HEADERSEARCH_H
