//===--- GlobalModuleIndex.h - Global Module Index --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the GlobalModuleIndex class, which manages a global index
// containing all of the identifiers known to the various modules within a given
// subdirectory of the module cache. It is used to improve the performance of
// queries such as "do any modules know about this identifier?"
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SERIALIZATION_GLOBAL_MODULE_INDEX_H
#define LLVM_CLANG_SERIALIZATION_GLOBAL_MODULE_INDEX_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <utility>

namespace llvm {
class BitstreamCursor;
class MemoryBuffer;
}

namespace clang {

class DirectoryEntry;
class FileEntry;
class FileManager;

namespace serialization {
  class ModuleFile;
};

using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;
using serialization::ModuleFile;

/// \brief Abstract class that resolves a module file name to a ModuleFile
/// pointer, which is used to uniquely describe a module file.
class ModuleFileNameResolver {
public:
  virtual ~ModuleFileNameResolver();

  /// \brief Attempt to resolve the given module file name to a specific,
  /// already-loaded module.
  ///
  /// \param FileName The name of the module file.
  ///
  /// \param ExpectedSize The size that the module file is expected to have.
  /// If the actual size differs, the resolver should return \c true.
  ///
  /// \param ExpectedModTime The modification time that the module file is
  /// expected to have. If the actual modification time differs, the resolver
  /// should return \c true.
  ///
  /// \param File Will be set to the module file if there is one, or null
  /// otherwise.
  ///
  /// \returns True if a module file exists but does not meet the size/
  /// modification time criteria, false if the module file is available or has
  /// not yet been loaded.
  virtual bool resolveModuleFileName(StringRef FileName,
                                     off_t ExpectedSize,
                                     time_t ExpectedModTime,
                                     ModuleFile *&File) = 0;
};

/// \brief A global index for a set of module files, providing information about
/// the identifiers within those module files.
///
/// The global index is an aid for name lookup into modules, offering a central
/// place where one can look for identifiers determine which
/// module files contain any information about that identifier. This
/// allows the client to restrict the search to only those module files known
/// to have a information about that identifier, improving performance. Moreover,
/// the global module index may know about module files that have not been
/// imported, and can be queried to determine which modules the current
/// translation could or should load to fix a problem.
class GlobalModuleIndex {
  /// \brief Buffer containing the index file, which is lazily accessed so long
  /// as the global module index is live.
  llvm::OwningPtr<llvm::MemoryBuffer> Buffer;

  /// \brief The module file name resolver.
  ModuleFileNameResolver *Resolver;

  /// \brief The hash table.
  ///
  /// This pointer actually points to a IdentifierIndexTable object,
  /// but that type is only accessible within the implementation of
  /// GlobalModuleIndex.
  void *IdentifierIndex;

  /// \brief Information about a given module file.
  struct ModuleInfo {
    ModuleInfo() : File(), Size(), ModTime() { }

    /// \brief The module file, if it is known.
    ModuleFile *File;

    /// \brief The module file name.
    std::string FileName;

    /// \brief Size of the module file at the time the global index was built.
    off_t Size;

    /// \brief Modification time of the module file at the time the global
    /// index was built.
    time_t ModTime;

    /// \brief The module IDs on which this module directly depends.
    /// FIXME: We don't really need a vector here.
    llvm::SmallVector<unsigned, 4> Dependencies;

    /// \brief The module IDs that directly depend on this module.
    llvm::SmallVector<unsigned, 4> ImportedBy;
  };

  /// \brief A mapping from module IDs to information about each module.
  ///
  /// This vector may have gaps, if module files have been removed or have
  /// been updated since the index was built. A gap is indicated by an empty
  /// file name.
  llvm::SmallVector<ModuleInfo, 16> Modules;

  /// \brief Lazily-populated mapping from module files to their
  /// corresponding index into the \c Modules vector.
  llvm::DenseMap<ModuleFile *, unsigned> ModulesByFile;

  /// \brief The number of identifier lookups we performed.
  unsigned NumIdentifierLookups;

  /// \brief The number of identifier lookup hits, where we recognize the
  /// identifier.
  unsigned NumIdentifierLookupHits;

  /// \brief Internal constructor. Use \c readIndex() to read an index.
  explicit GlobalModuleIndex(llvm::MemoryBuffer *Buffer,
                             llvm::BitstreamCursor Cursor);

  GlobalModuleIndex(const GlobalModuleIndex &) LLVM_DELETED_FUNCTION;
  GlobalModuleIndex &operator=(const GlobalModuleIndex &) LLVM_DELETED_FUNCTION;

public:
  ~GlobalModuleIndex();

  /// \brief An error code returned when trying to read an index.
  enum ErrorCode {
    /// \brief No error occurred.
    EC_None,
    /// \brief No index was found.
    EC_NotFound,
    /// \brief Some other process is currently building the index; it is not
    /// available yet.
    EC_Building,
    /// \brief There was an unspecified I/O error reading or writing the index.
    EC_IOError
  };

  /// \brief Read a global index file for the given directory.
  ///
  /// \param Path The path to the specific module cache where the module files
  /// for the intended configuration reside.
  ///
  /// \returns A pair containing the global module index (if it exists) and
  /// the error code.
  static std::pair<GlobalModuleIndex *, ErrorCode>
  readIndex(StringRef Path);

  /// \brief Retrieve the set of modules that have up-to-date indexes.
  ///
  /// \param ModuleFiles Will be populated with the set of module files that
  /// have been indexed.
  void getKnownModules(SmallVectorImpl<ModuleFile *> &ModuleFiles);

  /// \brief Retrieve the set of module files on which the given module file
  /// directly depends.
  void getModuleDependencies(ModuleFile *File,
                             SmallVectorImpl<ModuleFile *> &Dependencies);

  /// \brief A set of module files in which we found a result.
  typedef llvm::SmallPtrSet<ModuleFile *, 4> HitSet;
  
  /// \brief Look for all of the module files with information about the given
  /// identifier, e.g., a global function, variable, or type with that name.
  ///
  /// \param Name The identifier to look for.
  ///
  /// \param Hits Will be populated with the set of module files that have
  /// information about this name.
  ///
  /// \returns true if the identifier is known to the index, false otherwise.
  bool lookupIdentifier(StringRef Name, HitSet &Hits);

  /// \brief Set the module file name resolver.
  void setResolver(ModuleFileNameResolver *Resolver) {
    this->Resolver = Resolver;
  }

  /// \brief Note that additional modules have been loaded, which invalidates
  /// the module file -> module cache.
  void noteAdditionalModulesLoaded() {
    ModulesByFile.clear();
  }

  /// \brief Resolve the module file for the module with the given ID.
  ModuleFile *resolveModuleFile(unsigned ID);

  /// \brief Print statistics to standard error.
  void printStats();

  /// \brief Write a global index into the given
  ///
  /// \param FileMgr The file manager to use to load module files.
  ///
  /// \param Path The path to the directory containing module files, into
  /// which the global index will be written.
  static ErrorCode writeIndex(FileManager &FileMgr, StringRef Path);
};

}

#endif
