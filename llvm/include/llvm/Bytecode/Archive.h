//===-- llvm/Bytecode/Archive.h - LLVM Bytecode Archive ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header file defines the interface to LLVM Archive files. The interface
// is provided by the Archive class implemented by the lib/Bytecode/Archive
// library.  This library is used to read and write archive (*.a) files that 
// contain LLVM bytecode files (or others). It provides rudimentary capabilities 
// to construct an archive file from a set of files, read the archive members 
// into memory, search the archive for member files that fulfill unresolved 
// symbols, and extract the archive back to the file system.  Full 
// symbol table support is provided for loading only those files that resolve 
// symbols. Note that read performance of this library is _crucial_ for 
// performance of JIT type applications and the linkers. Consequently, the 
// library is optimized for reading.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_ARCHIVE_H
#define LLVM_BYTECODE_ARCHIVE_H

#include "llvm/System/Path.h"
#include <map>

namespace llvm {

class ModuleProvider;
class Module;

/// This class represents an archive file. It abstracts away the file format,
/// and logical operations that can be done on an archive file. It also provides
/// utilities for controlling the runtime loading/reading of the archive file
/// to provide efficient access mechanisms for JIT systems and linkers.
class Archive {

  /// @name Types
  /// @{
  public:
    /// The user's interface to the archive symbol table. This multimap allows
    /// symbols to be looked up and modules to be instantiated from the file
    /// lazily via the ModuleProvider::materializeModule method.
    typedef std::map<std::string,ModuleProvider*> SymTab;

    /// This typedef is just shorthand for a vector of Path names
    typedef std::vector<sys::Path> PathList;

    /// This typedef is just shorthand for a vector of Modules
    typedef std::vector<Module*> ModuleList;

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// Create an empty archive file, \p Filename. The returned Archive object
    /// will have no file members and an empty symbol table. The actual archive
    /// file is not created until the returned Archive object is destructed.
    /// @throws std::string if an error occurs
    /// @returns An Archive* that represents the new archive file.
    /// @brief Create an empty archive file.
    static Archive* CreateEmpty(
      const sys::Path& Filename        ///< Name of archive file
    );

    /// Create a new archive file, \p Filename, from the LLVM modules \p Modules.
    /// The module's externally visible linkage symbols will be added to the 
    /// archive's symbol table.  The names of the file members will be obtained 
    /// from the Module::getModuleId() method. If that name is not unique, it will
    /// be made unique by appending a monotonically increasing integer to it. If 
    /// \p StripName is non-empty, it specifies a prefix to be stripped from the 
    /// name of the file members. This allows archives with relative path names 
    /// to be created. The actual archive file is not created until the
    /// returned Archive object is destructed. 
    /// @returns An Archive* that that represents the newly created archive file. 
    /// @throws std::string if an error occurs
    /// @brief Create an archive file from modules.
    static Archive* CreateFromModules(
      const sys::Path& Filename,       ///< Name of archive file
      const ModuleList& Modules,       ///< Modules to be put in archive
      const std::string& StripName=""  ///< Prefix to strip from member names
    );

    /// Create a new archive file, \p Filename, from a set of existing \p Files.
    /// Each entry in \p Files will be added to the archive. If any file is an
    /// llvm bytecode file, its externally visible linkage symbols will be added
    /// to the archive's symbol table. If any of the paths in \p Files refer to
    /// directories, those directories will be ignored. Full path names to 
    /// files must be provided. However, if \p StripName is non-empty, it 
    /// specifies a prefix string to be removed from each path before it is 
    /// written as the name of the file member. Any path names that do not have 
    /// the \p StripName as a prefix will be saved with the full path name
    /// provided in \p Files. This permits archives relative to a top level 
    /// directory to be created.
    /// @throws std::string if an error occurs
    /// @brief Create an archive file from files.
    static Archive* CreateFromFiles(
      const sys::Path& Filename,       ///< Name of archive file
      const PathList& Files,           ///< File paths to be put in archive
      const std::string& StripName=""  ///< Prefix path name to strip from names
    );

    /// Open an existing archive file from \p Filename. The necessary
    /// arrangements to read the file are made, but nothing much is actually 
    /// read from the file. Use the Accessor methods below to lazily obtain 
    /// those portions of the file that are of interest.
    /// @throws std::string if an error occurs
    /// @brief Open an existing archive.
    static Archive* Open(
      const sys::Path& Filename       ///< Name of the archive file
    );

    /// This destructor "finalizes" the archive. Regardless of whether the 
    /// archive was created or opened, all memory associated with the archive 
    /// is released, including any SymTab* returned by the getSymbolTable()
    /// method, any ModuleProviders and their associated Modules returned by
    /// any interface method, etc.  Additionally, if the archive was created
    /// using one of the Create* methods, the archive is written to disk in
    /// its final format. After this method exits, none of the memory 
    /// associated with the archive is valid. It is the user's responsibility 
    /// to ensure that all references to such memory is removed before the
    /// Archive is destructed.
    /// @throws std::string if an error occurs
    /// @brief Destruct in-memory archive 
    ~Archive();

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// This accessor looks up the \p symbol in the archive's symbol table and 
    /// returns the associated module that defines that symbol. This method can
    /// be called as many times as necessary. This is handy for linking the 
    /// archive into another module based on unresolved symbols. Note that the
    /// ModuleProvider returned by this accessor is constant and it may refer
    /// to the same ModuleProvider object returned by this accessor in a 
    /// previous call (because the associated module defines both symbols). To
    /// use the returned ModuleProvider* you must make a copy and call the
    /// materializeModule method on the copy.
    /// @throws std::string if an error occurs
    /// @returns The ModuleProvider* found or null if the archive does not 
    /// contain a module that defines the \p symbol.
    /// @brief Look up a module by symbol name.
    const ModuleProvider* findModuleContainingSymbol(
      const std::string& symbol        ///< Symbol to be sought
    ) const;

    /// Return the list of all the \p Paths for the file members in the archive.
    /// This is handy for generating the table of contents of the archive. Note
    /// that \p Paths is *not* cleared before it is populated. New entries are 
    /// appended to the end of the PathList.
    /// @throw std::string if an error occurs
    /// @returns nothing
    /// @brief Get all the paths in the archive
    void getAllPaths(
      PathList& Paths                ///< The list of paths returned
    );

    /// This method returns a caller readable SymTab object which is a map
    /// of symbol names to ModuleProviders. Callers can traverse this symbol
    /// table, look up specific symbols, etc. and materialize any Modules they 
    /// want with the associated ModuleProviders.  It is unnecessary to call
    /// this accessor more than once as the same object is alway returned, even
    /// if changes have been made.
    /// @returns a constant SymTab* that is the same for every invocation. The
    /// caller should not attempt to free or modify this SymTab object, just use
    /// it. 
    /// @brief Get the archive's symbol table.
    const SymTab* getSymbolTable();

    /// Extract the contents of the archive back to the file system using \p
    /// RootDir as the directory at the root of the extraction.  Each file 
    /// member is written back as a separate file. If \p flat is false, then
    /// directories are created as necessary to restore the files to the correct
    /// sub-directory of \p root as specified in the full path of the file
    /// member. Otherwise, paths are ignored and all file members will be 
    /// extracted to the \p root directory using only the filename portion of
    /// the path (directories ignored). If \p symtab is true, the archive's
    /// symbol table will also be extracted to a file named __SYMTAB.
    /// @returns nothing
    /// @throws std::string if an error occurs
    /// @brief Extract archive contents to a file.
    void extractAllToDirectory(
      const sys::Path& RootDir,      ///< The root directory for extraction 
      bool Flat = true,              ///< false = recreate directory structure
      bool Symtab = false            ///< true = extract symbol table too
    );

    /// Extract one file in the archive back to the file system using \p RootDir
    /// as the directory at the root of the extraction. The file \p name is 
    /// extracted and placed into \p RootDir. If \p Flat is false, then any
    /// intermediate directory names in the file member's path will also be
    /// created. Otherwise, the member's file is created in RootDir ignoring
    /// the leading path and using only the member's file name.
    /// @throws std::string if an error occurs.
    /// @returns nothing
    /// @brief Extract a single archive file.
    void extractOneFileToDirectory(
      const sys::Path& RootDir,      ///< The root directory for extraction
      const sys::Path& name,         ///< Name of file to extract
      bool Flat = true               ///< false = recreate directories
    );

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// Each entry in \p Files will be added to the archive. If any file is an
    /// llvm bytecode file, its externally visible linkage symbols will be added
    /// to the archive's symbol table. If any of the paths in \p Files refer to
    /// directories, those directories will be ignored. Full path names to 
    /// files must be provided. However, if \p StripName is non-empty, it 
    /// specifies a prefix string to be removed from each path before it is 
    /// written as the name of the file member. Any path names that do not have 
    /// the \p StripName as a prefix will be saved with the full path name
    /// provided. This permits archives relative to a top level directory 
    /// to be created.
    /// @throws std::string if an error occurs
    /// @returns nothing
    /// @brief Add a set of files to the archive.
    void addFiles(
      const PathList& Files,           ///< The names of files to add to archive
      const std::string& StripName=""  ///< Prefix path to strip from names
    );

    /// Add a set of Modules to the archive.  Names of member files will
    /// be taken from the Module identifier (Module::getModuleIdentifier) if it
    /// is unique. Non-unique member names will be made unique by appending a 
    /// number. The symbol table will be augmented with the global symbols of 
    /// all the modules provided. If \p StripName is non-empty, it 
    /// specifies a prefix string to be removed from each moduleId  before it is 
    /// written as the name of the file member. Any path names that do not have 
    /// the \p StripName as a prefix will be saved with the full path name
    /// provided. This permits archives relative to a top level directory  to
    /// be created.
    /// @throws std::string if an error occurs
    /// @returns nothing
    /// @brief Add a set of modules to the archive.
    void addModules(
      const ModuleList& Modules,       ///< The modules to add to the archive
      const std::string& StripName=""  ///< Prefix path to strip from names
    );

  /// @}
};

} // End llvm namespace

// vim: sw=2 ai 

#endif
