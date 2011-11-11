//===--- ModuleMap.h - Describe the layout of modules -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ModuleMap interface, which describes the layout of a
// module as it relates to headers.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CLANG_LEX_MODULEMAP_H
#define LLVM_CLANG_LEX_MODULEMAP_H

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include <string>

namespace clang {
  
class FileEntry;
class FileManager;
class DiagnosticConsumer;
class DiagnosticsEngine;
class ModuleMapParser;
  
class ModuleMap {
public:
  /// \brief Describes a module or submodule.
  struct Module {
    /// \brief The name of this module.
    std::string Name;
    
    /// \brief The location of the module definition.
    SourceLocation DefinitionLoc;
    
    /// \brief The parent of this module. This will be NULL for the top-level
    /// module.
    Module *Parent;

    /// \brief The umbrella header, if any.
    ///
    /// Only the top-level module can have an umbrella header.
    const FileEntry *UmbrellaHeader;

    /// \brief The submodules of this module, indexed by name.
    llvm::StringMap<Module *> SubModules;

    /// \brief The headers that are part of this module.
    llvm::SmallVector<const FileEntry *, 2> Headers;
    
    /// \brief Whether this is an explicit submodule.
    bool IsExplicit;
    
    /// \brief Construct a top-level module.
    explicit Module(StringRef Name, SourceLocation DefinitionLoc)
      : Name(Name), DefinitionLoc(DefinitionLoc), Parent(0), UmbrellaHeader(0),
        IsExplicit(false) { }
    
    /// \brief Construct  a new module or submodule.
    Module(StringRef Name, SourceLocation DefinitionLoc, Module *Parent, 
           bool IsExplicit)
      : Name(Name), DefinitionLoc(DefinitionLoc), Parent(Parent), 
        UmbrellaHeader(0), IsExplicit(IsExplicit) {
    }
           
    /// \brief Determine whether this module is a submodule.
    bool isSubModule() const { return Parent != 0; }
    
    /// \brief Retrieve the full name of this module, including the path from
    /// its top-level module.
    std::string getFullModuleName() const;
    
    /// \brief Retrieve the name of the top-level module.
    StringRef getTopLevelModuleName() const;
  };
  
private:
  SourceManager *SourceMgr;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags;
  LangOptions LangOpts;
  
  /// \brief The top-level modules that are known.
  llvm::StringMap<Module *> Modules;
  
  /// \brief Mapping from each header to the module that owns the contents of the
  /// that header.
  llvm::DenseMap<const FileEntry *, Module *> Headers;
  
  friend class ModuleMapParser;
  
public:
  /// \brief Construct a new module map.
  ///
  /// \param FileMgr The file manager used to find module files and headers.
  /// This file manager should be shared with the header-search mechanism, since
  /// they will refer to the same headers.
  ///
  /// \param DC A diagnostic consumer that will be cloned for use in generating
  /// diagnostics.
  ModuleMap(FileManager &FileMgr, const DiagnosticConsumer &DC);

  /// \brief Destroy the module map.
  ///
  ~ModuleMap();
  
  /// \brief Retrieve the module that owns the given header file, if any.
  ///
  /// \param File The header file that is likely to be included.
  ///
  /// \returns The module that owns the given header file, or null to indicate
  /// that no module owns this header file.
  Module *findModuleForHeader(const FileEntry *File);

  /// \brief Retrieve a module with the given name.
  ///
  /// \param The name of the module to look up.
  ///
  /// \returns The named module, if known; otherwise, returns null.
  Module *findModule(StringRef Name);
  
  /// \brief Parse the given module map file, and record any modules we 
  /// encounter.
  ///
  /// \param File The file to be parsed.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool parseModuleMapFile(const FileEntry *File);
    
  /// \brief Dump the contents of the module map, for debugging purposes.
  void dump();
};
  
}
#endif
