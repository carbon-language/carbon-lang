//===--- Module.h - Describe a module ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Module class, which describes a module in the source
// code.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_MODULE_H
#define LLVM_CLANG_BASIC_MODULE_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>

namespace llvm {
  class raw_ostream;
}

namespace clang {
  
class FileEntry;

/// \brief Describes the name of a module.
typedef llvm::SmallVector<std::pair<std::string, SourceLocation>, 2>
  ModuleId;
  
/// \brief Describes a module or submodule.
class Module {
public:
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
  
  /// \brief Whether this is a framework module.
  bool IsFramework;
  
  /// \brief Whether this is an explicit submodule.
  bool IsExplicit;
  
  /// \brief Describes the visibility of the various names within a
  /// particular module.
  enum NameVisibilityKind {
    /// \brief All of the names in this module are hidden.
    ///
    Hidden,
    /// \brief Only the macro names in this module are visible.
    MacrosVisible,
    /// \brief All of the names in this module are visible.
    AllVisible
  };  
  
  ///\ brief The visibility of names within this particular module.
  NameVisibilityKind NameVisibility;
  
  /// \brief Describes an exported module.
  ///
  /// The pointer is the module being re-exported, while the bit will be true
  /// to indicate that this is a wildcard export.
  typedef llvm::PointerIntPair<Module *, 1, bool> ExportDecl;
  
  /// \brief The set of export declarations.
  llvm::SmallVector<ExportDecl, 2> Exports;
  
  /// \brief Describes an exported module that has not yet been resolved
  /// (perhaps because the module it refers to has not yet been loaded).
  struct UnresolvedExportDecl {
    /// \brief The location of the 'export' keyword in the module map file.
    SourceLocation ExportLoc;
    
    /// \brief The name of the module.
    ModuleId Id;
    
    /// \brief Whether this export declaration ends in a wildcard, indicating
    /// that all of its submodules should be exported (rather than the named
    /// module itself).
    bool Wildcard;
  };
  
  /// \brief The set of export declarations that have yet to be resolved.
  llvm::SmallVector<UnresolvedExportDecl, 2> UnresolvedExports;
  
  /// \brief Construct a top-level module.
  explicit Module(StringRef Name, SourceLocation DefinitionLoc,
                  bool IsFramework)
    : Name(Name), DefinitionLoc(DefinitionLoc), Parent(0), UmbrellaHeader(0),
      IsFramework(IsFramework), IsExplicit(false), NameVisibility(Hidden) { }
  
  /// \brief Construct  a new module or submodule.
  Module(StringRef Name, SourceLocation DefinitionLoc, Module *Parent, 
         bool IsFramework, bool IsExplicit)
    : Name(Name), DefinitionLoc(DefinitionLoc), Parent(Parent), 
      UmbrellaHeader(0), IsFramework(IsFramework), IsExplicit(IsExplicit), 
      NameVisibility(Hidden) { }
  
  ~Module();
  
  /// \brief Determine whether this module is a submodule.
  bool isSubModule() const { return Parent != 0; }
  
  /// \brief Determine whether this module is a part of a framework,
  /// either because it is a framework module or because it is a submodule
  /// of a framework module.
  bool isPartOfFramework() const {
    for (const Module *Mod = this; Mod; Mod = Mod->Parent) 
      if (Mod->IsFramework)
        return true;
    
    return false;
  }
  
  /// \brief Retrieve the full name of this module, including the path from
  /// its top-level module.
  std::string getFullModuleName() const;
  
  /// \brief Retrieve the name of the top-level module.
  ///
  StringRef getTopLevelModuleName() const;
  
  /// \brief Print the module map for this module to the given stream. 
  ///
  void print(llvm::raw_ostream &OS, unsigned Indent = 0) const;
  
  /// \brief Dump the contents of this module to the given output stream.
  void dump() const;
};

} // end namespace clang


#endif // LLVM_CLANG_BASIC_MODULE_H
