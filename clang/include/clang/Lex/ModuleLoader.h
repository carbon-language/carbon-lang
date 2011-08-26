//===--- ModuleLoader.h - Module Loader Interface ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ModuleLoader interface, which is responsible for 
//  loading named modules.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LEX_MODULE_LOADER_H
#define LLVM_CLANG_LEX_MODULE_LOADER_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class IdentifierInfo;
  
/// \brief An opaque key that is used to describe the module and can be 
/// interpreted by the module loader itself.
typedef void *ModuleKey;
  
/// \brief Abstract interface for a module loader.
///
/// This abstract interface describes a module loader, which is responsible
/// for resolving a module name (e.g., "std") to an actual module file, and
/// then loading that module.
class ModuleLoader {
public:
  virtual ~ModuleLoader();
  
  /// \brief Attempt to load the given module.
  ///
  /// This routine attempts to load the module described by the given 
  /// parameters.
  ///
  /// \param ImportLoc The location of the 'import' keyword.
  /// \param ModuleName The name of the module to be loaded.
  /// \param ModuleNameLoc The location of the module name.
  ///
  /// \returns If successful, a non-NULL module key describing this module.
  /// Otherwise, returns NULL to indicate that the module could not be
  /// loaded.
  virtual ModuleKey loadModule(SourceLocation ImportLoc, 
                               IdentifierInfo &ModuleName,
                               SourceLocation ModuleNameLoc) = 0;
};
  
}

#endif
