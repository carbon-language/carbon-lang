//===--- ForceLinkage.h - The LLVM Compiler Driver --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  A bit of preprocessor magic to force references to static libraries. Needed
//  because plugin initialization is done via static variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_FORCE_LINKAGE_H
#define LLVM_INCLUDE_COMPILER_DRIVER_FORCE_LINKAGE_H

#include "llvm/CompilerDriver/ForceLinkageMacros.h"

namespace llvmc {

// Declare all ForceLinkage$(PluginName) functions.

#ifdef LLVMC_BUILTIN_PLUGIN_1
      LLVMC_FORCE_LINKAGE_DECL(LLVMC_BUILTIN_PLUGIN_1);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_2
      LLVMC_FORCE_LINKAGE_DECL(LLVMC_BUILTIN_PLUGIN_2);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_3
      LLVMC_FORCE_LINKAGE_DECL(LLVMC_BUILTIN_PLUGIN_3);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_4
      LLVMC_FORCE_LINKAGE_DECL(LLVMC_BUILTIN_PLUGIN_4);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_5
      LLVMC_FORCE_LINKAGE_DECL(LLVMC_BUILTIN_PLUGIN_5);
#endif

namespace force_linkage {

  struct LinkageForcer {

    LinkageForcer() {

// Call all ForceLinkage$(PluginName) functions.
#ifdef LLVMC_BUILTIN_PLUGIN_1
      LLVMC_FORCE_LINKAGE_CALL(LLVMC_BUILTIN_PLUGIN_1);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_2
      LLVMC_FORCE_LINKAGE_CALL(LLVMC_BUILTIN_PLUGIN_2);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_3
      LLVMC_FORCE_LINKAGE_CALL(LLVMC_BUILTIN_PLUGIN_3);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_4
      LLVMC_FORCE_LINKAGE_CALL(LLVMC_BUILTIN_PLUGIN_4);
#endif

#ifdef LLVMC_BUILTIN_PLUGIN_5
      LLVMC_FORCE_LINKAGE_CALL(LLVMC_BUILTIN_PLUGIN_5);
#endif

    }
  };
} // End namespace force_linkage.

// The only externally used bit.
void ForceLinkage() {
  force_linkage::LinkageForcer dummy;
}

} // End namespace llvmc.

#endif // LLVM_INCLUDE_COMPILER_DRIVER_FORCE_LINKAGE_H
