//===--- ForceLinkageMacros.h - The LLVM Compiler Driver --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Preprocessor magic that forces references to static libraries - common
//  macros used by both driver and plugins.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_FORCE_LINKAGE_MACROS_H
#define LLVM_INCLUDE_COMPILER_DRIVER_FORCE_LINKAGE_MACROS_H

#define LLVMC_FORCE_LINKAGE_PREFIX(PluginName) ForceLinkage ## PluginName

#define LLVMC_FORCE_LINKAGE_FUN(PluginName) \
  LLVMC_FORCE_LINKAGE_PREFIX(PluginName)

#define LLVMC_FORCE_LINKAGE_DECL(PluginName) \
  void LLVMC_FORCE_LINKAGE_FUN(PluginName) ()

#define LLVMC_FORCE_LINKAGE_CALL(PluginName) \
  LLVMC_FORCE_LINKAGE_FUN(PluginName) ()

#endif // LLVM_INCLUDE_COMPILER_DRIVER_FORCE_LINKAGE_MACROS_H
