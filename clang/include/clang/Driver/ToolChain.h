//===--- ToolChain.h - Collections of tools for one platform ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_TOOLCHAIN_H_
#define CLANG_DRIVER_TOOLCHAIN_H_

#include "llvm/System/Path.h"
#include <string>

namespace clang {
namespace driver {
  class ArgList;
  class Compilation;
  class Driver;
  class JobAction;
  class Tool;

/// ToolChain - Access to tools for a single platform.
class ToolChain {
  Driver &TheDriver;

  std::string Arch, Platform, OS;

protected:
  ToolChain(Driver &D, const char *_Arch, const char *_Platform, 
            const char *_OS);

public:
  virtual ~ToolChain();

  // Accessors

  const std::string &getArchName() const { return Arch; }
  const std::string &getPlatform() const { return Platform; }
  const std::string &getOS() const { return OS; }

  // Tool access.

  /// TranslateArgs - Create a new derived argument list for any
  /// argument translations this ToolChain may wish to perform.
  virtual ArgList *TranslateArgs(const ArgList &Args) const = 0;

  /// SelectTool - Choose a tool to use to handle the action \arg JA.
  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const = 0;

  // Helper methods

  llvm::sys::Path GetFilePath(const Compilation &C, const char *Name) const;
  llvm::sys::Path GetProgramPath(const Compilation &C, const char *Name) const;

  /// ShouldUseClangCompilar - Should the clang compiler be used to
  /// handle this action.
  bool ShouldUseClangCompiler(const Compilation &C, const JobAction &JA) const;

  // Platform defaults information

  /// IsMathErrnoDefault - Does this tool chain set -fmath-errno by
  /// default.
  virtual bool IsMathErrnoDefault() const = 0;

  /// IsUnwindTablesDefault - Does this tool chain use -funwind-tables
  /// by default.
  virtual bool IsUnwindTablesDefault() const = 0;

  /// GetDefaultRelocationModel - Return the LLVM name of the default
  /// relocation model for this tool chain.
  virtual const char *GetDefaultRelocationModel() const = 0;

  /// GetForcedPicModel - Return the LLVM name of the forced PIC model
  /// for this tool chain, or 0 if this tool chain does not force a
  /// particular PIC mode.
  virtual const char *GetForcedPicModel() const = 0;
};

} // end namespace driver
} // end namespace clang

#endif
