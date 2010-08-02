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

#include "clang/Driver/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/System/Path.h"
#include <string>

namespace clang {
namespace driver {
  class Compilation;
  class DerivedArgList;
  class Driver;
  class HostInfo;
  class InputArgList;
  class JobAction;
  class Tool;

/// ToolChain - Access to tools for a single platform.
class ToolChain {
public:
  typedef llvm::SmallVector<std::string, 4> path_list;

private:
  const HostInfo &Host;
  const llvm::Triple Triple;

  /// The list of toolchain specific path prefixes to search for
  /// files.
  path_list FilePaths;

  /// The list of toolchain specific path prefixes to search for
  /// programs.
  path_list ProgramPaths;

protected:
  ToolChain(const HostInfo &Host, const llvm::Triple &_Triple);

public:
  virtual ~ToolChain();

  // Accessors

  const Driver &getDriver() const;
  const llvm::Triple &getTriple() const { return Triple; }

  llvm::StringRef getArchName() const { return Triple.getArchName(); }
  llvm::StringRef getPlatform() const { return Triple.getVendorName(); }
  llvm::StringRef getOS() const { return Triple.getOSName(); }

  std::string getTripleString() const {
    return Triple.getTriple();
  }

  path_list &getFilePaths() { return FilePaths; }
  const path_list &getFilePaths() const { return FilePaths; }

  path_list &getProgramPaths() { return ProgramPaths; }
  const path_list &getProgramPaths() const { return ProgramPaths; }

  // Tool access.

  /// TranslateArgs - Create a new derived argument list for any argument
  /// translations this ToolChain may wish to perform, or 0 if no tool chain
  /// specific translations are needed.
  ///
  /// \param BoundArch - The bound architecture name, or 0.
  virtual DerivedArgList *TranslateArgs(const DerivedArgList &Args,
                                        const char *BoundArch) const {
    return 0;
  }

  /// SelectTool - Choose a tool to use to handle the action \arg JA.
  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const = 0;

  // Helper methods

  std::string GetFilePath(const char *Name) const;
  std::string GetProgramPath(const char *Name, bool WantFile = false) const;

  // Platform defaults information

  /// LookupTypeForExtension - Return the default language type to use for the
  /// given extension.
  virtual types::ID LookupTypeForExtension(const char *Ext) const;

  /// IsBlocksDefault - Does this tool chain enable -fblocks by default.
  virtual bool IsBlocksDefault() const { return false; }

  /// IsIntegratedAssemblerDefault - Does this tool chain enable -integrated-as
  /// by default.
  virtual bool IsIntegratedAssemblerDefault() const { return false; }

  /// IsObjCNonFragileABIDefault - Does this tool chain set
  /// -fobjc-nonfragile-abi by default.
  virtual bool IsObjCNonFragileABIDefault() const { return false; }

  /// IsObjCLegacyDispatchDefault - Does this tool chain set
  /// -fobjc-legacy-dispatch by default (this is only used with the non-fragile
  /// ABI).
  virtual bool IsObjCLegacyDispatchDefault() const { return false; }

  /// UseObjCMixedDispatchDefault - When using non-legacy dispatch, should the
  /// mixed dispatch method be used?
  virtual bool UseObjCMixedDispatch() const { return false; }

  /// GetDefaultStackProtectorLevel - Get the default stack protector level for
  /// this tool chain (0=off, 1=on, 2=all).
  virtual unsigned GetDefaultStackProtectorLevel() const { return 0; }

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

  /// Does this tool chain support Objective-C garbage collection.
  virtual bool SupportsObjCGC() const { return false; }

  /// UseDwarfDebugFlags - Embed the compile options to clang into the Dwarf
  /// compile unit information.
  virtual bool UseDwarfDebugFlags() const { return false; }

  /// UseSjLjExceptions - Does this tool chain use SjLj exceptions.
  virtual bool UseSjLjExceptions() const { return false; }
};

} // end namespace driver
} // end namespace clang

#endif
