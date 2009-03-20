//===--- ToolChains.h - ToolChain Implementations ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_DRIVER_TOOLCHAINS_H_
#define CLANG_LIB_DRIVER_TOOLCHAINS_H_

#include "clang/Driver/Action.h"
#include "clang/Driver/ToolChain.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"

#include "Tools.h"

namespace clang {
namespace driver {
namespace toolchains {

  /// Generic_GCC - A tool chain using the 'gcc' command to perform
  /// all subcommands; this relies on gcc translating the majority of
  /// command line options.
class VISIBILITY_HIDDEN Generic_GCC : public ToolChain {
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

public:
  Generic_GCC(const HostInfo &Host, const char *Arch, const char *Platform, 
              const char *OS) : ToolChain(Host, Arch, Platform, OS) {}
  ~Generic_GCC();

  virtual ArgList *TranslateArgs(ArgList &Args) const { return &Args; }

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;

  virtual bool IsMathErrnoDefault() const;
  virtual bool IsUnwindTablesDefault() const;
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
