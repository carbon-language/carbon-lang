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

#include "clang/Driver/ToolChain.h"

#include "llvm/Support/Compiler.h"

namespace clang {
namespace driver {
namespace toolchains VISIBILITY_HIDDEN {

class Generic_GCC : public ToolChain {
public:
  Generic_GCC(const HostInfo &Host, const char *Arch, const char *Platform, 
              const char *OS) : ToolChain(Host, Arch, Platform, OS) {
  }

  virtual ArgList *TranslateArgs(ArgList &Args) const { return &Args; }

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const {
    return *((Tool*) 0);
  }

  virtual bool IsMathErrnoDefault() const { return true; }

  virtual bool IsUnwindTablesDefault() const { 
    // FIXME: Gross; we should probably have some separate target definition,
    // possibly even reusing the one in clang.
    return getArchName() == "x86_64";
  }

  virtual const char *GetDefaultRelocationModel() const {
    return "static";
  }

  virtual const char *GetForcedPicModel() const {
    return 0;
  }
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
