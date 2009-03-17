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

  virtual ArgList *TranslateArgs(ArgList &Args) const { return &Args; }

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const {
    Action::ActionClass Key;
    if (ShouldUseClangCompiler(C, JA))
      Key = Action::AnalyzeJobClass;
    else
      Key = JA.getKind();

    Tool *&T = Tools[Key];
    if (!T) {
      switch (Key) {
      default:
        assert(0 && "Invalid tool kind.");
      case Action::PreprocessJobClass: 
        T = new tools::gcc::Preprocess(*this); break;
      case Action::PrecompileJobClass: 
        T = new tools::gcc::Precompile(*this); break;
      case Action::AnalyzeJobClass: 
        T = new tools::Clang(*this); break;
      case Action::CompileJobClass: 
        T = new tools::gcc::Compile(*this); break;
      case Action::AssembleJobClass: 
        T = new tools::gcc::Assemble(*this); break;
      case Action::LinkJobClass: 
        T = new tools::gcc::Link(*this); break;
      }
    }

    return *T;
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
