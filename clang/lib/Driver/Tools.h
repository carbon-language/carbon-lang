//===--- Tools.h - Tool Implementations -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_DRIVER_TOOLS_H_
#define CLANG_LIB_DRIVER_TOOLS_H_

#include "clang/Driver/Tool.h"
#include "clang/Driver/Util.h"

#include "llvm/Support/Compiler.h"

namespace clang {
namespace driver {
namespace tools {

  class VISIBILITY_HIDDEN Clang : public Tool {
  public:
    Clang(const ToolChain &TC) : Tool("clang", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              Job &Dest,
                              const InputInfo &Output, 
                              const InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };

  /// gcc - Generic GCC tool implementations.
namespace gcc {
  class VISIBILITY_HIDDEN Common : public Tool {
  public:
    Common(const char *Name, const ToolChain &TC) : Tool(Name, TC) {}

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              Job &Dest,
                              const InputInfo &Output, 
                              const InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;

    /// RenderExtraToolArgs - Render any arguments necessary to force
    /// the particular tool mode.
    virtual void RenderExtraToolArgs(ArgStringList &CmdArgs) const = 0;
  };

  
  class VISIBILITY_HIDDEN Preprocess : public Common {
  public:
    Preprocess(const ToolChain &TC) : Common("gcc::Preprocess", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void RenderExtraToolArgs(ArgStringList &CmdArgs) const;
  };

  class VISIBILITY_HIDDEN Precompile : public Common  {
  public:
    Precompile(const ToolChain &TC) : Common("gcc::Precompile", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void RenderExtraToolArgs(ArgStringList &CmdArgs) const;
  };

  class VISIBILITY_HIDDEN Compile : public Common  {
  public:
    Compile(const ToolChain &TC) : Common("gcc::Compile", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void RenderExtraToolArgs(ArgStringList &CmdArgs) const;
  };

  class VISIBILITY_HIDDEN Assemble : public Common  {
  public:
    Assemble(const ToolChain &TC) : Common("gcc::Assemble", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void RenderExtraToolArgs(ArgStringList &CmdArgs) const;
  };

  class VISIBILITY_HIDDEN Link : public Common  {
  public:
    Link(const ToolChain &TC) : Common("gcc::Link", TC) {}

    virtual bool acceptsPipedInput() const { return false; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void RenderExtraToolArgs(ArgStringList &CmdArgs) const;
  };
} // end namespace gcc

namespace darwin {
  class VISIBILITY_HIDDEN Lipo : public Tool  {
  public:
    Lipo(const ToolChain &TC) : Tool("gcc::Link", TC) {}

    virtual bool acceptsPipedInput() const { return false; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              Job &Dest,
                              const InputInfo &Output, 
                              const InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };
}

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
