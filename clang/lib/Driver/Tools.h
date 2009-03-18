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
                              InputInfo &Output, InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };

  /// gcc - Generic GCC tool implementations.
namespace gcc {
  class VISIBILITY_HIDDEN Preprocess : public Tool {
  public:
    Preprocess(const ToolChain &TC) : Tool("gcc::Preprocess", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              InputInfo &Output, InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };

  class VISIBILITY_HIDDEN Precompile : public Tool  {
  public:
    Precompile(const ToolChain &TC) : Tool("gcc::Precompile", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              InputInfo &Output, InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };

  class VISIBILITY_HIDDEN Compile : public Tool  {
  public:
    Compile(const ToolChain &TC) : Tool("gcc::Compile", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              InputInfo &Output, InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };

  class VISIBILITY_HIDDEN Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("gcc::Assemble", TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              InputInfo &Output, InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };

  class VISIBILITY_HIDDEN Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("gcc::Link", TC) {}

    virtual bool acceptsPipedInput() const { return false; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              InputInfo &Output, InputInfoList &Inputs, 
                              const ArgList &TCArgs, 
                              const char *LinkingOutput) const;
  };
} // end namespace gcc

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
