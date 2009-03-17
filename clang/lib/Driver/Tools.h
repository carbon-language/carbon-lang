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
namespace tools VISIBILITY_HIDDEN {

  class Clang : public Tool {
  public:
    Clang(const ToolChain &TC) : Tool(TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }
  };

  class GCC_Preprocess : public Tool {
  public:
    GCC_Preprocess(const ToolChain &TC) : Tool(TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return false; }
  };

  class GCC_Precompile : public Tool  {
  public:
    GCC_Precompile(const ToolChain &TC) : Tool(TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return true; }
  };

  class GCC_Compile : public Tool  {
  public:
    GCC_Compile(const ToolChain &TC) : Tool(TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }
  };

  class GCC_Assemble : public Tool  {
  public:
    GCC_Assemble(const ToolChain &TC) : Tool(TC) {}

    virtual bool acceptsPipedInput() const { return true; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }
  };

  class GCC_Link : public Tool  {
  public:
    GCC_Link(const ToolChain &TC) : Tool(TC) {}

    virtual bool acceptsPipedInput() const { return false; }
    virtual bool canPipeOutput() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }
  };

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
