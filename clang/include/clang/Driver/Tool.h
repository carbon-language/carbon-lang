//===--- Tool.h - Compilation Tools -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_TOOL_H_
#define CLANG_DRIVER_TOOL_H_

namespace clang {
namespace driver {
  class ToolChain;
  
/// Tool - Information on a specific compilation tool.
class Tool {
  /// The tool name (for debugging).
  const char *Name;

  /// The tool chain this tool is a part of.
  const ToolChain &TheToolChain;

public:
  Tool(const char *Name, const ToolChain &TC);

public:
  virtual ~Tool();

  const char *getName() const { return Name; }

  const ToolChain &getToolChain() const { return TheToolChain; }

  virtual bool acceptsPipedInput() const = 0;
  virtual bool canPipeOutput() const = 0;
  virtual bool hasIntegratedCPP() const = 0;
};

} // end namespace driver
} // end namespace clang

#endif
