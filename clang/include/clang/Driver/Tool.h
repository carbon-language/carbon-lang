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

/// Tool - Information on a specific compilation tool.
class Tool {
protected:
  Tool();

public:
  virtual ~Tool();

  virtual bool acceptsPipedInput() const = 0;
  virtual bool canPipeOutput() const = 0;
  virtual bool hasIntegratedCPP() const = 0;
};

} // end namespace driver
} // end namespace clang

#endif
