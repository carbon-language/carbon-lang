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

namespace llvm {
  template<typename T, unsigned N> class SmallVector;
}

namespace clang {
namespace driver {
  class ArgList;
  class Compilation;
  class InputInfo;
  class Job;
  class JobAction;
  class ToolChain;

  typedef llvm::SmallVector<InputInfo, 4> InputInfoList;

/// Tool - Information on a specific compilation tool.
class Tool {
  /// The tool name (for debugging).
  const char *Name;

  /// The human readable name for the tool, for use in diagnostics.
  const char *ShortName;

  /// The tool chain this tool is a part of.
  const ToolChain &TheToolChain;

public:
  Tool(const char *Name, const char *ShortName,
       const ToolChain &TC);

public:
  virtual ~Tool();

  const char *getName() const { return Name; }

  const char *getShortName() const { return ShortName; }

  const ToolChain &getToolChain() const { return TheToolChain; }

  virtual bool acceptsPipedInput() const = 0;
  virtual bool canPipeOutput() const = 0;
  virtual bool hasIntegratedAssembler() const { return false; }
  virtual bool hasIntegratedCPP() const = 0;

  /// \brief Does this tool have "good" standardized diagnostics, or should the
  /// driver add an additional "command failed" diagnostic on failures.
  virtual bool hasGoodDiagnostics() const { return false; }

  /// ConstructJob - Construct jobs to perform the action \arg JA,
  /// writing to \arg Output and with \arg Inputs.
  ///
  /// \param Dest - Where to put the resulting commands.
  /// \param TCArgs - The argument list for this toolchain, with any
  /// tool chain specific translations applied.
  /// \param LinkingOutput - If this output will eventually feed the
  /// linker, then this is the final output name of the linked image.
  virtual void ConstructJob(Compilation &C, const JobAction &JA,
                            Job &Dest,
                            const InputInfo &Output,
                            const InputInfoList &Inputs,
                            const ArgList &TCArgs,
                            const char *LinkingOutput) const = 0;
};

} // end namespace driver
} // end namespace clang

#endif
