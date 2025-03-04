// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/llvm_tools.h"

#include "common/command_line.h"

// NOLINTBEGIN(readability-identifier-naming): External library name.
#define CARBON_LLVM_MAIN_TOOL(Identifier, Name, BinName, MainFn)              \
  extern auto MainFn(int argc, char** argv, const llvm::ToolContext& context) \
      -> int;
#include "toolchain/base/llvm_tools.def"
// NOLINTEND(readability-identifier-naming): External library name.

namespace Carbon {

CARBON_DEFINE_ENUM_CLASS_NAMES(LLVMTool) = {
#define CARBON_LLVM_TOOL(Identifier, Name, BinName, MainFn) Name,
#include "toolchain/base/llvm_tools.def"
};

constexpr llvm::StringLiteral LLVMTool::BinNames[] = {
#define CARBON_LLVM_TOOL(Identifier, Name, BinName, MainFn) BinName,
#include "toolchain/base/llvm_tools.def"
};

constexpr LLVMTool::MainFnT* LLVMTool::MainFns[] = {
#define CARBON_LLVM_TOOL(Identifier, Name, BinName, MainFn) &::MainFn,
#include "toolchain/base/llvm_tools.def"
};

constexpr CommandLine::CommandInfo LLVMTool::SubcommandInfos[] = {
#define CARBON_LLVM_TOOL(Identifier, Name, BinName, MainFn) \
  {.name = Name,                                            \
   .help = "Runs the LLVM " Name                            \
           " command line tool with the provided arguments."},
#include "toolchain/base/llvm_tools.def"
};

}  // namespace Carbon
