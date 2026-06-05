// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LINK_OPTIONS_H_
#define CARBON_TOOLCHAIN_DRIVER_LINK_OPTIONS_H_

#include <memory>

#include "common/command_line.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

// Options for the link subcommand.
//
// See the implementation of `link` for documentation on members.
struct LinkOptions {
  auto BuildForLinkSubcommand(CommandLine::CommandBuilder& b) -> void;
  auto BuildForBuildSubcommand(CommandLine::CommandBuilder& b) -> void;

  std::shared_ptr<CodegenOptions> codegen_options;
  llvm::StringRef output_filename;
  llvm::SmallVector<llvm::StringRef> object_filenames;

  llvm::SmallVector<llvm::StringRef> extra_clang_args;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LINK_OPTIONS_H_
