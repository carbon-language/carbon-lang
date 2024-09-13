// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LINK_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_LINK_SUBCOMMAND_H_

#include "common/command_line.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/codegen_options.h"

namespace Carbon {

// Options for the link subcommand.
//
// See the implementation of `Build` for documentation on members.
struct LinkOptions {
  static const CommandLine::CommandInfo Info;

  auto Build(CommandLine::CommandBuilder& b, CodegenOptions& codegen_options)
      -> void;

  llvm::StringRef output_filename;
  llvm::SmallVector<llvm::StringRef> object_filenames;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LINK_SUBCOMMAND_H_
