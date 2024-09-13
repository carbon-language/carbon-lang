// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_COMPILE_SUBCOMMAND_H_
#define CARBON_TOOLCHAIN_DRIVER_COMPILE_SUBCOMMAND_H_

#include "common/command_line.h"
#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/codegen_options.h"

namespace Carbon {

// Options for the compile subcommand.
//
// See the implementation of `Build` for documentation on members.
struct CompileOptions {
  static const CommandLine::CommandInfo Info;

  enum class Phase : int8_t {
    Lex,
    Parse,
    Check,
    Lower,
    CodeGen,
  };

  friend auto operator<<(llvm::raw_ostream& out, Phase phase)
      -> llvm::raw_ostream&;

  auto Build(CommandLine::CommandBuilder& b, CodegenOptions& codegen_options)
      -> void;

  Phase phase;

  llvm::StringRef output_filename;
  llvm::SmallVector<llvm::StringRef> input_filenames;

  bool asm_output = false;
  bool force_obj_output = false;
  bool dump_shared_values = false;
  bool dump_tokens = false;
  bool dump_parse_tree = false;
  bool dump_raw_sem_ir = false;
  bool dump_sem_ir = false;
  bool dump_llvm_ir = false;
  bool dump_asm = false;
  bool dump_mem_usage = false;
  bool stream_errors = false;
  bool preorder_parse_tree = false;
  bool builtin_sem_ir = false;
  bool prelude_import = false;
  bool include_debug_info = true;

  llvm::StringRef exclude_dump_file_prefix;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_COMPILE_SUBCOMMAND_H_
