// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_COMPILE_OPTIONS_H_
#define CARBON_TOOLCHAIN_DRIVER_COMPILE_OPTIONS_H_

#include <memory>

#include "common/command_line.h"
#include "common/error.h"
#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "toolchain/check/check.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/lower/lower.h"

namespace Carbon {

// Options for Carbon compilation. This struct is shared between the
// `build` and `compile` subcommands, supporting different flags
// for each subcomand.
//
// Note that the each subcommand supports its own `Build*()` function,
// supporting the different use cases for command-line control of compilation.
//
// Members are documented in their respective `Build` functions.
struct CompileOptions {
  enum class Phase : int8_t {
    Lex,
    Parse,
    Check,
    Lower,
    Optimize,
    CodeGen,
  };

  friend auto operator<<(llvm::raw_ostream& out, Phase phase)
      -> llvm::raw_ostream&;

  auto BuildForCompileSubcommand(CommandLine::CommandBuilder& b) -> void;
  auto BuildForBuildSubcommand(CommandLine::CommandBuilder& b) -> void;

  // Validate that the compile options make sense for the compilation phase
  // selected.
  auto ValidatePhase(Diagnostics::NoLocEmitter& emitter) const -> bool;

  // Validate the target before passing to clang.
  auto ValidateTarget(Diagnostics::NoLocEmitter& emitter)
      -> ErrorOr<const llvm::Target*>;

  // Build a clang invocation. We do this regardless of whether we're running
  // check, because this is essentially performing further option validation,
  // and we generally validate all options even if we're not using them for the
  // selected phases of compilation. We also use Clang's target option handling
  // to configure our target, to ensure that we are using the same ABI for both
  // the C++ and Carbon parts of the compilation.
  // TODO: Share any arguments we specify here with the `carbon clang`
  // subcommand.
  auto BuildClangInvocation(DriverEnv& driver_env)
      -> ErrorOr<std::shared_ptr<clang::CompilerInvocation>>;

  Lower::OptimizationLevel opt_level = Lower::OptimizationLevel::Debug;
  std::shared_ptr<CodegenOptions> codegen_options =
      std::make_shared<CodegenOptions>();

  llvm::SmallVector<llvm::StringRef> input_filenames;
  llvm::SmallVector<llvm::StringRef> clang_args;

  bool include_debug_info = true;
  bool run_llvm_verifier = true;

  Phase phase = Phase::CodeGen;
  Check::CheckParseTreesOptions::DumpSemIRRanges dump_sem_ir_ranges;

  llvm::StringRef output_filename;

  bool asm_output = false;
  bool force_obj_output = false;
  bool custom_core = false;
  bool dump_shared_values = false;
  bool dump_tokens = false;
  bool omit_file_boundary_tokens = false;
  bool dump_parse_tree = false;
  bool dump_raw_sem_ir = false;
  bool dump_sem_ir = false;
  bool dump_cpp_ast = false;
  bool dump_llvm_ir = false;
  bool dump_asm = false;
  bool dump_mem_usage = false;
  bool dump_timings = false;
  bool stream_errors = false;
  bool preorder_parse_tree = false;
  bool builtin_sem_ir = false;
  bool prelude_import = false;
  bool output_last_input_only = false;

  llvm::SmallVector<llvm::StringRef> exclude_dump_file_prefixes;

  llvm::StringRef sem_ir_crash_dump;

  bool mangle_string_fingerprint = false;

  // Get the LLVM optimization level corresponding to a Carbon optimization
  // level.
  static auto GetLLVMOptimizationLevel(Lower::OptimizationLevel opt_level)
      -> llvm::OptimizationLevel;

  // Get the `-O` flag corresponding to an optimization level.
  static auto GetClangOptimizationFlag(Lower::OptimizationLevel opt_level)
      -> llvm::StringLiteral;

  // Returns a string for printing the phase in a diagnostic.
  static auto PhaseToString(CompileOptions::Phase phase) -> std::string;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_COMPILE_OPTIONS_H_
