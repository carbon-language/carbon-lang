// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <algorithm>
#include <memory>
#include <optional>

#include "common/command_line.h"
#include "common/version.h"
#include "toolchain/driver/clang_subcommand.h"
#include "toolchain/driver/compile_subcommand.h"
#include "toolchain/driver/format_subcommand.h"
#include "toolchain/driver/language_server_subcommand.h"
#include "toolchain/driver/link_subcommand.h"
#include "toolchain/driver/lld_subcommand.h"

namespace Carbon {

namespace {
struct Options {
  static const CommandLine::CommandInfo Info;

  auto Build(CommandLine::CommandBuilder& b) -> void;

  bool verbose = false;
  bool fuzzing = false;
  bool include_diagnostic_kind = false;

  ClangSubcommand clang;
  CompileSubcommand compile;
  FormatSubcommand format;
  LanguageServerSubcommand language_server;
  LinkSubcommand link;
  LldSubcommand lld;

  // On success, this is set to the subcommand to run.
  DriverSubcommand* selected_subcommand = nullptr;
};
}  // namespace

// Note that this is not constexpr so that it can include information generated
// in separate translation units and potentially overridden at link time in the
// version string.
const CommandLine::CommandInfo Options::Info = {
    .name = "carbon",
    .version = Version::ToolchainInfo,
    .help = R"""(
This is the unified Carbon Language toolchain driver. Its subcommands provide
all of the core behavior of the toolchain, including compilation, linking, and
developer tools. Each of these has its own subcommand, and you can pass a
specific subcommand to the `help` subcommand to get details about its usage.
)""",
    .help_epilogue = R"""(
For questions, issues, or bug reports, please use our GitHub project:

  https://github.com/carbon-language/carbon-lang
)""",
};

auto Options::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddFlag(
      {
          .name = "verbose",
          .short_name = "v",
          .help = "Enable verbose logging to the stderr stream.",
      },
      [&](CommandLine::FlagBuilder& arg_b) { arg_b.Set(&verbose); });

  b.AddFlag(
      {
          .name = "fuzzing",
          .help = "Configure the command line for fuzzing.",
      },
      [&](CommandLine::FlagBuilder& arg_b) { arg_b.Set(&fuzzing); });

  b.AddFlag(
      {
          .name = "include-diagnostic-kind",
          .help = R"""(
When printing diagnostics, include the diagnostic kind as part of output. This
applies to each message that forms a diagnostic, not just the primary message.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&include_diagnostic_kind); });

  clang.AddTo(b, &selected_subcommand);
  compile.AddTo(b, &selected_subcommand);
  format.AddTo(b, &selected_subcommand);
  language_server.AddTo(b, &selected_subcommand);
  link.AddTo(b, &selected_subcommand);
  lld.AddTo(b, &selected_subcommand);

  b.RequiresSubcommand();
}

auto Driver::RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> DriverResult {
  if (driver_env_.installation->error()) {
    CARBON_DIAGNOSTIC(DriverInstallInvalid, Error, "{0}", std::string);
    driver_env_.emitter.Emit(DriverInstallInvalid,
                             driver_env_.installation->error()->str());
    return {.success = false};
  }

  Options options;

  ErrorOr<CommandLine::ParseResult> result = CommandLine::Parse(
      args, *driver_env_.output_stream, Options::Info,
      [&](CommandLine::CommandBuilder& b) { options.Build(b); });

  // Regardless of whether the parse succeeded, try to use the diagnostic kind
  // flag.
  driver_env_.consumer.set_include_diagnostic_kind(
      options.include_diagnostic_kind);

  if (!result.ok()) {
    CARBON_DIAGNOSTIC(DriverCommandLineParseFailed, Error, "{0}", std::string);
    driver_env_.emitter.Emit(DriverCommandLineParseFailed,
                             PrintToString(result.error()));
    return {.success = false};
  } else if (*result == CommandLine::ParseResult::MetaSuccess) {
    return {.success = true};
  }

  if (options.verbose) {
    // Note this implies streamed output in order to interleave.
    driver_env_.vlog_stream = driver_env_.error_stream;
  }
  if (options.fuzzing) {
    driver_env_.fuzzing = true;
  }

  CARBON_CHECK(options.selected_subcommand != nullptr);
  return options.selected_subcommand->Run(driver_env_);
}

}  // namespace Carbon
