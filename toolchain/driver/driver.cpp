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
#include "toolchain/driver/link_subcommand.h"

namespace Carbon {

namespace {
struct Options {
  static const CommandLine::CommandInfo Info;

  auto Build(CommandLine::CommandBuilder& b) -> void;

  bool verbose;
  bool fuzzing;

  ClangSubcommand clang;
  CompileSubcommand compile;
  FormatSubcommand format;
  LinkSubcommand link;

  // On success, this is set to the subcommand to run.
  DriverSubcommand* subcommand = nullptr;
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

  b.AddSubcommand(ClangOptions::Info, [&](CommandLine::CommandBuilder& sub_b) {
    clang.BuildOptions(sub_b);
    sub_b.Do([&] { subcommand = &clang; });
  });

  b.AddSubcommand(CompileOptions::Info,
                  [&](CommandLine::CommandBuilder& sub_b) {
                    compile.BuildOptions(sub_b);
                    sub_b.Do([&] { subcommand = &compile; });
                  });

  b.AddSubcommand(FormatOptions::Info, [&](CommandLine::CommandBuilder& sub_b) {
    format.BuildOptions(sub_b);
    sub_b.Do([&] { subcommand = &format; });
  });

  b.AddSubcommand(LinkOptions::Info, [&](CommandLine::CommandBuilder& sub_b) {
    link.BuildOptions(sub_b);
    sub_b.Do([&] { subcommand = &link; });
  });

  b.RequiresSubcommand();
}

auto Driver::RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> DriverResult {
  if (driver_env_.installation->error()) {
    llvm::errs() << "error: " << *driver_env_.installation->error() << "\n";
    return {.success = false};
  }

  Options options;

  CommandLine::ParseResult result = CommandLine::Parse(
      args, driver_env_.output_stream, driver_env_.error_stream, Options::Info,
      [&](CommandLine::CommandBuilder& b) { options.Build(b); });

  if (result == CommandLine::ParseResult::Error) {
    return {.success = false};
  } else if (result == CommandLine::ParseResult::MetaSuccess) {
    return {.success = true};
  }

  if (options.verbose) {
    // Note this implies streamed output in order to interleave.
    driver_env_.vlog_stream = &driver_env_.error_stream;
  }
  if (options.fuzzing) {
    SetFuzzing();
  }

  CARBON_CHECK(options.subcommand != nullptr);
  return options.subcommand->Run(driver_env_);
}

auto Driver::SetFuzzing() -> void { driver_env_.fuzzing = true; }

}  // namespace Carbon
