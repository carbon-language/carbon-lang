// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <algorithm>
#include <memory>
#include <optional>

#include "common/command_line.h"
#include "common/version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Path.h"
#include "toolchain/base/value_store.h"

namespace Carbon {

struct Driver::Options {
  static const CommandLine::CommandInfo Info;

  enum class Subcommand : int8_t {
    Compile,
    Link,
  };

  void Build(CommandLine::CommandBuilder& b) {
    b.AddFlag(
        {
            .name = "verbose",
            .short_name = "v",
            .help = "Enable verbose logging to the stderr stream.",
        },
        [&](CommandLine::FlagBuilder& arg_b) { arg_b.Set(&verbose); });

    b.AddSubcommand(CompileOptions::Info,
                    [&](CommandLine::CommandBuilder& sub_b) {
                      compile_options.Build(sub_b, codegen_options);
                      sub_b.Do([&] { subcommand = Subcommand::Compile; });
                    });

    b.AddSubcommand(LinkOptions::Info, [&](CommandLine::CommandBuilder& sub_b) {
      link_options.Build(sub_b, codegen_options);
      sub_b.Do([&] { subcommand = Subcommand::Link; });
    });

    b.RequiresSubcommand();
  }

  bool verbose;
  Subcommand subcommand;

  CodegenOptions codegen_options;
  CompileOptions compile_options;
  LinkOptions link_options;
};

// Note that this is not constexpr so that it can include information generated
// in separate translation units and potentially overridden at link time in the
// version string.
const CommandLine::CommandInfo Driver::Options::Info = {
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

auto Driver::ParseArgs(llvm::ArrayRef<llvm::StringRef> args, Options& options)
    -> CommandLine::ParseResult {
  return CommandLine::Parse(
      args, driver_env_.output_stream, driver_env_.error_stream, Options::Info,
      [&](CommandLine::CommandBuilder& b) { options.Build(b); });
}

auto Driver::RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> RunResult {
  Options options;
  CommandLine::ParseResult result = ParseArgs(args, options);
  if (result == CommandLine::ParseResult::Error) {
    return {.success = false};
  } else if (result == CommandLine::ParseResult::MetaSuccess) {
    return {.success = true};
  }

  if (options.verbose) {
    // Note this implies streamed output in order to interleave.
    driver_env_.vlog_stream = &driver_env_.error_stream;
  }

  switch (options.subcommand) {
    case Options::Subcommand::Compile:
      return Compile(options.compile_options, options.codegen_options);
    case Options::Subcommand::Link:
      return Link(options.link_options, options.codegen_options);
  }
  llvm_unreachable("All subcommands handled!");
}

}  // namespace Carbon
