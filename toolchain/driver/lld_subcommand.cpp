// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/lld_subcommand.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/lld_runner.h"

namespace Carbon {

auto LldOptions::Build(CommandLine::CommandBuilder& b) -> void {
  // We want to select a default platform based on the default target. Since
  // that requires some dynamic inspection of the target, do that here.
  std::string default_target = llvm::sys::getDefaultTargetTriple();
  llvm::Triple default_triple(default_target);
  switch (default_triple.getObjectFormat()) {
    case llvm::Triple::MachO:
      platform = Platform::MachO;
      break;

      // We default to the GNU or Unix platform as ELF is a plausible default
      // and LLD doesn't support any generic invocations.
    default:
    case llvm::Triple::ELF:
      platform = Platform::Elf;
      break;
  }

  b.AddOneOfOption(
      {
          .name = "platform",
          .help = R"""(
Platform linking style to use. The default is selected to match the default
target's platform.
)""",
      },
      [&](auto& arg_b) {
        arg_b.SetOneOf(
            {
                arg_b.OneOfValue("elf", Platform::Elf),
                // Some of LLD documentation uses "Unix" or "GNU", so
                // include an alias here.
                arg_b.OneOfValue("gnu", Platform::Elf),
                arg_b.OneOfValue("unix", Platform::Elf),

                arg_b.OneOfValue("macho", Platform::MachO),
                // Darwin is also sometimes used, include it as an alias here.
                arg_b.OneOfValue("darwin", Platform::MachO),
            },
            &platform);
      });
  b.AddStringPositionalArg(
      {
          .name = "ARG",
          .help = R"""(
Arguments passed to LLD.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&args); });
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "lld",
    .help = R"""(
Runs LLD with the provided arguments.

Note that a specific LLD platform must be selected, and it is actually that
particular platform's LLD-driver that is run with the arguments. There is no
generic LLD command line.

For a given platform, this is equivalent to running that platform's LLD alias
directly, and provides the full command line interface.

Use `carbon lld --platform=elf -- ARGS` to separate the `ARGS` forwarded to LLD
from the flags passed to the Carbon subcommand.

Note that typically it is better to use a higher level command to link code,
such as invoking `carbon link` with the relevant flags. However, this subcommand
supports when you already have a specific invocation using existing command line
syntaxes, as well as testing and debugging of the underlying tool.
)""",
};

LldSubcommand::LldSubcommand() : DriverSubcommand(SubcommandInfo) {}

// TODO: This lacks a lot of features from the main driver code. We may need to
// add more.
// https://github.com/llvm/llvm-project/blob/main/clang/tools/driver/driver.cpp
auto LldSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  LldRunner runner(driver_env.installation, driver_env.vlog_stream);

  // Don't run LLD when fuzzing, as we're not currently in a good position to
  // debug and fix fuzzer-found bugs within LLD.
  if (driver_env.fuzzing) {
    CARBON_DIAGNOSTIC(
        LLDFuzzingDisallowed, Error,
        "preventing fuzzing of `lld` subcommand due to external library");
    driver_env.emitter.Emit(LLDFuzzingDisallowed);
    return {.success = false};
  }

  switch (options_.platform) {
    case LldOptions::Platform::Elf:
      return {.success = runner.ElfLink(options_.args)};
    case LldOptions::Platform::MachO:
      return {.success = runner.MachOLink(options_.args)};
  }
  CARBON_FATAL("Failed to find and run a valid LLD platform link!");
}

}  // namespace Carbon
