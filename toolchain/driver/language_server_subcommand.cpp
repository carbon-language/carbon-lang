// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/language_server_subcommand.h"

#include "toolchain/language_server/language_server.h"

namespace Carbon {

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "language-server",
    .help = R"""(
Runs the language server.
)""",
};

LanguageServerSubcommand::LanguageServerSubcommand()
    : DriverSubcommand(SubcommandInfo) {}

auto LanguageServerSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  // TODO: Consider a way to override stdin, but it's a `FILE*` so less
  // convenient to work with.
  auto err = LanguageServer::Run(stdin, driver_env.output_stream);
  if (!err.ok()) {
    driver_env.error_stream << "error: " << err.error() << "\n";
  }
  return {.success = err.ok()};
}

}  // namespace Carbon
