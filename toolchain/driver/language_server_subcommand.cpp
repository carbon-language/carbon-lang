// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/language_server_subcommand.h"

#include "toolchain/diagnostics/diagnostic_consumer.h"
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
  if (!driver_env.input_stream) {
    CARBON_DIAGNOSTIC(LanguageServerMissingInputStream, Error,
                      "language-server requires input_stream");
    driver_env.emitter.Emit(LanguageServerMissingInputStream);
    return {.success = false};
  }

  bool success = LanguageServer::Run(
      driver_env.input_stream, *driver_env.output_stream,
      *driver_env.error_stream, driver_env.vlog_stream, driver_env.consumer);
  return {.success = success};
}

}  // namespace Carbon
