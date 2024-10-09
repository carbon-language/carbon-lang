// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/format_subcommand.h"

#include <string>

#include "toolchain/base/value_store.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"
#include "toolchain/format/format.h"
#include "toolchain/lex/lex.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

constexpr CommandLine::CommandInfo FormatOptions::Info = {
    .name = "format",
    .help = R"""(
Format Carbon source code.
)""",
};

auto FormatOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringPositionalArg(
      {
          .name = "FILE",
          .help = R"""(
The input Carbon source file(s) to format.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Append(&input_filenames);
      });
  b.AddStringOption(
      {
          .name = "output",
          .value_name = "FILE",
          .help = R"""(
The output filename for formatted output.

By default, the input file is formatted. Passing `--output=-` will write the
output to stdout.

Not valid when multiple files are passed for formatting.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_filename); });
}

auto FormatSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  DriverResult result = {.success = true};
  if (options_.input_filenames.size() > 1 &&
      !options_.output_filename.empty()) {
    driver_env.error_stream << "error: cannot format multiple input files when "
                               "--output is set\n";
    result.success = false;
    return result;
  }

  auto mark_per_file_error = [&]() {
    result.success = false;
    result.per_file_success.back().second = false;
  };

  StreamDiagnosticConsumer consumer(driver_env.error_stream);
  for (auto& f : options_.input_filenames) {
    // Push a result, which we'll update on failure.
    result.per_file_success.push_back({f.str(), true});

    // TODO: Consider refactoring this for sharing with compile.
    // TODO: Decide what to do with `-` when there are multiple arguments.
    auto source = SourceBuffer::MakeFromFileOrStdin(driver_env.fs, f, consumer);
    if (!source) {
      mark_per_file_error();
      continue;
    }
    SharedValueStores value_stores;
    auto tokens = Lex::Lex(value_stores, *source, consumer);

    std::string buffer_str;
    llvm::raw_string_ostream buffer(buffer_str);

    if (Format::Format(tokens, buffer)) {
      // TODO: Figure out a multi-file output setup that supports good
      // multi-file testing.
      // TODO: Use --output values (and default to overwrite).
      driver_env.output_stream << buffer_str;
    } else {
      mark_per_file_error();
      driver_env.output_stream << source->text();
    }
  }

  return result;
}

}  // namespace Carbon
