// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/format_subcommand.h"

#include <string>

#include "common/raw_string_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/format/format.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/parse.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

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

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "format",
    .help = R"""(
Format Carbon source code.
)""",
};

FormatSubcommand::FormatSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto FormatSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  DriverResult result = {.success = true};
  if (options_.input_filenames.size() > 1 &&
      !options_.output_filename.empty()) {
    CARBON_DIAGNOSTIC(FormatMultipleFilesToOneOutput, Error,
                      "multiple input files provided; `--output` only works "
                      "with one input file");
    driver_env.emitter.Emit(FormatMultipleFilesToOneOutput);
    result.success = false;
    return result;
  }

  auto mark_per_file_error = [&]() {
    result.success = false;
    result.per_file_success.back().second = false;
  };

  for (llvm::StringRef filename : options_.input_filenames) {
    // Push a result, which we'll update on failure.
    result.per_file_success.push_back({filename.str(), true});

    // TODO: Consider refactoring this for sharing with compile.
    // TODO: Decide what to do with `-` when there are multiple arguments.
    auto source = SourceBuffer::MakeFromFileOrStdin(*driver_env.fs, filename,
                                                    *driver_env.consumer);
    if (!source) {
      mark_per_file_error();
      continue;
    }
    SharedValueStores value_stores;
    Lex::LexOptions lex_options;
    lex_options.consumer = driver_env.consumer;
    auto tokens = Lex::Lex(value_stores, *source, lex_options);

    Parse::ParseOptions parse_options;
    parse_options.consumer = driver_env.consumer;
    auto tree = Parse::Parse(tokens, parse_options);

    // Formatting is best-effort: it always produces output, even for input with
    // errors. The return value reports whether the input was error-free; the
    // best-effort output is used regardless, but a file with errors is still
    // marked as a failure (for example, for the exit code).
    RawStringOstream buffer;
    bool formatted_cleanly = Format::Format(tree, buffer);
    std::string formatted = buffer.TakeStr();

    // Decide where the formatted output goes:
    //   --output=-     -> stdout,
    //   --output=FILE  -> FILE,
    //   (no --output)  -> overwrite the input file in place.
    // When the input came from stdin (`-`) and no `--output` was given there is
    // no file to overwrite, so `dest` is `-` and the output goes to stdout.
    llvm::StringRef dest = options_.output_filename.empty()
                               ? filename
                               : llvm::StringRef(options_.output_filename);
    if (dest == "-") {
      *driver_env.output_stream << formatted;
    } else if (options_.output_filename.empty() &&
               formatted == source->text()) {
      // Already formatted: skip rewriting the input in place. This keeps the
      // file's timestamp unchanged for build systems and watchers, and avoids
      // any window where the file is truncated.
    } else {
      // Opening the destination truncates it, and the default destination is
      // the input file itself, so from here through a successful write is the
      // window where an I/O error loses the input. Check the stream's error
      // state explicitly after writing: the destructor would otherwise crash
      // the whole driver on an unchecked error, with no per-file diagnostic.
      //
      // TODO: Write to a temporary file in the same directory and rename it
      // into place, so a partial write can never clobber the input.
      std::error_code ec;
      llvm::raw_fd_ostream output_file(dest, ec, llvm::sys::fs::OF_None);
      if (ec) {
        CARBON_DIAGNOSTIC(FormatOutputFileOpenError, Error,
                          "could not open output file `{0}`: {1}", std::string,
                          std::string);
        driver_env.emitter.Emit(FormatOutputFileOpenError, dest.str(),
                                ec.message());
        mark_per_file_error();
        continue;
      }
      output_file << formatted;
      output_file.close();
      if (output_file.has_error()) {
        CARBON_DIAGNOSTIC(FormatOutputFileWriteError, Error,
                          "error writing output file `{0}`: {1}", std::string,
                          std::string);
        driver_env.emitter.Emit(FormatOutputFileWriteError, dest.str(),
                                output_file.error().message());
        output_file.clear_error();
        mark_per_file_error();
        continue;
      }
    }
    if (!formatted_cleanly) {
      mark_per_file_error();
    }
  }

  return result;
}

}  // namespace Carbon
