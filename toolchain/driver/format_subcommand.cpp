// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/format_subcommand.h"

#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/format/format.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/parse.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

namespace {

// Writes `contents` to `dest`, using a temporary file and atomic rename for
// regular files so an interrupted write or an `mmap`ped input buffer cannot
// corrupt or truncate the destination.
auto WriteOutputFile(DriverEnv& driver_env, llvm::StringRef dest,
                     llvm::StringRef contents) -> bool {
  CARBON_DIAGNOSTIC(FormatOutputFileOpenError, Error,
                    "could not open output file `{0}`: {1}", std::string,
                    std::string);
  CARBON_DIAGNOSTIC(FormatOutputFileWriteError, Error,
                    "error writing output file `{0}`: {1}", std::string,
                    std::string);

  // Follow symlinks so formatting a symlink in place updates its target file
  // rather than replacing the symlink itself.
  llvm::SmallString<256> resolved_dest;
  llvm::StringRef target = dest;
  llvm::sys::fs::file_status link_status;
  if (!llvm::sys::fs::status(dest, link_status, /*follow=*/false) &&
      llvm::sys::fs::is_symlink_file(link_status) &&
      !llvm::sys::fs::real_path(dest, resolved_dest)) {
    target = resolved_dest;
  }

  llvm::sys::fs::file_status dest_status;
  std::error_code status_ec = llvm::sys::fs::status(target, dest_status);
  if (!status_ec && llvm::sys::fs::is_directory(dest_status)) {
    driver_env.emitter.Emit(
        FormatOutputFileOpenError, dest.str(),
        std::make_error_code(std::errc::is_a_directory).message());
    return false;
  }

  if (!status_ec && llvm::sys::fs::is_other(dest_status)) {
    // Character devices, FIFOs, and similar special files cannot be replaced
    // by `rename`; write to them directly.
    std::error_code ec;
    llvm::raw_fd_ostream output_file(target, ec, llvm::sys::fs::OF_None);
    if (ec) {
      driver_env.emitter.Emit(FormatOutputFileOpenError, dest.str(),
                              ec.message());
      return false;
    }
    output_file << contents;
    output_file.close();
    if (output_file.has_error()) {
      driver_env.emitter.Emit(FormatOutputFileWriteError, dest.str(),
                              output_file.error().message());
      output_file.clear_error();
      return false;
    }
    return true;
  }

  llvm::Expected<llvm::sys::fs::TempFile> temp =
      llvm::sys::fs::TempFile::create(llvm::Twine(target) + ".tmp%%%%%%%");
  if (!temp) {
    driver_env.emitter.Emit(FormatOutputFileOpenError, dest.str(),
                            llvm::errorToErrorCode(temp.takeError()).message());
    return false;
  }

  {
    llvm::raw_fd_ostream output_file(temp->FD, /*shouldClose=*/false);
    output_file << contents;
    output_file.flush();
    if (output_file.has_error()) {
      std::string message = output_file.error().message();
      output_file.clear_error();
      llvm::consumeError(temp->discard());
      driver_env.emitter.Emit(FormatOutputFileWriteError, dest.str(), message);
      return false;
    }
  }

  if (!status_ec && llvm::sys::fs::is_regular_file(dest_status)) {
    // Best-effort preservation of existing file permissions.
    (void)llvm::sys::fs::setPermissions(temp->FD, dest_status.permissions());
  }

  if (llvm::Error err = temp->keep(target)) {
    std::string message = llvm::errorToErrorCode(std::move(err)).message();
    llvm::consumeError(temp->discard());
    driver_env.emitter.Emit(FormatOutputFileWriteError, dest.str(), message);
    return false;
  }
  return true;
}

}  // namespace

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

Not valid when multiple files are passed for formatting, except `--output=-`.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_filename); });
  b.AddStringOption(
      {
          .name = "lines",
          .value_name = "START:END",
          .help = R"""(
Format only the given inclusive range of 1-based line numbers, leaving the rest
of the file unchanged. By default the whole file is formatted.

Not valid when multiple files are passed for formatting.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&lines); });
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "format",
    .help = R"""(
Format Carbon source code.
)""",
};

// Parses a `START:END` line-range argument, returning nullopt if it is
// malformed (non-numeric, or not 1 <= START <= END).
static auto ParseLineRange(llvm::StringRef arg)
    -> std::optional<Format::LineRange> {
  auto [first_str, last_str] = arg.split(':');
  int first = 0;
  int last = 0;
  if (first_str.getAsInteger(10, first) || last_str.getAsInteger(10, last) ||
      first < 1 || last < first) {
    return std::nullopt;
  }
  return Format::LineRange{.first_line = first, .last_line = last};
}

FormatSubcommand::FormatSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto FormatSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  DriverResult result = {.success = true};
  if (options_.input_filenames.size() > 1 &&
      !options_.output_filename.empty() && options_.output_filename != "-") {
    CARBON_DIAGNOSTIC(FormatMultipleFilesToOneOutput, Error,
                      "multiple input files provided; `--output` only works "
                      "with one input file");
    driver_env.emitter.Emit(FormatMultipleFilesToOneOutput);
    result.success = false;
    return result;
  }
  if (llvm::count(options_.input_filenames, "-") > 1) {
    CARBON_DIAGNOSTIC(FormatMultipleStdinInputs, Error,
                      "`-` (standard input) cannot be specified more than "
                      "once");
    driver_env.emitter.Emit(FormatMultipleStdinInputs);
    result.success = false;
    return result;
  }
  if (options_.input_filenames.size() > 1 && !options_.lines.empty()) {
    CARBON_DIAGNOSTIC(FormatMultipleFilesWithLines, Error,
                      "multiple input files provided; `--lines` only works "
                      "with one input file");
    driver_env.emitter.Emit(FormatMultipleFilesWithLines);
    result.success = false;
    return result;
  }

  std::optional<Format::LineRange> lines;
  if (!options_.lines.empty()) {
    lines = ParseLineRange(options_.lines);
    if (!lines) {
      CARBON_DIAGNOSTIC(FormatInvalidLineRange, Error,
                        "invalid `--lines` value `{0}`; expected `START:END` "
                        "with 1 <= START <= END",
                        std::string);
      driver_env.emitter.Emit(FormatInvalidLineRange, options_.lines.str());
      result.success = false;
      return result;
    }
  }

  auto mark_per_file_error = [&]() {
    result.success = false;
    result.per_file_success.back().second = false;
  };

  for (llvm::StringRef filename : options_.input_filenames) {
    // Push a result, which we'll update on failure.
    result.per_file_success.push_back({filename.str(), true});

    // TODO: Consider refactoring this for sharing with compile.
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
    std::string formatted;
    bool formatted_cleanly = true;
    if (!lines) {
      // Format the whole file.
      RawStringOstream buffer;
      formatted_cleanly = Format::Format(tree, buffer);
      formatted = buffer.TakeStr();
    } else {
      // Format only the requested line range, leaving the rest unchanged.
      llvm::SmallVector<Format::Replacement> replacements;
      formatted_cleanly =
          Format::FormatReplacements(tree, replacements, *lines);
      formatted = Format::ApplyReplacements(source->text(), replacements);
    }

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
      // Already formatted: skip rewriting the input in place so the file's
      // modification time is preserved for build systems and watchers.
    } else if (!WriteOutputFile(driver_env, dest, formatted)) {
      mark_per_file_error();
      continue;
    }
    if (!formatted_cleanly) {
      mark_per_file_error();
    }
  }

  return result;
}

}  // namespace Carbon
