// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/bazel_working_dir.h"
#include "common/command_line.h"
#include "common/init_llvm.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "testing/base/source_gen.h"

namespace Carbon::Testing {
namespace {

constexpr CommandLine::CommandInfo Info = {
    .name = "source_gen",
    .help = R"""(
A source generator for Carbon.
)""",
};

constexpr CommandLine::ArgInfo OutputArgInfo = {
    .name = "output",
    .value_name = "FILE",
    .help = R"""(
Writes the generate source code to a file rather than stdout.
)""",
};

constexpr CommandLine::ArgInfo LinesArgInfo = {
    .name = "lines",
    .value_name = "N",
    .help = R"""(
The number of lines of code to target for a generated source file.
)""",
};

constexpr CommandLine::ArgInfo LanguageArgInfo = {
    .name = "language",
    //.value_name = "[carbon|cpp]",
    .help = R"""(
The language of source code to generate. The C++ source generation is best
effort to try to provide as much comparable benchmarking as possible, but the
primary language focus is generating Carbon.
)""",
};

auto Run(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  // Default to outputting to stdout and writing 10k lines of source code.
  llvm::StringRef output_filename = "-";
  int lines = 10'000;
  SourceGen::Language language;

  auto parse_result = CommandLine::Parse(
      args, llvm::outs(), Info, [&](CommandLine::CommandBuilder& b) {
        b.AddStringOption(OutputArgInfo,
                          [&](auto& arg_b) { arg_b.Set(&output_filename); });
        b.AddIntegerOption(LinesArgInfo,
                           [&](auto& arg_b) { arg_b.Set(&lines); });
        b.AddOneOfOption(LanguageArgInfo, [&](auto& arg_b) {
          arg_b.SetOneOf(
              {
                  arg_b.OneOfValue("carbon", SourceGen::Language::Carbon)
                      .Default(true),
                  arg_b.OneOfValue("cpp", SourceGen::Language::Cpp),
              },
              &language);
        });

        // No-op action as there is only one operation for this command.
        b.Do([] {});
      });
  if (!parse_result.ok()) {
    llvm::errs() << "error: " << *parse_result << "\n";
    return false;
  } else if (*parse_result == CommandLine::ParseResult::MetaSuccess) {
    // Fully handled by the CLI library.
    return true;
  }

  std::optional<llvm::raw_fd_ostream> output_file;
  llvm::raw_fd_ostream* output = &llvm::outs();
  if (output_filename != "-") {
    std::error_code ec;
    output_file.emplace(output_filename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "ERROR: Unable to open output file '" << output_filename
                   << "': " << ec.message() << "\n";
      return false;
    }
    output = &*output_file;
  }

  SourceGen gen(language);
  *output << gen.GenAPIFileDenseDecls(lines, SourceGen::DenseDeclParams{});
  output->flush();
  return true;
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  // Do LLVM's initialization first, this will also transform UTF-16 to UTF-8.
  Carbon::InitLLVM init_llvm(argc, argv);

  Carbon::SetWorkingDirForBazel();

  llvm::SmallVector<llvm::StringRef> args(argv + 1, argv + argc);
  bool success = Carbon::Testing::Run(args);
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
