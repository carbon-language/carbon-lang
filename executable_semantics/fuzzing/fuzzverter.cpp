// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An utility for converting between fuzzer protos and Carbon sources.
//
// For example, to convert a crashing input in text proto to carbon source:
// `fuzzverter --mode=proto_to_carbon --input file.textproto`
//
// To generate a new text proto from carbon source for seeding the corpus:
// `fuzzverter --mode=carbon_to_proto --input file.carbon`

#include <google/protobuf/text_format.h>

#include <cstdlib>
#include <fstream>
#include <ios>
#include <sstream>

#include "common/error.h"
#include "common/fuzzing/carbon.pb.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/fuzzing/ast_to_proto.h"
#include "executable_semantics/fuzzing/fuzzer_util.h"
#include "executable_semantics/syntax/parse.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace Carbon {

// Reads a file and returns its contents as a string.
static auto ReadFile(std::string_view file_name) -> ErrorOr<std::string> {
  std::ifstream file(file_name, std::ios::in);
  if (!file.is_open()) {
    return ErrorBuilder() << "Could not open " << file_name << " for reading";
  }
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

// Writes string `s` to `file_name`.
static auto WriteFile(std::string_view s, std::string_view file_name)
    -> ErrorOr<Success> {
  std::ofstream file(file_name, std::ios::out);
  if (!file.is_open()) {
    return ErrorBuilder() << "Could not open " << file_name << " for writing";
  }
  file << s;
  return Success();
}

// Converts text proto to Carbon source.
static auto TextProtoToCarbon(std::string_view input_file_name,
                              std::string_view output_file_name)
    -> ErrorOr<Success> {
  ASSIGN_OR_RETURN(const std::string input_contents, ReadFile(input_file_name));
  Fuzzing::Carbon carbon_proto;
  if (!google::protobuf::TextFormat::ParseFromString(input_contents,
                                                     &carbon_proto)) {
    return Error("Could not parse text proto");
  }
  const std::string carbon_source =
      ProtoToCarbonWithMain(carbon_proto.compilation_unit());
  return WriteFile(carbon_source, output_file_name);
}

// Converts Carbon source to text proto.
static auto CarbonToTextProto(std::string_view input_file_name,
                              std::string_view output_file_name)
    -> ErrorOr<Success> {
  Carbon::Arena arena;
  const ErrorOr<AST> ast = Carbon::Parse(&arena, input_file_name,
                                         /*trace=*/false);
  if (!ast.ok()) {
    return ErrorBuilder() << "Parsing failed: " << ast.error().message();
  }
  Fuzzing::Carbon carbon_proto;
  *carbon_proto.mutable_compilation_unit() = AstToProto(*ast);

  std::string proto_string;
  google::protobuf::TextFormat::Printer p;
  if (!p.PrintToString(carbon_proto, &proto_string)) {
    return Error("Failed to convert to text proto");
  }
  return WriteFile(proto_string, output_file_name);
}

// Command line options for defining input/output format.
enum class ConversionMode { TextProtoToCarbon, CarbonToTextProto };

// Returns string representation of an enum option.
static auto GetEnumString(llvm::cl::opt<ConversionMode>& o) -> llvm::StringRef {
  // TODO: is there a better way?
  return o.getParser().getOption(static_cast<int>(ConversionMode(o)));
}

// Performs the conversion specified by `mode`.
auto Convert(const ConversionMode mode, std::string_view input_file_name,
             std::string_view output_file_name) -> ErrorOr<Success> {
  switch (mode) {
    case ConversionMode::TextProtoToCarbon:
      return TextProtoToCarbon(input_file_name, output_file_name);
    case ConversionMode::CarbonToTextProto:
      return CarbonToTextProto(input_file_name, output_file_name);
  }
}

}  // namespace Carbon

auto main(int argc, char* argv[]) -> int {
  llvm::InitLLVM init_llvm(argc, argv);

  llvm::cl::opt<Carbon::ConversionMode> mode(
      "mode", llvm::cl::desc("Conversion mode"),
      llvm::cl::values(
          clEnumValN(Carbon::ConversionMode::TextProtoToCarbon,
                     "proto_to_carbon", "Convert text proto to Carbon source"),
          clEnumValN(Carbon::ConversionMode::CarbonToTextProto,
                     "carbon_to_proto",
                     "Convert Carbon source to text proto")));
  llvm::cl::opt<std::string> input_file_name(
      "input", llvm::cl::desc("<input file>"), llvm::cl::init("/dev/stdin"));
  llvm::cl::opt<std::string> output_file_name(
      "output", llvm::cl::desc("<output file>"), llvm::cl::init("/dev/stdout"));
  llvm::cl::ParseCommandLineOptions(argc, argv);

  if (const auto result =
          Carbon::Convert(mode, input_file_name, output_file_name);
      !result.ok()) {
    llvm::errs() << GetEnumString(mode)
                 << " conversion failed: " << result.error().message() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
