// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An utility for converting between fuzzer protos and Carbon sources.
//
// For example, to convert a crashing input in binary proto to carbon source:
// `fuzzverter --from=binary_proto --input file.binaryproto --to=carbon_source`
//
// To generate a new binary proto from carbon source for seeding the corpus:
// `fuzzverter --from=carbon_source --input file.carbon --to=binary_proto`
//
// To convert a binary proto to text proto:
// `fuzzverter --from=binary_proto --input file.binaryproto --to=text_proto`

#include <google/protobuf/text_format.h>

#include <cstdlib>
#include <fstream>
#include <ios>
#include <sstream>

#include "common/error.h"
#include "common/fuzzing/carbon.pb.h"
#include "common/fuzzing/proto_to_carbon.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/fuzzing/ast_to_proto.h"
#include "executable_semantics/fuzzing/fuzzer_util.h"
#include "executable_semantics/syntax/parse.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace Carbon {

// Reads a file and returns its contents as a string.
static auto ReadFile(std::string_view file_name, bool binary = false)
    -> ErrorOr<std::string> {
  auto mode = std::ios::in;
  if (binary) {
    mode |= std::ios::binary;
  }
  std::ifstream file(file_name, mode);
  if (!file.is_open()) {
    return ErrorBuilder() << "Could not open " << file_name << " for reading";
  }
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

// Writes string `s` to `file_name`.
static auto WriteFile(std::string_view s, std::string_view file_name,
                      bool binary = false) -> ErrorOr<Success> {
  auto mode = std::ios::out;
  if (binary) {
    mode |= std::ios::binary;
  }
  std::ofstream file(file_name, mode);
  if (!file.is_open()) {
    return ErrorBuilder() << "Could not open " << file_name << " for writing";
  }
  file << s;
  return Success();
}

// Reads text Carbon proto from a file.
static auto ReadTextProto(std::string_view input_file_name)
    -> ErrorOr<Fuzzing::Carbon> {
  ASSIGN_OR_RETURN(const std::string input_contents, ReadFile(input_file_name));
  Fuzzing::Carbon carbon_proto;
  if (!google::protobuf::TextFormat::ParseFromString(input_contents,
                                                     &carbon_proto)) {
    return Error("Could not parse text proto");
  }
  return carbon_proto;
}

// Writes text representation of `carbon_proto` to a file.
static auto WriteTextProto(const Fuzzing::Carbon& carbon_proto,
                           std::string_view output_file_name)
    -> ErrorOr<Success> {
  std::string proto_string;
  google::protobuf::TextFormat::Printer p;
  if (!p.PrintToString(carbon_proto, &proto_string)) {
    return Error("Failed to convert to text proto");
  }
  return WriteFile(proto_string, output_file_name);
}

// Reads binary Carbon proto from a file.
static auto ReadBinaryProto(std::string_view input_file_name)
    -> ErrorOr<Fuzzing::Carbon> {
  ASSIGN_OR_RETURN(const std::string input_contents,
                   ReadFile(input_file_name, /*binary=*/true));
  Fuzzing::Carbon carbon_proto;
  if (!carbon_proto.ParseFromString(input_contents)) {
    return Error("Could not parse binary proto");
  }
  return carbon_proto;
}

// Writes a binary representation of `carbon_proto` to a file.
auto WriteBinaryProto(const Fuzzing::Carbon& carbon_proto,
                      std::string_view output_file_name) -> ErrorOr<Success> {
  std::string proto_string;
  if (!carbon_proto.SerializeToString(&proto_string)) {
    return Error("Failed to convert to binary proto");
  }
  return WriteFile(proto_string, output_file_name, /*binary=*/true);
}

// Reads Carbon source from a file, and converts to Carbon proto.
static auto ReadCarbonAsProto(std::string_view input_file_name)
    -> ErrorOr<Fuzzing::Carbon> {
  Carbon::Arena arena;
  const ErrorOr<AST> ast = Carbon::Parse(&arena, input_file_name,
                                         /*trace=*/false);
  if (!ast.ok()) {
    return ErrorBuilder() << "Parsing failed: " << ast.error().message();
  }
  Fuzzing::Carbon carbon_proto;
  *carbon_proto.mutable_compilation_unit() = AstToProto(*ast);
  return carbon_proto;
}

// Converts Carbon proto to Carbon source, and writes to a file.
static auto WriteProtoAsCarbon(Fuzzing::Carbon& carbon_proto,
                               std::string_view output_file_name)
    -> ErrorOr<Success> {
  MaybeAddMain(*carbon_proto.mutable_compilation_unit());
  const std::string carbon_source =
      ProtoToCarbon(carbon_proto.compilation_unit());
  return WriteFile(carbon_source, output_file_name);
}

// Unsupported conversion.
static auto Unsupported(std::string_view input_file_name,
                        std::string_view output_file_name) -> ErrorOr<Success> {
  return Error("Unsupported");
}

// Conversion routines from text_proto.

static auto TextProtoToBinaryProto(std::string_view input_file_name,
                                   std::string_view output_file_name)
    -> ErrorOr<Success> {
  ASSIGN_OR_RETURN(const Fuzzing::Carbon carbon_proto,
                   ReadTextProto(input_file_name));
  return WriteBinaryProto(carbon_proto, output_file_name);
}

static auto TextProtoToCarbon(std::string_view input_file_name,
                              std::string_view output_file_name)
    -> ErrorOr<Success> {
  ASSIGN_OR_RETURN(Fuzzing::Carbon carbon_proto,
                   ReadTextProto(input_file_name));
  return WriteProtoAsCarbon(carbon_proto, output_file_name);
}

// Conversion routines from binary_proto.

static auto BinaryProtoToTextProto(std::string_view input_file_name,
                                   std::string_view output_file_name)
    -> ErrorOr<Success> {
  ASSIGN_OR_RETURN(Fuzzing::Carbon carbon_proto,
                   ReadBinaryProto(input_file_name));
  return WriteTextProto(carbon_proto, output_file_name);
}

static auto BinaryProtoToCarbon(std::string_view input_file_name,
                                std::string_view output_file_name)
    -> ErrorOr<Success> {
  ASSIGN_OR_RETURN(Fuzzing::Carbon carbon_proto,
                   ReadBinaryProto(input_file_name));
  return WriteProtoAsCarbon(carbon_proto, output_file_name);
}

// Conversion routines from carbon_source.

static auto CarbonToTextProto(std::string_view input_file_name,
                              std::string_view output_file_name)
    -> ErrorOr<Success> {
  ASSIGN_OR_RETURN(const Fuzzing::Carbon carbon_proto,
                   ReadCarbonAsProto(input_file_name));
  return WriteTextProto(carbon_proto, output_file_name);
}

static auto CarbonToBinaryProto(std::string_view input_file_name,
                                std::string_view output_file_name)
    -> ErrorOr<Success> {
  ASSIGN_OR_RETURN(const Fuzzing::Carbon carbon_proto,
                   ReadCarbonAsProto(input_file_name));
  return WriteBinaryProto(carbon_proto, output_file_name);
}

}  // namespace Carbon

// Command line options for defining input/output format.
enum FileFormat { text_proto = 0, binary_proto, carbon_source };

// Returns string representation of an enum option.
template <typename T>
static auto GetEnumString(llvm::cl::opt<T>& o) -> llvm::StringRef {
  // TODO: is there a better way?
  return o.getParser().getOption(o);
}

auto main(int argc, char* argv[]) -> int {
  llvm::InitLLVM init_llvm(argc, argv);

  const auto file_format_values =
      llvm::cl::values(clEnumVal(text_proto, "Text protocol buffer"),
                       clEnumVal(binary_proto, "Binary protocol buffer"),
                       clEnumVal(carbon_source, "Carbon source string"));

  llvm::cl::opt<FileFormat> input_format(
      "from", llvm::cl::desc("Input file format"), file_format_values);
  llvm::cl::opt<FileFormat> output_format(
      "to", llvm::cl::desc("Output file format"), file_format_values);
  llvm::cl::opt<std::string> input_file_name(
      "input", llvm::cl::desc("<input file>"), llvm::cl::init("/dev/stdin"));
  llvm::cl::opt<std::string> output_file_name(
      "output", llvm::cl::desc("<output file>"), llvm::cl::init("/dev/stdout"));
  llvm::cl::ParseCommandLineOptions(argc, argv);

  using ConverterFunc = std::function<Carbon::ErrorOr<Carbon::Success>(
      std::string_view input_file_name, std::string_view output_file_name)>;
  ConverterFunc converters[][3] = {
      // From text_proto.
      {
          Carbon::Unsupported,
          Carbon::TextProtoToBinaryProto,
          Carbon::TextProtoToCarbon,
      },

      // From binary_proto.
      {
          Carbon::BinaryProtoToTextProto,
          Carbon::Unsupported,
          Carbon::BinaryProtoToCarbon,
      },

      // From carbon_source
      {
          Carbon::CarbonToTextProto,
          Carbon::CarbonToBinaryProto,
          Carbon::Unsupported,
      },
  };

  const auto result = converters[input_format][output_format](input_file_name,
                                                              output_file_name);
  if (!result.ok()) {
    llvm::errs() << "Conversion from " << GetEnumString(input_format) << " to "
                 << GetEnumString(output_format)
                 << " failed: " << result.error().message() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
