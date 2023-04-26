// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// To convert a Carbon file to a text proto:
// `ast_to_proto <file.carbon>`

#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "common/bazel_working_dir.h"
#include "common/error.h"
#include "common/fuzzing/carbon.pb.h"
#include "explorer/ast/ast.h"
#include "explorer/common/arena.h"
#include "explorer/fuzzing/ast_to_proto.h"
#include "explorer/syntax/parse.h"

namespace Carbon::Testing {

auto Main(int argc, char** argv) -> ErrorOr<Success> {
  SetWorkingDirForBazel();

  if (argc != 2) {
    return Error("Syntax: ast_to_proto <file.carbon>");
  }
  if (!std::filesystem::is_regular_file(argv[1])) {
    return Error("Argument must be a file.");
  }

  // Read the input file.
  std::ifstream proto_file(argv[1]);
  std::stringstream buffer;
  buffer << proto_file.rdbuf();
  proto_file.close();

  Arena arena;
  const ErrorOr<AST> ast = Parse(&arena, argv[1],
                                 /*parser_debug=*/false);
  if (!ast.ok()) {
    return ErrorBuilder() << "Parsing failed: " << ast.error().message();
  }
  Fuzzing::Carbon carbon_proto = AstToProto(*ast);

  std::string proto_string;
  google::protobuf::TextFormat::Printer p;
  if (!p.PrintToString(carbon_proto, &proto_string)) {
    return Error("Failed to convert to text proto");
  }
  std::cout << proto_string;
  return Success();
}

}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  auto err = Carbon::Testing::Main(argc, argv);
  if (!err.ok()) {
    std::cerr << err.error().message() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
