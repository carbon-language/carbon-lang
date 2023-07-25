// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// To convert a crashing input in text proto to Carbon source:
// `proto_to_carbon <file.textproto>`

#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "common/bazel_working_dir.h"
#include "common/error.h"
#include "testing/fuzzing/proto_to_carbon.h"

namespace Carbon {

auto Main(int argc, char** argv) -> ErrorOr<Success> {
  Carbon::SetWorkingDirForBazel();

  if (argc != 2) {
    return Error("Syntax: proto_to_carbon <file.textproto>");
  }
  if (!std::filesystem::is_regular_file(argv[1])) {
    return Error("Argument must be a file.");
  }

  // Read the input file.
  std::ifstream proto_file(argv[1]);
  std::stringstream buffer;
  buffer << proto_file.rdbuf();
  proto_file.close();

  CARBON_ASSIGN_OR_RETURN(Fuzzing::Carbon proto,
                          Carbon::ParseCarbonTextProto(buffer.str()));
  std::cout << Carbon::ProtoToCarbon(proto, /*maybe_add_main=*/true);
  return Success();
}

}  // namespace Carbon

auto main(int argc, char** argv) -> int {
  auto err = Carbon::Main(argc, argv);
  if (!err.ok()) {
    std::cerr << err.error().message() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
