// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// To convert a crashing input in text proto to Carbon source:
// `proto_to_carbon file.textproto`

#include <google/protobuf/text_format.h>

#include <fstream>
#include <sstream>

#include "common/bazel_working_dir.h"

auto main(int argc, char** argv) -> int {
  Carbon::SetWorkingDirForBazel();
  if (argc != 2) {
    llvm::report_fatal_error("Syntax: proto_to_carbon <textproto>");
  }

  std::ifstream proto_file(argv[1]);
  std::stringstream buffer;
  buffer << proto_file.rdbuf();
  proto_file.close();
}
/*
CARBON_ASSIGN_OR_RETURN(const std::string input_contents,
                        ReadFile(input_file_name));
CARBON_ASSIGN_OR_RETURN(const Fuzzing::Carbon carbon_proto,
                        ParseCarbonTextProto(input_contents));
const std::string carbon_source =
    ProtoToCarbonWithMain(carbon_proto.compilation_unit());
return WriteFile(carbon_source, output_file_name);
*/
