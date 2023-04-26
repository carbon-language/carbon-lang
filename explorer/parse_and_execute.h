// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_PARSE_AND_EXECUTE_H_
#define CARBON_EXPLORER_PARSE_AND_EXECUTE_H_

#include "common/error.h"
#include "explorer/interpreter/trace_stream.h"

namespace Carbon {

// Parses and executes source code.
// Returns program result if execution was successful.
auto ParseAndExecute(const std::string& prelude_path,
                     const std::string& input_file_name, bool parser_debug,
                     TraceStream* trace_stream) -> ErrorOr<int>;

}  // namespace Carbon

namespace Carbon::Testing {

// Parses and executes source code.
// Returns program result if execution was successful.
auto ParseAndExecute(const std::string& prelude_path, const std::string& source)
    -> ErrorOr<int>;

}  // namespace Carbon::Testing

#endif  // CARBON_EXPLORER_PARSE_AND_EXECUTE_H_
