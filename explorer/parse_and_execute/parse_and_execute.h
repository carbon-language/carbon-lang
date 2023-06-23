// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_PARSE_AND_EXECUTE_PARSE_AND_EXECUTE_H_
#define CARBON_EXPLORER_PARSE_AND_EXECUTE_PARSE_AND_EXECUTE_H_

#include "common/error.h"
#include "explorer/common/trace_stream.h"

namespace Carbon {

// Parses and executes the input file, returning the program result on success.
// This API is intended for use by main execution.
auto ParseAndExecuteFile(std::string_view prelude_path,
                         std::string_view input_file_name, bool parser_debug,
                         Nonnull<TraceStream*> trace_stream,
                         Nonnull<llvm::raw_ostream*> print_stream)
    -> ErrorOr<int>;

// Parses and executes the source, returning the program result on success.
// Discards output.
auto ParseAndExecute(std::string_view prelude_path,
                     std::string_view input_file_name,
                     std::string_view file_contents, bool parser_debug,
                     Nonnull<TraceStream*> trace_stream,
                     Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_PARSE_AND_EXECUTE_PARSE_AND_EXECUTE_H_
