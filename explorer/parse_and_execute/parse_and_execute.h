// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_PARSE_AND_EXECUTE_PARSE_AND_EXECUTE_H_
#define CARBON_EXPLORER_PARSE_AND_EXECUTE_PARSE_AND_EXECUTE_H_

#include "common/error.h"
#include "explorer/base/trace_stream.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace Carbon {

// Parses and executes the input file, returning the program result on success.
auto ParseAndExecute(llvm::vfs::FileSystem& fs, std::string_view prelude_path,
                     std::string_view input_file_name, bool parser_debug,
                     Nonnull<TraceStream*> trace_stream,
                     Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_PARSE_AND_EXECUTE_PARSE_AND_EXECUTE_H_
