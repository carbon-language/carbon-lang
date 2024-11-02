// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/language_server.h"

#include "toolchain/language_server/server.h"

namespace Carbon::LanguageServer {

auto Run(std::FILE* input_stream, llvm::raw_ostream& output_stream)
    -> ErrorOr<Success> {
  Server server(input_stream, output_stream);
  return server.Run();
}

}  // namespace Carbon::LanguageServer
