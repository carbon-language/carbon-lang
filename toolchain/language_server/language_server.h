// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_LANGUAGE_SERVER_H_

#include "common/error.h"
#include "common/ostream.h"

namespace Carbon::LanguageServer {

// Start the language server. input_stream and output_stream are used by LSP;
// error_stream is primarily for errors that don't fit into LSP.
auto Run(FILE* input_stream, llvm::raw_ostream& output_stream,
         llvm::raw_ostream& error_stream) -> ErrorOr<Success>;

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
