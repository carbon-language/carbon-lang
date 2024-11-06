// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_LANGUAGE_SERVER_H_

#include "common/error.h"
#include "common/ostream.h"

namespace Carbon::LanguageServer {

// Start the language server.
auto Run(std::FILE* input_stream, llvm::raw_ostream& output_stream)
    -> ErrorOr<Success>;

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
