// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLE_SEM_IR_TEXT_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLE_SEM_IR_TEXT_H_

#include <optional>
#include <vector>

#include "clang-tools-extra/clangd/Protocol.h"
#include "toolchain/language_server/context.h"

namespace Carbon::LanguageServer {

// Answers `textDocument/hover` from the formatted SemIR in a test file's
// expected output. Returns `nullopt` if `position` isn't within a SemIR name,
// leaving the request to be answered from the file's Carbon source.
auto GetSemIRTextHover(const Context::File& file,
                       const clang::clangd::Position& position)
    -> std::optional<clang::clangd::Hover>;

// What a goto-style request should find for a name in formatted SemIR.
enum class SemIRTextGoto : int8_t {
  // The lines that define the name, for `textDocument/definition`.
  Definition,

  // The source text that the name's `.loc<line>_<column>` suffix points at,
  // for `textDocument/declaration`. A SemIR name is derived from the source it
  // was checked from, so this is where the name came from, in the input file
  // the enclosing block of output was compiled from.
  Source,

  // The rows of `specific` blocks that give the name its value, for
  // `textDocument/implementation`. A name defined in a generic is symbolic,
  // and each specific of that generic is one way of making it concrete, which
  // is as close to an implementation as SemIR has.
  Specifics,

  // Everywhere the name is written in the same block of output, for
  // `textDocument/references`.
  References,
};

// Answers a goto-style request from the formatted SemIR in a test file's
// expected output. Returns `nullopt` if `position` isn't within a SemIR name,
// leaving the request to be answered from the file's Carbon source.
auto GetSemIRTextLocations(const Context::File& file,
                           const clang::clangd::Position& position,
                           SemIRTextGoto goto_kind)
    -> std::optional<std::vector<clang::clangd::Location>>;

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLE_SEM_IR_TEXT_H_
