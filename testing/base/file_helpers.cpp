// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/file_helpers.h"

#include <fstream>
#include <sstream>

namespace Carbon::Testing {

auto ReadFile(std::filesystem::path path) -> ErrorOr<std::string> {
  std::ifstream file_stream(path);
  if (file_stream.fail()) {
    return Error(llvm::formatv("Error opening file: {0}", path));
  }
  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  if (file_stream.fail()) {
    return Error(llvm::formatv("Error reading file: {0}", path));
  }
  return buffer.str();
}

}  // namespace Carbon::Testing
