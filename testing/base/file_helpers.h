// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_FILE_HELPERS_H_
#define CARBON_TESTING_BASE_FILE_HELPERS_H_

#include <filesystem>
#include <string>

#include "common/error.h"

namespace Carbon::Testing {

// Reads a file to string.
auto ReadFile(std::filesystem::path path) -> ErrorOr<std::string>;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_FILE_HELPERS_H_
