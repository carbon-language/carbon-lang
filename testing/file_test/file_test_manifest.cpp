// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

namespace Carbon::Testing {

auto GetFileTestManifestPath() -> std::filesystem::path {
  return CARBON_FILE_TEST_MANIFEST;
}

}  // namespace Carbon::Testing
