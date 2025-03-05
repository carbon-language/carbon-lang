// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/manifest.h"

// The test manifest, produced by `manifest_as_cpp`.
// NOLINTNEXTLINE(readability-identifier-naming): Constant in practice.
extern const char* CarbonFileTestManifest[];

namespace Carbon::Testing {

auto GetFileTestManifest() -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> manifest;
  for (int i = 0; CarbonFileTestManifest[i]; ++i) {
    manifest.push_back(CarbonFileTestManifest[i]);
  }
  return manifest;
}

}  // namespace Carbon::Testing
