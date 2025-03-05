// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_MANIFEST_H_
#define CARBON_TESTING_FILE_TEST_MANIFEST_H_

#include <string>

#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {

// Returns the manifest path, which is provided by rules.bzl and manifest.cpp.
// This is exposed separately so that the explorer sharding approach can use a
// different implementation.
auto GetFileTestManifest() -> llvm::SmallVector<std::string>;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_MANIFEST_H_
