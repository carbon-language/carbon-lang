// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_FILE_SYSTEM_H_
#define CARBON_TESTING_FILE_TEST_FILE_SYSTEM_H_

#include "common/error.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace Carbon::Testing {

// Adds a file to the fs.
auto AddFile(llvm::vfs::InMemoryFileSystem& fs, llvm::StringRef path)
    -> ErrorOr<Success>;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_FILE_SYSTEM_H_
