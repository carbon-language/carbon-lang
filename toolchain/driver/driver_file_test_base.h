// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_

#include <cstdio>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "testing/file_test/file_test_base.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {

// Provides common test support for the driver. This is used by file tests in
// phase subdirectories.
class DriverFileTestBase : public FileTestBase {
 public:
  using FileTestBase::FileTestBase;

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           llvm::vfs::InMemoryFileSystem& fs, llvm::raw_pwrite_stream& stdout,
           llvm::raw_pwrite_stream& stderr) -> ErrorOr<bool> override {
    Driver driver(fs, stdout, stderr);
    return driver.RunCommand(test_args);
  }
};

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
