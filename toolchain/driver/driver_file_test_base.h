// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "testing/file_test/file_test_base.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {

// Provides common test support for the driver. This is used by file tests in
// phase subdirectories.
class DriverFileTestBase : public FileTestBase {
 public:
  using FileTestBase::FileTestBase;

  auto RunWithFiles(const llvm::SmallVector<std::string>& test_files,
                    llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool override {
    Driver driver(stdout, stderr);
    return driver.RunFullCommand(MakeArgs(test_files));
  }

  virtual auto MakeArgs(const llvm::SmallVector<std::string>& test_files)
      -> llvm::SmallVector<llvm::StringRef> = 0;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
