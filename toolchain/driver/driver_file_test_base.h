// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_

#include <filesystem>
#include <fstream>

#include "llvm/ADT/ScopeExit.h"
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

  auto RunWithFiles(const llvm::SmallVector<TestFile>& test_files,
                    llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool override {
    // TODO: The working dir and file creation logic should be replaced with vfs
    // use. That will require modifying Driver and SourceBuffer to use vfs.
    // Get the working dir.
    std::error_code ec;
    auto orig_dir = std::filesystem::current_path(ec);
    CARBON_CHECK(!ec) << ec.message();

    // Set the working dir.
    const char* tmpdir = getenv("TEST_TMPDIR");
    CARBON_CHECK(tmpdir);
    std::filesystem::current_path(tmpdir, ec);
    CARBON_CHECK(!ec) << ec.message();

    // Prepare a list of filenames for MakeArgs. Also create the files.
    llvm::SmallVector<llvm::StringRef> test_file_names;
    for (const auto& test_file : test_files) {
      test_file_names.push_back(test_file.filename);

      std::ofstream f(test_file.filename);
      f << test_file.content;
    }

    auto cleanup = llvm::make_scope_exit([&]() {
      // Remove the files.
      for (const auto& test_file : test_files) {
        std::filesystem::remove(test_file.filename, ec);
        CARBON_CHECK(!ec) << ec.message();
      }
      // Restore the working dir.
      std::filesystem::current_path(orig_dir, ec);
      CARBON_CHECK(!ec) << ec.message();
    });

    Driver driver(stdout, stderr);
    return driver.RunFullCommand(MakeArgs(test_file_names));
  }

  virtual auto MakeArgs(const llvm::SmallVector<llvm::StringRef>& test_files)
      -> llvm::SmallVector<llvm::StringRef> = 0;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
