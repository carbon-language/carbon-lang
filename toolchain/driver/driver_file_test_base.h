// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_

#include <fstream>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
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
    // Prepare a list of filenames for MakeArgs. Also create the files
    // in-memory.
    llvm::SmallVector<llvm::StringRef> test_file_names;
    llvm::vfs::InMemoryFileSystem fs;
    for (const auto& test_file : test_files) {
      test_file_names.push_back(test_file.filename);

      if (!fs.addFile(test_file.filename, /*ModificationTime=*/0,
                      llvm::MemoryBuffer::getMemBuffer(test_file.content))) {
        ADD_FAILURE() << "File is repeated: " << test_file.filename;
        return false;
      }
    }

    Driver driver(fs, llvm::outs(), llvm::errs());
    return driver.RunFullCommand(MakeArgs(test_file_names));
  }

  virtual auto MakeArgs(const llvm::SmallVector<llvm::StringRef>& test_files)
      -> llvm::SmallVector<llvm::StringRef> = 0;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
