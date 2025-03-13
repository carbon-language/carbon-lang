// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_system.h"

namespace Carbon::Testing {

auto AddFile(llvm::vfs::InMemoryFileSystem& fs, llvm::StringRef path)
    -> ErrorOr<Success> {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(path);
  if (file.getError()) {
    return ErrorBuilder() << "Getting `" << path
                          << "`: " << file.getError().message();
  }
  if (!fs.addFile(path, /*ModificationTime=*/0, std::move(*file))) {
    return ErrorBuilder() << "Duplicate file: `" << path << "`";
  }
  return Success();
}

}  // namespace Carbon::Testing
