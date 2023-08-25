// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/source/source_buffer.h"

#include <limits>

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

auto SourceBuffer::CreateFromFile(llvm::vfs::FileSystem& fs,
                                  llvm::StringRef filename)
    -> ErrorOr<SourceBuffer> {
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> file =
      fs.openFileForRead(filename);
  if (file.getError()) {
    return Error(file.getError().message());
  }

  llvm::ErrorOr<llvm::vfs::Status> status = (*file)->status();
  if (status.getError()) {
    return Error(status.getError().message());
  }
  auto size = status->getSize();
  if (size >= std::numeric_limits<int32_t>::max()) {
    return Error(
        llvm::formatv("`{0}` is over the 2GiB input limit.", filename));
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      (*file)->getBuffer(filename, size, /*RequiresNullTerminator=*/false);
  if (buffer.getError()) {
    return Error(buffer.getError().message());
  }

  return SourceBuffer(filename.str(), std::move(buffer.get()));
}

}  // namespace Carbon
