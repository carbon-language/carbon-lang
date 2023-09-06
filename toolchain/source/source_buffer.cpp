// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/source/source_buffer.h"

#include <limits>

#include "llvm/Support/ErrorOr.h"

namespace Carbon {

auto SourceBuffer::CreateFromFile(llvm::vfs::FileSystem& fs,
                                  llvm::raw_ostream& error_stream,
                                  llvm::StringRef filename)
    -> std::optional<SourceBuffer> {
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> file =
      fs.openFileForRead(filename);
  if (file.getError()) {
    error_stream << "Error opening `" << filename
                 << "`: " << file.getError().message();
    return std::nullopt;
  }

  llvm::ErrorOr<llvm::vfs::Status> status = (*file)->status();
  if (status.getError()) {
    error_stream << "Error getting status for `" << filename
                 << "`: " << file.getError().message();
    return std::nullopt;
  }
  auto size = status->getSize();
  if (size >= std::numeric_limits<int32_t>::max()) {
    error_stream << "Cannot load `" << filename
                 << "`: file is over the 2GiB input limit.";
    return std::nullopt;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      (*file)->getBuffer(filename, size, /*RequiresNullTerminator=*/false);
  if (buffer.getError()) {
    error_stream << "Error reading `" << filename
                 << "`: " << file.getError().message();
    return std::nullopt;
  }

  return SourceBuffer(filename.str(), std::move(buffer.get()));
}

}  // namespace Carbon
