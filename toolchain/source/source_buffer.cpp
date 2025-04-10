// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/source/source_buffer.h"

#include <limits>

#include "llvm/Support/ErrorOr.h"
#include "toolchain/diagnostics/file_diagnostics.h"

namespace Carbon {

auto SourceBuffer::MakeFromStdin(Diagnostics::Consumer& consumer)
    -> std::optional<SourceBuffer> {
  return MakeFromMemoryBuffer(llvm::MemoryBuffer::getSTDIN(), "<stdin>",
                              /*is_regular_file=*/false, consumer);
}

auto SourceBuffer::MakeFromFile(llvm::vfs::FileSystem& fs,
                                llvm::StringRef filename,
                                Diagnostics::Consumer& consumer)
    -> std::optional<SourceBuffer> {
  Diagnostics::FileEmitter emitter(&consumer);

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> file =
      fs.openFileForRead(filename);
  if (file.getError()) {
    CARBON_DIAGNOSTIC(ErrorOpeningFile, Error,
                      "error opening file for read: {0}", std::string);
    emitter.Emit(filename, ErrorOpeningFile, file.getError().message());
    return std::nullopt;
  }

  llvm::ErrorOr<llvm::vfs::Status> status = (*file)->status();
  if (status.getError()) {
    CARBON_DIAGNOSTIC(ErrorStattingFile, Error, "error statting file: {0}",
                      std::string);
    emitter.Emit(filename, ErrorStattingFile, file.getError().message());
    return std::nullopt;
  }

  // `stat` on a file without a known size gives a size of 0, which causes
  // `llvm::vfs::File::getBuffer` to produce an empty buffer. Use a size of -1
  // in this case so we get the complete file contents.
  bool is_regular_file = status->isRegularFile();
  int64_t size = is_regular_file ? status->getSize() : -1;

  return MakeFromMemoryBuffer(
      (*file)->getBuffer(filename, size, /*RequiresNullTerminator=*/false),
      filename, is_regular_file, consumer);
}

auto SourceBuffer::MakeFromStringCopy(llvm::StringRef filename,
                                      llvm::StringRef text,
                                      Diagnostics::Consumer& consumer)
    -> std::optional<SourceBuffer> {
  return MakeFromMemoryBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(text, filename), filename,
      /*is_regular_file=*/true, consumer);
}

auto SourceBuffer::MakeFromMemoryBuffer(
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer,
    llvm::StringRef filename, bool is_regular_file,
    Diagnostics::Consumer& consumer) -> std::optional<SourceBuffer> {
  Diagnostics::FileEmitter emitter(&consumer);

  if (buffer.getError()) {
    CARBON_DIAGNOSTIC(ErrorReadingFile, Error, "error reading file: {0}",
                      std::string);
    emitter.Emit(filename, ErrorReadingFile, buffer.getError().message());
    return std::nullopt;
  }

  if (buffer.get()->getBufferSize() >= std::numeric_limits<int32_t>::max()) {
    CARBON_DIAGNOSTIC(FileTooLarge, Error,
                      "file is over the 2GiB input limit; size is {0} bytes",
                      int64_t);
    emitter.Emit(filename, FileTooLarge, buffer.get()->getBufferSize());
    return std::nullopt;
  }

  return SourceBuffer(filename.str(), std::move(buffer.get()), is_regular_file);
}

}  // namespace Carbon
