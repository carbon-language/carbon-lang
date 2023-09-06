// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/source/source_buffer.h"

#include <limits>

#include "llvm/Support/ErrorOr.h"

namespace Carbon {

namespace {
struct FilenameTranslator : DiagnosticLocationTranslator<llvm::StringRef> {
  auto GetLocation(llvm::StringRef filename) -> DiagnosticLocation override {
    return {.file_name = filename};
  }
};
}  // namespace

auto SourceBuffer::CreateFromFile(llvm::vfs::FileSystem& fs,
                                  llvm::StringRef filename,
                                  DiagnosticConsumer& consumer)
    -> std::optional<SourceBuffer> {
  FilenameTranslator translator;
  DiagnosticEmitter<llvm::StringRef> emitter(translator, consumer);

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> file =
      fs.openFileForRead(filename);
  if (file.getError()) {
    CARBON_DIAGNOSTIC(ErrorOpeningFile, Error,
                      "Error opening file for read: {0}", std::string);
    emitter.Emit(filename, ErrorOpeningFile, file.getError().message());
    return std::nullopt;
  }

  llvm::ErrorOr<llvm::vfs::Status> status = (*file)->status();
  if (status.getError()) {
    CARBON_DIAGNOSTIC(ErrorStattingFile, Error, "Error statting file: {0}",
                      std::string);
    emitter.Emit(filename, ErrorStattingFile, file.getError().message());
    return std::nullopt;
  }
  int64_t size = status->getSize();
  if (size >= std::numeric_limits<int32_t>::max()) {
    CARBON_DIAGNOSTIC(FileTooLarge, Error,
                      "File is over the 2GiB input limit; size is {0} bytes.",
                      int64_t);
    emitter.Emit(filename, FileTooLarge, size);
    return std::nullopt;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      (*file)->getBuffer(filename, size, /*RequiresNullTerminator=*/false);
  if (buffer.getError()) {
    CARBON_DIAGNOSTIC(ErrorReadingFile, Error, "Error reading file: {0}",
                      std::string);
    emitter.Emit(filename, ErrorReadingFile, file.getError().message());
    return std::nullopt;
  }

  return SourceBuffer(filename.str(), std::move(buffer.get()));
}

}  // namespace Carbon
