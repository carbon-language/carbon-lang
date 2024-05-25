// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/source/source_buffer.h"

#include <limits>

#include "llvm/Support/ErrorOr.h"

namespace Carbon {

namespace {
struct FilenameConverter : DiagnosticConverter<llvm::StringRef> {
  auto ConvertLoc(llvm::StringRef filename, ContextFnT /*context_fn*/) const
      -> DiagnosticLoc override {
    return {.filename = filename};
  }
};
}  // namespace

auto SourceBuffer::MakeFromStdin(DiagnosticConsumer& consumer)
    -> std::optional<SourceBuffer> {
  return MakeFromMemoryBuffer(llvm::MemoryBuffer::getSTDIN(), "<stdin>",
                              /*is_regular_file=*/false, consumer);
}

auto SourceBuffer::MakeFromFile(llvm::vfs::FileSystem& fs,
                                llvm::StringRef filename,
                                DiagnosticConsumer& consumer)
    -> std::optional<SourceBuffer> {
  FilenameConverter converter;
  DiagnosticEmitter<llvm::StringRef> emitter(converter, consumer);

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

  // `stat` on a file without a known size gives a size of 0, which causes
  // `llvm::vfs::File::getBuffer` to produce an empty buffer. Use a size of -1
  // in this case so we get the complete file contents.
  bool is_regular_file = status->isRegularFile();
  int64_t size = is_regular_file ? status->getSize() : -1;

  return MakeFromMemoryBuffer(
      (*file)->getBuffer(filename, size, /*RequiresNullTerminator=*/false),
      filename, is_regular_file, consumer);
}

auto SourceBuffer::MakeFromMemoryBuffer(
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer,
    llvm::StringRef filename, bool is_regular_file,
    DiagnosticConsumer& consumer) -> std::optional<SourceBuffer> {
  FilenameConverter converter;
  DiagnosticEmitter<llvm::StringRef> emitter(converter, consumer);

  if (buffer.getError()) {
    CARBON_DIAGNOSTIC(ErrorReadingFile, Error, "Error reading file: {0}",
                      std::string);
    emitter.Emit(filename, ErrorReadingFile, buffer.getError().message());
    return std::nullopt;
  }

  if (buffer.get()->getBufferSize() >= std::numeric_limits<int32_t>::max()) {
    CARBON_DIAGNOSTIC(FileTooLarge, Error,
                      "File is over the 2GiB input limit; size is {0} bytes.",
                      int64_t);
    emitter.Emit(filename, FileTooLarge, buffer.get()->getBufferSize());
    return std::nullopt;
  }

  return SourceBuffer(filename.str(), std::move(buffer.get()), is_regular_file);
}

}  // namespace Carbon
