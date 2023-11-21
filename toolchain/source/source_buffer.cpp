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

// Checks that a memory buffer is suitable for use as a source buffer. Returns
// true if so; diagnoses and returns false if not.
static auto CheckBuffer(
    const llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>& buffer,
    llvm::StringRef filename, DiagnosticEmitter<llvm::StringRef>& emitter)
    -> bool {
  if (buffer.getError()) {
    CARBON_DIAGNOSTIC(ErrorReadingFile, Error, "Error reading file: {0}",
                      std::string);
    emitter.Emit(filename, ErrorReadingFile, buffer.getError().message());
    return false;
  }

  if (buffer.get()->getBufferSize() >= std::numeric_limits<int32_t>::max()) {
    CARBON_DIAGNOSTIC(FileTooLarge, Error,
                      "File is over the 2GiB input limit; size is {0} bytes.",
                      int64_t);
    emitter.Emit(filename, FileTooLarge, buffer.get()->getBufferSize());
    return false;
  }

  return true;
}

auto SourceBuffer::CreateFromStdin(DiagnosticConsumer& consumer)
    -> std::optional<SourceBuffer> {
  FilenameTranslator translator;
  DiagnosticEmitter<llvm::StringRef> emitter(translator, consumer);

  static const llvm::StringLiteral StdinName = "<stdin>";

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getSTDIN();

  if (!CheckBuffer(buffer, StdinName, emitter)) {
    return std::nullopt;
  }

  return SourceBuffer(StdinName.str(), std::move(buffer.get()),
                      /*is_regular_file=*/false);
}

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

  // `stat` on a file without a known size gives a size of 0, which causes
  // `llvm::vfs::File::getBuffer` to produce an empty buffer. Use a size of -1
  // in this case so we get the complete file contents.
  bool is_regular_file = status->isRegularFile();
  int64_t size = is_regular_file ? status->getSize() : -1;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      (*file)->getBuffer(filename, size, /*RequiresNullTerminator=*/false);

  if (!CheckBuffer(buffer, "<stdin>", emitter)) {
    return std::nullopt;
  }

  return SourceBuffer(filename.str(), std::move(buffer.get()), is_regular_file);
}

}  // namespace Carbon
