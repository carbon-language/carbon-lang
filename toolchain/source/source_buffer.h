// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SOURCE_SOURCE_BUFFER_H_
#define CARBON_TOOLCHAIN_SOURCE_SOURCE_BUFFER_H_

#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// A buffer of Carbon source code.
//
// This class holds a buffer of Carbon source code as text and makes it
// available for use in the rest of the Carbon compiler. It owns the memory for
// the underlying source code text and ensures it lives as long as the buffer
// objects.
//
// Every buffer of source code text is notionally loaded from a Carbon source
// file, even if provided directly when constructing the buffer. The name that
// should be used for that Carbon source file is also retained and made
// available.
//
// Because the underlying memory for the source code text may have been read
// from a file, and we may want to use facilities like `mmap` to simply map that
// file into memory, the buffer itself is not copyable to avoid needing to
// define copy semantics for a mapped file. We can relax this restriction with
// some implementation complexity in the future if needed.
class SourceBuffer {
 public:
  // Opens the requested file. Returns a SourceBuffer on success. Prints an
  // error and returns nullopt on failure.
  static auto CreateFromFile(llvm::vfs::FileSystem& fs,
                             llvm::StringRef filename,
                             DiagnosticConsumer& consumer)
      -> std::optional<SourceBuffer>;

  // Use one of the factory functions above to create a source buffer.
  SourceBuffer() = delete;

  [[nodiscard]] auto filename() const -> llvm::StringRef { return filename_; }

  [[nodiscard]] auto text() const -> llvm::StringRef {
    return text_->getBuffer();
  }

 private:
  explicit SourceBuffer(std::string filename,
                        std::unique_ptr<llvm::MemoryBuffer> text)
      : filename_(std::move(filename)), text_(std::move(text)) {}

  std::string filename_;
  std::unique_ptr<llvm::MemoryBuffer> text_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SOURCE_SOURCE_BUFFER_H_
