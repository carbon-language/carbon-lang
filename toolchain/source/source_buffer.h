// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SOURCE_SOURCE_BUFFER_H_
#define TOOLCHAIN_SOURCE_SOURCE_BUFFER_H_

#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

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
  static auto CreateFromText(llvm::Twine text,
                             llvm::StringRef filename = "/text")
      -> llvm::Expected<SourceBuffer>;
  static auto CreateFromFile(llvm::StringRef filename)
      -> llvm::Expected<SourceBuffer>;

  // Use one of the factory functions above to create a source buffer.
  SourceBuffer() = delete;

  // Cannot copy as there may be non-trivial owned file data; see the class
  // comment for details.
  SourceBuffer(const SourceBuffer& arg) = delete;

  SourceBuffer(SourceBuffer&& arg) noexcept;

  ~SourceBuffer();

  [[nodiscard]] auto filename() const -> llvm::StringRef { return filename_; }

  [[nodiscard]] auto text() const -> llvm::StringRef { return text_; }

 private:
  enum class ContentMode {
    Uninitialized,
    MMapped,
    Owned,
  };

  // Constructor for mmapped content.
  SourceBuffer(std::string filename, llvm::StringRef text);
  // Constructor for owned content.
  SourceBuffer(std::string filename, std::string text);

  ContentMode content_mode_;
  std::string filename_;
  std::string text_storage_;
  llvm::StringRef text_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SOURCE_SOURCE_BUFFER_H_
