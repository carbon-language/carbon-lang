// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SOURCE_SOURCEBUFFER_H_
#define TOOLCHAIN_SOURCE_SOURCEBUFFER_H_

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
      -> SourceBuffer;
  static auto CreateFromFile(llvm::StringRef filename)
      -> llvm::Expected<SourceBuffer>;

  // Use one of the factory functions above to create a source buffer.
  SourceBuffer() = delete;

  // Cannot copy as there may be non-trivial owned file data, see the class
  // comment for details.
  SourceBuffer(const SourceBuffer& arg) = delete;

  SourceBuffer(SourceBuffer&& arg) noexcept
      : filename_(std::move(arg.filename_)),
        text_(arg.text_),
        is_string_rep_(arg.is_string_rep_) {
    // The easy case in when we don't need to transfer an allocated string
    // representation.
    if (!arg.is_string_rep_) {
      // Take ownership of a non-string representation by clearing its text.
      arg.text_ = llvm::StringRef();
      return;
    }

    // If the argument is using a string rep we need to move that storage over
    // and recreate our text `StringRef` to point at our storage.
    new (&string_storage_) std::string(std::move(arg.string_storage_));
    text_ = string_storage_;
  }

  ~SourceBuffer();

  [[nodiscard]] auto Filename() const -> llvm::StringRef { return filename_; }

  [[nodiscard]] auto Text() const -> llvm::StringRef { return text_; }

 private:
  SourceBuffer(llvm::StringRef fake_filename, std::string buffer_text)
      : filename_(fake_filename.str()),
        is_string_rep_(true),
        string_storage_(std::move(buffer_text)) {
    text_ = string_storage_;
  }

  explicit SourceBuffer(llvm::StringRef filename)
      : filename_(filename.str()), text_(), is_string_rep_(false) {}

  std::string filename_;

  llvm::StringRef text_;

  bool is_string_rep_;

  // We use a transparent union to avoid constructing the storage.
  // FIXME: We should replace this and the boolean with an optional which would
  // be much simpler.
  union {
    std::string string_storage_;
  };
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SOURCE_SOURCEBUFFER_H_
