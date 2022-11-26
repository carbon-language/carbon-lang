// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/source/source_buffer.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>

#include "common/check.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"

namespace Carbon {

// Verifies that the content size is within limits.
static auto CheckContentSize(int64_t size) -> llvm::Error {
  if (size < std::numeric_limits<int32_t>::max()) {
    return llvm::Error::success();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Input too large!");
}

auto SourceBuffer::CreateFromText(llvm::Twine text, llvm::StringRef filename)
    -> llvm::Expected<SourceBuffer> {
  std::string buffer = text.str();
  auto size_check = CheckContentSize(buffer.size());
  if (size_check) {
    return std::move(size_check);
  }
  return SourceBuffer(filename.str(), std::move(buffer));
}

static auto ErrnoToError(int errno_value) -> llvm::Error {
  return llvm::errorCodeToError(
      std::error_code(errno_value, std::generic_category()));
}

auto SourceBuffer::CreateFromFile(llvm::StringRef filename)
    -> llvm::Expected<SourceBuffer> {
  // Add storage to ensure there's a nul-terminator for open().
  std::string filename_str = filename.str();

  errno = 0;
  int file_descriptor = open(filename_str.c_str(), O_RDONLY);
  if (file_descriptor == -1) {
    return ErrnoToError(errno);
  }

  // Now that we have an open file, we need to close it on any error.
  auto closer =
      llvm::make_scope_exit([file_descriptor] { close(file_descriptor); });

  struct stat stat_buffer = {};
  errno = 0;
  if (fstat(file_descriptor, &stat_buffer) == -1) {
    return ErrnoToError(errno);
  }

  int64_t size = stat_buffer.st_size;
  if (size == 0) {
    // Rather than opening an empty file, create an empty buffer.
    return SourceBuffer(std::move(filename_str), std::string());
  }
  auto size_check = CheckContentSize(size);
  if (size_check) {
    return std::move(size_check);
  }

  errno = 0;
  void* mapped_text = mmap(nullptr, size, PROT_READ,
#if defined(__linux__)
                           MAP_PRIVATE | MAP_POPULATE,
#else
                           MAP_PRIVATE,
#endif
                           file_descriptor, /*offset=*/0);
  // The `MAP_FAILED` macro may expand to a cast to pointer that `clang-tidy`
  // complains about.
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  if (mapped_text == MAP_FAILED) {
    return ErrnoToError(errno);
  }

  errno = 0;
  closer.release();
  if (close(file_descriptor) == -1) {
    // Try to unmap the text. No errer handling as this is just best-effort
    // cleanup.
    munmap(mapped_text, size);
    return ErrnoToError(errno);
  }

  return SourceBuffer(
      std::move(filename_str),
      llvm::StringRef(static_cast<const char*>(mapped_text), size));
}

SourceBuffer::SourceBuffer(SourceBuffer&& arg) noexcept
    // Sets Uninitialized to ensure the input doesn't release mmapped data.
    : content_mode_(
          std::exchange(arg.content_mode_, ContentMode::Uninitialized)),
      filename_(std::move(arg.filename_)),
      text_storage_(std::move(arg.text_storage_)),
      text_(content_mode_ == ContentMode::Owned ? text_storage_ : arg.text_) {}

SourceBuffer::SourceBuffer(std::string filename, std::string text)
    : content_mode_(ContentMode::Owned),
      filename_(std::move(filename)),
      text_storage_(std::move(text)),
      text_(text_storage_) {}

SourceBuffer::SourceBuffer(std::string filename, llvm::StringRef text)
    : content_mode_(ContentMode::MMapped),
      filename_(std::move(filename)),
      text_(text) {
  CARBON_CHECK(!text.empty())
      << "Must not have an empty text when we have mapped data from a file!";
}

SourceBuffer::~SourceBuffer() {
  if (content_mode_ == ContentMode::MMapped) {
    errno = 0;
    int result =
        munmap(const_cast<void*>(static_cast<const void*>(text_.data())),
               text_.size());
    CARBON_CHECK(result != -1) << "Unmapping text failed!";
  }
}

}  // namespace Carbon
