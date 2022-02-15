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
#include <system_error>

#include "common/check.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Locale.h"

namespace Carbon {

// Requires that all characters in filenames be printable.
static auto IsPrintable(llvm::StringRef filename) -> bool {
  llvm::UTF32 c;
  unsigned const char* end = filename.bytes_end();
  for (unsigned const char* begin = filename.bytes_begin(); begin < end;) {
    unsigned const char* cp_end = begin + llvm::getNumBytesForUTF8(*begin);
    if (cp_end > end) {
      return false;
    }
    llvm::UTF32* cptr = &c;
    llvm::ConversionResult res = llvm::ConvertUTF8toUTF32(
        &begin, cp_end, &cptr, cptr + 1, llvm::strictConversion);
    if (res != llvm::conversionOK || !llvm::sys::locale::isPrint(c)) {
      return false;
    }
  }
  return true;
}

static auto ErrnoToError(int errno_value) -> llvm::Error {
  return llvm::errorCodeToError(
      std::error_code(errno_value, std::generic_category()));
}

auto SourceBuffer::CreateFromTextWithFilename(llvm::Twine text,
                                              llvm::StringRef filename)
    -> llvm::Expected<SourceBuffer> {
  if (!IsPrintable(filename)) {
    return ErrnoToError(EINVAL);
  }
  return SourceBuffer(filename, text.str());
}

auto SourceBuffer::CreateFromFile(llvm::StringRef filename)
    -> llvm::Expected<SourceBuffer> {
  if (!IsPrintable(filename)) {
    return ErrnoToError(EINVAL);
  }

  SourceBuffer buffer(filename);

  errno = 0;
  int file_descriptor = open(buffer.filename_.c_str(), O_RDONLY);
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
    // Nothing to do for an empty file.
    return {std::move(buffer)};
  }

  errno = 0;
  void* mapped_text = mmap(nullptr, size, PROT_READ,
#ifdef __APPLE__
                           MAP_PRIVATE,
#else
                           MAP_PRIVATE | MAP_POPULATE,
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

  buffer.text_ = llvm::StringRef(static_cast<const char*>(mapped_text), size);
  CHECK(!buffer.text_.empty())
      << "Must not have an empty text when we have mapped data from a file!";
  return {std::move(buffer)};
}

SourceBuffer::~SourceBuffer() {
  if (is_string_rep_) {
    string_storage_.~decltype(string_storage_)();
    return;
  }

  if (!text_.empty()) {
    errno = 0;
    int result =
        munmap(const_cast<void*>(static_cast<const void*>(text_.data())),
               text_.size());
    (void)result;
    CHECK(result != -1) << "Unmapping text failed!";
  }
}

}  // namespace Carbon
