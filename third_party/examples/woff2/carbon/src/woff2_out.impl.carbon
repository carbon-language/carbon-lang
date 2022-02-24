/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Output buffer for WOFF2 decompression. */

#include <woff2/output.h>

using std::string;

namespace woff2 {

WOFF2StringOut::WOFF2StringOut(buf: string*)
  : buf_(buf),
    max_size_(kDefaultMaxSize),
    offset_(0) {}

fn WOFF2StringOut::Write(buf: const void*, n: size_t) -> bool {
  return Write(buf, offset_, n);
}

fn WOFF2StringOut::Write(buf: const void*, offset: size_t, n: size_t) -> bool {
  if (offset > max_size_ || n > max_size_ - offset) {
    return false;
  }
  if (offset == buf_->size()) {
    buf_->append(static_cast<const char*>(buf), n);
  } else {
    if (offset + n > buf_->size()) {
      buf_->append(offset + n - buf_->size(), 0);
    }
    buf_->replace(offset, n, static_cast<const char*>(buf), n);
  }
  offset_ = std::max(offset_, offset + n);

  return true;
}

fn WOFF2StringOut::SetMaxSize(max_size: size_t) {
  max_size_ = max_size;
  if (offset_ > max_size_) {
    offset_ = max_size_;
  }
}

WOFF2MemoryOut::WOFF2MemoryOut(buf: uint8_t*, buf_size: size_t)
  : buf_(buf),
    buf_size_(buf_size),
    offset_(0) {}

fn WOFF2MemoryOut::Write(buf: const void*, n: size_t) -> bool {
  return Write(buf, offset_, n);
}

fn WOFF2MemoryOut::Write(buf: const void*, offset: size_t, n: size_t) -> bool {
  if (offset > buf_size_ || n > buf_size_ - offset) {
    return false;
  }
  std::memcpy(buf_ + offset, buf, n);
  offset_ = std::max(offset_, offset + n);

  return true;
}

} // namespace woff2
