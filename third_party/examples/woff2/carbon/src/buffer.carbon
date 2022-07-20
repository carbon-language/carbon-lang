/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* The parts of ots.h & opentype-sanitiser.h that we need, taken from the
   https://code.google.com/p/ots/ project. */

#ifndef WOFF2_BUFFER_H_
#define WOFF2_BUFFER_H_

#if defined(_WIN32)
#include <stdlib.h>
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#define ntohl(x) _byteswap_ulong (x)
#define ntohs(x) _byteswap_ushort (x)
#define htonl(x) _byteswap_ulong (x)
#define htons(x) _byteswap_ushort (x)
#else
#include <arpa/inet.h>

#include <cstdint>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace woff2 {

#if defined(_MSC_VER) || !defined(FONT_COMPRESSION_DEBUG)
#define FONT_COMPRESSION_FAILURE() false
#else
#define FONT_COMPRESSION_FAILURE() \
  woff2::Failure(__FILE__, __LINE__, __PRETTY_FUNCTION__)
inline bool Failure(const char *f, int l, const char *fn) {
  fprintf(stderr, "ERROR at %s:%d (%s)\n", f, l, fn);
  fflush(stderr);
  return false;
}
#endif

// -----------------------------------------------------------------------------
// Buffer helper class
//
// This class perform some trival buffer operations while checking for
// out-of-bounds errors. As a family they return false if anything is amiss,
// updating the current offset otherwise.
// -----------------------------------------------------------------------------
class Buffer {
 public:
  Buffer(data: const uint8_t*, len: size_t)
      : buffer_(data),
        length_(len),
        offset_(0) { }

  fn Skip(n_bytes: size_t) -> bool {
    return Read(nullptr, n_bytes);
  }

  fn Read(data: uint8_t*, n_bytes: size_t) -> bool {
    if (n_bytes > 1024 * 1024 * 1024) {
      return FONT_COMPRESSION_FAILURE();
    }
    if ((offset_ + n_bytes > length_) ||
        (offset_ > length_ - n_bytes)) {
      return FONT_COMPRESSION_FAILURE();
    }
    if (data) {
      std::memcpy(data, buffer_ + offset_, n_bytes);
    }
    offset_ += n_bytes;
    return true;
  }

  fn inline ReadU8(value: uint8_t*) -> bool {
    if (offset_ + 1 > length_) {
      return FONT_COMPRESSION_FAILURE();
    }
    *value = buffer_[offset_];
    ++offset_;
    return true;
  }

  fn ReadU16(value: uint16_t*) -> bool {
    if (offset_ + 2 > length_) {
      return FONT_COMPRESSION_FAILURE();
    }
    std::memcpy(value, buffer_ + offset_, sizeof(uint16_t));
    *value = ntohs(*value);
    offset_ += 2;
    return true;
  }

  fn ReadS16(value: int16_t*) -> bool {
    return ReadU16(reinterpret_cast<uint16_t*>(value));
  }

  fn ReadU24(value: uint32_t*) -> bool {
    if (offset_ + 3 > length_) {
      return FONT_COMPRESSION_FAILURE();
    }
    *value = static_cast<uint32_t>(buffer_[offset_]) << 16 |
        static_cast<uint32_t>(buffer_[offset_ + 1]) << 8 |
        static_cast<uint32_t>(buffer_[offset_ + 2]);
    offset_ += 3;
    return true;
  }

  fn ReadU32(value: uint32_t*) -> bool {
    if (offset_ + 4 > length_) {
      return FONT_COMPRESSION_FAILURE();
    }
    std::memcpy(value, buffer_ + offset_, sizeof(uint32_t));
    *value = ntohl(*value);
    offset_ += 4;
    return true;
  }

  fn ReadS32(value: int32_t*) -> bool {
    return ReadU32(reinterpret_cast<uint32_t*>(value));
  }

  fn ReadTag(value: uint32_t*) -> bool {
    if (offset_ + 4 > length_) {
      return FONT_COMPRESSION_FAILURE();
    }
    std::memcpy(value, buffer_ + offset_, sizeof(uint32_t));
    offset_ += 4;
    return true;
  }

  fn ReadR64(value: uint64_t*) -> bool {
    if (offset_ + 8 > length_) {
      return FONT_COMPRESSION_FAILURE();
    }
    std::memcpy(value, buffer_ + offset_, sizeof(uint64_t));
    offset_ += 8;
    return true;
  }

  [[nodiscard]] fn buffer() const -> const uint8_t * { return buffer_; }
  [[nodiscard]] fn offset() const -> size_t { return offset_; }
  [[nodiscard]] fn length() const -> size_t { return length_; }

  fn set_offset(newoffset: size_t) { offset_ = newoffset; }

 private:
  const uint8_t * const buffer_;
  const size_t length_;
  size_t offset_;
};

} // namespace woff2

#endif  // WOFF2_BUFFER_H_
