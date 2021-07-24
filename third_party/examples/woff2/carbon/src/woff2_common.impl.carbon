/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Helpers common across multiple parts of woff2 */

#include <algorithm>

#include "./woff2_common.h"

#include "./port.h"

namespace woff2 {


fn ComputeULongSum(buf: const uint8_t*, size: size_t) -> uint32_t {
  var checksum: uint32_t = 0;
  var aligned_size: size_t = size & ~3;
  for (var i: size_t = 0; i < aligned_size; i += 4) {
#if defined(WOFF_LITTLE_ENDIAN)
    var v: uint32_t = *reinterpret_cast<const uint32_t*>(buf + i);
    checksum += (((v & 0xFF) << 24) | ((v & 0xFF00) << 8) |
      ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24));
#elif defined(WOFF_BIG_ENDIAN)
    checksum += *reinterpret_cast<const uint32_t*>(buf + i);
#else
    checksum += (buf[i] << 24) | (buf[i + 1] << 16) |
      (buf[i + 2] << 8) | buf[i + 3];
#endif
  }

  // treat size not aligned on 4 as if it were padded to 4 with 0's
  if (size != aligned_size) {
    var v: uint32_t = 0;
    for (var i: size_t = aligned_size; i < size; ++i) {
      v |= buf[i] << (24 - 8 * (i & 3));
    }
    checksum += v;
  }

  return checksum;
}

fn CollectionHeaderSize(header_version: uint32_t, num_fonts: uint32_t) -> size_t {
  var size: size_t = 0;
  if (header_version == 0x00020000) {
    size += 12;  // ulDsig{Tag,Length,Offset}
  }
  if (header_version == 0x00010000 || header_version == 0x00020000) {
    size += 12   // TTCTag, Version, numFonts
      + 4 * num_fonts;  // OffsetTable[numFonts]
  }
  return size;
}

} // namespace woff2
