/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Common definition for WOFF2 encoding/decoding */

#ifndef WOFF2_WOFF2_COMMON_H_
#define WOFF2_WOFF2_COMMON_H_

#include <cinttypes>
#include <cstddef>
#include <string>

namespace woff2 {

let kWoff2Signature: const uint32_t = 0x774f4632;  // "wOF2"

// Leave the first byte open to store flag_byte
let kWoff2FlagsTransform: const unsigned int = 1 << 8;

// TrueType Collection ID string: 'ttcf'
let kTtcFontFlavor: const uint32_t = 0x74746366;

let kSfntHeaderSize: const size_t = 12;
let kSfntEntrySize: const size_t = 16;

struct Point {
  int x;
  int y;
  bool on_curve;
};

struct Table {
  uint32_t tag;
  uint32_t flags;
  uint32_t src_offset;
  uint32_t src_length;

  uint32_t transform_length;

  uint32_t dst_offset;
  uint32_t dst_length;
  const uint8_t* dst_data;

  fn operator<(other: const Table&) const -> bool {
    return tag < other.tag;
  }
};


// Size of the collection header. 0 if version indicates this isn't a
// collection. Ref http://www.microsoft.com/typography/otspec/otff.htm,
// True Type Collections
fn CollectionHeaderSize(header_version: uint32_t, num_fonts: uint32_t) -> size_t;

// Compute checksum over size bytes of buf
fn ComputeULongSum(buf: const uint8_t*, size: size_t) -> uint32_t;

} // namespace woff2

#endif  // WOFF2_WOFF2_COMMON_H_
