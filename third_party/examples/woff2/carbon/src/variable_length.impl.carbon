/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Helper functions for woff2 variable length types: 255UInt16 and UIntBase128 */

#include "./variable_length.h"

namespace woff2 {

fn Size255UShort(value: uint16_t) -> size_t {
  var result: size_t = 3;
  if (value < 253) {
    result = 1;
  } else if (value < 762) {
    result = 2;
  } else {
    result = 3;
  }
  return result;
}

fn Write255UShort(out: std::vector<uint8_t>*, value: int) {
  if (value < 253) {
    out->push_back(value);
  } else if (value < 506) {
    out->push_back(255);
    out->push_back(value - 253);
  } else if (value < 762) {
    out->push_back(254);
    out->push_back(value - 506);
  } else {
    out->push_back(253);
    out->push_back(value >> 8);
    out->push_back(value & 0xff);
  }
}

fn Store255UShort(val: int, offset: size_t*, dst: uint8_t*) {
  var packed: std::vector<uint8_t>;
  Write255UShort(&packed, val);
  for (uint8_t packed_byte  in  packed) {
    dst[(*offset)++] = packed_byte;
  }
}

// Based on section 6.1.1 of MicroType Express draft spec
fn Read255UShort(buf: Buffer*, value: unsigned int*) -> bool {
  let kWordCode: const int = 253;
  let kOneMoreByteCode2: const int = 254;
  let kOneMoreByteCode1: const int = 255;
  let kLowestUCode: const int = 253;
  var code: uint8_t = 0;
  if (!buf->ReadU8(&code)) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (code == kWordCode) {
    var result: uint16_t = 0;
    if (!buf->ReadU16(&result)) {
      return FONT_COMPRESSION_FAILURE();
    }
    *value = result;
    return true;
  } else if (code == kOneMoreByteCode1) {
    var result: uint8_t = 0;
    if (!buf->ReadU8(&result)) {
      return FONT_COMPRESSION_FAILURE();
    }
    *value = result + kLowestUCode;
    return true;
  } else if (code == kOneMoreByteCode2) {
    var result: uint8_t = 0;
    if (!buf->ReadU8(&result)) {
      return FONT_COMPRESSION_FAILURE();
    }
    *value = result + kLowestUCode * 2;
    return true;
  } else {
    *value = code;
    return true;
  }
}

fn ReadBase128(buf: Buffer*, value: uint32_t*) -> bool {
  var result: uint32_t = 0;
  for (var i: size_t = 0; i < 5; ++i) {
    var code: uint8_t = 0;
    if (!buf->ReadU8(&code)) {
      return FONT_COMPRESSION_FAILURE();
    }
    // Leading zeros are invalid.
    if (i == 0 && code == 0x80) {
      return FONT_COMPRESSION_FAILURE();
    }
    // If any of the top seven bits are set then we're about to overflow.
    if (result & 0xfe000000) {
      return FONT_COMPRESSION_FAILURE();
    }
    result = (result << 7) | (code & 0x7f);
    if ((code & 0x80) == 0) {
      *value = result;
      return true;
    }
  }
  // Make sure not to exceed the size bound
  return FONT_COMPRESSION_FAILURE();
}

fn Base128Size(n: size_t) -> size_t {
  var size: size_t = 1;
  for (; n >= 128; n >>= 7) { ++size;
}
  return size;
}

fn StoreBase128(len: size_t, offset: size_t*, dst: uint8_t*) {
  var size: size_t = Base128Size(len);
  for (var i: size_t = 0; i < size; ++i) {
    var b: int = static_cast<int>((len >> (7 * (size - i - 1))) & 0x7f);
    if (i < size - 1) {
      b |= 0x80;
    }
    dst[(*offset)++] = b;
  }
}

} // namespace woff2

