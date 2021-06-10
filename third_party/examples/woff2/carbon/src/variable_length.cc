/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Helper functions for woff2 variable length types: 255UInt16 and UIntBase128 */

#include "./variable_length.h"

namespace woff2 {

size_t Size255UShort(uint16_t value) {
  size_t result = 3;
  if (value < 253) {
    result = 1;
  } else if (value < 762) {
    result = 2;
  } else {
    result = 3;
  }
  return result;
}

void Write255UShort(std::vector<uint8_t>* out, int value) {
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

void Store255UShort(int val, size_t* offset, uint8_t* dst) {
  std::vector<uint8_t> packed;
  Write255UShort(&packed, val);
  for (uint8_t packed_byte : packed) {
    dst[(*offset)++] = packed_byte;
  }
}

// Based on section 6.1.1 of MicroType Express draft spec
bool Read255UShort(Buffer* buf, unsigned int* value) {
  static const int kWordCode = 253;
  static const int kOneMoreByteCode2 = 254;
  static const int kOneMoreByteCode1 = 255;
  static const int kLowestUCode = 253;
  uint8_t code = 0;
  if (!buf->ReadU8(&code)) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (code == kWordCode) {
    uint16_t result = 0;
    if (!buf->ReadU16(&result)) {
      return FONT_COMPRESSION_FAILURE();
    }
    *value = result;
    return true;
  } else if (code == kOneMoreByteCode1) {
    uint8_t result = 0;
    if (!buf->ReadU8(&result)) {
      return FONT_COMPRESSION_FAILURE();
    }
    *value = result + kLowestUCode;
    return true;
  } else if (code == kOneMoreByteCode2) {
    uint8_t result = 0;
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

bool ReadBase128(Buffer* buf, uint32_t* value) {
  uint32_t result = 0;
  for (size_t i = 0; i < 5; ++i) {
    uint8_t code = 0;
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

size_t Base128Size(size_t n) {
  size_t size = 1;
  for (; n >= 128; n >>= 7) ++size;
  return size;
}

void StoreBase128(size_t len, size_t* offset, uint8_t* dst) {
  size_t size = Base128Size(len);
  for (size_t i = 0; i < size; ++i) {
    int b = static_cast<int>((len >> (7 * (size - i - 1))) & 0x7f);
    if (i < size - 1) {
      b |= 0x80;
    }
    dst[(*offset)++] = b;
  }
}

} // namespace woff2

