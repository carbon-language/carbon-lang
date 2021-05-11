/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting WOFF2 format font files to their TTF versions. */

#ifndef WOFF2_WOFF2_ENC_H_
#define WOFF2_WOFF2_ENC_H_

#include <cstddef>
#include <cinttypes>
#include <string>

namespace woff2 {

struct WOFF2Params {
  WOFF2Params() : extended_metadata("")
                  {}

  std::string extended_metadata;
  int brotli_quality{11};
  bool allow_transforms{true};
};

// Returns an upper bound on the size of the compressed file.
auto MaxWOFF2CompressedSize(const uint8_t* data, size_t length) -> size_t;
auto MaxWOFF2CompressedSize(const uint8_t* data, size_t length,
                              const std::string& extended_metadata) -> size_t;

// Compresses the font into the target buffer. *result_length should be at least
// the value returned by MaxWOFF2CompressedSize(), upon return, it is set to the
// actual compressed size. Returns true on successful compression.
auto ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length) -> bool;
auto ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length,
                       const WOFF2Params& params) -> bool;

} // namespace woff2

#endif  // WOFF2_WOFF2_ENC_H_
