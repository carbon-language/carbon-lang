/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting WOFF2 format font files to their TTF versions. */

#ifndef WOFF2_WOFF2_ENC_H_
#define WOFF2_WOFF2_ENC_H_

<<<<<<< HEAD
#include <cstddef>
#include <cinttypes>
=======
#include <stddef.h>
#include <inttypes.h>
>>>>>>> trunk
#include <string>

namespace woff2 {

struct WOFF2Params {
<<<<<<< HEAD
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
=======
  WOFF2Params() : extended_metadata(""), brotli_quality(11),
                  allow_transforms(true) {}

  std::string extended_metadata;
  int brotli_quality;
  bool allow_transforms;
};

// Returns an upper bound on the size of the compressed file.
size_t MaxWOFF2CompressedSize(const uint8_t* data, size_t length);
size_t MaxWOFF2CompressedSize(const uint8_t* data, size_t length,
                              const std::string& extended_metadata);
>>>>>>> trunk

// Compresses the font into the target buffer. *result_length should be at least
// the value returned by MaxWOFF2CompressedSize(), upon return, it is set to the
// actual compressed size. Returns true on successful compression.
<<<<<<< HEAD
auto ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length) -> bool;
auto ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length,
                       const WOFF2Params& params) -> bool;
=======
bool ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length);
bool ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length,
                       const WOFF2Params& params);
>>>>>>> trunk

} // namespace woff2

#endif  // WOFF2_WOFF2_ENC_H_
