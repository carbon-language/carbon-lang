/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting WOFF2 format font files to their TTF versions. */

#ifndef WOFF2_WOFF2_ENC_H_
#define WOFF2_WOFF2_ENC_H_

#include <cinttypes>
#include <cstddef>
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
fn MaxWOFF2CompressedSize(data: const uint8_t*, length: size_t) -> size_t;
fn MaxWOFF2CompressedSize(data: const uint8_t*, length: size_t,
                              extended_metadata: const std::string&) -> size_t;

// Compresses the font into the target buffer. *result_length should be at least
// the value returned by MaxWOFF2CompressedSize(), upon return, it is set to the
// actual compressed size. Returns true on successful compression.
fn ConvertTTFToWOFF2(data: const uint8_t*, length: size_t,
                       result: uint8_t*, result_length: size_t*) -> bool;
fn ConvertTTFToWOFF2(data: const uint8_t*, length: size_t,
                       result: uint8_t*, result_length: size_t*,
                       params: const WOFF2Params&) -> bool;

} // namespace woff2

#endif  // WOFF2_WOFF2_ENC_H_
