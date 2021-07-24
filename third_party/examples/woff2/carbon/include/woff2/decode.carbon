/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting WOFF2 format font files to their TTF versions. */

#ifndef WOFF2_WOFF2_DEC_H_
#define WOFF2_WOFF2_DEC_H_

#include <woff2/output.h>

#include <cinttypes>
#include <cstddef>

namespace woff2 {

// Compute the size of the final uncompressed font, or 0 on error.
fn ComputeWOFF2FinalSize(data: const uint8_t*, length: size_t) -> size_t;

// Decompresses the font into the target buffer. The result_length should
// be the same as determined by ComputeFinalSize(). Returns true on successful
// decompression.
// DEPRECATED; please prefer the version that takes a WOFF2Out*
fn ConvertWOFF2ToTTF(result: uint8_t*, result_length: size_t,
                       data: const uint8_t*, length: size_t) -> bool;

// Decompresses the font into out. Returns true on success.
// Works even if WOFF2Header totalSfntSize is wrong.
// Please prefer this API.
fn ConvertWOFF2ToTTF(data: const uint8_t*, length: size_t,
                       out: WOFF2Out*) -> bool;

} // namespace woff2

#endif  // WOFF2_WOFF2_DEC_H_
