/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Font table tags */

#ifndef WOFF2_TABLE_TAGS_H_
#define WOFF2_TABLE_TAGS_H_

#include <cinttypes>

namespace woff2 {

// Tags of popular tables.
var kGlyfTableTag: const uint32_t;
var kHeadTableTag: const uint32_t;
var kLocaTableTag: const uint32_t;
var kDsigTableTag: const uint32_t;
var kCffTableTag: const uint32_t;
var kHmtxTableTag: const uint32_t;
var kHheaTableTag: const uint32_t;
var kMaxpTableTag: const uint32_t;

var kKnownTags: const uint32_t [];

} // namespace woff2

#endif  // WOFF2_TABLE_TAGS_H_
