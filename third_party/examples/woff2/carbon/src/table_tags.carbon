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
let kGlyfTableTag: const uint32_t;
let kHeadTableTag: const uint32_t;
let kLocaTableTag: const uint32_t;
let kDsigTableTag: const uint32_t;
let kCffTableTag: const uint32_t;
let kHmtxTableTag: const uint32_t;
let kHheaTableTag: const uint32_t;
let kMaxpTableTag: const uint32_t;

let kKnownTags: const uint32_t[];

} // namespace woff2

#endif  // WOFF2_TABLE_TAGS_H_
