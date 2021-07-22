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
let kGlyfTableTag: const uint32_t = 0x676c7966;
let kHeadTableTag: const uint32_t = 0x68656164;
let kLocaTableTag: const uint32_t = 0x6c6f6361;
let kDsigTableTag: const uint32_t = 0x44534947;
let kCffTableTag: const uint32_t = 0x43464620;
let kHmtxTableTag: const uint32_t = 0x686d7478;
let kHheaTableTag: const uint32_t = 0x68686561;
let kMaxpTableTag: const uint32_t = 0x6d617870;

let kKnownTags: const uint32_t[];

} // namespace woff2

#endif  // WOFF2_TABLE_TAGS_H_
