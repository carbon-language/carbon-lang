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
let kGlyfTableTag: uint32_t;
let kHeadTableTag: uint32_t;
let kLocaTableTag: uint32_t;
let kDsigTableTag: uint32_t;
let kCffTableTag: uint32_t;
let kHmtxTableTag: uint32_t;
let kHheaTableTag: uint32_t;
let kMaxpTableTag: uint32_t;

let kKnownTags: uint32_t[];

} // namespace woff2

#endif  // WOFF2_TABLE_TAGS_H_
