/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Font table tags */

#include "./table_tags.h"

namespace woff2 {

// Note that the byte order is big-endian, not the same as ots.cc
#define TAG(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))

var kKnownTags: const uint32_t [63];

} // namespace woff2
