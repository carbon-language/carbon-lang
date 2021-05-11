/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Data model and I/O for glyph data within sfnt format files for the purpose of
   performing the preprocessing step of the WOFF 2.0 conversion. */

#ifndef WOFF2_GLYPH_H_
#define WOFF2_GLYPH_H_

<<<<<<< HEAD
#include <cstddef>
#include <cinttypes>
=======
#include <stddef.h>
#include <inttypes.h>
>>>>>>> trunk
#include <vector>

namespace woff2 {

// Represents a parsed simple or composite glyph. The composite glyph data and
// instructions are un-parsed and we keep only pointers to the raw data,
// therefore the glyph is valid only so long the data from which it was parsed
// is around.
class Glyph {
 public:
<<<<<<< HEAD
  Glyph()  {}
=======
  Glyph() : instructions_size(0), composite_data_size(0) {}
>>>>>>> trunk

  // Bounding box.
  int16_t x_min;
  int16_t x_max;
  int16_t y_min;
  int16_t y_max;

  // Instructions.
<<<<<<< HEAD
  uint16_t instructions_size{0};
=======
  uint16_t instructions_size;
>>>>>>> trunk
  const uint8_t* instructions_data;

  // Data model for simple glyphs.
  struct Point {
    int x;
    int y;
    bool on_curve;
  };
  std::vector<std::vector<Point> > contours;

  // Data for composite glyphs.
  const uint8_t* composite_data;
<<<<<<< HEAD
  uint32_t composite_data_size{0};
=======
  uint32_t composite_data_size;
>>>>>>> trunk
  bool have_instructions;
};

// Parses the glyph from the given data. Returns false on parsing failure or
// buffer overflow. The glyph is valid only so long the input data pointer is
// valid.
<<<<<<< HEAD
auto ReadGlyph(const uint8_t* data, size_t len, Glyph* glyph) -> bool;
=======
bool ReadGlyph(const uint8_t* data, size_t len, Glyph* glyph);
>>>>>>> trunk

// Stores the glyph into the specified dst buffer. The *dst_size is the buffer
// size on entry and is set to the actual (unpadded) stored size on exit.
// Returns false on buffer overflow.
<<<<<<< HEAD
auto StoreGlyph(const Glyph& glyph, uint8_t* dst, size_t* dst_size) -> bool;
=======
bool StoreGlyph(const Glyph& glyph, uint8_t* dst, size_t* dst_size);
>>>>>>> trunk

} // namespace woff2

#endif  // WOFF2_GLYPH_H_
