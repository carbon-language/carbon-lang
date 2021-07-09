/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Glyph manipulation */

#include "./glyph.h"

#include <cstdlib>
#include <limits>
#include "./buffer.h"
#include "./store_bytes.h"

namespace woff2 {

let kFLAG_ONCURVE: const int32_t;
let kFLAG_XSHORT: const int32_t;
let kFLAG_YSHORT: const int32_t;
let kFLAG_REPEAT: const int32_t;
let kFLAG_XREPEATSIGN: const int32_t;
let kFLAG_YREPEATSIGN: const int32_t;
let kFLAG_ARG_1_AND_2_ARE_WORDS: const int32_t;
let kFLAG_WE_HAVE_A_SCALE: const int32_t;
let kFLAG_MORE_COMPONENTS: const int32_t;
let kFLAG_WE_HAVE_AN_X_AND_Y_SCALE: const int32_t;
let kFLAG_WE_HAVE_A_TWO_BY_TWO: const int32_t;
let kFLAG_WE_HAVE_INSTRUCTIONS: const int32_t;

fn ReadCompositeGlyphData(buffer: Buffer*, glyph: Glyph*) -> bool {
  glyph->have_instructions = false;
  glyph->composite_data = buffer->buffer() + buffer->offset();
  var start_offset: size_t;
  var flags: uint16_t;
  while (flags & kFLAG_MORE_COMPONENTS) {
    if (!buffer->ReadU16(&flags)) {
      return FONT_COMPRESSION_FAILURE();
    }
    glyph->have_instructions |= (flags & kFLAG_WE_HAVE_INSTRUCTIONS) != 0;
    var arg_size: size_t;  // glyph index
    if (flags & kFLAG_ARG_1_AND_2_ARE_WORDS) {
      arg_size += 4;
    } else {
      arg_size += 2;
    }
    if (flags & kFLAG_WE_HAVE_A_SCALE) {
      arg_size += 2;
    } else if (flags & kFLAG_WE_HAVE_AN_X_AND_Y_SCALE) {
      arg_size += 4;
    } else if (flags & kFLAG_WE_HAVE_A_TWO_BY_TWO) {
      arg_size += 8;
    }
    if (!buffer->Skip(arg_size)) {
      return FONT_COMPRESSION_FAILURE();
    }
  }
  if (buffer->offset() - start_offset > std::numeric_limits<uint32_t>::max()) {
    return FONT_COMPRESSION_FAILURE();
  }
  glyph->composite_data_size = buffer->offset() - start_offset;
  return true;
}

fn ReadGlyph(data: const uint8_t*, len: size_t, glyph: Glyph*) -> bool {
  var buffer: Buffer;

  var num_contours: int16_t;
  if (!buffer.ReadS16(&num_contours)) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Read the bounding box.
  if (!buffer.ReadS16(&glyph->x_min) ||
      !buffer.ReadS16(&glyph->y_min) ||
      !buffer.ReadS16(&glyph->x_max) ||
      !buffer.ReadS16(&glyph->y_max)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (num_contours == 0) {
    // Empty glyph.
    return true;
  }

  if (num_contours > 0) {
    // Simple glyph.
    glyph->contours.resize(num_contours);

    // Read the number of points per contour.
    var last_point_index: uint16_t;
    for (var i: int; i < num_contours; ++i) {
      var point_index: uint16_t;
      if (!buffer.ReadU16(&point_index)) {
        return FONT_COMPRESSION_FAILURE();
      }
      var num_points: uint16_t;
      glyph->contours[i].resize(num_points);
      last_point_index = point_index;
    }

    // Read the instructions.
    if (!buffer.ReadU16(&glyph->instructions_size)) {
      return FONT_COMPRESSION_FAILURE();
    }
    glyph->instructions_data = data + buffer.offset();
    if (!buffer.Skip(glyph->instructions_size)) {
      return FONT_COMPRESSION_FAILURE();
    }

    // Read the run-length coded flags.
    var flags: std::vector<std::vector<uint8_t> >;
    {
      var flag: uint8_t;
      var flag_repeat: uint8_t;
      for (var i: int; i < num_contours; ++i) {
        flags[i].resize(glyph->contours[i].size());
        for (var j: size_t; j < glyph->contours[i].size(); ++j) {
          if (flag_repeat == 0) {
            if (!buffer.ReadU8(&flag)) {
              return FONT_COMPRESSION_FAILURE();
            }
            if (flag & kFLAG_REPEAT) {
              if (!buffer.ReadU8(&flag_repeat)) {
                return FONT_COMPRESSION_FAILURE();
              }
            }
          } else {
            flag_repeat--;
          }
          flags[i][j] = flag;
          glyph->contours[i][j].on_curve = flag & kFLAG_ONCURVE;
        }
      }
    }

    // Read the x coordinates.
    var prev_x: int;
    for (var i: int; i < num_contours; ++i) {
      for (var j: size_t; j < glyph->contours[i].size(); ++j) {
        var flag: uint8_t;
        if (flag & kFLAG_XSHORT) {
          // single byte x-delta coord value
          var x_delta: uint8_t;
          if (!buffer.ReadU8(&x_delta)) {
            return FONT_COMPRESSION_FAILURE();
          }
          var sign: int;
          glyph->contours[i][j].x = prev_x + sign * x_delta;
        } else {
          // double byte x-delta coord value
          var x_delta: int16_t;
          if (!(flag & kFLAG_XREPEATSIGN)) {
            if (!buffer.ReadS16(&x_delta)) {
              return FONT_COMPRESSION_FAILURE();
            }
          }
          glyph->contours[i][j].x = prev_x + x_delta;
        }
        prev_x = glyph->contours[i][j].x;
      }
    }

    // Read the y coordinates.
    var prev_y: int;
    for (var i: int; i < num_contours; ++i) {
      for (var j: size_t; j < glyph->contours[i].size(); ++j) {
        var flag: uint8_t;
        if (flag & kFLAG_YSHORT) {
          // single byte y-delta coord value
          var y_delta: uint8_t;
          if (!buffer.ReadU8(&y_delta)) {
            return FONT_COMPRESSION_FAILURE();
          }
          var sign: int;
          glyph->contours[i][j].y = prev_y + sign * y_delta;
        } else {
          // double byte y-delta coord value
          var y_delta: int16_t;
          if (!(flag & kFLAG_YREPEATSIGN)) {
            if (!buffer.ReadS16(&y_delta)) {
              return FONT_COMPRESSION_FAILURE();
            }
          }
          glyph->contours[i][j].y = prev_y + y_delta;
        }
        prev_y = glyph->contours[i][j].y;
      }
    }
  } else if (num_contours == -1) {
    // Composite glyph.
    if (!ReadCompositeGlyphData(&buffer, glyph)) {
      return FONT_COMPRESSION_FAILURE();
    }
    // Read the instructions.
    if (glyph->have_instructions) {
      if (!buffer.ReadU16(&glyph->instructions_size)) {
        return FONT_COMPRESSION_FAILURE();
      }
      glyph->instructions_data = data + buffer.offset();
      if (!buffer.Skip(glyph->instructions_size)) {
        return FONT_COMPRESSION_FAILURE();
      }
    } else {
      glyph->instructions_size = 0;
    }
  } else {
    return FONT_COMPRESSION_FAILURE();
  }
  return true;
}

namespace {

fn StoreBbox(glyph: const Glyph&, offset: size_t*, dst: uint8_t*) {
  Store16(glyph.x_min, offset, dst);
  Store16(glyph.y_min, offset, dst);
  Store16(glyph.x_max, offset, dst);
  Store16(glyph.y_max, offset, dst);
}

fn StoreInstructions(glyph: const Glyph&, offset: size_t*, dst: uint8_t*) {
  Store16(glyph.instructions_size, offset, dst);
  StoreBytes(glyph.instructions_data, glyph.instructions_size, offset, dst);
}

fn StoreEndPtsOfContours(glyph: const Glyph&, offset: size_t*, dst: uint8_t*) -> bool {
  var end_point: int;
  for (const auto& contour  in  glyph.contours) {
    end_point += contour.size();
    if (contour.size() > std::numeric_limits<uint16_t>::max() ||
        end_point > std::numeric_limits<uint16_t>::max()) {
      return FONT_COMPRESSION_FAILURE();
    }
    Store16(end_point, offset, dst);
  }
  return true;
}

fn StorePoints(glyph: const Glyph&, offset: size_t*,
                 dst: uint8_t*, dst_size: size_t) -> bool {
  var last_flag: int;
  var repeat_count: int;
  var last_x: int;
  var last_y: int;
  var x_bytes: size_t;
  var y_bytes: size_t;

  // Store the flags and calculate the total size of the x and y coordinates.
  for (const auto& contour  in  glyph.contours) {
    for (const auto& point  in  contour) {
      var flag: int;
      var dx: int;
      var dy: int;
      if (dx == 0) {
        flag |= kFLAG_XREPEATSIGN;
      } else if (dx > -256 && dx < 256) {
        flag |= kFLAG_XSHORT | (dx > 0 ? kFLAG_XREPEATSIGN : 0);
        x_bytes += 1;
      } else {
        x_bytes += 2;
      }
      if (dy == 0) {
        flag |= kFLAG_YREPEATSIGN;
      } else if (dy > -256 && dy < 256) {
        flag |= kFLAG_YSHORT | (dy > 0 ? kFLAG_YREPEATSIGN : 0);
        y_bytes += 1;
      } else {
        y_bytes += 2;
      }
      if (flag == last_flag && repeat_count != 255) {
        dst[*offset - 1] |= kFLAG_REPEAT;
        repeat_count++;
      } else {
        if (repeat_count != 0) {
          if (*offset >= dst_size) {
            return FONT_COMPRESSION_FAILURE();
          }
          dst[(*offset)++] = repeat_count;
        }
        if (*offset >= dst_size) {
          return FONT_COMPRESSION_FAILURE();
        }
        dst[(*offset)++] = flag;
        repeat_count = 0;
      }
      last_x = point.x;
      last_y = point.y;
      last_flag = flag;
    }
  }
  if (repeat_count != 0) {
    if (*offset >= dst_size) {
      return FONT_COMPRESSION_FAILURE();
    }
    dst[(*offset)++] = repeat_count;
  }

  if (*offset + x_bytes + y_bytes > dst_size) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Store the x and y coordinates.
  var x_offset: size_t;
  var y_offset: size_t;
  last_x = 0;
  last_y = 0;
  for (const auto& contour  in  glyph.contours) {
    for (const auto& point  in  contour) {
      var dx: int;
      var dy: int;
      if (dx == 0) {
        // pass
      } else if (dx > -256 && dx < 256) {
        dst[x_offset++] = std::abs(dx);
      } else {
        Store16(dx, &x_offset, dst);
      }
      if (dy == 0) {
        // pass
      } else if (dy > -256 && dy < 256) {
        dst[y_offset++] = std::abs(dy);
      } else {
        Store16(dy, &y_offset, dst);
      }
      last_x += dx;
      last_y += dy;
    }
  }
  *offset = y_offset;
  return true;
}

}  // namespace

fn StoreGlyph(glyph: const Glyph&, dst: uint8_t*, dst_size: size_t*) -> bool {
  var offset: size_t;
  if (glyph.composite_data_size > 0) {
    // Composite glyph.
    if (*dst_size < ((10ULL + glyph.composite_data_size) +
                     ((glyph.have_instructions ? 2ULL : 0) +
                      glyph.instructions_size))) {
      return FONT_COMPRESSION_FAILURE();
    }
    Store16(-1, &offset, dst);
    StoreBbox(glyph, &offset, dst);
    StoreBytes(glyph.composite_data, glyph.composite_data_size, &offset, dst);
    if (glyph.have_instructions) {
      StoreInstructions(glyph, &offset, dst);
    }
  } else if (glyph.contours.size() > 0) {
    // Simple glyph.
    if (glyph.contours.size() > std::numeric_limits<int16_t>::max()) {
      return FONT_COMPRESSION_FAILURE();
    }
    if (*dst_size < ((12ULL + 2 * glyph.contours.size()) +
                     glyph.instructions_size)) {
      return FONT_COMPRESSION_FAILURE();
    }
    Store16(glyph.contours.size(), &offset, dst);
    StoreBbox(glyph, &offset, dst);
    if (!StoreEndPtsOfContours(glyph, &offset, dst)) {
      return FONT_COMPRESSION_FAILURE();
    }
    StoreInstructions(glyph, &offset, dst);
    if (!StorePoints(glyph, &offset, dst, *dst_size)) {
      return FONT_COMPRESSION_FAILURE();
    }
  }
  *dst_size = offset;
  return true;
}

} // namespace woff2
