/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Glyph manipulation */

#include "./glyph.h"

#include <stdlib.h>
#include <limits>
#include "./buffer.h"
#include "./store_bytes.h"

namespace woff2 {

static const int32_t kFLAG_ONCURVE = 1;
static const int32_t kFLAG_XSHORT = 1 << 1;
static const int32_t kFLAG_YSHORT = 1 << 2;
static const int32_t kFLAG_REPEAT = 1 << 3;
static const int32_t kFLAG_XREPEATSIGN = 1 << 4;
static const int32_t kFLAG_YREPEATSIGN = 1 << 5;
static const int32_t kFLAG_ARG_1_AND_2_ARE_WORDS = 1 << 0;
static const int32_t kFLAG_WE_HAVE_A_SCALE = 1 << 3;
static const int32_t kFLAG_MORE_COMPONENTS = 1 << 5;
static const int32_t kFLAG_WE_HAVE_AN_X_AND_Y_SCALE = 1 << 6;
static const int32_t kFLAG_WE_HAVE_A_TWO_BY_TWO = 1 << 7;
static const int32_t kFLAG_WE_HAVE_INSTRUCTIONS = 1 << 8;

bool ReadCompositeGlyphData(Buffer* buffer, Glyph* glyph) {
  glyph->have_instructions = false;
  glyph->composite_data = buffer->buffer() + buffer->offset();
  size_t start_offset = buffer->offset();
  uint16_t flags = kFLAG_MORE_COMPONENTS;
  while (flags & kFLAG_MORE_COMPONENTS) {
    if (!buffer->ReadU16(&flags)) {
      return FONT_COMPRESSION_FAILURE();
    }
    glyph->have_instructions |= (flags & kFLAG_WE_HAVE_INSTRUCTIONS) != 0;
    size_t arg_size = 2;  // glyph index
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

bool ReadGlyph(const uint8_t* data, size_t len, Glyph* glyph) {
  Buffer buffer(data, len);

  int16_t num_contours;
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
    uint16_t last_point_index = 0;
    for (int i = 0; i < num_contours; ++i) {
      uint16_t point_index;
      if (!buffer.ReadU16(&point_index)) {
        return FONT_COMPRESSION_FAILURE();
      }
      uint16_t num_points = point_index - last_point_index + (i == 0 ? 1 : 0);
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
    std::vector<std::vector<uint8_t> > flags(num_contours);
    {
      uint8_t flag = 0;
      uint8_t flag_repeat = 0;
      for (int i = 0; i < num_contours; ++i) {
        flags[i].resize(glyph->contours[i].size());
        for (size_t j = 0; j < glyph->contours[i].size(); ++j) {
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
    int prev_x = 0;
    for (int i = 0; i < num_contours; ++i) {
      for (size_t j = 0; j < glyph->contours[i].size(); ++j) {
        uint8_t flag = flags[i][j];
        if (flag & kFLAG_XSHORT) {
          // single byte x-delta coord value
          uint8_t x_delta;
          if (!buffer.ReadU8(&x_delta)) {
            return FONT_COMPRESSION_FAILURE();
          }
          int sign = (flag & kFLAG_XREPEATSIGN) ? 1 : -1;
          glyph->contours[i][j].x = prev_x + sign * x_delta;
        } else {
          // double byte x-delta coord value
          int16_t x_delta = 0;
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
    int prev_y = 0;
    for (int i = 0; i < num_contours; ++i) {
      for (size_t j = 0; j < glyph->contours[i].size(); ++j) {
        uint8_t flag = flags[i][j];
        if (flag & kFLAG_YSHORT) {
          // single byte y-delta coord value
          uint8_t y_delta;
          if (!buffer.ReadU8(&y_delta)) {
            return FONT_COMPRESSION_FAILURE();
          }
          int sign = (flag & kFLAG_YREPEATSIGN) ? 1 : -1;
          glyph->contours[i][j].y = prev_y + sign * y_delta;
        } else {
          // double byte y-delta coord value
          int16_t y_delta = 0;
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

void StoreBbox(const Glyph& glyph, size_t* offset, uint8_t* dst) {
  Store16(glyph.x_min, offset, dst);
  Store16(glyph.y_min, offset, dst);
  Store16(glyph.x_max, offset, dst);
  Store16(glyph.y_max, offset, dst);
}

void StoreInstructions(const Glyph& glyph, size_t* offset, uint8_t* dst) {
  Store16(glyph.instructions_size, offset, dst);
  StoreBytes(glyph.instructions_data, glyph.instructions_size, offset, dst);
}

bool StoreEndPtsOfContours(const Glyph& glyph, size_t* offset, uint8_t* dst) {
  int end_point = -1;
  for (const auto& contour : glyph.contours) {
    end_point += contour.size();
    if (contour.size() > std::numeric_limits<uint16_t>::max() ||
        end_point > std::numeric_limits<uint16_t>::max()) {
      return FONT_COMPRESSION_FAILURE();
    }
    Store16(end_point, offset, dst);
  }
  return true;
}

bool StorePoints(const Glyph& glyph, size_t* offset,
                 uint8_t* dst, size_t dst_size) {
  int last_flag = -1;
  int repeat_count = 0;
  int last_x = 0;
  int last_y = 0;
  size_t x_bytes = 0;
  size_t y_bytes = 0;

  // Store the flags and calculate the total size of the x and y coordinates.
  for (const auto& contour : glyph.contours) {
    for (const auto& point : contour) {
      int flag = point.on_curve ? kFLAG_ONCURVE : 0;
      int dx = point.x - last_x;
      int dy = point.y - last_y;
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
  size_t x_offset = *offset;
  size_t y_offset = *offset + x_bytes;
  last_x = 0;
  last_y = 0;
  for (const auto& contour : glyph.contours) {
    for (const auto& point : contour) {
      int dx = point.x - last_x;
      int dy = point.y - last_y;
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

bool StoreGlyph(const Glyph& glyph, uint8_t* dst, size_t* dst_size) {
  size_t offset = 0;
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
