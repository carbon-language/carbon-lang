/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting WOFF2 format font files to their TTF versions. */

#include <woff2/decode.h>

#include <stdlib.h>
#include <algorithm>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>

#include <brotli/decode.h>
#include "./buffer.h"
#include "./port.h"
#include "./round.h"
#include "./store_bytes.h"
#include "./table_tags.h"
#include "./variable_length.h"
#include "./woff2_common.h"

namespace woff2 {

namespace {

using std::string;
using std::vector;


// simple glyph flags
const int kGlyfOnCurve = 1 << 0;
const int kGlyfXShort = 1 << 1;
const int kGlyfYShort = 1 << 2;
const int kGlyfRepeat = 1 << 3;
const int kGlyfThisXIsSame = 1 << 4;
const int kGlyfThisYIsSame = 1 << 5;

// composite glyph flags
// See CompositeGlyph.java in sfntly for full definitions
const int FLAG_ARG_1_AND_2_ARE_WORDS = 1 << 0;
const int FLAG_WE_HAVE_A_SCALE = 1 << 3;
const int FLAG_MORE_COMPONENTS = 1 << 5;
const int FLAG_WE_HAVE_AN_X_AND_Y_SCALE = 1 << 6;
const int FLAG_WE_HAVE_A_TWO_BY_TWO = 1 << 7;
const int FLAG_WE_HAVE_INSTRUCTIONS = 1 << 8;

const size_t kCheckSumAdjustmentOffset = 8;

const size_t kEndPtsOfContoursOffset = 10;
const size_t kCompositeGlyphBegin = 10;

// 98% of Google Fonts have no glyph above 5k bytes
// Largest glyph ever observed was 72k bytes
const size_t kDefaultGlyphBuf = 5120;

// Over 14k test fonts the max compression ratio seen to date was ~20.
// >100 suggests you wrote a bad uncompressed size.
const float kMaxPlausibleCompressionRatio = 100.0;

// metadata for a TTC font entry
struct TtcFont {
  uint32_t flavor;
  uint32_t dst_offset;
  uint32_t header_checksum;
  std::vector<uint16_t> table_indices;
};

struct WOFF2Header {
  uint32_t flavor;
  uint32_t header_version;
  uint16_t num_tables;
  uint64_t compressed_offset;
  uint32_t compressed_length;
  uint32_t uncompressed_size;
  std::vector<Table> tables;  // num_tables unique tables
  std::vector<TtcFont> ttc_fonts;  // metadata to help rebuild font
};

/**
 * Accumulates data we may need to reconstruct a single font. One per font
 * created for a TTC.
 */
struct WOFF2FontInfo {
  uint16_t num_glyphs;
  uint16_t index_format;
  uint16_t num_hmetrics;
  std::vector<int16_t> x_mins;
  std::map<uint32_t, uint32_t> table_entry_by_tag;
};

// Accumulates metadata as we rebuild the font
struct RebuildMetadata {
  uint32_t header_checksum;  // set by WriteHeaders
  std::vector<WOFF2FontInfo> font_infos;
  // checksums for tables that have been written.
  // (tag, src_offset) => checksum. Need both because 0-length loca.
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> checksums;
};

int WithSign(int flag, int baseval) {
  // Precondition: 0 <= baseval < 65536 (to avoid integer overflow)
  return (flag & 1) ? baseval : -baseval;
}

bool _SafeIntAddition(int a, int b, int* result) {
  if (PREDICT_FALSE(
          ((a > 0) && (b > std::numeric_limits<int>::max() - a)) ||
          ((a < 0) && (b < std::numeric_limits<int>::min() - a)))) {
    return false;
  }
  *result = a + b;
  return true;
}

bool TripletDecode(const uint8_t* flags_in, const uint8_t* in, size_t in_size,
    unsigned int n_points, Point* result, size_t* in_bytes_consumed) {
  int x = 0;
  int y = 0;

  if (PREDICT_FALSE(n_points > in_size)) {
    return FONT_COMPRESSION_FAILURE();
  }
  unsigned int triplet_index = 0;

  for (unsigned int i = 0; i < n_points; ++i) {
    uint8_t flag = flags_in[i];
    bool on_curve = !(flag >> 7);
    flag &= 0x7f;
    unsigned int n_data_bytes;
    if (flag < 84) {
      n_data_bytes = 1;
    } else if (flag < 120) {
      n_data_bytes = 2;
    } else if (flag < 124) {
      n_data_bytes = 3;
    } else {
      n_data_bytes = 4;
    }
    if (PREDICT_FALSE(triplet_index + n_data_bytes > in_size ||
        triplet_index + n_data_bytes < triplet_index)) {
      return FONT_COMPRESSION_FAILURE();
    }
    int dx, dy;
    if (flag < 10) {
      dx = 0;
      dy = WithSign(flag, ((flag & 14) << 7) + in[triplet_index]);
    } else if (flag < 20) {
      dx = WithSign(flag, (((flag - 10) & 14) << 7) + in[triplet_index]);
      dy = 0;
    } else if (flag < 84) {
      int b0 = flag - 20;
      int b1 = in[triplet_index];
      dx = WithSign(flag, 1 + (b0 & 0x30) + (b1 >> 4));
      dy = WithSign(flag >> 1, 1 + ((b0 & 0x0c) << 2) + (b1 & 0x0f));
    } else if (flag < 120) {
      int b0 = flag - 84;
      dx = WithSign(flag, 1 + ((b0 / 12) << 8) + in[triplet_index]);
      dy = WithSign(flag >> 1,
                    1 + (((b0 % 12) >> 2) << 8) + in[triplet_index + 1]);
    } else if (flag < 124) {
      int b2 = in[triplet_index + 1];
      dx = WithSign(flag, (in[triplet_index] << 4) + (b2 >> 4));
      dy = WithSign(flag >> 1, ((b2 & 0x0f) << 8) + in[triplet_index + 2]);
    } else {
      dx = WithSign(flag, (in[triplet_index] << 8) + in[triplet_index + 1]);
      dy = WithSign(flag >> 1,
          (in[triplet_index + 2] << 8) + in[triplet_index + 3]);
    }
    triplet_index += n_data_bytes;
    if (!_SafeIntAddition(x, dx, &x)) {
      return false;
    }
    if (!_SafeIntAddition(y, dy, &y)) {
      return false;
    }
    *result++ = {x, y, on_curve};
  }
  *in_bytes_consumed = triplet_index;
  return true;
}

// This function stores just the point data. On entry, dst points to the
// beginning of a simple glyph. Returns true on success.
bool StorePoints(unsigned int n_points, const Point* points,
    unsigned int n_contours, unsigned int instruction_length,
    uint8_t* dst, size_t dst_size, size_t* glyph_size) {
  // I believe that n_contours < 65536, in which case this is safe. However, a
  // comment and/or an assert would be good.
  unsigned int flag_offset = kEndPtsOfContoursOffset + 2 * n_contours + 2 +
    instruction_length;
  int last_flag = -1;
  int repeat_count = 0;
  int last_x = 0;
  int last_y = 0;
  unsigned int x_bytes = 0;
  unsigned int y_bytes = 0;

  for (unsigned int i = 0; i < n_points; ++i) {
    const Point& point = points[i];
    int flag = point.on_curve ? kGlyfOnCurve : 0;
    int dx = point.x - last_x;
    int dy = point.y - last_y;
    if (dx == 0) {
      flag |= kGlyfThisXIsSame;
    } else if (dx > -256 && dx < 256) {
      flag |= kGlyfXShort | (dx > 0 ? kGlyfThisXIsSame : 0);
      x_bytes += 1;
    } else {
      x_bytes += 2;
    }
    if (dy == 0) {
      flag |= kGlyfThisYIsSame;
    } else if (dy > -256 && dy < 256) {
      flag |= kGlyfYShort | (dy > 0 ? kGlyfThisYIsSame : 0);
      y_bytes += 1;
    } else {
      y_bytes += 2;
    }

    if (flag == last_flag && repeat_count != 255) {
      dst[flag_offset - 1] |= kGlyfRepeat;
      repeat_count++;
    } else {
      if (repeat_count != 0) {
        if (PREDICT_FALSE(flag_offset >= dst_size)) {
          return FONT_COMPRESSION_FAILURE();
        }
        dst[flag_offset++] = repeat_count;
      }
      if (PREDICT_FALSE(flag_offset >= dst_size)) {
        return FONT_COMPRESSION_FAILURE();
      }
      dst[flag_offset++] = flag;
      repeat_count = 0;
    }
    last_x = point.x;
    last_y = point.y;
    last_flag = flag;
  }

  if (repeat_count != 0) {
    if (PREDICT_FALSE(flag_offset >= dst_size)) {
      return FONT_COMPRESSION_FAILURE();
    }
    dst[flag_offset++] = repeat_count;
  }
  unsigned int xy_bytes = x_bytes + y_bytes;
  if (PREDICT_FALSE(xy_bytes < x_bytes ||
      flag_offset + xy_bytes < flag_offset ||
      flag_offset + xy_bytes > dst_size)) {
    return FONT_COMPRESSION_FAILURE();
  }

  int x_offset = flag_offset;
  int y_offset = flag_offset + x_bytes;
  last_x = 0;
  last_y = 0;
  for (unsigned int i = 0; i < n_points; ++i) {
    int dx = points[i].x - last_x;
    if (dx == 0) {
      // pass
    } else if (dx > -256 && dx < 256) {
      dst[x_offset++] = std::abs(dx);
    } else {
      // will always fit for valid input, but overflow is harmless
      x_offset = Store16(dst, x_offset, dx);
    }
    last_x += dx;
    int dy = points[i].y - last_y;
    if (dy == 0) {
      // pass
    } else if (dy > -256 && dy < 256) {
      dst[y_offset++] = std::abs(dy);
    } else {
      y_offset = Store16(dst, y_offset, dy);
    }
    last_y += dy;
  }
  *glyph_size = y_offset;
  return true;
}

// Compute the bounding box of the coordinates, and store into a glyf buffer.
// A precondition is that there are at least 10 bytes available.
// dst should point to the beginning of a 'glyf' record.
void ComputeBbox(unsigned int n_points, const Point* points, uint8_t* dst) {
  int x_min = 0;
  int y_min = 0;
  int x_max = 0;
  int y_max = 0;

  if (n_points > 0) {
    x_min = points[0].x;
    x_max = points[0].x;
    y_min = points[0].y;
    y_max = points[0].y;
  }
  for (unsigned int i = 1; i < n_points; ++i) {
    int x = points[i].x;
    int y = points[i].y;
    x_min = std::min(x, x_min);
    x_max = std::max(x, x_max);
    y_min = std::min(y, y_min);
    y_max = std::max(y, y_max);
  }
  size_t offset = 2;
  offset = Store16(dst, offset, x_min);
  offset = Store16(dst, offset, y_min);
  offset = Store16(dst, offset, x_max);
  offset = Store16(dst, offset, y_max);
}


bool SizeOfComposite(Buffer composite_stream, size_t* size,
                     bool* have_instructions) {
  size_t start_offset = composite_stream.offset();
  bool we_have_instructions = false;

  uint16_t flags = FLAG_MORE_COMPONENTS;
  while (flags & FLAG_MORE_COMPONENTS) {
    if (PREDICT_FALSE(!composite_stream.ReadU16(&flags))) {
      return FONT_COMPRESSION_FAILURE();
    }
    we_have_instructions |= (flags & FLAG_WE_HAVE_INSTRUCTIONS) != 0;
    size_t arg_size = 2;  // glyph index
    if (flags & FLAG_ARG_1_AND_2_ARE_WORDS) {
      arg_size += 4;
    } else {
      arg_size += 2;
    }
    if (flags & FLAG_WE_HAVE_A_SCALE) {
      arg_size += 2;
    } else if (flags & FLAG_WE_HAVE_AN_X_AND_Y_SCALE) {
      arg_size += 4;
    } else if (flags & FLAG_WE_HAVE_A_TWO_BY_TWO) {
      arg_size += 8;
    }
    if (PREDICT_FALSE(!composite_stream.Skip(arg_size))) {
      return FONT_COMPRESSION_FAILURE();
    }
  }

  *size = composite_stream.offset() - start_offset;
  *have_instructions = we_have_instructions;

  return true;
}

bool Pad4(WOFF2Out* out) {
  uint8_t zeroes[] = {0, 0, 0};
  if (PREDICT_FALSE(out->Size() + 3 < out->Size())) {
    return FONT_COMPRESSION_FAILURE();
  }
  uint32_t pad_bytes = Round4(out->Size()) - out->Size();
  if (pad_bytes > 0) {
    if (PREDICT_FALSE(!out->Write(&zeroes, pad_bytes))) {
      return FONT_COMPRESSION_FAILURE();
    }
  }
  return true;
}

// Build TrueType loca table
bool StoreLoca(const std::vector<uint32_t>& loca_values, int index_format,
               uint32_t* checksum, WOFF2Out* out) {
  // TODO(user) figure out what index format to use based on whether max
  // offset fits into uint16_t or not
  const uint64_t loca_size = loca_values.size();
  const uint64_t offset_size = index_format ? 4 : 2;
  if (PREDICT_FALSE((loca_size << 2) >> 2 != loca_size)) {
    return FONT_COMPRESSION_FAILURE();
  }
  std::vector<uint8_t> loca_content(loca_size * offset_size);
  uint8_t* dst = &loca_content[0];
  size_t offset = 0;
  for (size_t i = 0; i < loca_values.size(); ++i) {
    uint32_t value = loca_values[i];
    if (index_format) {
      offset = StoreU32(dst, offset, value);
    } else {
      offset = Store16(dst, offset, value >> 1);
    }
  }
  *checksum = ComputeULongSum(&loca_content[0], loca_content.size());
  if (PREDICT_FALSE(!out->Write(&loca_content[0], loca_content.size()))) {
    return FONT_COMPRESSION_FAILURE();
  }
  return true;
}

// Reconstruct entire glyf table based on transformed original
bool ReconstructGlyf(const uint8_t* data, Table* glyf_table,
                     uint32_t* glyf_checksum, Table * loca_table,
                     uint32_t* loca_checksum, WOFF2FontInfo* info,
                     WOFF2Out* out) {
  static const int kNumSubStreams = 7;
  Buffer file(data, glyf_table->transform_length);
  uint32_t version;
  std::vector<std::pair<const uint8_t*, size_t> > substreams(kNumSubStreams);
  const size_t glyf_start = out->Size();

  if (PREDICT_FALSE(!file.ReadU32(&version))) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (PREDICT_FALSE(!file.ReadU16(&info->num_glyphs) ||
      !file.ReadU16(&info->index_format))) {
    return FONT_COMPRESSION_FAILURE();
  }

  // https://dev.w3.org/webfonts/WOFF2/spec/#conform-mustRejectLoca
  // dst_length here is origLength in the spec
  uint32_t expected_loca_dst_length = (info->index_format ? 4 : 2)
    * (static_cast<uint32_t>(info->num_glyphs) + 1);
  if (PREDICT_FALSE(loca_table->dst_length != expected_loca_dst_length)) {
    return FONT_COMPRESSION_FAILURE();
  }

  unsigned int offset = (2 + kNumSubStreams) * 4;
  if (PREDICT_FALSE(offset > glyf_table->transform_length)) {
    return FONT_COMPRESSION_FAILURE();
  }
  // Invariant from here on: data_size >= offset
  for (int i = 0; i < kNumSubStreams; ++i) {
    uint32_t substream_size;
    if (PREDICT_FALSE(!file.ReadU32(&substream_size))) {
      return FONT_COMPRESSION_FAILURE();
    }
    if (PREDICT_FALSE(substream_size > glyf_table->transform_length - offset)) {
      return FONT_COMPRESSION_FAILURE();
    }
    substreams[i] = std::make_pair(data + offset, substream_size);
    offset += substream_size;
  }
  Buffer n_contour_stream(substreams[0].first, substreams[0].second);
  Buffer n_points_stream(substreams[1].first, substreams[1].second);
  Buffer flag_stream(substreams[2].first, substreams[2].second);
  Buffer glyph_stream(substreams[3].first, substreams[3].second);
  Buffer composite_stream(substreams[4].first, substreams[4].second);
  Buffer bbox_stream(substreams[5].first, substreams[5].second);
  Buffer instruction_stream(substreams[6].first, substreams[6].second);

  std::vector<uint32_t> loca_values(info->num_glyphs + 1);
  std::vector<unsigned int> n_points_vec;
  std::unique_ptr<Point[]> points;
  size_t points_size = 0;
  const uint8_t* bbox_bitmap = bbox_stream.buffer();
  // Safe because num_glyphs is bounded
  unsigned int bitmap_length = ((info->num_glyphs + 31) >> 5) << 2;
  if (!bbox_stream.Skip(bitmap_length)) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Temp buffer for glyph's.
  size_t glyph_buf_size = kDefaultGlyphBuf;
  std::unique_ptr<uint8_t[]> glyph_buf(new uint8_t[glyph_buf_size]);

  info->x_mins.resize(info->num_glyphs);
  for (unsigned int i = 0; i < info->num_glyphs; ++i) {
    size_t glyph_size = 0;
    uint16_t n_contours = 0;
    bool have_bbox = false;
    if (bbox_bitmap[i >> 3] & (0x80 >> (i & 7))) {
      have_bbox = true;
    }
    if (PREDICT_FALSE(!n_contour_stream.ReadU16(&n_contours))) {
      return FONT_COMPRESSION_FAILURE();
    }

    if (n_contours == 0xffff) {
      // composite glyph
      bool have_instructions = false;
      unsigned int instruction_size = 0;
      if (PREDICT_FALSE(!have_bbox)) {
        // composite glyphs must have an explicit bbox
        return FONT_COMPRESSION_FAILURE();
      }

      size_t composite_size;
      if (PREDICT_FALSE(!SizeOfComposite(composite_stream, &composite_size,
                                         &have_instructions))) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (have_instructions) {
        if (PREDICT_FALSE(!Read255UShort(&glyph_stream, &instruction_size))) {
          return FONT_COMPRESSION_FAILURE();
        }
      }

      size_t size_needed = 12 + composite_size + instruction_size;
      if (PREDICT_FALSE(glyph_buf_size < size_needed)) {
        glyph_buf.reset(new uint8_t[size_needed]);
        glyph_buf_size = size_needed;
      }

      glyph_size = Store16(glyph_buf.get(), glyph_size, n_contours);
      if (PREDICT_FALSE(!bbox_stream.Read(glyph_buf.get() + glyph_size, 8))) {
        return FONT_COMPRESSION_FAILURE();
      }
      glyph_size += 8;

      if (PREDICT_FALSE(!composite_stream.Read(glyph_buf.get() + glyph_size,
            composite_size))) {
        return FONT_COMPRESSION_FAILURE();
      }
      glyph_size += composite_size;
      if (have_instructions) {
        glyph_size = Store16(glyph_buf.get(), glyph_size, instruction_size);
        if (PREDICT_FALSE(!instruction_stream.Read(glyph_buf.get() + glyph_size,
              instruction_size))) {
          return FONT_COMPRESSION_FAILURE();
        }
        glyph_size += instruction_size;
      }
    } else if (n_contours > 0) {
      // simple glyph
      n_points_vec.clear();
      unsigned int total_n_points = 0;
      unsigned int n_points_contour;
      for (unsigned int j = 0; j < n_contours; ++j) {
        if (PREDICT_FALSE(
            !Read255UShort(&n_points_stream, &n_points_contour))) {
          return FONT_COMPRESSION_FAILURE();
        }
        n_points_vec.push_back(n_points_contour);
        if (PREDICT_FALSE(total_n_points + n_points_contour < total_n_points)) {
          return FONT_COMPRESSION_FAILURE();
        }
        total_n_points += n_points_contour;
      }
      unsigned int flag_size = total_n_points;
      if (PREDICT_FALSE(
          flag_size > flag_stream.length() - flag_stream.offset())) {
        return FONT_COMPRESSION_FAILURE();
      }
      const uint8_t* flags_buf = flag_stream.buffer() + flag_stream.offset();
      const uint8_t* triplet_buf = glyph_stream.buffer() +
        glyph_stream.offset();
      size_t triplet_size = glyph_stream.length() - glyph_stream.offset();
      size_t triplet_bytes_consumed = 0;
      if (points_size < total_n_points) {
        points_size = total_n_points;
        points.reset(new Point[points_size]);
      }
      if (PREDICT_FALSE(!TripletDecode(flags_buf, triplet_buf, triplet_size,
          total_n_points, points.get(), &triplet_bytes_consumed))) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (PREDICT_FALSE(!flag_stream.Skip(flag_size))) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (PREDICT_FALSE(!glyph_stream.Skip(triplet_bytes_consumed))) {
        return FONT_COMPRESSION_FAILURE();
      }
      unsigned int instruction_size;
      if (PREDICT_FALSE(!Read255UShort(&glyph_stream, &instruction_size))) {
        return FONT_COMPRESSION_FAILURE();
      }

      if (PREDICT_FALSE(total_n_points >= (1 << 27)
                        || instruction_size >= (1 << 30))) {
        return FONT_COMPRESSION_FAILURE();
      }
      size_t size_needed = 12 + 2 * n_contours + 5 * total_n_points
                           + instruction_size;
      if (PREDICT_FALSE(glyph_buf_size < size_needed)) {
        glyph_buf.reset(new uint8_t[size_needed]);
        glyph_buf_size = size_needed;
      }

      glyph_size = Store16(glyph_buf.get(), glyph_size, n_contours);
      if (have_bbox) {
        if (PREDICT_FALSE(!bbox_stream.Read(glyph_buf.get() + glyph_size, 8))) {
          return FONT_COMPRESSION_FAILURE();
        }
      } else {
        ComputeBbox(total_n_points, points.get(), glyph_buf.get());
      }
      glyph_size = kEndPtsOfContoursOffset;
      int end_point = -1;
      for (unsigned int contour_ix = 0; contour_ix < n_contours; ++contour_ix) {
        end_point += n_points_vec[contour_ix];
        if (PREDICT_FALSE(end_point >= 65536)) {
          return FONT_COMPRESSION_FAILURE();
        }
        glyph_size = Store16(glyph_buf.get(), glyph_size, end_point);
      }

      glyph_size = Store16(glyph_buf.get(), glyph_size, instruction_size);
      if (PREDICT_FALSE(!instruction_stream.Read(glyph_buf.get() + glyph_size,
                                                 instruction_size))) {
        return FONT_COMPRESSION_FAILURE();
      }
      glyph_size += instruction_size;

      if (PREDICT_FALSE(!StorePoints(total_n_points, points.get(), n_contours,
            instruction_size, glyph_buf.get(), glyph_buf_size, &glyph_size))) {
        return FONT_COMPRESSION_FAILURE();
      }
    } else {
      // n_contours == 0; empty glyph. Must NOT have a bbox.
      if (PREDICT_FALSE(have_bbox)) {
#ifdef FONT_COMPRESSION_BIN
        fprintf(stderr, "Empty glyph has a bbox\n");
#endif
        return FONT_COMPRESSION_FAILURE();
      }
    }

    loca_values[i] = out->Size() - glyf_start;
    if (PREDICT_FALSE(!out->Write(glyph_buf.get(), glyph_size))) {
      return FONT_COMPRESSION_FAILURE();
    }

    // TODO(user) Old code aligned glyphs ... but do we actually need to?
    if (PREDICT_FALSE(!Pad4(out))) {
      return FONT_COMPRESSION_FAILURE();
    }

    *glyf_checksum += ComputeULongSum(glyph_buf.get(), glyph_size);

    // We may need x_min to reconstruct 'hmtx'
    if (n_contours > 0) {
      Buffer x_min_buf(glyph_buf.get() + 2, 2);
      if (PREDICT_FALSE(!x_min_buf.ReadS16(&info->x_mins[i]))) {
        return FONT_COMPRESSION_FAILURE();
      }
    }
  }

  // glyf_table dst_offset was set by ReconstructFont
  glyf_table->dst_length = out->Size() - glyf_table->dst_offset;
  loca_table->dst_offset = out->Size();
  // loca[n] will be equal the length of the glyph data ('glyf') table
  loca_values[info->num_glyphs] = glyf_table->dst_length;
  if (PREDICT_FALSE(!StoreLoca(loca_values, info->index_format, loca_checksum,
      out))) {
    return FONT_COMPRESSION_FAILURE();
  }
  loca_table->dst_length = out->Size() - loca_table->dst_offset;

  return true;
}

Table* FindTable(std::vector<Table*>* tables, uint32_t tag) {
  for (Table* table : *tables) {
    if (table->tag == tag) {
      return table;
    }
  }
  return NULL;
}

// Get numberOfHMetrics, https://www.microsoft.com/typography/otspec/hhea.htm
bool ReadNumHMetrics(const uint8_t* data, size_t data_size,
                     uint16_t* num_hmetrics) {
  // Skip 34 to reach 'hhea' numberOfHMetrics
  Buffer buffer(data, data_size);
  if (PREDICT_FALSE(!buffer.Skip(34) || !buffer.ReadU16(num_hmetrics))) {
    return FONT_COMPRESSION_FAILURE();
  }
  return true;
}

// http://dev.w3.org/webfonts/WOFF2/spec/Overview.html#hmtx_table_format
bool ReconstructTransformedHmtx(const uint8_t* transformed_buf,
                                size_t transformed_size,
                                uint16_t num_glyphs,
                                uint16_t num_hmetrics,
                                const std::vector<int16_t>& x_mins,
                                uint32_t* checksum,
                                WOFF2Out* out) {
  Buffer hmtx_buff_in(transformed_buf, transformed_size);

  uint8_t hmtx_flags;
  if (PREDICT_FALSE(!hmtx_buff_in.ReadU8(&hmtx_flags))) {
    return FONT_COMPRESSION_FAILURE();
  }

  std::vector<uint16_t> advance_widths;
  std::vector<int16_t> lsbs;
  bool has_proportional_lsbs = (hmtx_flags & 1) == 0;
  bool has_monospace_lsbs = (hmtx_flags & 2) == 0;

  // Bits 2-7 are reserved and MUST be zero.
  if ((hmtx_flags & 0xFC) != 0) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "Illegal hmtx flags; bits 2-7 must be 0\n");
#endif
    return FONT_COMPRESSION_FAILURE();
  }

  // you say you transformed but there is little evidence of it
  if (has_proportional_lsbs && has_monospace_lsbs) {
    return FONT_COMPRESSION_FAILURE();
  }

  assert(x_mins.size() == num_glyphs);

  // num_glyphs 0 is OK if there is no 'glyf' but cannot then xform 'hmtx'.
  if (PREDICT_FALSE(num_hmetrics > num_glyphs)) {
    return FONT_COMPRESSION_FAILURE();
  }

  // https://www.microsoft.com/typography/otspec/hmtx.htm
  // "...only one entry need be in the array, but that entry is required."
  if (PREDICT_FALSE(num_hmetrics < 1)) {
    return FONT_COMPRESSION_FAILURE();
  }

  for (uint16_t i = 0; i < num_hmetrics; i++) {
    uint16_t advance_width;
    if (PREDICT_FALSE(!hmtx_buff_in.ReadU16(&advance_width))) {
      return FONT_COMPRESSION_FAILURE();
    }
    advance_widths.push_back(advance_width);
  }

  for (uint16_t i = 0; i < num_hmetrics; i++) {
    int16_t lsb;
    if (has_proportional_lsbs) {
      if (PREDICT_FALSE(!hmtx_buff_in.ReadS16(&lsb))) {
        return FONT_COMPRESSION_FAILURE();
      }
    } else {
      lsb = x_mins[i];
    }
    lsbs.push_back(lsb);
  }

  for (uint16_t i = num_hmetrics; i < num_glyphs; i++) {
    int16_t lsb;
    if (has_monospace_lsbs) {
      if (PREDICT_FALSE(!hmtx_buff_in.ReadS16(&lsb))) {
        return FONT_COMPRESSION_FAILURE();
      }
    } else {
      lsb = x_mins[i];
    }
    lsbs.push_back(lsb);
  }

  // bake me a shiny new hmtx table
  uint32_t hmtx_output_size = 2 * num_glyphs + 2 * num_hmetrics;
  std::vector<uint8_t> hmtx_table(hmtx_output_size);
  uint8_t* dst = &hmtx_table[0];
  size_t dst_offset = 0;
  for (uint32_t i = 0; i < num_glyphs; i++) {
    if (i < num_hmetrics) {
      Store16(advance_widths[i], &dst_offset, dst);
    }
    Store16(lsbs[i], &dst_offset, dst);
  }

  *checksum = ComputeULongSum(&hmtx_table[0], hmtx_output_size);
  if (PREDICT_FALSE(!out->Write(&hmtx_table[0], hmtx_output_size))) {
    return FONT_COMPRESSION_FAILURE();
  }

  return true;
}

bool Woff2Uncompress(uint8_t* dst_buf, size_t dst_size,
  const uint8_t* src_buf, size_t src_size) {
  size_t uncompressed_size = dst_size;
  BrotliDecoderResult result = BrotliDecoderDecompress(
      src_size, src_buf, &uncompressed_size, dst_buf);
  if (PREDICT_FALSE(result != BROTLI_DECODER_RESULT_SUCCESS ||
                    uncompressed_size != dst_size)) {
    return FONT_COMPRESSION_FAILURE();
  }
  return true;
}

bool ReadTableDirectory(Buffer* file, std::vector<Table>* tables,
    size_t num_tables) {
  uint32_t src_offset = 0;
  for (size_t i = 0; i < num_tables; ++i) {
    Table* table = &(*tables)[i];
    uint8_t flag_byte;
    if (PREDICT_FALSE(!file->ReadU8(&flag_byte))) {
      return FONT_COMPRESSION_FAILURE();
    }
    uint32_t tag;
    if ((flag_byte & 0x3f) == 0x3f) {
      if (PREDICT_FALSE(!file->ReadU32(&tag))) {
        return FONT_COMPRESSION_FAILURE();
      }
    } else {
      tag = kKnownTags[flag_byte & 0x3f];
    }
    uint32_t flags = 0;
    uint8_t xform_version = (flag_byte >> 6) & 0x03;

    // 0 means xform for glyph/loca, non-0 for others
    if (tag == kGlyfTableTag || tag == kLocaTableTag) {
      if (xform_version == 0) {
        flags |= kWoff2FlagsTransform;
      }
    } else if (xform_version != 0) {
      flags |= kWoff2FlagsTransform;
    }
    flags |= xform_version;

    uint32_t dst_length;
    if (PREDICT_FALSE(!ReadBase128(file, &dst_length))) {
      return FONT_COMPRESSION_FAILURE();
    }
    uint32_t transform_length = dst_length;
    if ((flags & kWoff2FlagsTransform) != 0) {
      if (PREDICT_FALSE(!ReadBase128(file, &transform_length))) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (PREDICT_FALSE(tag == kLocaTableTag && transform_length)) {
        return FONT_COMPRESSION_FAILURE();
      }
    }
    if (PREDICT_FALSE(src_offset + transform_length < src_offset)) {
      return FONT_COMPRESSION_FAILURE();
    }
    table->src_offset = src_offset;
    table->src_length = transform_length;
    src_offset += transform_length;

    table->tag = tag;
    table->flags = flags;
    table->transform_length = transform_length;
    table->dst_length = dst_length;
  }
  return true;
}

// Writes a single Offset Table entry
size_t StoreOffsetTable(uint8_t* result, size_t offset, uint32_t flavor,
                        uint16_t num_tables) {
  offset = StoreU32(result, offset, flavor);  // sfnt version
  offset = Store16(result, offset, num_tables);  // num_tables
  unsigned max_pow2 = 0;
  while (1u << (max_pow2 + 1) <= num_tables) {
    max_pow2++;
  }
  const uint16_t output_search_range = (1u << max_pow2) << 4;
  offset = Store16(result, offset, output_search_range);  // searchRange
  offset = Store16(result, offset, max_pow2);  // entrySelector
  // rangeShift
  offset = Store16(result, offset, (num_tables << 4) - output_search_range);
  return offset;
}

size_t StoreTableEntry(uint8_t* result, uint32_t offset, uint32_t tag) {
  offset = StoreU32(result, offset, tag);
  offset = StoreU32(result, offset, 0);
  offset = StoreU32(result, offset, 0);
  offset = StoreU32(result, offset, 0);
  return offset;
}

// First table goes after all the headers, table directory, etc
uint64_t ComputeOffsetToFirstTable(const WOFF2Header& hdr) {
  uint64_t offset = kSfntHeaderSize +
    kSfntEntrySize * static_cast<uint64_t>(hdr.num_tables);
  if (hdr.header_version) {
    offset = CollectionHeaderSize(hdr.header_version, hdr.ttc_fonts.size())
      + kSfntHeaderSize * hdr.ttc_fonts.size();
    for (const auto& ttc_font : hdr.ttc_fonts) {
      offset += kSfntEntrySize * ttc_font.table_indices.size();
    }
  }
  return offset;
}

std::vector<Table*> Tables(WOFF2Header* hdr, size_t font_index) {
  std::vector<Table*> tables;
  if (PREDICT_FALSE(hdr->header_version)) {
    for (auto index : hdr->ttc_fonts[font_index].table_indices) {
      tables.push_back(&hdr->tables[index]);
    }
  } else {
    for (auto& table : hdr->tables) {
      tables.push_back(&table);
    }
  }
  return tables;
}

// Offset tables assumed to have been written in with 0's initially.
// WOFF2Header isn't const so we can use [] instead of at() (which upsets FF)
bool ReconstructFont(uint8_t* transformed_buf,
                     const uint32_t transformed_buf_size,
                     RebuildMetadata* metadata,
                     WOFF2Header* hdr,
                     size_t font_index,
                     WOFF2Out* out) {
  size_t dest_offset = out->Size();
  uint8_t table_entry[12];
  WOFF2FontInfo* info = &metadata->font_infos[font_index];
  std::vector<Table*> tables = Tables(hdr, font_index);

  // 'glyf' without 'loca' doesn't make sense
  const Table* glyf_table = FindTable(&tables, kGlyfTableTag);
  const Table* loca_table = FindTable(&tables, kLocaTableTag);
  if (PREDICT_FALSE(static_cast<bool>(glyf_table) !=
                    static_cast<bool>(loca_table))) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Cannot have just one of glyf/loca\n");
#endif
    return FONT_COMPRESSION_FAILURE();
  }

  if (glyf_table != NULL) {
    if (PREDICT_FALSE((glyf_table->flags & kWoff2FlagsTransform)
                      != (loca_table->flags & kWoff2FlagsTransform))) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Cannot transform just one of glyf/loca\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
  }

  uint32_t font_checksum = metadata->header_checksum;
  if (hdr->header_version) {
    font_checksum = hdr->ttc_fonts[font_index].header_checksum;
  }

  uint32_t loca_checksum = 0;
  for (size_t i = 0; i < tables.size(); i++) {
    Table& table = *tables[i];

    std::pair<uint32_t, uint32_t> checksum_key = {table.tag, table.src_offset};
    bool reused = metadata->checksums.find(checksum_key)
               != metadata->checksums.end();
    if (PREDICT_FALSE(font_index == 0 && reused)) {
      return FONT_COMPRESSION_FAILURE();
    }

    // TODO(user) a collection with optimized hmtx that reused glyf/loca
    // would fail. We don't optimize hmtx for collections yet.
    if (PREDICT_FALSE(static_cast<uint64_t>(table.src_offset) + table.src_length
        > transformed_buf_size)) {
      return FONT_COMPRESSION_FAILURE();
    }

    if (table.tag == kHheaTableTag) {
      if (!ReadNumHMetrics(transformed_buf + table.src_offset,
          table.src_length, &info->num_hmetrics)) {
        return FONT_COMPRESSION_FAILURE();
      }
    }

    uint32_t checksum = 0;
    if (!reused) {
      if ((table.flags & kWoff2FlagsTransform) != kWoff2FlagsTransform) {
        if (table.tag == kHeadTableTag) {
          if (PREDICT_FALSE(table.src_length < 12)) {
            return FONT_COMPRESSION_FAILURE();
          }
          // checkSumAdjustment = 0
          StoreU32(transformed_buf + table.src_offset, 8, 0);
        }
        table.dst_offset = dest_offset;
        checksum = ComputeULongSum(transformed_buf + table.src_offset,
                                   table.src_length);
        if (PREDICT_FALSE(!out->Write(transformed_buf + table.src_offset,
            table.src_length))) {
          return FONT_COMPRESSION_FAILURE();
        }
      } else {
        if (table.tag == kGlyfTableTag) {
          table.dst_offset = dest_offset;

          Table* loca_table = FindTable(&tables, kLocaTableTag);
          if (PREDICT_FALSE(!ReconstructGlyf(transformed_buf + table.src_offset,
              &table, &checksum, loca_table, &loca_checksum, info, out))) {
            return FONT_COMPRESSION_FAILURE();
          }
        } else if (table.tag == kLocaTableTag) {
          // All the work was done by ReconstructGlyf. We already know checksum.
          checksum = loca_checksum;
        } else if (table.tag == kHmtxTableTag) {
          table.dst_offset = dest_offset;
          // Tables are sorted so all the info we need has been gathered.
          if (PREDICT_FALSE(!ReconstructTransformedHmtx(
              transformed_buf + table.src_offset, table.src_length,
              info->num_glyphs, info->num_hmetrics, info->x_mins, &checksum,
              out))) {
            return FONT_COMPRESSION_FAILURE();
          }
        } else {
          return FONT_COMPRESSION_FAILURE();  // transform unknown
        }
      }
      metadata->checksums[checksum_key] = checksum;
    } else {
      checksum = metadata->checksums[checksum_key];
    }
    font_checksum += checksum;

    // update the table entry with real values.
    StoreU32(table_entry, 0, checksum);
    StoreU32(table_entry, 4, table.dst_offset);
    StoreU32(table_entry, 8, table.dst_length);
    if (PREDICT_FALSE(!out->Write(table_entry,
        info->table_entry_by_tag[table.tag] + 4, 12))) {
      return FONT_COMPRESSION_FAILURE();
    }

    // We replaced 0's. Update overall checksum.
    font_checksum += ComputeULongSum(table_entry, 12);

    if (PREDICT_FALSE(!Pad4(out))) {
      return FONT_COMPRESSION_FAILURE();
    }

    if (PREDICT_FALSE(static_cast<uint64_t>(table.dst_offset + table.dst_length)
        > out->Size())) {
      return FONT_COMPRESSION_FAILURE();
    }
    dest_offset = out->Size();
  }

  // Update 'head' checkSumAdjustment. We already set it to 0 and summed font.
  Table* head_table = FindTable(&tables, kHeadTableTag);
  if (head_table) {
    if (PREDICT_FALSE(head_table->dst_length < 12)) {
      return FONT_COMPRESSION_FAILURE();
    }
    uint8_t checksum_adjustment[4];
    StoreU32(checksum_adjustment, 0, 0xB1B0AFBA - font_checksum);
    if (PREDICT_FALSE(!out->Write(checksum_adjustment,
                                  head_table->dst_offset + 8, 4))) {
      return FONT_COMPRESSION_FAILURE();
    }
  }

  return true;
}

bool ReadWOFF2Header(const uint8_t* data, size_t length, WOFF2Header* hdr) {
  Buffer file(data, length);

  uint32_t signature;
  if (PREDICT_FALSE(!file.ReadU32(&signature) || signature != kWoff2Signature ||
      !file.ReadU32(&hdr->flavor))) {
    return FONT_COMPRESSION_FAILURE();
  }

  // TODO(user): Should call IsValidVersionTag() here.

  uint32_t reported_length;
  if (PREDICT_FALSE(
      !file.ReadU32(&reported_length) || length != reported_length)) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (PREDICT_FALSE(!file.ReadU16(&hdr->num_tables) || !hdr->num_tables)) {
    return FONT_COMPRESSION_FAILURE();
  }

  // We don't care about these fields of the header:
  //   uint16_t reserved
  //   uint32_t total_sfnt_size, we don't believe this, will compute later
  if (PREDICT_FALSE(!file.Skip(6))) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (PREDICT_FALSE(!file.ReadU32(&hdr->compressed_length))) {
    return FONT_COMPRESSION_FAILURE();
  }
  // We don't care about these fields of the header:
  //   uint16_t major_version, minor_version
  if (PREDICT_FALSE(!file.Skip(2 * 2))) {
    return FONT_COMPRESSION_FAILURE();
  }
  uint32_t meta_offset;
  uint32_t meta_length;
  uint32_t meta_length_orig;
  if (PREDICT_FALSE(!file.ReadU32(&meta_offset) ||
      !file.ReadU32(&meta_length) ||
      !file.ReadU32(&meta_length_orig))) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (meta_offset) {
    if (PREDICT_FALSE(
        meta_offset >= length || length - meta_offset < meta_length)) {
      return FONT_COMPRESSION_FAILURE();
    }
  }
  uint32_t priv_offset;
  uint32_t priv_length;
  if (PREDICT_FALSE(!file.ReadU32(&priv_offset) ||
      !file.ReadU32(&priv_length))) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (priv_offset) {
    if (PREDICT_FALSE(
        priv_offset >= length || length - priv_offset < priv_length)) {
      return FONT_COMPRESSION_FAILURE();
    }
  }
  hdr->tables.resize(hdr->num_tables);
  if (PREDICT_FALSE(!ReadTableDirectory(
          &file, &hdr->tables, hdr->num_tables))) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Before we sort for output the last table end is the uncompressed size.
  Table& last_table = hdr->tables.back();
  hdr->uncompressed_size = last_table.src_offset + last_table.src_length;
  if (PREDICT_FALSE(hdr->uncompressed_size < last_table.src_offset)) {
    return FONT_COMPRESSION_FAILURE();
  }

  hdr->header_version = 0;

  if (hdr->flavor == kTtcFontFlavor) {
    if (PREDICT_FALSE(!file.ReadU32(&hdr->header_version))) {
      return FONT_COMPRESSION_FAILURE();
    }
    if (PREDICT_FALSE(hdr->header_version != 0x00010000
                   && hdr->header_version != 0x00020000)) {
      return FONT_COMPRESSION_FAILURE();
    }
    uint32_t num_fonts;
    if (PREDICT_FALSE(!Read255UShort(&file, &num_fonts) || !num_fonts)) {
      return FONT_COMPRESSION_FAILURE();
    }
    hdr->ttc_fonts.resize(num_fonts);

    for (uint32_t i = 0; i < num_fonts; i++) {
      TtcFont& ttc_font = hdr->ttc_fonts[i];
      uint32_t num_tables;
      if (PREDICT_FALSE(!Read255UShort(&file, &num_tables) || !num_tables)) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (PREDICT_FALSE(!file.ReadU32(&ttc_font.flavor))) {
        return FONT_COMPRESSION_FAILURE();
      }

      ttc_font.table_indices.resize(num_tables);


      unsigned int glyf_idx = 0;
      unsigned int loca_idx = 0;

      for (uint32_t j = 0; j < num_tables; j++) {
        unsigned int table_idx;
        if (PREDICT_FALSE(!Read255UShort(&file, &table_idx)) ||
            table_idx >= hdr->tables.size()) {
          return FONT_COMPRESSION_FAILURE();
        }
        ttc_font.table_indices[j] = table_idx;

        const Table& table = hdr->tables[table_idx];
        if (table.tag == kLocaTableTag) {
          loca_idx = table_idx;
        }
        if (table.tag == kGlyfTableTag) {
          glyf_idx = table_idx;
        }

      }

      // if we have both glyf and loca make sure they are consecutive
      // if we have just one we'll reject the font elsewhere
      if (glyf_idx > 0 || loca_idx > 0) {
        if (PREDICT_FALSE(glyf_idx > loca_idx || loca_idx - glyf_idx != 1)) {
#ifdef FONT_COMPRESSION_BIN
        fprintf(stderr, "TTC font %d has non-consecutive glyf/loca\n", i);
#endif
          return FONT_COMPRESSION_FAILURE();
        }
      }
    }
  }

  const uint64_t first_table_offset = ComputeOffsetToFirstTable(*hdr);

  hdr->compressed_offset = file.offset();
  if (PREDICT_FALSE(hdr->compressed_offset >
                    std::numeric_limits<uint32_t>::max())) {
    return FONT_COMPRESSION_FAILURE();
  }
  uint64_t src_offset = Round4(hdr->compressed_offset + hdr->compressed_length);
  uint64_t dst_offset = first_table_offset;


  if (PREDICT_FALSE(src_offset > length)) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "offset fail; src_offset %" PRIu64 " length %lu "
      "dst_offset %" PRIu64 "\n",
      src_offset, length, dst_offset);
#endif
    return FONT_COMPRESSION_FAILURE();
  }
  if (meta_offset) {
    if (PREDICT_FALSE(src_offset != meta_offset)) {
      return FONT_COMPRESSION_FAILURE();
    }
    src_offset = Round4(meta_offset + meta_length);
    if (PREDICT_FALSE(src_offset > std::numeric_limits<uint32_t>::max())) {
      return FONT_COMPRESSION_FAILURE();
    }
  }

  if (priv_offset) {
    if (PREDICT_FALSE(src_offset != priv_offset)) {
      return FONT_COMPRESSION_FAILURE();
    }
    src_offset = Round4(priv_offset + priv_length);
    if (PREDICT_FALSE(src_offset > std::numeric_limits<uint32_t>::max())) {
      return FONT_COMPRESSION_FAILURE();
    }
  }

  if (PREDICT_FALSE(src_offset != Round4(length))) {
    return FONT_COMPRESSION_FAILURE();
  }

  return true;
}

// Write everything before the actual table data
bool WriteHeaders(const uint8_t* data, size_t length, RebuildMetadata* metadata,
                  WOFF2Header* hdr, WOFF2Out* out) {
  std::vector<uint8_t> output(ComputeOffsetToFirstTable(*hdr), 0);

  // Re-order tables in output (OTSpec) order
  std::vector<Table> sorted_tables(hdr->tables);
  if (hdr->header_version) {
    // collection; we have to sort the table offset vector in each font
    for (auto& ttc_font : hdr->ttc_fonts) {
      std::map<uint32_t, uint16_t> sorted_index_by_tag;
      for (auto table_index : ttc_font.table_indices) {
        sorted_index_by_tag[hdr->tables[table_index].tag] = table_index;
      }
      uint16_t index = 0;
      for (auto& i : sorted_index_by_tag) {
        ttc_font.table_indices[index++] = i.second;
      }
    }
  } else {
    // non-collection; we can just sort the tables
    std::sort(sorted_tables.begin(), sorted_tables.end());
  }

  // Start building the font
  uint8_t* result = &output[0];
  size_t offset = 0;
  if (hdr->header_version) {
    // TTC header
    offset = StoreU32(result, offset, hdr->flavor);  // TAG TTCTag
    offset = StoreU32(result, offset, hdr->header_version);  // FIXED Version
    offset = StoreU32(result, offset, hdr->ttc_fonts.size());  // ULONG numFonts
    // Space for ULONG OffsetTable[numFonts] (zeroed initially)
    size_t offset_table = offset;  // keep start of offset table for later
    for (size_t i = 0; i < hdr->ttc_fonts.size(); i++) {
      offset = StoreU32(result, offset, 0);  // will fill real values in later
    }
    // space for DSIG fields for header v2
    if (hdr->header_version == 0x00020000) {
      offset = StoreU32(result, offset, 0);  // ULONG ulDsigTag
      offset = StoreU32(result, offset, 0);  // ULONG ulDsigLength
      offset = StoreU32(result, offset, 0);  // ULONG ulDsigOffset
    }

    // write Offset Tables and store the location of each in TTC Header
    metadata->font_infos.resize(hdr->ttc_fonts.size());
    for (size_t i = 0; i < hdr->ttc_fonts.size(); i++) {
      TtcFont& ttc_font = hdr->ttc_fonts[i];

      // write Offset Table location into TTC Header
      offset_table = StoreU32(result, offset_table, offset);

      // write the actual offset table so our header doesn't lie
      ttc_font.dst_offset = offset;
      offset = StoreOffsetTable(result, offset, ttc_font.flavor,
                                ttc_font.table_indices.size());

      for (const auto table_index : ttc_font.table_indices) {
        uint32_t tag = hdr->tables[table_index].tag;
        metadata->font_infos[i].table_entry_by_tag[tag] = offset;
        offset = StoreTableEntry(result, offset, tag);
      }

      ttc_font.header_checksum = ComputeULongSum(&output[ttc_font.dst_offset],
                                                 offset - ttc_font.dst_offset);
    }
  } else {
    metadata->font_infos.resize(1);
    offset = StoreOffsetTable(result, offset, hdr->flavor, hdr->num_tables);
    for (uint16_t i = 0; i < hdr->num_tables; ++i) {
      metadata->font_infos[0].table_entry_by_tag[sorted_tables[i].tag] = offset;
      offset = StoreTableEntry(result, offset, sorted_tables[i].tag);
    }
  }

  if (PREDICT_FALSE(!out->Write(&output[0], output.size()))) {
    return FONT_COMPRESSION_FAILURE();
  }
  metadata->header_checksum = ComputeULongSum(&output[0], output.size());
  return true;
}

}  // namespace

size_t ComputeWOFF2FinalSize(const uint8_t* data, size_t length) {
  Buffer file(data, length);
  uint32_t total_length;

  if (!file.Skip(16) ||
      !file.ReadU32(&total_length)) {
    return 0;
  }
  return total_length;
}

bool ConvertWOFF2ToTTF(uint8_t *result, size_t result_length,
                       const uint8_t *data, size_t length) {
  WOFF2MemoryOut out(result, result_length);
  return ConvertWOFF2ToTTF(data, length, &out);
}

bool ConvertWOFF2ToTTF(const uint8_t* data, size_t length,
                       WOFF2Out* out) {
  RebuildMetadata metadata;
  WOFF2Header hdr;
  if (!ReadWOFF2Header(data, length, &hdr)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (!WriteHeaders(data, length, &metadata, &hdr, out)) {
    return FONT_COMPRESSION_FAILURE();
  }

  const float compression_ratio = (float) hdr.uncompressed_size / length;
  if (compression_ratio > kMaxPlausibleCompressionRatio) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "Implausible compression ratio %.01f\n", compression_ratio);
#endif
    return FONT_COMPRESSION_FAILURE();
  }

  const uint8_t* src_buf = data + hdr.compressed_offset;
  std::vector<uint8_t> uncompressed_buf(hdr.uncompressed_size);
  if (PREDICT_FALSE(hdr.uncompressed_size < 1)) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (PREDICT_FALSE(!Woff2Uncompress(&uncompressed_buf[0],
                                     hdr.uncompressed_size, src_buf,
                                     hdr.compressed_length))) {
    return FONT_COMPRESSION_FAILURE();
  }

  for (size_t i = 0; i < metadata.font_infos.size(); i++) {
    if (PREDICT_FALSE(!ReconstructFont(&uncompressed_buf[0],
                                       hdr.uncompressed_size,
                                       &metadata, &hdr, i, out))) {
      return FONT_COMPRESSION_FAILURE();
    }
  }

  return true;
}

} // namespace woff2
