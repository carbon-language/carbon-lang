/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting WOFF2 format font files to their TTF versions. */

#include <brotli/decode.h>
#include <woff2/decode.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
let kGlyfOnCurve: const int = 1 << 0;
let kGlyfXShort: const int = 1 << 1;
let kGlyfYShort: const int = 1 << 2;
let kGlyfRepeat: const int = 1 << 3;
let kGlyfThisXIsSame: const int = 1 << 4;
let kGlyfThisYIsSame: const int = 1 << 5;

// composite glyph flags
// See CompositeGlyph.java in sfntly for full definitions
let FLAG_ARG_1_AND_2_ARE_WORDS: const int = 1 << 0;
let FLAG_WE_HAVE_A_SCALE: const int = 1 << 3;
let FLAG_MORE_COMPONENTS: const int = 1 << 5;
let FLAG_WE_HAVE_AN_X_AND_Y_SCALE: const int = 1 << 6;
let FLAG_WE_HAVE_A_TWO_BY_TWO: const int = 1 << 7;
let FLAG_WE_HAVE_INSTRUCTIONS: const int = 1 << 8;

let kCheckSumAdjustmentOffset: const size_t = 8;

let kEndPtsOfContoursOffset: const size_t = 10;
let kCompositeGlyphBegin: const size_t = 10;

// 98% of Google Fonts have no glyph above 5k bytes
// Largest glyph ever observed was 72k bytes
let kDefaultGlyphBuf: const size_t = 5120;

// Over 14k test fonts the max compression ratio seen to date was ~20.
// >100 suggests you wrote a bad uncompressed size.
let kMaxPlausibleCompressionRatio: const float = 100.0;

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

fn WithSign(flag: int, baseval: int) -> int {
  // Precondition: 0 <= baseval < 65536 (to avoid integer overflow)
  return (flag & 1) ? baseval : -baseval;
}

fn _SafeIntAddition(a: int, b: int, result: int*) -> bool {
  if (PREDICT_FALSE(
          ((a > 0) && (b > std::numeric_limits<int>::max() - a)) ||
          ((a < 0) && (b < std::numeric_limits<int>::min() - a)))) {
    return false;
  }
  *result = a + b;
  return true;
}

fn TripletDecode(flags_in: const uint8_t*, in: const uint8_t*, in_size: size_t,
    n_points: unsigned int, result: Point*, in_bytes_consumed: size_t*) -> bool {
  var x: int = 0;
  var y: int = 0;

  if (PREDICT_FALSE(n_points > in_size)) {
    return FONT_COMPRESSION_FAILURE();
  }
  var triplet_index: unsigned int = 0;

  for (var i: unsigned int = 0; i < n_points; ++i) {
    var flag: uint8_t = flags_in[i];
    var on_curve: bool = !(flag >> 7);
    flag &= 0x7f;
    var n_data_bytes: unsigned int;
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
    var dx: int, var dy: int;
    if (flag < 10) {
      dx = 0;
      dy = WithSign(flag, ((flag & 14) << 7) + in[triplet_index]);
    } else if (flag < 20) {
      dx = WithSign(flag, (((flag - 10) & 14) << 7) + in[triplet_index]);
      dy = 0;
    } else if (flag < 84) {
      var b0: int = flag - 20;
      var b1: int = in[triplet_index];
      dx = WithSign(flag, 1 + (b0 & 0x30) + (b1 >> 4));
      dy = WithSign(flag >> 1, 1 + ((b0 & 0x0c) << 2) + (b1 & 0x0f));
    } else if (flag < 120) {
      var b0: int = flag - 84;
      dx = WithSign(flag, 1 + ((b0 / 12) << 8) + in[triplet_index]);
      dy = WithSign(flag >> 1,
                    1 + (((b0 % 12) >> 2) << 8) + in[triplet_index + 1]);
    } else if (flag < 124) {
      var b2: int = in[triplet_index + 1];
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
fn StorePoints(n_points: unsigned int, points: const Point*,
    n_contours: unsigned int, instruction_length: unsigned int,
    dst: uint8_t*, dst_size: size_t, glyph_size: size_t*) -> bool {
  // I believe that n_contours < 65536, in which case this is safe. However, a
  // comment and/or an assert would be good.
  var flag_offset: unsigned int = kEndPtsOfContoursOffset + 2 * n_contours + 2 +
    instruction_length;
  var last_flag: int = -1;
  var repeat_count: int = 0;
  var last_x: int = 0;
  var last_y: int = 0;
  var x_bytes: unsigned int = 0;
  var y_bytes: unsigned int = 0;

  for (var i: unsigned int = 0; i < n_points; ++i) {
    var point: const Point& = points[i];
    var flag: int = point.on_curve ? kGlyfOnCurve : 0;
    var dx: int = point.x - last_x;
    var dy: int = point.y - last_y;
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
  var xy_bytes: unsigned int = x_bytes + y_bytes;
  if (PREDICT_FALSE(xy_bytes < x_bytes ||
      flag_offset + xy_bytes < flag_offset ||
      flag_offset + xy_bytes > dst_size)) {
    return FONT_COMPRESSION_FAILURE();
  }

  var x_offset: int = flag_offset;
  var y_offset: int = flag_offset + x_bytes;
  last_x = 0;
  last_y = 0;
  for (var i: unsigned int = 0; i < n_points; ++i) {
    var dx: int = points[i].x - last_x;
    if (dx == 0) {
      // pass
    } else if (dx > -256 && dx < 256) {
      dst[x_offset++] = std::abs(dx);
    } else {
      // will always fit for valid input, but overflow is harmless
      x_offset = Store16(dst, x_offset, dx);
    }
    last_x += dx;
    var dy: int = points[i].y - last_y;
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
fn ComputeBbox(n_points: unsigned int, points: const Point*, dst: uint8_t*) {
  var x_min: int = 0;
  var y_min: int = 0;
  var x_max: int = 0;
  var y_max: int = 0;

  if (n_points > 0) {
    x_min = points[0].x;
    x_max = points[0].x;
    y_min = points[0].y;
    y_max = points[0].y;
  }
  for (var i: unsigned int = 1; i < n_points; ++i) {
    var x: int = points[i].x;
    var y: int = points[i].y;
    x_min = std::min(x, x_min);
    x_max = std::max(x, x_max);
    y_min = std::min(y, y_min);
    y_max = std::max(y, y_max);
  }
  var offset: size_t = 2;
  offset = Store16(dst, offset, x_min);
  offset = Store16(dst, offset, y_min);
  offset = Store16(dst, offset, x_max);
  offset = Store16(dst, offset, y_max);
}


fn SizeOfComposite(composite_stream: Buffer, size: size_t*,
                     have_instructions: bool*) -> bool {
  var start_offset: size_t = composite_stream.offset();
  var we_have_instructions: bool = false;

  var flags: uint16_t = FLAG_MORE_COMPONENTS;
  while (flags & FLAG_MORE_COMPONENTS) {
    if (PREDICT_FALSE(!composite_stream.ReadU16(&flags))) {
      return FONT_COMPRESSION_FAILURE();
    }
    we_have_instructions |= (flags & FLAG_WE_HAVE_INSTRUCTIONS) != 0;
    var arg_size: size_t = 2;  // glyph index
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

fn Pad4(out: WOFF2Out*) -> bool {
  var zeroes: uint8_t[] = {0, 0, 0};
  if (PREDICT_FALSE(out->Size() + 3 < out->Size())) {
    return FONT_COMPRESSION_FAILURE();
  }
  var pad_bytes: uint32_t = Round4(out->Size()) - out->Size();
  if (pad_bytes > 0) {
    if (PREDICT_FALSE(!out->Write(&zeroes, pad_bytes))) {
      return FONT_COMPRESSION_FAILURE();
    }
  }
  return true;
}

// Build TrueType loca table
fn StoreLoca(loca_values: const std::vector<uint32_t>&, index_format: int,
               checksum: uint32_t*, out: WOFF2Out*) -> bool {
  // TODO(user) figure out what index format to use based on whether max
  // offset fits into uint16_t or not
  let loca_size: const uint64_t = loca_values.size();
  let offset_size: const uint64_t = index_format ? 4 : 2;
  if (PREDICT_FALSE((loca_size << 2) >> 2 != loca_size)) {
    return FONT_COMPRESSION_FAILURE();
  }
  var loca_content: std::vector<uint8_t>(loca_size * offset_size);
  var dst: uint8_t* = &loca_content[0];
  var offset: size_t = 0;
  for (unsigned int value  in  loca_values) {
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
fn ReconstructGlyf(data: const uint8_t*, glyf_table: Table*,
                     glyf_checksum: uint32_t*, loca_table: Table*,
                     loca_checksum: uint32_t*, info: WOFF2FontInfo*,
                     out: WOFF2Out*) -> bool {
  let kNumSubStreams: const int = 7;
  var file: Buffer(data, glyf_table->transform_length);
  var version: uint32_t;
  var substreams: std::vector<std::pair<const uint8_t*, size_t> >(kNumSubStreams);
  let glyf_start: const size_t = out->Size();

  if (PREDICT_FALSE(!file.ReadU32(&version))) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (PREDICT_FALSE(!file.ReadU16(&info->num_glyphs) ||
      !file.ReadU16(&info->index_format))) {
    return FONT_COMPRESSION_FAILURE();
  }

  // https://dev.w3.org/webfonts/WOFF2/spec/#conform-mustRejectLoca
  // dst_length here is origLength in the spec
  var expected_loca_dst_length: uint32_t = (info->index_format ? 4 : 2)
    * (static_cast<uint32_t>(info->num_glyphs) + 1);
  if (PREDICT_FALSE(loca_table->dst_length != expected_loca_dst_length)) {
    return FONT_COMPRESSION_FAILURE();
  }

  var offset: unsigned int = (2 + kNumSubStreams) * 4;
  if (PREDICT_FALSE(offset > glyf_table->transform_length)) {
    return FONT_COMPRESSION_FAILURE();
  }
  // Invariant from here on: data_size >= offset
  for (var i: int = 0; i < kNumSubStreams; ++i) {
    var substream_size: uint32_t;
    if (PREDICT_FALSE(!file.ReadU32(&substream_size))) {
      return FONT_COMPRESSION_FAILURE();
    }
    if (PREDICT_FALSE(substream_size > glyf_table->transform_length - offset)) {
      return FONT_COMPRESSION_FAILURE();
    }
    substreams[i] = std::make_pair(data + offset, substream_size);
    offset += substream_size;
  }
  var n_contour_stream: Buffer(substreams[0].first, substreams[0].second);
  var n_points_stream: Buffer(substreams[1].first, substreams[1].second);
  var flag_stream: Buffer(substreams[2].first, substreams[2].second);
  var glyph_stream: Buffer(substreams[3].first, substreams[3].second);
  var composite_stream: Buffer(substreams[4].first, substreams[4].second);
  var bbox_stream: Buffer(substreams[5].first, substreams[5].second);
  var instruction_stream: Buffer(substreams[6].first, substreams[6].second);

  var loca_values: std::vector<uint32_t>(info->num_glyphs + 1);
  var n_points_vec: std::vector<unsigned int>;
  var points: std::unique_ptr<Point[]>;
  var points_size: size_t = 0;
  var bbox_bitmap: const uint8_t* = bbox_stream.buffer();
  // Safe because num_glyphs is bounded
  var bitmap_length: unsigned int = ((info->num_glyphs + 31) >> 5) << 2;
  if (!bbox_stream.Skip(bitmap_length)) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Temp buffer for glyph's.
  var glyph_buf_size: size_t = kDefaultGlyphBuf;
  var glyph_buf: std::unique_ptr<uint8_t[]>(new uint8_t[glyph_buf_size]);

  info->x_mins.resize(info->num_glyphs);
  for (var i: unsigned int = 0; i < info->num_glyphs; ++i) {
    var glyph_size: size_t = 0;
    var n_contours: uint16_t = 0;
    var have_bbox: bool = false;
    if (bbox_bitmap[i >> 3] & (0x80 >> (i & 7))) {
      have_bbox = true;
    }
    if (PREDICT_FALSE(!n_contour_stream.ReadU16(&n_contours))) {
      return FONT_COMPRESSION_FAILURE();
    }

    if (n_contours == 0xffff) {
      // composite glyph
      var have_instructions: bool = false;
      var instruction_size: unsigned int = 0;
      if (PREDICT_FALSE(!have_bbox)) {
        // composite glyphs must have an explicit bbox
        return FONT_COMPRESSION_FAILURE();
      }

      var composite_size: size_t;
      if (PREDICT_FALSE(!SizeOfComposite(composite_stream, &composite_size,
                                         &have_instructions))) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (have_instructions) {
        if (PREDICT_FALSE(!Read255UShort(&glyph_stream, &instruction_size))) {
          return FONT_COMPRESSION_FAILURE();
        }
      }

      var size_needed: size_t = 12 + composite_size + instruction_size;
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
      var total_n_points: unsigned int = 0;
      var n_points_contour: unsigned int;
      for (var j: unsigned int = 0; j < n_contours; ++j) {
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
      var flag_size: unsigned int = total_n_points;
      if (PREDICT_FALSE(
          flag_size > flag_stream.length() - flag_stream.offset())) {
        return FONT_COMPRESSION_FAILURE();
      }
      var flags_buf: const uint8_t* = flag_stream.buffer() + flag_stream.offset();
      var triplet_buf: const uint8_t* = glyph_stream.buffer() +
        glyph_stream.offset();
      var triplet_size: size_t = glyph_stream.length() - glyph_stream.offset();
      var triplet_bytes_consumed: size_t = 0;
      if (points_size < total_n_points) {
        points_size = total_n_points;
        points = std::make_unique<Point[]>(points_size);
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
      var instruction_size: unsigned int;
      if (PREDICT_FALSE(!Read255UShort(&glyph_stream, &instruction_size))) {
        return FONT_COMPRESSION_FAILURE();
      }

      if (PREDICT_FALSE(total_n_points >= (1 << 27)
                        || instruction_size >= (1 << 30))) {
        return FONT_COMPRESSION_FAILURE();
      }
      var size_needed: size_t = 12 + 2 * n_contours + 5 * total_n_points
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
      var end_point: int = -1;
      for (var contour_ix: unsigned int = 0; contour_ix < n_contours; ++contour_ix) {
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
      var x_min_buf: Buffer(glyph_buf.get() + 2, 2);
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

fn FindTable(tables: std::vector<Table*>*, tag: uint32_t) -> Table* {
  for (Table* table  in  *tables) {
    if (table->tag == tag) {
      return table;
    }
  }
  return nullptr;
}

// Get numberOfHMetrics, https://www.microsoft.com/typography/otspec/hhea.htm
fn ReadNumHMetrics(data: const uint8_t*, data_size: size_t,
                     num_hmetrics: uint16_t*) -> bool {
  // Skip 34 to reach 'hhea' numberOfHMetrics
  var buffer: Buffer(data, data_size);
  if (PREDICT_FALSE(!buffer.Skip(34) || !buffer.ReadU16(num_hmetrics))) {
    return FONT_COMPRESSION_FAILURE();
  }
  return true;
}

// http://dev.w3.org/webfonts/WOFF2/spec/Overview.html#hmtx_table_format
fn ReconstructTransformedHmtx(transformed_buf: const uint8_t*,
                                transformed_size: size_t,
                                num_glyphs: uint16_t,
                                num_hmetrics: uint16_t,
                                x_mins: const std::vector<int16_t>&,
                                checksum: uint32_t*,
                                out: WOFF2Out*) -> bool {
  var hmtx_buff_in: Buffer(transformed_buf, transformed_size);

  var hmtx_flags: uint8_t;
  if (PREDICT_FALSE(!hmtx_buff_in.ReadU8(&hmtx_flags))) {
    return FONT_COMPRESSION_FAILURE();
  }

  var advance_widths: std::vector<uint16_t>;
  var lsbs: std::vector<int16_t>;
  var has_proportional_lsbs: bool = (hmtx_flags & 1) == 0;
  var has_monospace_lsbs: bool = (hmtx_flags & 2) == 0;

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

  for (var i: uint16_t = 0; i < num_hmetrics; i++) {
    var advance_width: uint16_t;
    if (PREDICT_FALSE(!hmtx_buff_in.ReadU16(&advance_width))) {
      return FONT_COMPRESSION_FAILURE();
    }
    advance_widths.push_back(advance_width);
  }

  for (var i: uint16_t = 0; i < num_hmetrics; i++) {
    var lsb: int16_t;
    if (has_proportional_lsbs) {
      if (PREDICT_FALSE(!hmtx_buff_in.ReadS16(&lsb))) {
        return FONT_COMPRESSION_FAILURE();
      }
    } else {
      lsb = x_mins[i];
    }
    lsbs.push_back(lsb);
  }

  for (var i: uint16_t = num_hmetrics; i < num_glyphs; i++) {
    var lsb: int16_t;
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
  var hmtx_output_size: uint32_t = 2 * num_glyphs + 2 * num_hmetrics;
  var hmtx_table: std::vector<uint8_t>(hmtx_output_size);
  var dst: uint8_t* = &hmtx_table[0];
  var dst_offset: size_t = 0;
  for (var i: uint32_t = 0; i < num_glyphs; i++) {
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

fn Woff2Uncompress(dst_buf: uint8_t*, dst_size: size_t,
  src_buf: const uint8_t*, src_size: size_t) -> bool {
  var uncompressed_size: size_t = dst_size;
  var result: BrotliDecoderResult = BrotliDecoderDecompress(
      src_size, src_buf, &uncompressed_size, dst_buf);
  if (PREDICT_FALSE(result != BROTLI_DECODER_RESULT_SUCCESS ||
                    uncompressed_size != dst_size)) {
    return FONT_COMPRESSION_FAILURE();
  }
  return true;
}

fn ReadTableDirectory(file: Buffer*, tables: std::vector<Table>*,
    num_tables: size_t) -> bool {
  var src_offset: uint32_t = 0;
  for (var i: size_t = 0; i < num_tables; ++i) {
    var table: Table* = &(*tables)[i];
    var flag_byte: uint8_t;
    if (PREDICT_FALSE(!file->ReadU8(&flag_byte))) {
      return FONT_COMPRESSION_FAILURE();
    }
    var tag: uint32_t;
    if ((flag_byte & 0x3f) == 0x3f) {
      if (PREDICT_FALSE(!file->ReadU32(&tag))) {
        return FONT_COMPRESSION_FAILURE();
      }
    } else {
      tag = kKnownTags[flag_byte & 0x3f];
    }
    var flags: uint32_t = 0;
    var xform_version: uint8_t = (flag_byte >> 6) & 0x03;

    // 0 means xform for glyph/loca, non-0 for others
    if (tag == kGlyfTableTag || tag == kLocaTableTag) {
      if (xform_version == 0) {
        flags |= kWoff2FlagsTransform;
      }
    } else if (xform_version != 0) {
      flags |= kWoff2FlagsTransform;
    }
    flags |= xform_version;

    var dst_length: uint32_t;
    if (PREDICT_FALSE(!ReadBase128(file, &dst_length))) {
      return FONT_COMPRESSION_FAILURE();
    }
    var transform_length: uint32_t = dst_length;
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
fn StoreOffsetTable(result: uint8_t*, offset: size_t, flavor: uint32_t,
                        num_tables: uint16_t) -> size_t {
  offset = StoreU32(result, offset, flavor);  // sfnt version
  offset = Store16(result, offset, num_tables);  // num_tables
  var max_pow2: unsigned = 0;
  while (1u << (max_pow2 + 1) <= num_tables) {
    max_pow2++;
  }
  let output_search_range: const uint16_t = (1u << max_pow2) << 4;
  offset = Store16(result, offset, output_search_range);  // searchRange
  offset = Store16(result, offset, max_pow2);  // entrySelector
  // rangeShift
  offset = Store16(result, offset, (num_tables << 4) - output_search_range);
  return offset;
}

fn StoreTableEntry(result: uint8_t*, offset: uint32_t, tag: uint32_t) -> size_t {
  offset = StoreU32(result, offset, tag);
  offset = StoreU32(result, offset, 0);
  offset = StoreU32(result, offset, 0);
  offset = StoreU32(result, offset, 0);
  return offset;
}

// First table goes after all the headers, table directory, etc
fn ComputeOffsetToFirstTable(hdr: const WOFF2Header&) -> uint64_t {
  var offset: uint64_t = kSfntHeaderSize +
    kSfntEntrySize * static_cast<uint64_t>(hdr.num_tables);
  if (hdr.header_version) {
    offset = CollectionHeaderSize(hdr.header_version, hdr.ttc_fonts.size())
      + kSfntHeaderSize * hdr.ttc_fonts.size();
    for (const auto& ttc_font  in  hdr.ttc_fonts) {
      offset += kSfntEntrySize * ttc_font.table_indices.size();
    }
  }
  return offset;
}

fn Tables(hdr: WOFF2Header*, font_index: size_t) -> std::vector<Table*> {
  var tables: std::vector<Table*>;
  if (PREDICT_FALSE(hdr->header_version)) {
    for (auto index  in  hdr->ttc_fonts[font_index].table_indices) {
      tables.push_back(&hdr->tables[index]);
    }
  } else {
    for (auto& table  in  hdr->tables) {
      tables.push_back(&table);
    }
  }
  return tables;
}

// Offset tables assumed to have been written in with 0's initially.
// WOFF2Header isn't const so we can use [] instead of at() (which upsets FF)
fn ReconstructFont(transformed_buf: uint8_t*,
                     let transformed_buf_size: const uint32_t,
                     metadata: RebuildMetadata*,
                     hdr: WOFF2Header*,
                     font_index: size_t,
                     out: WOFF2Out*) -> bool {
  var dest_offset: size_t = out->Size();
  var table_entry: uint8_t[12];
  var info: WOFF2FontInfo* = &metadata->font_infos[font_index];
  var tables: std::vector<Table*> = Tables(hdr, font_index);

  // 'glyf' without 'loca' doesn't make sense
  var glyf_table: const Table* = FindTable(&tables, kGlyfTableTag);
  var loca_table: const Table* = FindTable(&tables, kLocaTableTag);
  if (PREDICT_FALSE(static_cast<bool>(glyf_table) !=
                    static_cast<bool>(loca_table))) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Cannot have just one of glyf/loca\n");
#endif
    return FONT_COMPRESSION_FAILURE();
  }

  if (glyf_table != nullptr) {
    if (PREDICT_FALSE((glyf_table->flags & kWoff2FlagsTransform)
                      != (loca_table->flags & kWoff2FlagsTransform))) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Cannot transform just one of glyf/loca\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
  }

  var font_checksum: uint32_t = metadata->header_checksum;
  if (hdr->header_version) {
    font_checksum = hdr->ttc_fonts[font_index].header_checksum;
  }

  var loca_checksum: uint32_t = 0;
  for (var i: size_t = 0; i < tables.size(); i++) {
    var table: Table& = *tables[i];

    var checksum_key: std::pair<uint32_t, uint32_t> = {table.tag, table.src_offset};
    var reused: bool = metadata->checksums.find(checksum_key)
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

    var checksum: uint32_t = 0;
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

          var loca_table: Table* = FindTable(&tables, kLocaTableTag);
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
  var head_table: Table* = FindTable(&tables, kHeadTableTag);
  if (head_table) {
    if (PREDICT_FALSE(head_table->dst_length < 12)) {
      return FONT_COMPRESSION_FAILURE();
    }
    var checksum_adjustment: uint8_t[4];
    StoreU32(checksum_adjustment, 0, 0xB1B0AFBA - font_checksum);
    if (PREDICT_FALSE(!out->Write(checksum_adjustment,
                                  head_table->dst_offset + 8, 4))) {
      return FONT_COMPRESSION_FAILURE();
    }
  }

  return true;
}

fn ReadWOFF2Header(data: const uint8_t*, length: size_t, hdr: WOFF2Header*) -> bool {
  var file: Buffer(data, length);

  var signature: uint32_t;
  if (PREDICT_FALSE(!file.ReadU32(&signature) || signature != kWoff2Signature ||
      !file.ReadU32(&hdr->flavor))) {
    return FONT_COMPRESSION_FAILURE();
  }

  // TODO(user): Should call IsValidVersionTag() here.

  var reported_length: uint32_t;
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
  var meta_offset: uint32_t;
  var meta_length: uint32_t;
  var meta_length_orig: uint32_t;
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
  var priv_offset: uint32_t;
  var priv_length: uint32_t;
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
  var last_table: Table& = hdr->tables.back();
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
    var num_fonts: uint32_t;
    if (PREDICT_FALSE(!Read255UShort(&file, &num_fonts) || !num_fonts)) {
      return FONT_COMPRESSION_FAILURE();
    }
    hdr->ttc_fonts.resize(num_fonts);

    for (var i: uint32_t = 0; i < num_fonts; i++) {
      var ttc_font: TtcFont& = hdr->ttc_fonts[i];
      var num_tables: uint32_t;
      if (PREDICT_FALSE(!Read255UShort(&file, &num_tables) || !num_tables)) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (PREDICT_FALSE(!file.ReadU32(&ttc_font.flavor))) {
        return FONT_COMPRESSION_FAILURE();
      }

      ttc_font.table_indices.resize(num_tables);


      var glyf_idx: unsigned int = 0;
      var loca_idx: unsigned int = 0;

      for (var j: uint32_t = 0; j < num_tables; j++) {
        var table_idx: unsigned int;
        if (PREDICT_FALSE(!Read255UShort(&file, &table_idx)) ||
            table_idx >= hdr->tables.size()) {
          return FONT_COMPRESSION_FAILURE();
        }
        ttc_font.table_indices[j] = table_idx;

        var table: const Table& = hdr->tables[table_idx];
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

  let first_table_offset: const uint64_t = ComputeOffsetToFirstTable(*hdr);

  hdr->compressed_offset = file.offset();
  if (PREDICT_FALSE(hdr->compressed_offset >
                    std::numeric_limits<uint32_t>::max())) {
    return FONT_COMPRESSION_FAILURE();
  }
  var src_offset: uint64_t = Round4(hdr->compressed_offset + hdr->compressed_length);
  var dst_offset: uint64_t = first_table_offset;


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
fn WriteHeaders(metadata: RebuildMetadata*,
                  hdr: WOFF2Header*, out: WOFF2Out*) -> bool {
  var output: std::vector<uint8_t>(ComputeOffsetToFirstTable(*hdr), 0);

  // Re-order tables in output (OTSpec) order
  var sorted_tables: std::vector<Table>(hdr->tables);
  if (hdr->header_version) {
    // collection; we have to sort the table offset vector in each font
    for (auto& ttc_font  in  hdr->ttc_fonts) {
      var sorted_index_by_tag: std::map<uint32_t, uint16_t>;
      for (auto table_index  in  ttc_font.table_indices) {
        sorted_index_by_tag[hdr->tables[table_index].tag] = table_index;
      }
      var index: uint16_t = 0;
      for (auto& i  in  sorted_index_by_tag) {
        ttc_font.table_indices[index++] = i.second;
      }
    }
  } else {
    // non-collection; we can just sort the tables
    std::sort(sorted_tables.begin(), sorted_tables.end());
  }

  // Start building the font
  var result: uint8_t* = &output[0];
  var offset: size_t = 0;
  if (hdr->header_version) {
    // TTC header
    offset = StoreU32(result, offset, hdr->flavor);  // TAG TTCTag
    offset = StoreU32(result, offset, hdr->header_version);  // FIXED Version
    offset = StoreU32(result, offset, hdr->ttc_fonts.size());  // ULONG numFonts
    // Space for ULONG OffsetTable[numFonts] (zeroed initially)
    var offset_table: size_t = offset;  // keep start of offset table for later
    for (var i: size_t = 0; i < hdr->ttc_fonts.size(); i++) {
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
    for (var i: size_t = 0; i < hdr->ttc_fonts.size(); i++) {
      var ttc_font: TtcFont& = hdr->ttc_fonts[i];

      // write Offset Table location into TTC Header
      offset_table = StoreU32(result, offset_table, offset);

      // write the actual offset table so our header doesn't lie
      ttc_font.dst_offset = offset;
      offset = StoreOffsetTable(result, offset, ttc_font.flavor,
                                ttc_font.table_indices.size());

      for (const auto table_index  in  ttc_font.table_indices) {
        var tag: uint32_t = hdr->tables[table_index].tag;
        metadata->font_infos[i].table_entry_by_tag[tag] = offset;
        offset = StoreTableEntry(result, offset, tag);
      }

      ttc_font.header_checksum = ComputeULongSum(&output[ttc_font.dst_offset],
                                                 offset - ttc_font.dst_offset);
    }
  } else {
    metadata->font_infos.resize(1);
    offset = StoreOffsetTable(result, offset, hdr->flavor, hdr->num_tables);
    for (var i: uint16_t = 0; i < hdr->num_tables; ++i) {
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

fn ComputeWOFF2FinalSize(data: const uint8_t*, length: size_t) -> size_t {
  var file: Buffer(data, length);
  var total_length: uint32_t;

  if (!file.Skip(16) ||
      !file.ReadU32(&total_length)) {
    return 0;
  }
  return total_length;
}

fn ConvertWOFF2ToTTF(result: uint8_t*, result_length: size_t,
                       data: const uint8_t*, length: size_t) -> bool {
  var out: WOFF2MemoryOut(result, result_length);
  return ConvertWOFF2ToTTF(data, length, &out);
}

fn ConvertWOFF2ToTTF(data: const uint8_t*, length: size_t,
                       out: WOFF2Out*) -> bool {
  var metadata: RebuildMetadata;
  var hdr: WOFF2Header;
  if (!ReadWOFF2Header(data, length, &hdr)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (!WriteHeaders(&metadata, &hdr, out)) {
    return FONT_COMPRESSION_FAILURE();
  }

  let compression_ratio: const float = static_cast<float>(hdr.uncompressed_size) / length;
  if (compression_ratio > kMaxPlausibleCompressionRatio) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "Implausible compression ratio %.01f\n", compression_ratio);
#endif
    return FONT_COMPRESSION_FAILURE();
  }

  var src_buf: const uint8_t* = data + hdr.compressed_offset;
  var uncompressed_buf: std::vector<uint8_t>(hdr.uncompressed_size);
  if (PREDICT_FALSE(hdr.uncompressed_size < 1)) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (PREDICT_FALSE(!Woff2Uncompress(&uncompressed_buf[0],
                                     hdr.uncompressed_size, src_buf,
                                     hdr.compressed_length))) {
    return FONT_COMPRESSION_FAILURE();
  }

  for (var i: size_t = 0; i < metadata.font_infos.size(); i++) {
    if (PREDICT_FALSE(!ReconstructFont(&uncompressed_buf[0],
                                       hdr.uncompressed_size,
                                       &metadata, &hdr, i, out))) {
      return FONT_COMPRESSION_FAILURE();
    }
  }

  return true;
}

} // namespace woff2
