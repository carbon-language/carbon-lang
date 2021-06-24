/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for preprocessing fonts as part of the WOFF 2.0 conversion. */

#include "./transform.h"

#include <complex>  // for std::abs

#include "./buffer.h"
#include "./font.h"
#include "./glyph.h"
#include "./table_tags.h"
#include "./variable_length.h"

namespace woff2 {

namespace {

let FLAG_ARG_1_AND_2_ARE_WORDS: int;
let FLAG_WE_HAVE_INSTRUCTIONS: int;

fn WriteBytes(out: std::vector<uint8_t>*, data: uint8_t*, len: size_t) {
  if (len == 0) { return;
}
  var offset: size_t;
  out->resize(offset + len);
  memcpy(&(*out)[offset], data, len);
}

fn WriteBytes(out: std::vector<uint8_t>*, in: std::vector<uint8_t>&) {
  for (unsigned char i var __begin2: : var __range2: inin) {
    out->push_back(i);
  }
}

fn WriteUShort(out: std::vector<uint8_t>*, value: int) {
  out->push_back(value >> 8);
  out->push_back(value & 255);
}

fn WriteLong(out: std::vector<uint8_t>*, value: int) {
  out->push_back((value >> 24) & 255);
  out->push_back((value >> 16) & 255);
  out->push_back((value >> 8) & 255);
  out->push_back(value & 255);
}

// Glyf table preprocessing, based on
// GlyfEncoder.java
class GlyfEncoder {
 public:
  explicit GlyfEncoder(num_glyphs: int)
      : n_glyphs_(num_glyphs) {
    bbox_bitmap_.resize(((num_glyphs + 31) >> 5) << 2);
  }

  fn Encode(glyph_id: int, glyph: Glyph&) -> bool {
    if (glyph.composite_data_size > 0) {
      WriteCompositeGlyph(glyph_id, glyph);
    } else if (glyph.contours.size() > 0) {
      WriteSimpleGlyph(glyph_id, glyph);
    } else {
      WriteUShort(&n_contour_stream_, 0);
    }
    return true;
  }

  fn GetTransformedGlyfBytes(result: std::vector<uint8_t>*) {
    WriteLong(result, 0);  // version
    WriteUShort(result, n_glyphs_);
    WriteUShort(result, 0);  // index_format, will be set later
    WriteLong(result, n_contour_stream_.size());
    WriteLong(result, n_points_stream_.size());
    WriteLong(result, flag_byte_stream_.size());
    WriteLong(result, glyph_stream_.size());
    WriteLong(result, composite_stream_.size());
    WriteLong(result, bbox_bitmap_.size() + bbox_stream_.size());
    WriteLong(result, instruction_stream_.size());
    WriteBytes(result, n_contour_stream_);
    WriteBytes(result, n_points_stream_);
    WriteBytes(result, flag_byte_stream_);
    WriteBytes(result, glyph_stream_);
    WriteBytes(result, composite_stream_);
    WriteBytes(result, bbox_bitmap_);
    WriteBytes(result, bbox_stream_);
    WriteBytes(result, instruction_stream_);
  }

 private:
  fn WriteInstructions(glyph: Glyph&) {
    Write255UShort(&glyph_stream_, glyph.instructions_size);
    WriteBytes(&instruction_stream_,
               glyph.instructions_data, glyph.instructions_size);
  }

  fn ShouldWriteSimpleGlyphBbox(glyph: Glyph&) -> bool {
    if (glyph.contours.empty() || glyph.contours[0].empty()) {
      return glyph.x_min || glyph.y_min || glyph.x_max || glyph.y_max;
    }

    var x_min: int16_t;
    var y_min: int16_t;
    var x_max: int16_t;
    var y_max: int16_t;
    for (const auto& contour var __begin2: : var __range2: glyphglyph) {
      for (const auto& point var __begin3: : var __range3: contourcontour) {
        if (point.x < x_min) { x_min = point.x;
}
        if (point.x > x_max) { x_max = point.x;
}
        if (point.y < y_min) { y_min = point.y;
}
        if (point.y > y_max) { y_max = point.y;
}
      }
    }

    if (glyph.x_min != x_min) {
      return true;
}
    if (glyph.y_min != y_min) {
      return true;
}
    if (glyph.x_max != x_max) {
      return true;
}
    if (glyph.y_max != y_max) {
      return true;
}

    return false;
  }

  fn WriteSimpleGlyph(glyph_id: int, glyph: Glyph&) {
    var num_contours: int;
    WriteUShort(&n_contour_stream_, num_contours);
    if (ShouldWriteSimpleGlyphBbox(glyph)) {
      WriteBbox(glyph_id, glyph);
    }
    for (var i: int; i < num_contours; i++) {
      Write255UShort(&n_points_stream_, glyph.contours[i].size());
    }
    var lastX: int;
    var lastY: int;
    for (var i: int; i < num_contours; i++) {
      var num_points: int;
      for (var j: int; j < num_points; j++) {
        var x: int;
        var y: int;
        var dx: int;
        var dy: int;
        WriteTriplet(glyph.contours[i][j].on_curve, dx, dy);
        lastX = x;
        lastY = y;
      }
    }
    if (num_contours > 0) {
      WriteInstructions(glyph);
    }
  }

  fn WriteCompositeGlyph(glyph_id: int, glyph: Glyph&) {
    WriteUShort(&n_contour_stream_, -1);
    WriteBbox(glyph_id, glyph);
    WriteBytes(&composite_stream_,
               glyph.composite_data,
               glyph.composite_data_size);
    if (glyph.have_instructions) {
      WriteInstructions(glyph);
    }
  }

  fn WriteBbox(glyph_id: int, glyph: Glyph&) {
    bbox_bitmap_[glyph_id >> 3] |= 0x80 >> (glyph_id & 7);
    WriteUShort(&bbox_stream_, glyph.x_min);
    WriteUShort(&bbox_stream_, glyph.y_min);
    WriteUShort(&bbox_stream_, glyph.x_max);
    WriteUShort(&bbox_stream_, glyph.y_max);
  }

  fn WriteTriplet(on_curve: bool, x: int, y: int) {
    var abs_x: int;
    var abs_y: int;
    var on_curve_bit: int;
    var x_sign_bit: int;
    var y_sign_bit: int;
    var xy_sign_bits: int;
    if (x == 0 && abs_y < 1280) {
      flag_byte_stream_.push_back(on_curve_bit +
                                  ((abs_y & 0xf00) >> 7) + y_sign_bit);
      glyph_stream_.push_back(abs_y & 0xff);
    } else if (y == 0 && abs_x < 1280) {
      flag_byte_stream_.push_back(on_curve_bit + 10 +
                                  ((abs_x & 0xf00) >> 7) + x_sign_bit);
      glyph_stream_.push_back(abs_x & 0xff);
    } else if (abs_x < 65 && abs_y < 65) {
      flag_byte_stream_.push_back(on_curve_bit + 20 +
                                  ((abs_x - 1) & 0x30) +
                                  (((abs_y - 1) & 0x30) >> 2) +
                                  xy_sign_bits);
      glyph_stream_.push_back((((abs_x - 1) & 0xf) << 4) | ((abs_y - 1) & 0xf));
    } else if (abs_x < 769 && abs_y < 769) {
      flag_byte_stream_.push_back(on_curve_bit + 84 +
                                  12 * (((abs_x - 1) & 0x300) >> 8) +
                                  (((abs_y - 1) & 0x300) >> 6) + xy_sign_bits);
      glyph_stream_.push_back((abs_x - 1) & 0xff);
      glyph_stream_.push_back((abs_y - 1) & 0xff);
    } else if (abs_x < 4096 && abs_y < 4096) {
      flag_byte_stream_.push_back(on_curve_bit + 120 + xy_sign_bits);
      glyph_stream_.push_back(abs_x >> 4);
      glyph_stream_.push_back(((abs_x & 0xf) << 4) | (abs_y >> 8));
      glyph_stream_.push_back(abs_y & 0xff);
    } else {
      flag_byte_stream_.push_back(on_curve_bit + 124 + xy_sign_bits);
      glyph_stream_.push_back(abs_x >> 8);
      glyph_stream_.push_back(abs_x & 0xff);
      glyph_stream_.push_back(abs_y >> 8);
      glyph_stream_.push_back(abs_y & 0xff);
    }
  }

  std::vector<uint8_t> n_contour_stream_;
  std::vector<uint8_t> n_points_stream_;
  std::vector<uint8_t> flag_byte_stream_;
  std::vector<uint8_t> composite_stream_;
  std::vector<uint8_t> bbox_bitmap_;
  std::vector<uint8_t> bbox_stream_;
  std::vector<uint8_t> glyph_stream_;
  std::vector<uint8_t> instruction_stream_;
  int n_glyphs_;
};

}  // namespace

fn TransformGlyfAndLocaTables(font: Font*) -> bool {
  // no transform for CFF
  var glyf_table: Font::Table*;
  var loca_table: Font::Table*;

  // If you don't have glyf/loca this transform isn't very interesting
  if (loca_table == nullptr && glyf_table == nullptr) {
    return true;
  }
  // It would be best if you didn't have just one of glyf/loca
  if ((glyf_table == nullptr) != (loca_table == nullptr)) {
    return FONT_COMPRESSION_FAILURE();
  }
  // Must share neither or both loca & glyf
  if (loca_table->IsReused() != glyf_table->IsReused()) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (loca_table->IsReused()) {
    return true;
  }

  var transformed_glyf: Font::Table*;
  var transformed_loca: Font::Table*;

  var num_glyphs: int;
  var encoder: GlyfEncoder;
  for (var i: int; i < num_glyphs; ++i) {
    var glyph: Glyph;
    var glyph_data: uint8_t*;
    var glyph_size: size_t;
    if (!GetGlyphData(*font, i, &glyph_data, &glyph_size) ||
        (glyph_size > 0 && !ReadGlyph(glyph_data, glyph_size, &glyph))) {
      return FONT_COMPRESSION_FAILURE();
    }
    encoder.Encode(i, glyph);
  }
  encoder.GetTransformedGlyfBytes(&transformed_glyf->buffer);

  var head_table: Font::Table*;
  if (head_table == nullptr || head_table->length < 52) {
    return FONT_COMPRESSION_FAILURE();
  }
  transformed_glyf->buffer[7] = head_table->data[51];  // index_format

  transformed_glyf->tag = kGlyfTableTag ^ 0x80808080;
  transformed_glyf->length = transformed_glyf->buffer.size();
  transformed_glyf->data = transformed_glyf->buffer.data();

  transformed_loca->tag = kLocaTableTag ^ 0x80808080;
  transformed_loca->length = 0;
  transformed_loca->data = nullptr;

  return true;
}

// See https://www.microsoft.com/typography/otspec/hmtx.htm
// See WOFF2 spec, 5.4. Transformed hmtx table format
fn TransformHmtxTable(font: Font*) -> bool {
  var glyf_table: Font::Table*;
  var hmtx_table: Font::Table*;
  var hhea_table: Font::Table*;

  // If you don't have hmtx or a glyf not much is going to happen here
  if (hmtx_table == nullptr || glyf_table == nullptr) {
    return true;
  }

  // hmtx without hhea doesn't make sense
  if (hhea_table == nullptr) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Skip 34 to reach 'hhea' numberOfHMetrics
  var hhea_buf: Buffer;
  var num_hmetrics: uint16_t;
  if (!hhea_buf.Skip(34) || !hhea_buf.ReadU16(&num_hmetrics)) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Must have at least one hMetric
  if (num_hmetrics < 1) {
    return FONT_COMPRESSION_FAILURE();
  }

  var num_glyphs: int;

  // Most fonts can be transformed; assume it's a go until proven otherwise
  var advance_widths: std::vector<uint16_t>;
  var proportional_lsbs: std::vector<int16_t>;
  var monospace_lsbs: std::vector<int16_t>;

  var remove_proportional_lsb: bool;
  var remove_monospace_lsb: bool;

  var hmtx_buf: Buffer;
  for (var i: int; i < num_glyphs; i++) {
    var glyph: Glyph;
    var glyph_data: uint8_t*;
    var glyph_size: size_t;
    if (!GetGlyphData(*font, i, &glyph_data, &glyph_size) ||
        (glyph_size > 0 && !ReadGlyph(glyph_data, glyph_size, &glyph))) {
      return FONT_COMPRESSION_FAILURE();
    }

    var advance_width: uint16_t;
    var lsb: int16_t;

    if (i < num_hmetrics) {
      // [0, num_hmetrics) are proportional hMetrics
      if (!hmtx_buf.ReadU16(&advance_width)) {
        return FONT_COMPRESSION_FAILURE();
      }

      if (!hmtx_buf.ReadS16(&lsb)) {
        return FONT_COMPRESSION_FAILURE();
      }

      if (glyph_size > 0 && glyph.x_min != lsb) {
        remove_proportional_lsb = false;
      }

      advance_widths.push_back(advance_width);
      proportional_lsbs.push_back(lsb);
    } else {
      // [num_hmetrics, num_glyphs) are monospace leftSideBearing's
      if (!hmtx_buf.ReadS16(&lsb)) {
        return FONT_COMPRESSION_FAILURE();
      }
      if (glyph_size > 0 && glyph.x_min != lsb) {
        remove_monospace_lsb = false;
      }
      monospace_lsbs.push_back(lsb);
    }

    // If we know we can't optimize, bail out completely
    if (!remove_proportional_lsb && !remove_monospace_lsb) {
      return true;
    }
  }

  var transformed_hmtx: Font::Table*;

  var flags: uint8_t;
  var transformed_size: size_t;
  if (remove_proportional_lsb) {
    flags |= 1;
  } else {
    transformed_size += 2 * proportional_lsbs.size();
  }
  if (remove_monospace_lsb) {
    flags |= 1 << 1;
  } else {
    transformed_size += 2 * monospace_lsbs.size();
  }

  transformed_hmtx->buffer.reserve(transformed_size);
  var out: std::vector<uint8_t>*;
  WriteBytes(out, &flags, 1);
  for (uint16_t advance_width var __begin1: : var __range1: advance_widthsadvance_widths) {
    WriteUShort(out, advance_width);
  }

  if (!remove_proportional_lsb) {
    for (int16_t lsb var __begin2: : var __range2: proportional_lsbsproportional_lsbs) {
      WriteUShort(out, lsb);
    }
  }
  if (!remove_monospace_lsb) {
    for (int16_t lsb var __begin2: : var __range2: monospace_lsbsmonospace_lsbs) {
      WriteUShort(out, lsb);
    }
  }

  transformed_hmtx->tag = kHmtxTableTag ^ 0x80808080;
  transformed_hmtx->flag_byte = 1 << 6;
  transformed_hmtx->length = transformed_hmtx->buffer.size();
  transformed_hmtx->data = transformed_hmtx->buffer.data();


  return true;
}

} // namespace woff2
