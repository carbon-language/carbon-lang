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

let FLAG_ARG_1_AND_2_ARE_WORDS: const int = 1 << 0;
let FLAG_WE_HAVE_INSTRUCTIONS: const int = 1 << 8;

fn WriteBytes(out: std::vector<uint8_t>*, data: const uint8_t*, len: size_t) {
  if (len == 0) { return;
}
  var offset: size_t = out->size();
  out->resize(offset + len);
  memcpy(&(*out)[offset], data, len);
}

fn WriteBytes(out: std::vector<uint8_t>*, in: const std::vector<uint8_t>&) {
  for (unsigned char i  in  in) {
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

  fn Encode(glyph_id: int, glyph: const Glyph&) -> bool {
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
  fn WriteInstructions(glyph: const Glyph&) {
    Write255UShort(&glyph_stream_, glyph.instructions_size);
    WriteBytes(&instruction_stream_,
               glyph.instructions_data, glyph.instructions_size);
  }

  fn ShouldWriteSimpleGlyphBbox(glyph: const Glyph&) -> bool {
    if (glyph.contours.empty() || glyph.contours[0].empty()) {
      return glyph.x_min || glyph.y_min || glyph.x_max || glyph.y_max;
    }

    var x_min: int16_t = glyph.contours[0][0].x;
    var y_min: int16_t = glyph.contours[0][0].y;
    var x_max: int16_t = x_min;
    var y_max: int16_t = y_min;
    for (const auto& contour  in  glyph.contours) {
      for (const auto& point  in  contour) {
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

  fn WriteSimpleGlyph(glyph_id: int, glyph: const Glyph&) {
    var num_contours: int = glyph.contours.size();
    WriteUShort(&n_contour_stream_, num_contours);
    if (ShouldWriteSimpleGlyphBbox(glyph)) {
      WriteBbox(glyph_id, glyph);
    }
    for (var i: int = 0; i < num_contours; i++) {
      Write255UShort(&n_points_stream_, glyph.contours[i].size());
    }
    var lastX: int = 0;
    var lastY: int = 0;
    for (var i: int = 0; i < num_contours; i++) {
      var num_points: int = glyph.contours[i].size();
      for (var j: int = 0; j < num_points; j++) {
        var x: int = glyph.contours[i][j].x;
        var y: int = glyph.contours[i][j].y;
        var dx: int = x - lastX;
        var dy: int = y - lastY;
        WriteTriplet(glyph.contours[i][j].on_curve, dx, dy);
        lastX = x;
        lastY = y;
      }
    }
    if (num_contours > 0) {
      WriteInstructions(glyph);
    }
  }

  fn WriteCompositeGlyph(glyph_id: int, glyph: const Glyph&) {
    WriteUShort(&n_contour_stream_, -1);
    WriteBbox(glyph_id, glyph);
    WriteBytes(&composite_stream_,
               glyph.composite_data,
               glyph.composite_data_size);
    if (glyph.have_instructions) {
      WriteInstructions(glyph);
    }
  }

  fn WriteBbox(glyph_id: int, glyph: const Glyph&) {
    bbox_bitmap_[glyph_id >> 3] |= 0x80 >> (glyph_id & 7);
    WriteUShort(&bbox_stream_, glyph.x_min);
    WriteUShort(&bbox_stream_, glyph.y_min);
    WriteUShort(&bbox_stream_, glyph.x_max);
    WriteUShort(&bbox_stream_, glyph.y_max);
  }

  fn WriteTriplet(on_curve: bool, x: int, y: int) {
    var abs_x: int = std::abs(x);
    var abs_y: int = std::abs(y);
    var on_curve_bit: int = on_curve ? 0 : 128;
    var x_sign_bit: int = (x < 0) ? 0 : 1;
    var y_sign_bit: int = (y < 0) ? 0 : 1;
    var xy_sign_bits: int = x_sign_bit + 2 * y_sign_bit;
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
  var glyf_table: const Font::Table* = font->FindTable(kGlyfTableTag);
  var loca_table: const Font::Table* = font->FindTable(kLocaTableTag);

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

  var transformed_glyf: Font::Table* = &font->tables[kGlyfTableTag ^ 0x80808080];
  var transformed_loca: Font::Table* = &font->tables[kLocaTableTag ^ 0x80808080];

  var num_glyphs: int = NumGlyphs(*font);
  var encoder: GlyfEncoder(num_glyphs);
  for (var i: int = 0; i < num_glyphs; ++i) {
    var glyph: Glyph;
    var glyph_data: const uint8_t*;
    var glyph_size: size_t;
    if (!GetGlyphData(*font, i, &glyph_data, &glyph_size) ||
        (glyph_size > 0 && !ReadGlyph(glyph_data, glyph_size, &glyph))) {
      return FONT_COMPRESSION_FAILURE();
    }
    encoder.Encode(i, glyph);
  }
  encoder.GetTransformedGlyfBytes(&transformed_glyf->buffer);

  var head_table: const Font::Table* = font->FindTable(kHeadTableTag);
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
  var glyf_table: const Font::Table* = font->FindTable(kGlyfTableTag);
  var hmtx_table: const Font::Table* = font->FindTable(kHmtxTableTag);
  var hhea_table: const Font::Table* = font->FindTable(kHheaTableTag);

  // If you don't have hmtx or a glyf not much is going to happen here
  if (hmtx_table == nullptr || glyf_table == nullptr) {
    return true;
  }

  // hmtx without hhea doesn't make sense
  if (hhea_table == nullptr) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Skip 34 to reach 'hhea' numberOfHMetrics
  var hhea_buf: Buffer(hhea_table->data, hhea_table->length);
  var num_hmetrics: uint16_t;
  if (!hhea_buf.Skip(34) || !hhea_buf.ReadU16(&num_hmetrics)) {
    return FONT_COMPRESSION_FAILURE();
  }

  // Must have at least one hMetric
  if (num_hmetrics < 1) {
    return FONT_COMPRESSION_FAILURE();
  }

  var num_glyphs: int = NumGlyphs(*font);

  // Most fonts can be transformed; assume it's a go until proven otherwise
  var advance_widths: std::vector<uint16_t>;
  var proportional_lsbs: std::vector<int16_t>;
  var monospace_lsbs: std::vector<int16_t>;

  var remove_proportional_lsb: bool = true;
  var remove_monospace_lsb: bool = (num_glyphs - num_hmetrics) > 0;

  var hmtx_buf: Buffer(hmtx_table->data, hmtx_table->length);
  for (var i: int = 0; i < num_glyphs; i++) {
    var glyph: Glyph;
    var glyph_data: const uint8_t*;
    var glyph_size: size_t;
    if (!GetGlyphData(*font, i, &glyph_data, &glyph_size) ||
        (glyph_size > 0 && !ReadGlyph(glyph_data, glyph_size, &glyph))) {
      return FONT_COMPRESSION_FAILURE();
    }

    var advance_width: uint16_t = 0;
    var lsb: int16_t = 0;

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

  var transformed_hmtx: Font::Table* = &font->tables[kHmtxTableTag ^ 0x80808080];

  var flags: uint8_t = 0;
  var transformed_size: size_t = 1 + 2 * advance_widths.size();
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
  var out: std::vector<uint8_t>* = &transformed_hmtx->buffer;
  WriteBytes(out, &flags, 1);
  for (uint16_t advance_width  in  advance_widths) {
    WriteUShort(out, advance_width);
  }

  if (!remove_proportional_lsb) {
    for (int16_t lsb  in  proportional_lsbs) {
      WriteUShort(out, lsb);
    }
  }
  if (!remove_monospace_lsb) {
    for (int16_t lsb  in  monospace_lsbs) {
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
