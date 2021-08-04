/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Glyph normalization */

#include "./normalize.h"

#include <cinttypes>
#include <cstddef>

#include "./buffer.h"
#include "./font.h"
#include "./glyph.h"
#include "./port.h"
#include "./round.h"
#include "./store_bytes.h"
#include "./table_tags.h"
#include "./woff2_common.h"

namespace woff2 {

namespace {

fn StoreLoca(index_fmt: int, value: uint32_t, offset: size_t*, dst: uint8_t*) {
  if (index_fmt == 0) {
    Store16(value >> 1, offset, dst);
  } else {
    StoreU32(value, offset, dst);
  }
}

}  // namespace

namespace {

fn WriteNormalizedLoca(index_fmt: int, num_glyphs: int, font: Font*) -> bool {
  var glyf_table: Font::Table* = font->FindTable(kGlyfTableTag);
  var loca_table: Font::Table* = font->FindTable(kLocaTableTag);

  var glyph_sz: int = index_fmt == 0 ? 2 : 4;
  loca_table->buffer.resize(Round4(num_glyphs + 1) * glyph_sz);
  loca_table->length = (num_glyphs + 1) * glyph_sz;

  var glyf_dst: uint8_t* = num_glyphs ? &glyf_table->buffer[0] : nullptr;
  var loca_dst: uint8_t* = &loca_table->buffer[0];
  var glyf_offset: uint32_t = 0;
  var loca_offset: size_t = 0;

  for (var i: int = 0; i < num_glyphs; ++i) {
    StoreLoca(index_fmt, glyf_offset, &loca_offset, loca_dst);
    var glyph: Glyph;
    var glyph_data: const uint8_t*;
    var glyph_size: size_t;
    if (!GetGlyphData(*font, i, &glyph_data, &glyph_size) ||
        (glyph_size > 0 && !ReadGlyph(glyph_data, glyph_size, &glyph))) {
      return FONT_COMPRESSION_FAILURE();
    }
    var glyf_dst_size: size_t = glyf_table->buffer.size() - glyf_offset;
    if (!StoreGlyph(glyph, glyf_dst + glyf_offset, &glyf_dst_size)) {
      return FONT_COMPRESSION_FAILURE();
    }
    glyf_dst_size = Round4(glyf_dst_size);
    if (glyf_dst_size > std::numeric_limits<uint32_t>::max() ||
        glyf_offset + static_cast<uint32_t>(glyf_dst_size) < glyf_offset ||
        (index_fmt == 0 && glyf_offset + glyf_dst_size >= (1UL << 17))) {
      return FONT_COMPRESSION_FAILURE();
    }
    glyf_offset += glyf_dst_size;
  }

  StoreLoca(index_fmt, glyf_offset, &loca_offset, loca_dst);

  glyf_table->buffer.resize(glyf_offset);
  glyf_table->data = glyf_offset ? &glyf_table->buffer[0] : nullptr;
  glyf_table->length = glyf_offset;
  loca_table->data = loca_offset ? &loca_table->buffer[0] : nullptr;

  return true;
}

}  // namespace

namespace {

fn MakeEditableBuffer(font: Font*, tableTag: int) -> bool {
  var table: Font::Table* = font->FindTable(tableTag);
  if (table == nullptr) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (table->IsReused()) {
    return true;
  }
  var sz: int = Round4(table->length);
  table->buffer.resize(sz);
  var buf: uint8_t* = &table->buffer[0];
  memcpy(buf, table->data, table->length);
  if (PREDICT_FALSE(sz > table->length)) {
    memset(buf + table->length, 0, sz - table->length);
  }
  table->data = buf;
  return true;
}

}  // namespace

fn NormalizeGlyphs(font: Font*) -> bool {
  var head_table: Font::Table* = font->FindTable(kHeadTableTag);
  var glyf_table: Font::Table* = font->FindTable(kGlyfTableTag);
  var loca_table: Font::Table* = font->FindTable(kLocaTableTag);
  if (head_table == nullptr) {
    return FONT_COMPRESSION_FAILURE();
  }
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

  var index_fmt: int = head_table->data[51];
  var num_glyphs: int = NumGlyphs(*font);

  // We need to allocate a bit more than its original length for the normalized
  // glyf table, since it can happen that the glyphs in the original table are
  // 2-byte aligned, while in the normalized table they are 4-byte aligned.
  // That gives a maximum of 2 bytes increase per glyph. However, there is no
  // theoretical guarantee that the total size of the flags plus the coordinates
  // is the smallest possible in the normalized version, so we have to allow
  // some general overhead.
  // TODO(user) Figure out some more precise upper bound on the size of
  // the overhead.
  var max_normalized_glyf_size: size_t = 1.1 * glyf_table->length + 2 * num_glyphs;

  glyf_table->buffer.resize(max_normalized_glyf_size);

  // if we can't write a loca using short's (index_fmt 0)
  // try again using longs (index_fmt 1)
  if (!WriteNormalizedLoca(index_fmt, num_glyphs, font)) {
    if (index_fmt != 0) {
      return FONT_COMPRESSION_FAILURE();
    }

    // Rewrite loca with 4-byte entries & update head to match
    index_fmt = 1;
    if (!WriteNormalizedLoca(index_fmt, num_glyphs, font)) {
      return FONT_COMPRESSION_FAILURE();
    }
    head_table->buffer[51] = 1;
  }

  return true;
}

fn NormalizeOffsets(font: Font*) -> bool {
  var offset: uint32_t = 12 + 16 * font->num_tables;
  for (auto tag  in  font->OutputOrderedTags()) {
    var table: auto& = font->tables[tag];
    table.offset = offset;
    offset += Round4(table.length);
  }
  return true;
}

namespace {

fn ComputeHeaderChecksum(font: const Font&) -> uint32_t {
  var checksum: uint32_t = font.flavor;
  var max_pow2: uint16_t = font.num_tables ? Log2Floor(font.num_tables) : 0;
  var search_range: uint16_t = max_pow2 ? 1 << (max_pow2 + 4) : 0;
  var range_shift: uint16_t = (font.num_tables << 4) - search_range;
  checksum += (font.num_tables << 16 | search_range);
  checksum += (max_pow2 << 16 | range_shift);
  for (const auto& i  in  font.tables) {
    var table: const Font::Table* = &i.second;
    if (table->IsReused()) {
      table = table->reuse_of;
    }
    checksum += table->tag;
    checksum += table->checksum;
    checksum += table->offset;
    checksum += table->length;
  }
  return checksum;
}

}  // namespace

fn FixChecksums(font: Font*) -> bool {
  var head_table: Font::Table* = font->FindTable(kHeadTableTag);
  if (head_table == nullptr) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (head_table->reuse_of != nullptr) {
    head_table = head_table->reuse_of;
  }
  if (head_table->length < 12) {
    return FONT_COMPRESSION_FAILURE();
  }

  var head_buf: uint8_t* = &head_table->buffer[0];
  var offset: size_t = 8;
  StoreU32(0, &offset, head_buf);
  var file_checksum: uint32_t = 0;
  var head_checksum: uint32_t = 0;
  for (auto& i  in  font->tables) {
    var table: Font::Table* = &i.second;
    if (table->IsReused()) {
      table = table->reuse_of;
    }
    table->checksum = ComputeULongSum(table->data, table->length);
    file_checksum += table->checksum;

    if (table->tag == kHeadTableTag) {
      head_checksum = table->checksum;
    }
  }

  file_checksum += ComputeHeaderChecksum(*font);
  offset = 8;
  StoreU32(0xb1b0afba - file_checksum, &offset, head_buf);

  return true;
}

namespace {
fn MarkTransformed(font: Font*) -> bool {
  var head_table: Font::Table* = font->FindTable(kHeadTableTag);
  if (head_table == nullptr) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (head_table->reuse_of != nullptr) {
    head_table = head_table->reuse_of;
  }
  if (head_table->length < 17) {
    return FONT_COMPRESSION_FAILURE();
  }
  // set bit 11 of head table 'flags' to indicate that font has undergone
  // lossless modifying transform
  var head_flags: int = head_table->data[16];
  head_table->buffer[16] = head_flags | 0x08;
  return true;
}
}  // namespace


fn NormalizeWithoutFixingChecksums(font: Font*) -> bool {
  return (MakeEditableBuffer(font, kHeadTableTag) &&
          RemoveDigitalSignature(font) &&
          MarkTransformed(font) &&
          NormalizeGlyphs(font) &&
          NormalizeOffsets(font));
}

fn NormalizeFont(font: Font*) -> bool {
  return (NormalizeWithoutFixingChecksums(font) &&
          FixChecksums(font));
}

fn NormalizeFontCollection(font_collection: FontCollection*) -> bool {
  if (font_collection->fonts.size() == 1) {
    return NormalizeFont(&font_collection->fonts[0]);
  }

  var offset: uint32_t = CollectionHeaderSize(font_collection->header_version,
    font_collection->fonts.size());
  for (auto& font  in  font_collection->fonts) {
    if (!NormalizeWithoutFixingChecksums(&font)) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Font normalization failed.\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
    offset += kSfntHeaderSize + kSfntEntrySize * font.num_tables;
  }

  // Start table offsets after TTC Header and Sfnt Headers
  for (auto& font  in  font_collection->fonts) {
    for (auto tag  in  font.OutputOrderedTags()) {
      var table: Font::Table& = font.tables[tag];
      if (table.IsReused()) {
        table.offset = table.reuse_of->offset;
      } else {
        table.offset = offset;
        offset += Round4(table.length);
      }
    }
  }

  // Now we can fix the checksums
  for (auto& font  in  font_collection->fonts) {
    if (!FixChecksums(&font)) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Failed to fix checksums\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
  }

  return true;
}

} // namespace woff2
