/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Glyph normalization */

#include "./normalize.h"

#include <inttypes.h>
#include <stddef.h>

#include "./buffer.h"
#include "./port.h"
#include "./font.h"
#include "./glyph.h"
#include "./round.h"
#include "./store_bytes.h"
#include "./table_tags.h"
#include "./woff2_common.h"

namespace woff2 {

namespace {

void StoreLoca(int index_fmt, uint32_t value, size_t* offset, uint8_t* dst) {
  if (index_fmt == 0) {
    Store16(value >> 1, offset, dst);
  } else {
    StoreU32(value, offset, dst);
  }
}

}  // namespace

namespace {

bool WriteNormalizedLoca(int index_fmt, int num_glyphs, Font* font) {
  Font::Table* glyf_table = font->FindTable(kGlyfTableTag);
  Font::Table* loca_table = font->FindTable(kLocaTableTag);

  int glyph_sz = index_fmt == 0 ? 2 : 4;
  loca_table->buffer.resize(Round4(num_glyphs + 1) * glyph_sz);
  loca_table->length = (num_glyphs + 1) * glyph_sz;

  uint8_t* glyf_dst = num_glyphs ? &glyf_table->buffer[0] : NULL;
  uint8_t* loca_dst = &loca_table->buffer[0];
  uint32_t glyf_offset = 0;
  size_t loca_offset = 0;

  for (int i = 0; i < num_glyphs; ++i) {
    StoreLoca(index_fmt, glyf_offset, &loca_offset, loca_dst);
    Glyph glyph;
    const uint8_t* glyph_data;
    size_t glyph_size;
    if (!GetGlyphData(*font, i, &glyph_data, &glyph_size) ||
        (glyph_size > 0 && !ReadGlyph(glyph_data, glyph_size, &glyph))) {
      return FONT_COMPRESSION_FAILURE();
    }
    size_t glyf_dst_size = glyf_table->buffer.size() - glyf_offset;
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
  glyf_table->data = glyf_offset ? &glyf_table->buffer[0] : NULL;
  glyf_table->length = glyf_offset;
  loca_table->data = loca_offset ? &loca_table->buffer[0] : NULL;

  return true;
}

}  // namespace

namespace {

bool MakeEditableBuffer(Font* font, int tableTag) {
  Font::Table* table = font->FindTable(tableTag);
  if (table == NULL) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (table->IsReused()) {
    return true;
  }
  int sz = Round4(table->length);
  table->buffer.resize(sz);
  uint8_t* buf = &table->buffer[0];
  memcpy(buf, table->data, table->length);
  if (PREDICT_FALSE(sz > table->length)) {
    memset(buf + table->length, 0, sz - table->length);
  }
  table->data = buf;
  return true;
}

}  // namespace

bool NormalizeGlyphs(Font* font) {
  Font::Table* head_table = font->FindTable(kHeadTableTag);
  Font::Table* glyf_table = font->FindTable(kGlyfTableTag);
  Font::Table* loca_table = font->FindTable(kLocaTableTag);
  if (head_table == NULL) {
    return FONT_COMPRESSION_FAILURE();
  }
  // If you don't have glyf/loca this transform isn't very interesting
  if (loca_table == NULL && glyf_table == NULL) {
    return true;
  }
  // It would be best if you didn't have just one of glyf/loca
  if ((glyf_table == NULL) != (loca_table == NULL)) {
    return FONT_COMPRESSION_FAILURE();
  }
  // Must share neither or both loca & glyf
  if (loca_table->IsReused() != glyf_table->IsReused()) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (loca_table->IsReused()) {
    return true;
  }

  int index_fmt = head_table->data[51];
  int num_glyphs = NumGlyphs(*font);

  // We need to allocate a bit more than its original length for the normalized
  // glyf table, since it can happen that the glyphs in the original table are
  // 2-byte aligned, while in the normalized table they are 4-byte aligned.
  // That gives a maximum of 2 bytes increase per glyph. However, there is no
  // theoretical guarantee that the total size of the flags plus the coordinates
  // is the smallest possible in the normalized version, so we have to allow
  // some general overhead.
  // TODO(user) Figure out some more precise upper bound on the size of
  // the overhead.
  size_t max_normalized_glyf_size = 1.1 * glyf_table->length + 2 * num_glyphs;

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

bool NormalizeOffsets(Font* font) {
  uint32_t offset = 12 + 16 * font->num_tables;
  for (auto tag : font->OutputOrderedTags()) {
    auto& table = font->tables[tag];
    table.offset = offset;
    offset += Round4(table.length);
  }
  return true;
}

namespace {

uint32_t ComputeHeaderChecksum(const Font& font) {
  uint32_t checksum = font.flavor;
  uint16_t max_pow2 = font.num_tables ? Log2Floor(font.num_tables) : 0;
  uint16_t search_range = max_pow2 ? 1 << (max_pow2 + 4) : 0;
  uint16_t range_shift = (font.num_tables << 4) - search_range;
  checksum += (font.num_tables << 16 | search_range);
  checksum += (max_pow2 << 16 | range_shift);
  for (const auto& i : font.tables) {
    const Font::Table* table = &i.second;
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

bool FixChecksums(Font* font) {
  Font::Table* head_table = font->FindTable(kHeadTableTag);
  if (head_table == NULL) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (head_table->reuse_of != NULL) {
    head_table = head_table->reuse_of;
  }
  if (head_table->length < 12) {
    return FONT_COMPRESSION_FAILURE();
  }

  uint8_t* head_buf = &head_table->buffer[0];
  size_t offset = 8;
  StoreU32(0, &offset, head_buf);
  uint32_t file_checksum = 0;
  uint32_t head_checksum = 0;
  for (auto& i : font->tables) {
    Font::Table* table = &i.second;
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
bool MarkTransformed(Font* font) {
  Font::Table* head_table = font->FindTable(kHeadTableTag);
  if (head_table == NULL) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (head_table->reuse_of != NULL) {
    head_table = head_table->reuse_of;
  }
  if (head_table->length < 17) {
    return FONT_COMPRESSION_FAILURE();
  }
  // set bit 11 of head table 'flags' to indicate that font has undergone
  // lossless modifying transform
  int head_flags = head_table->data[16];
  head_table->buffer[16] = head_flags | 0x08;
  return true;
}
}  // namespace


bool NormalizeWithoutFixingChecksums(Font* font) {
  return (MakeEditableBuffer(font, kHeadTableTag) &&
          RemoveDigitalSignature(font) &&
          MarkTransformed(font) &&
          NormalizeGlyphs(font) &&
          NormalizeOffsets(font));
}

bool NormalizeFont(Font* font) {
  return (NormalizeWithoutFixingChecksums(font) &&
          FixChecksums(font));
}

bool NormalizeFontCollection(FontCollection* font_collection) {
  if (font_collection->fonts.size() == 1) {
    return NormalizeFont(&font_collection->fonts[0]);
  }

  uint32_t offset = CollectionHeaderSize(font_collection->header_version,
    font_collection->fonts.size());
  for (auto& font : font_collection->fonts) {
    if (!NormalizeWithoutFixingChecksums(&font)) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Font normalization failed.\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
    offset += kSfntHeaderSize + kSfntEntrySize * font.num_tables;
  }

  // Start table offsets after TTC Header and Sfnt Headers
  for (auto& font : font_collection->fonts) {
    for (auto tag : font.OutputOrderedTags()) {
      Font::Table& table = font.tables[tag];
      if (table.IsReused()) {
        table.offset = table.reuse_of->offset;
      } else {
        table.offset = offset;
        offset += Round4(table.length);
      }
    }
  }

  // Now we can fix the checksums
  for (auto& font : font_collection->fonts) {
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
