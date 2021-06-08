/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Font management utilities */

#include "./font.h"

#include <algorithm>

#include "./buffer.h"
#include "./port.h"
#include "./store_bytes.h"
#include "./table_tags.h"
#include "./woff2_common.h"

namespace woff2 {

auto Font::FindTable(uint32_t tag) -> Font::Table* {
  auto it = tables.find(tag);
  return it == tables.end() ? nullptr : &it->second;
}

auto Font::FindTable(uint32_t tag) const -> const Font::Table* {
  auto it = tables.find(tag);
  return it == tables.end() ? nullptr : &it->second;
}

auto Font::OutputOrderedTags() const -> std::vector<uint32_t> {
  std::vector<uint32_t> output_order;

  for (const auto& i : tables) {
    const Font::Table& table = i.second;
    // This is a transformed table, we will write it together with the
    // original version.
    if (table.tag & 0x80808080) {
      continue;
    }
    output_order.push_back(table.tag);
  }

  // Alphabetize then put loca immediately after glyf
  auto glyf_loc = std::find(output_order.begin(), output_order.end(),
      kGlyfTableTag);
  auto loca_loc = std::find(output_order.begin(), output_order.end(),
      kLocaTableTag);
  if (glyf_loc != output_order.end() && loca_loc != output_order.end()) {
    output_order.erase(loca_loc);
    output_order.insert(std::find(output_order.begin(), output_order.end(),
      kGlyfTableTag) + 1, kLocaTableTag);
  }

  return output_order;
}

auto ReadTrueTypeFont(Buffer* file, const uint8_t* data, size_t len,
                      Font* font) -> bool {
  // We don't care about the search_range, entry_selector and range_shift
  // fields, they will always be computed upon writing the font.
  if (!file->ReadU16(&font->num_tables) ||
      !file->Skip(6)) {
    return FONT_COMPRESSION_FAILURE();
  }

  std::map<uint32_t, uint32_t> intervals;
  for (uint16_t i = 0; i < font->num_tables; ++i) {
    Font::Table table;
    table.flag_byte = 0;
    table.reuse_of = nullptr;
    if (!file->ReadU32(&table.tag) ||
        !file->ReadU32(&table.checksum) ||
        !file->ReadU32(&table.offset) ||
        !file->ReadU32(&table.length)) {
      return FONT_COMPRESSION_FAILURE();
    }
    if ((table.offset & 3) != 0 ||
        table.length > len ||
        len - table.length < table.offset) {
      return FONT_COMPRESSION_FAILURE();
    }
    intervals[table.offset] = table.length;
    table.data = data + table.offset;
    if (font->tables.find(table.tag) != font->tables.end()) {
      return FONT_COMPRESSION_FAILURE();
    }
    font->tables[table.tag] = table;
  }

  // Check that tables are non-overlapping.
  uint32_t last_offset = 12UL + 16UL * font->num_tables;
  for (const auto& i : intervals) {
    if (i.first < last_offset || i.first + i.second < i.first) {
      return FONT_COMPRESSION_FAILURE();
    }
    last_offset = i.first + i.second;
  }

  // Sanity check key tables
  const Font::Table* head_table = font->FindTable(kHeadTableTag);
  if (head_table != nullptr && head_table->length < 52) {
    return FONT_COMPRESSION_FAILURE();
  }

  return true;
}

auto ReadCollectionFont(Buffer* file, const uint8_t* data, size_t len,
                        Font* font,
                        std::map<uint32_t, Font::Table*>* all_tables) -> bool {
  if (!file->ReadU32(&font->flavor)) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (!ReadTrueTypeFont(file, data, len, font)) {
    return FONT_COMPRESSION_FAILURE();
  }

  for (auto& entry : font->tables) {
    Font::Table& table = entry.second;

    if (all_tables->find(table.offset) == all_tables->end()) {
      (*all_tables)[table.offset] = font->FindTable(table.tag);
    } else {
      table.reuse_of = (*all_tables)[table.offset];
      if (table.tag != table.reuse_of->tag) {
        return FONT_COMPRESSION_FAILURE();
      }
    }

  }
  return true;
}

auto ReadTrueTypeCollection(Buffer* file, const uint8_t* data, size_t len,
                            FontCollection* font_collection) -> bool {
    uint32_t num_fonts;

    if (!file->ReadU32(&font_collection->header_version) ||
        !file->ReadU32(&num_fonts)) {
      return FONT_COMPRESSION_FAILURE();
    }

    std::vector<uint32_t> offsets;
    for (size_t i = 0; i < num_fonts; i++) {
      uint32_t offset;
      if (!file->ReadU32(&offset)) {
        return FONT_COMPRESSION_FAILURE();
      }
      offsets.push_back(offset);
    }

    font_collection->fonts.resize(offsets.size());
    auto font_it = font_collection->fonts.begin();

    std::map<uint32_t, Font::Table*> all_tables;
    for (const auto offset : offsets) {
      file->set_offset(offset);
      Font& font = *font_it++;
      if (!ReadCollectionFont(file, data, len, &font, &all_tables)) {
        return FONT_COMPRESSION_FAILURE();
      }
    }

    return true;
}

auto ReadFont(const uint8_t* data, size_t len, Font* font) -> bool {
  Buffer file(data, len);

  if (!file.ReadU32(&font->flavor)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (font->flavor == kTtcFontFlavor) {
    return FONT_COMPRESSION_FAILURE();
  }
  return ReadTrueTypeFont(&file, data, len, font);
}

auto ReadFontCollection(const uint8_t* data, size_t len,
                        FontCollection* font_collection) -> bool {
  Buffer file(data, len);

  if (!file.ReadU32(&font_collection->flavor)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (font_collection->flavor != kTtcFontFlavor) {
    font_collection->fonts.resize(1);
    Font& font = font_collection->fonts[0];
    font.flavor = font_collection->flavor;
    return ReadTrueTypeFont(&file, data, len, &font);
  }
  return ReadTrueTypeCollection(&file, data, len, font_collection);
}

auto FontFileSize(const Font& font) -> size_t {
  size_t max_offset = 12ULL + 16ULL * font.num_tables;
  for (const auto& i : font.tables) {
    const Font::Table& table = i.second;
    size_t padding_size = (4 - (table.length & 3)) & 3;
    size_t end_offset = (padding_size + table.offset) + table.length;
    max_offset = std::max(max_offset, end_offset);
  }
  return max_offset;
}

auto FontCollectionFileSize(const FontCollection& font_collection) -> size_t {
  size_t max_offset = 0;
  for (auto& font : font_collection.fonts) {
    // font file size actually just finds max offset
    max_offset = std::max(max_offset, FontFileSize(font));
  }
  return max_offset;
}

auto WriteFont(const Font& font, uint8_t* dst, size_t dst_size) -> bool {
  size_t offset = 0;
  return WriteFont(font, &offset, dst, dst_size);
}

auto WriteTableRecord(const Font::Table* table, size_t* offset, uint8_t* dst,
                      size_t dst_size) -> bool {
  if (dst_size < *offset + kSfntEntrySize) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (table->IsReused()) {
    table = table->reuse_of;
  }
  StoreU32(table->tag, offset, dst);
  StoreU32(table->checksum, offset, dst);
  StoreU32(table->offset, offset, dst);
  StoreU32(table->length, offset, dst);
  return true;
}

auto WriteTable(const Font::Table& table, size_t* offset, uint8_t* dst,
                size_t dst_size) -> bool {
  if (!WriteTableRecord(&table, offset, dst, dst_size)) {
    return false;
  }

  // Write the actual table data if it's the first time we've seen it
  if (!table.IsReused()) {
    if (table.offset + table.length < table.offset ||
        dst_size < table.offset + table.length) {
      return FONT_COMPRESSION_FAILURE();
    }
    memcpy(dst + table.offset, table.data, table.length);
    size_t padding_size = (4 - (table.length & 3)) & 3;
    if (table.offset + table.length + padding_size < padding_size ||
        dst_size < table.offset + table.length + padding_size) {
      return FONT_COMPRESSION_FAILURE();
    }
    memset(dst + table.offset + table.length, 0, padding_size);
  }
  return true;
}

auto WriteFont(const Font& font, size_t* offset, uint8_t* dst,
               size_t dst_size) -> bool {
  if (dst_size < 12ULL + 16ULL * font.num_tables) {
    return FONT_COMPRESSION_FAILURE();
  }
  StoreU32(font.flavor, offset, dst);
  Store16(font.num_tables, offset, dst);
  uint16_t max_pow2 = font.num_tables ? Log2Floor(font.num_tables) : 0;
  uint16_t search_range = max_pow2 ? 1 << (max_pow2 + 4) : 0;
  uint16_t range_shift = (font.num_tables << 4) - search_range;
  Store16(search_range, offset, dst);
  Store16(max_pow2, offset, dst);
  Store16(range_shift, offset, dst);

  for (const auto& i : font.tables) {
    if (!WriteTable(i.second, offset, dst, dst_size)) {
      return false;
    }
  }

  return true;
}

auto WriteFontCollection(const FontCollection& font_collection, uint8_t* dst,
                         size_t dst_size) -> bool {
  size_t offset = 0;

  // It's simpler if this just a simple sfnt
  if (font_collection.flavor != kTtcFontFlavor) {
    return WriteFont(font_collection.fonts[0], &offset, dst, dst_size);
  }

  // Write TTC header
  StoreU32(kTtcFontFlavor, &offset, dst);
  StoreU32(font_collection.header_version, &offset, dst);
  StoreU32(font_collection.fonts.size(), &offset, dst);

  // Offset Table, zeroed for now
  size_t offset_table = offset;  // where to write offsets later
  for (size_t i = 0; i < font_collection.fonts.size(); i++) {
    StoreU32(0, &offset, dst);
  }

  if (font_collection.header_version == 0x00020000) {
    StoreU32(0, &offset, dst);  // ulDsigTag
    StoreU32(0, &offset, dst);  // ulDsigLength
    StoreU32(0, &offset, dst);  // ulDsigOffset
  }

  // Write fonts and their offsets.
  for (const auto & font : font_collection.fonts) {
    StoreU32(offset, &offset_table, dst);
    if (!WriteFont(font, &offset, dst, dst_size)) {
      return false;
    }
  }

  return true;
}

auto NumGlyphs(const Font& font) -> int {
  const Font::Table* head_table = font.FindTable(kHeadTableTag);
  const Font::Table* loca_table = font.FindTable(kLocaTableTag);
  if (head_table == nullptr || loca_table == nullptr || head_table->length < 52) {
    return 0;
  }
  int index_fmt = IndexFormat(font);
  int loca_record_size = (index_fmt == 0 ? 2 : 4);
  if (loca_table->length < loca_record_size) {
    return 0;
  }
  return (loca_table->length / loca_record_size) - 1;
}

auto IndexFormat(const Font& font) -> int {
  const Font::Table* head_table = font.FindTable(kHeadTableTag);
  if (head_table == nullptr) {
    return 0;
  }
  return head_table->data[51];
}

auto Font::Table::IsReused() const -> bool {
  return this->reuse_of != nullptr;
}

auto GetGlyphData(const Font& font, int glyph_index,
                  const uint8_t** glyph_data, size_t* glyph_size) -> bool {
  if (glyph_index < 0) {
    return FONT_COMPRESSION_FAILURE();
  }
  const Font::Table* head_table = font.FindTable(kHeadTableTag);
  const Font::Table* loca_table = font.FindTable(kLocaTableTag);
  const Font::Table* glyf_table = font.FindTable(kGlyfTableTag);
  if (head_table == nullptr || loca_table == nullptr || glyf_table == nullptr ||
      head_table->length < 52) {
    return FONT_COMPRESSION_FAILURE();
  }

  int index_fmt = IndexFormat(font);

  Buffer loca_buf(loca_table->data, loca_table->length);
  if (index_fmt == 0) {
    uint16_t offset1, offset2;
    if (!loca_buf.Skip(2 * glyph_index) ||
        !loca_buf.ReadU16(&offset1) ||
        !loca_buf.ReadU16(&offset2) ||
        offset2 < offset1 ||
        2 * offset2 > glyf_table->length) {
      return FONT_COMPRESSION_FAILURE();
    }
    *glyph_data = glyf_table->data + 2 * offset1;
    *glyph_size = 2 * (offset2 - offset1);
  } else {
    uint32_t offset1, offset2;
    if (!loca_buf.Skip(4 * glyph_index) ||
        !loca_buf.ReadU32(&offset1) ||
        !loca_buf.ReadU32(&offset2) ||
        offset2 < offset1 ||
        offset2 > glyf_table->length) {
      return FONT_COMPRESSION_FAILURE();
    }
    *glyph_data = glyf_table->data + offset1;
    *glyph_size = offset2 - offset1;
  }
  return true;
}

auto RemoveDigitalSignature(Font* font) -> bool {
  auto it =
      font->tables.find(kDsigTableTag);
  if (it != font->tables.end()) {
    font->tables.erase(it);
    font->num_tables = font->tables.size();
  }
  return true;
}

} // namespace woff2
