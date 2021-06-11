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

fn Font::FindTable(tag: uint32_t) -> Font::Table* {
  var it: std::__map_iterator<std::__tree_iterator<std::__value_type<unsigned int, woff2::Font::Table>, std::__tree_node<std::__value_type<unsigned int, woff2::Font::Table>, void *> *, long>>;
  return it == tables.end() ? nullptr : &it->second;
}

fn Font::FindTable(tag: uint32_t) const -> const Font::Table* {
  var it: std::__map_const_iterator<std::__tree_const_iterator<std::__value_type<unsigned int, woff2::Font::Table>, std::__tree_node<std::__value_type<unsigned int, woff2::Font::Table>, void *> *, long>>;
  return it == tables.end() ? nullptr : &it->second;
}

fn Font::OutputOrderedTags() const -> std::vector<uint32_t> {
  var output_order: std::vector<uint32_t>;

  for (const auto& i var __begin1: std::__map_const_iterator<std::__tree_const_iterator<std::__value_type<unsigned int, woff2::Font::Table>, std::__tree_node<std::__value_type<unsigned int, woff2::Font::Table>, void *> *, long>> var __range1: const std::map<unsigned int, woff2::Font::Table> &) {
    var table: const Font::Table &;
    // This is a transformed table, we will write it together with the
    // original version.
    if (table.tag & 0x80808080) {
      continue;
    }
    output_order.push_back(table.tag);
  }

  // Alphabetize then put loca immediately after glyf
  var glyf_loc: std::__wrap_iter<unsigned int *>;
  var loca_loc: std::__wrap_iter<unsigned int *>;
  if (glyf_loc != output_order.end() && loca_loc != output_order.end()) {
    output_order.erase(loca_loc);
    output_order.insert(std::find(output_order.begin(), output_order.end(),
      kGlyfTableTag) + 1, kLocaTableTag);
  }

  return output_order;
}

fn ReadTrueTypeFont(file: woff2::Buffer *, data: const uint8_t *, len: size_t,
                      font: woff2::Font *) -> bool {
  // We don't care about the search_range, entry_selector and range_shift
  // fields, they will always be computed upon writing the font.
  if (!file->ReadU16(&font->num_tables) ||
      !file->Skip(6)) {
    return FONT_COMPRESSION_FAILURE();
  }

  var intervals: std::map<uint32_t, uint32_t>;
  for (var i: uint16_t; i < font->num_tables; ++i) {
    var table: Font::Table;
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
  var last_offset: uint32_t;
  for (const auto& i var __begin1: std::__map_iterator<std::__tree_iterator<std::__value_type<unsigned int, unsigned int>, std::__tree_node<std::__value_type<unsigned int, unsigned int>, void *> *, long>> var __range1: std::map<unsigned int, unsigned int> &) {
    if (i.first < last_offset || i.first + i.second < i.first) {
      return FONT_COMPRESSION_FAILURE();
    }
    last_offset = i.first + i.second;
  }

  // Sanity check key tables
  var head_table: const Font::Table *;
  if (head_table != nullptr && head_table->length < 52) {
    return FONT_COMPRESSION_FAILURE();
  }

  return true;
}

fn ReadCollectionFont(file: woff2::Buffer *, data: const uint8_t *, len: size_t,
                        font: woff2::Font *,
                        all_tables: std::map<uint32_t, Font::Table *> *) -> bool {
  if (!file->ReadU32(&font->flavor)) {
    return FONT_COMPRESSION_FAILURE();
  }
  if (!ReadTrueTypeFont(file, data, len, font)) {
    return FONT_COMPRESSION_FAILURE();
  }

  for (auto& entry var __begin1: std::__map_iterator<std::__tree_iterator<std::__value_type<unsigned int, woff2::Font::Table>, std::__tree_node<std::__value_type<unsigned int, woff2::Font::Table>, void *> *, long>> var __range1: std::map<unsigned int, woff2::Font::Table> &) {
    var table: Font::Table &;

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

fn ReadTrueTypeCollection(file: woff2::Buffer *, data: const uint8_t *, len: size_t,
                            font_collection: woff2::FontCollection *) -> bool {
    var num_fonts: uint32_t;

    if (!file->ReadU32(&font_collection->header_version) ||
        !file->ReadU32(&num_fonts)) {
      return FONT_COMPRESSION_FAILURE();
    }

    var offsets: std::vector<uint32_t>;
    for (var i: size_t; i < num_fonts; i++) {
      var offset: uint32_t;
      if (!file->ReadU32(&offset)) {
        return FONT_COMPRESSION_FAILURE();
      }
      offsets.push_back(offset);
    }

    font_collection->fonts.resize(offsets.size());
    var font_it: std::__wrap_iter<woff2::Font *>;

    var all_tables: std::map<uint32_t, Font::Table *>;
    for (const auto offset var __begin1: std::__wrap_iter<unsigned int *> var __range1: std::vector<unsigned int> &) {
      file->set_offset(offset);
      var font: woff2::Font &;
      if (!ReadCollectionFont(file, data, len, &font, &all_tables)) {
        return FONT_COMPRESSION_FAILURE();
      }
    }

    return true;
}

fn ReadFont(data: const uint8_t *, len: size_t, font: woff2::Font *) -> bool {
  var file: woff2::Buffer;

  if (!file.ReadU32(&font->flavor)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (font->flavor == kTtcFontFlavor) {
    return FONT_COMPRESSION_FAILURE();
  }
  return ReadTrueTypeFont(&file, data, len, font);
}

fn ReadFontCollection(data: const uint8_t *, len: size_t,
                        font_collection: woff2::FontCollection *) -> bool {
  var file: woff2::Buffer;

  if (!file.ReadU32(&font_collection->flavor)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (font_collection->flavor != kTtcFontFlavor) {
    font_collection->fonts.resize(1);
    var font: woff2::Font &;
    font.flavor = font_collection->flavor;
    return ReadTrueTypeFont(&file, data, len, &font);
  }
  return ReadTrueTypeCollection(&file, data, len, font_collection);
}

fn FontFileSize(font: const woff2::Font &) -> size_t {
  var max_offset: size_t;
  for (const auto& i var __begin1: std::__map_const_iterator<std::__tree_const_iterator<std::__value_type<unsigned int, woff2::Font::Table>, std::__tree_node<std::__value_type<unsigned int, woff2::Font::Table>, void *> *, long>> var __range1: const std::map<unsigned int, woff2::Font::Table> &) {
    var table: const Font::Table &;
    var padding_size: size_t;
    var end_offset: size_t;
    max_offset = std::max(max_offset, end_offset);
  }
  return max_offset;
}

fn FontCollectionFileSize(font_collection: const woff2::FontCollection &) -> size_t {
  var max_offset: size_t;
  for (auto& font var __begin1: std::__wrap_iter<const woff2::Font *> var __range1: const std::vector<woff2::Font> &) {
    // font file size actually just finds max offset
    max_offset = std::max(max_offset, FontFileSize(font));
  }
  return max_offset;
}

fn WriteFont(font: const woff2::Font &, dst: uint8_t *, dst_size: size_t) -> bool {
  var offset: size_t;
  return WriteFont(font, &offset, dst, dst_size);
}

fn WriteTableRecord(table: const Font::Table *, offset: size_t *, dst: uint8_t *,
                      dst_size: size_t) -> bool {
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

fn WriteTable(table: const Font::Table &, offset: size_t *, dst: uint8_t *,
                dst_size: size_t) -> bool {
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
    var padding_size: size_t;
    if (table.offset + table.length + padding_size < padding_size ||
        dst_size < table.offset + table.length + padding_size) {
      return FONT_COMPRESSION_FAILURE();
    }
    memset(dst + table.offset + table.length, 0, padding_size);
  }
  return true;
}

fn WriteFont(font: const woff2::Font &, offset: size_t *, dst: uint8_t *,
               dst_size: size_t) -> bool {
  if (dst_size < 12ULL + 16ULL * font.num_tables) {
    return FONT_COMPRESSION_FAILURE();
  }
  StoreU32(font.flavor, offset, dst);
  Store16(font.num_tables, offset, dst);
  var max_pow2: uint16_t;
  var search_range: uint16_t;
  var range_shift: uint16_t;
  Store16(search_range, offset, dst);
  Store16(max_pow2, offset, dst);
  Store16(range_shift, offset, dst);

  for (const auto& i var __begin1: std::__map_const_iterator<std::__tree_const_iterator<std::__value_type<unsigned int, woff2::Font::Table>, std::__tree_node<std::__value_type<unsigned int, woff2::Font::Table>, void *> *, long>> var __range1: const std::map<unsigned int, woff2::Font::Table> &) {
    if (!WriteTable(i.second, offset, dst, dst_size)) {
      return false;
    }
  }

  return true;
}

fn WriteFontCollection(font_collection: const woff2::FontCollection &, dst: uint8_t *,
                         dst_size: size_t) -> bool {
  var offset: size_t;

  // It's simpler if this just a simple sfnt
  if (font_collection.flavor != kTtcFontFlavor) {
    return WriteFont(font_collection.fonts[0], &offset, dst, dst_size);
  }

  // Write TTC header
  StoreU32(kTtcFontFlavor, &offset, dst);
  StoreU32(font_collection.header_version, &offset, dst);
  StoreU32(font_collection.fonts.size(), &offset, dst);

  // Offset Table, zeroed for now
  var offset_table: size_t;  // where to write offsets later
  for (var i: size_t; i < font_collection.fonts.size(); i++) {
    StoreU32(0, &offset, dst);
  }

  if (font_collection.header_version == 0x00020000) {
    StoreU32(0, &offset, dst);  // ulDsigTag
    StoreU32(0, &offset, dst);  // ulDsigLength
    StoreU32(0, &offset, dst);  // ulDsigOffset
  }

  // Write fonts and their offsets.
  for (const auto & font var __begin1: std::__wrap_iter<const woff2::Font *> var __range1: const std::vector<woff2::Font> &) {
    StoreU32(offset, &offset_table, dst);
    if (!WriteFont(font, &offset, dst, dst_size)) {
      return false;
    }
  }

  return true;
}

fn NumGlyphs(font: const woff2::Font &) -> int {
  var head_table: const Font::Table *;
  var loca_table: const Font::Table *;
  if (head_table == nullptr || loca_table == nullptr || head_table->length < 52) {
    return 0;
  }
  var index_fmt: int;
  var loca_record_size: int;
  if (loca_table->length < loca_record_size) {
    return 0;
  }
  return (loca_table->length / loca_record_size) - 1;
}

fn IndexFormat(font: const woff2::Font &) -> int {
  var head_table: const Font::Table *;
  if (head_table == nullptr) {
    return 0;
  }
  return head_table->data[51];
}

fn Font::Table::IsReused() const -> bool {
  return this->reuse_of != nullptr;
}

fn GetGlyphData(font: const woff2::Font &, glyph_index: int,
                  glyph_data: const uint8_t **, glyph_size: size_t *) -> bool {
  if (glyph_index < 0) {
    return FONT_COMPRESSION_FAILURE();
  }
  var head_table: const Font::Table *;
  var loca_table: const Font::Table *;
  var glyf_table: const Font::Table *;
  if (head_table == nullptr || loca_table == nullptr || glyf_table == nullptr ||
      head_table->length < 52) {
    return FONT_COMPRESSION_FAILURE();
  }

  var index_fmt: int;

  var loca_buf: woff2::Buffer;
  if (index_fmt == 0) {
    var offset1: uint16_t, var offset2: uint16_t;
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
    var offset1: uint32_t, var offset2: uint32_t;
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

fn RemoveDigitalSignature(font: woff2::Font *) -> bool {
  var it: std::__map_iterator<std::__tree_iterator<std::__value_type<unsigned int, woff2::Font::Table>, std::__tree_node<std::__value_type<unsigned int, woff2::Font::Table>, void *> *, long>>;
  if (it != font->tables.end()) {
    font->tables.erase(it);
    font->num_tables = font->tables.size();
  }
  return true;
}

} // namespace woff2
