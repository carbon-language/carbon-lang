/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting TTF format font files to their WOFF2 versions. */

#include <woff2/encode.h>

#include <stdlib.h>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <brotli/encode.h>
#include "./buffer.h"
#include "./font.h"
#include "./normalize.h"
#include "./round.h"
#include "./store_bytes.h"
#include "./table_tags.h"
#include "./transform.h"
#include "./variable_length.h"
#include "./woff2_common.h"

namespace woff2 {

namespace {


using std::string;
using std::vector;


const size_t kWoff2HeaderSize = 48;
const size_t kWoff2EntrySize = 20;

bool Compress(const uint8_t* data, const size_t len, uint8_t* result,
              uint32_t* result_len, BrotliEncoderMode mode, int quality) {
  size_t compressed_len = *result_len;
  if (BrotliEncoderCompress(quality, BROTLI_DEFAULT_WINDOW, mode, len, data,
                            &compressed_len, result) == 0) {
    return false;
  }
  *result_len = compressed_len;
  return true;
}

bool Woff2Compress(const uint8_t* data, const size_t len,
                   uint8_t* result, uint32_t* result_len,
                   int quality) {
  return Compress(data, len, result, result_len,
                  BROTLI_MODE_FONT, quality);
}

bool TextCompress(const uint8_t* data, const size_t len,
                  uint8_t* result, uint32_t* result_len,
                  int quality) {
  return Compress(data, len, result, result_len,
                  BROTLI_MODE_TEXT, quality);
}

int KnownTableIndex(uint32_t tag) {
  for (int i = 0; i < 63; ++i) {
    if (tag == kKnownTags[i]) return i;
  }
  return 63;
}

void StoreTableEntry(const Table& table, size_t* offset, uint8_t* dst) {
  uint8_t flag_byte = (table.flags & 0xC0) | KnownTableIndex(table.tag);
  dst[(*offset)++] = flag_byte;
  // The index here is treated as a set of flag bytes because
  // bits 6 and 7 of the byte are reserved for future use as flags.
  // 0x3f or 63 means an arbitrary table tag.
  if ((flag_byte & 0x3f) == 0x3f) {
    StoreU32(table.tag, offset, dst);
  }
  StoreBase128(table.src_length, offset, dst);
  if ((table.flags & kWoff2FlagsTransform) != 0) {
    StoreBase128(table.transform_length, offset, dst);
  }
}

size_t TableEntrySize(const Table& table) {
  uint8_t flag_byte = KnownTableIndex(table.tag);
  size_t size = ((flag_byte & 0x3f) != 0x3f) ? 1 : 5;
  size += Base128Size(table.src_length);
  if ((table.flags & kWoff2FlagsTransform) != 0) {
     size += Base128Size(table.transform_length);
  }
  return size;
}

size_t ComputeWoff2Length(const FontCollection& font_collection,
                          const std::vector<Table>& tables,
                          std::map<std::pair<uint32_t, uint32_t>, uint16_t>
                            index_by_tag_offset,
                          size_t compressed_data_length,
                          size_t extended_metadata_length) {
  size_t size = kWoff2HeaderSize;

  for (const auto& table : tables) {
    size += TableEntrySize(table);
  }

  // for collections only, collection tables
  if (font_collection.flavor == kTtcFontFlavor) {
    size += 4;  // UInt32 Version of TTC Header
    size += Size255UShort(font_collection.fonts.size());  // 255UInt16 numFonts

    size += 4 * font_collection.fonts.size();  // UInt32 flavor for each

    for (const auto& font : font_collection.fonts) {
      size += Size255UShort(font.tables.size());  // 255UInt16 numTables
      for (const auto& entry : font.tables) {
        const Font::Table& table = entry.second;
        // no collection entry for xform table
        if (table.tag & 0x80808080) continue;

        std::pair<uint32_t, uint32_t> tag_offset(table.tag, table.offset);
        uint16_t table_index = index_by_tag_offset[tag_offset];
        size += Size255UShort(table_index);  // 255UInt16 index entry
      }
    }
  }

  // compressed data
  size += compressed_data_length;
  size = Round4(size);

  size += extended_metadata_length;
  return size;
}

size_t ComputeUncompressedLength(const Font& font) {
  // sfnt header + offset table
  size_t size = 12 + 16 * font.num_tables;
  for (const auto& entry : font.tables) {
    const Font::Table& table = entry.second;
    if (table.tag & 0x80808080) continue;  // xform tables don't stay
    if (table.IsReused()) continue;  // don't have to pay twice
    size += Round4(table.length);
  }
  return size;
}

size_t ComputeUncompressedLength(const FontCollection& font_collection) {
  if (font_collection.flavor != kTtcFontFlavor) {
    return ComputeUncompressedLength(font_collection.fonts[0]);
  }
  size_t size = CollectionHeaderSize(font_collection.header_version,
    font_collection.fonts.size());
  for (const auto& font : font_collection.fonts) {
    size += ComputeUncompressedLength(font);
  }
  return size;
}

size_t ComputeTotalTransformLength(const Font& font) {
  size_t total = 0;
  for (const auto& i : font.tables) {
    const Font::Table& table = i.second;
    if (table.IsReused()) {
      continue;
    }
    if (table.tag & 0x80808080 || !font.FindTable(table.tag ^ 0x80808080)) {
      // Count transformed tables and non-transformed tables that do not have
      // transformed versions.
      total += table.length;
    }
  }
  return total;
}

}  // namespace

size_t MaxWOFF2CompressedSize(const uint8_t* data, size_t length) {
  return MaxWOFF2CompressedSize(data, length, "");
}

size_t MaxWOFF2CompressedSize(const uint8_t* data, size_t length,
    const string& extended_metadata) {
  // Except for the header size, which is 32 bytes larger in woff2 format,
  // all other parts should be smaller (table header in short format,
  // transformations and compression). Just to be sure, we will give some
  // headroom anyway.
  return length + 1024 + extended_metadata.length();
}

uint32_t CompressedBufferSize(uint32_t original_size) {
  return 1.2 * original_size + 10240;
}

bool TransformFontCollection(FontCollection* font_collection) {
  for (auto& font : font_collection->fonts) {
    if (!TransformGlyfAndLocaTables(&font)) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "glyf/loca transformation failed.\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
  }

  return true;
}

bool ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length) {
  WOFF2Params params;
  return ConvertTTFToWOFF2(data, length, result, result_length,
                           params);
}

bool ConvertTTFToWOFF2(const uint8_t *data, size_t length,
                       uint8_t *result, size_t *result_length,
                       const WOFF2Params& params) {
  FontCollection font_collection;
  if (!ReadFontCollection(data, length, &font_collection)) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "Parsing of the input font failed.\n");
#endif
    return FONT_COMPRESSION_FAILURE();
  }

  if (!NormalizeFontCollection(&font_collection)) {
    return FONT_COMPRESSION_FAILURE();
  }

  if (params.allow_transforms && !TransformFontCollection(&font_collection)) {
    return FONT_COMPRESSION_FAILURE();
  } else {
    // glyf/loca use 11 to flag "not transformed"
    for (auto& font : font_collection.fonts) {
      Font::Table* glyf_table = font.FindTable(kGlyfTableTag);
      Font::Table* loca_table = font.FindTable(kLocaTableTag);
      if (glyf_table) {
        glyf_table->flag_byte |= 0xc0;
      }
      if (loca_table) {
        loca_table->flag_byte |= 0xc0;
      }
    }
  }

  // Although the compressed size of each table in the final woff2 file won't
  // be larger than its transform_length, we have to allocate a large enough
  // buffer for the compressor, since the compressor can potentially increase
  // the size. If the compressor overflows this, it should return false and
  // then this function will also return false.

  size_t total_transform_length = 0;
  for (const auto& font : font_collection.fonts) {
    total_transform_length += ComputeTotalTransformLength(font);
  }
  size_t compression_buffer_size = CompressedBufferSize(total_transform_length);
  std::vector<uint8_t> compression_buf(compression_buffer_size);
  uint32_t total_compressed_length = compression_buffer_size;

  // Collect all transformed data into one place in output order.
  std::vector<uint8_t> transform_buf(total_transform_length);
  size_t transform_offset = 0;
  for (const auto& font : font_collection.fonts) {
    for (const auto tag : font.OutputOrderedTags()) {
      const Font::Table& original = font.tables.at(tag);
      if (original.IsReused()) continue;
      if (tag & 0x80808080) continue;
      const Font::Table* table_to_store = font.FindTable(tag ^ 0x80808080);
      if (table_to_store == NULL) table_to_store = &original;

      StoreBytes(table_to_store->data, table_to_store->length,
                 &transform_offset, &transform_buf[0]);
    }
  }

  // Compress all transformed data in one stream.
  if (!Woff2Compress(transform_buf.data(), total_transform_length,
                     &compression_buf[0],
                     &total_compressed_length,
                     params.brotli_quality)) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "Compression of combined table failed.\n");
#endif
    return FONT_COMPRESSION_FAILURE();
  }

#ifdef FONT_COMPRESSION_BIN
  fprintf(stderr, "Compressed %zu to %u.\n", total_transform_length,
          total_compressed_length);
#endif

  // Compress the extended metadata
  // TODO(user): how does this apply to collections
  uint32_t compressed_metadata_buf_length =
    CompressedBufferSize(params.extended_metadata.length());
  std::vector<uint8_t> compressed_metadata_buf(compressed_metadata_buf_length);

  if (params.extended_metadata.length() > 0) {
    if (!TextCompress((const uint8_t*)params.extended_metadata.data(),
                      params.extended_metadata.length(),
                      compressed_metadata_buf.data(),
                      &compressed_metadata_buf_length,
                      params.brotli_quality)) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "Compression of extended metadata failed.\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
  } else {
    compressed_metadata_buf_length = 0;
  }

  std::vector<Table> tables;
  std::map<std::pair<uint32_t, uint32_t>, uint16_t> index_by_tag_offset;

  for (const auto& font : font_collection.fonts) {

    for (const auto tag : font.OutputOrderedTags()) {
      const Font::Table& src_table = font.tables.at(tag);
      if (src_table.IsReused()) {
        continue;
      }

      std::pair<uint32_t, uint32_t> tag_offset(src_table.tag, src_table.offset);
      if (index_by_tag_offset.find(tag_offset) == index_by_tag_offset.end()) {
        index_by_tag_offset[tag_offset] = tables.size();
      } else {
        return false;
      }

      Table table;
      table.tag = src_table.tag;
      table.flags = src_table.flag_byte;
      table.src_length = src_table.length;
      table.transform_length = src_table.length;
      const uint8_t* transformed_data = src_table.data;
      const Font::Table* transformed_table =
          font.FindTable(src_table.tag ^ 0x80808080);
      if (transformed_table != NULL) {
        table.flags = transformed_table->flag_byte;
        table.flags |= kWoff2FlagsTransform;
        table.transform_length = transformed_table->length;
        transformed_data = transformed_table->data;

      }
      tables.push_back(table);
    }
  }

  size_t woff2_length = ComputeWoff2Length(font_collection, tables,
      index_by_tag_offset, total_compressed_length,
      compressed_metadata_buf_length);
  if (woff2_length > *result_length) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "Result allocation was too small (%zd vs %zd bytes).\n",
           *result_length, woff2_length);
#endif
    return FONT_COMPRESSION_FAILURE();
  }
  *result_length = woff2_length;

  size_t offset = 0;

  // start of woff2 header (http://www.w3.org/TR/WOFF2/#woff20Header)
  StoreU32(kWoff2Signature, &offset, result);
  if (font_collection.flavor != kTtcFontFlavor) {
    StoreU32(font_collection.fonts[0].flavor, &offset, result);
  } else {
    StoreU32(kTtcFontFlavor, &offset, result);
  }
  StoreU32(woff2_length, &offset, result);
  Store16(tables.size(), &offset, result);
  Store16(0, &offset, result);  // reserved
  // totalSfntSize
  StoreU32(ComputeUncompressedLength(font_collection), &offset, result);
  StoreU32(total_compressed_length, &offset, result);  // totalCompressedSize

  // Let's just all be v1.0
  Store16(1, &offset, result);  // majorVersion
  Store16(0, &offset, result);  // minorVersion
  if (compressed_metadata_buf_length > 0) {
    StoreU32(woff2_length - compressed_metadata_buf_length,
             &offset, result);  // metaOffset
    StoreU32(compressed_metadata_buf_length, &offset, result);  // metaLength
    StoreU32(params.extended_metadata.length(),
             &offset, result);  // metaOrigLength
  } else {
    StoreU32(0, &offset, result);  // metaOffset
    StoreU32(0, &offset, result);  // metaLength
    StoreU32(0, &offset, result);  // metaOrigLength
  }
  StoreU32(0, &offset, result);  // privOffset
  StoreU32(0, &offset, result);  // privLength
  // end of woff2 header

  // table directory (http://www.w3.org/TR/WOFF2/#table_dir_format)
  for (const auto& table : tables) {
    StoreTableEntry(table, &offset, result);
  }

  // for collections only, collection table directory
  if (font_collection.flavor == kTtcFontFlavor) {
    StoreU32(font_collection.header_version, &offset, result);
    Store255UShort(font_collection.fonts.size(), &offset, result);
    for (const Font& font : font_collection.fonts) {

      uint16_t num_tables = 0;
      for (const auto& entry : font.tables) {
        const Font::Table& table = entry.second;
        if (table.tag & 0x80808080) continue;  // don't write xform tables
        num_tables++;
      }
      Store255UShort(num_tables, &offset, result);

      StoreU32(font.flavor, &offset, result);
      for (const auto& entry : font.tables) {
        const Font::Table& table = entry.second;
        if (table.tag & 0x80808080) continue;  // don't write xform tables

        // for reused tables, only the original has an updated offset
        uint32_t table_offset =
          table.IsReused() ? table.reuse_of->offset : table.offset;
        uint32_t table_length =
          table.IsReused() ? table.reuse_of->length : table.length;
        std::pair<uint32_t, uint32_t> tag_offset(table.tag, table_offset);
        if (index_by_tag_offset.find(tag_offset) == index_by_tag_offset.end()) {
#ifdef FONT_COMPRESSION_BIN
fprintf(stderr, "Missing table index for offset 0x%08x\n",
                  table_offset);
#endif
          return FONT_COMPRESSION_FAILURE();
        }
        uint16_t index = index_by_tag_offset[tag_offset];
        Store255UShort(index, &offset, result);

      }

    }
  }

  // compressed data format (http://www.w3.org/TR/WOFF2/#table_format)

  StoreBytes(&compression_buf[0], total_compressed_length, &offset, result);
  offset = Round4(offset);

  StoreBytes(compressed_metadata_buf.data(), compressed_metadata_buf_length,
             &offset, result);

  if (*result_length != offset) {
#ifdef FONT_COMPRESSION_BIN
    fprintf(stderr, "Mismatch between computed and actual length "
            "(%zd vs %zd)\n", *result_length, offset);
#endif
    return FONT_COMPRESSION_FAILURE();
  }
  return true;
}

} // namespace woff2
