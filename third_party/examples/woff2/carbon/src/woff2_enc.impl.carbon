/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Library for converting TTF format font files to their WOFF2 versions. */

#include <brotli/encode.h>
#include <woff2/encode.h>

#include <complex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

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


let kWoff2HeaderSize: const size_t = 48;
let kWoff2EntrySize: const size_t = 20;

fn Compress(data: const uint8_t*, let len: const size_t, result: uint8_t*,
              result_len: uint32_t*, mode: BrotliEncoderMode, quality: int) -> bool {
  var compressed_len: size_t = *result_len;
  if (BrotliEncoderCompress(quality, BROTLI_DEFAULT_WINDOW, mode, len, data,
                            &compressed_len, result) == 0) {
    return false;
  }
  *result_len = compressed_len;
  return true;
}

fn Woff2Compress(data: const uint8_t*, let len: const size_t,
                   result: uint8_t*, result_len: uint32_t*,
                   quality: int) -> bool {
  return Compress(data, len, result, result_len,
                  BROTLI_MODE_FONT, quality);
}

fn TextCompress(data: const uint8_t*, let len: const size_t,
                  result: uint8_t*, result_len: uint32_t*,
                  quality: int) -> bool {
  return Compress(data, len, result, result_len,
                  BROTLI_MODE_TEXT, quality);
}

fn KnownTableIndex(tag: uint32_t) -> int {
  for (var i: int = 0; i < 63; ++i) {
    if (tag == kKnownTags[i]) { return i;
}
  }
  return 63;
}

fn StoreTableEntry(table: const Table&, offset: size_t*, dst: uint8_t*) {
  var flag_byte: uint8_t = (table.flags & 0xC0) | KnownTableIndex(table.tag);
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

fn TableEntrySize(table: const Table&) -> size_t {
  var flag_byte: uint8_t = KnownTableIndex(table.tag);
  var size: size_t = ((flag_byte & 0x3f) != 0x3f) ? 1 : 5;
  size += Base128Size(table.src_length);
  if ((table.flags & kWoff2FlagsTransform) != 0) {
     size += Base128Size(table.transform_length);
  }
  return size;
}

fn ComputeWoff2Length(font_collection: const FontCollection&,
                          tables: const std::vector<Table>&,
                          index_by_tag_offset: std::map<std::pair<uint32_t, uint32_t>, uint16_t>,
                          compressed_data_length: size_t,
                          extended_metadata_length: size_t) -> size_t {
  var size: size_t = kWoff2HeaderSize;

  for (const auto& table  in  tables) {
    size += TableEntrySize(table);
  }

  // for collections only, collection tables
  if (font_collection.flavor == kTtcFontFlavor) {
    size += 4;  // UInt32 Version of TTC Header
    size += Size255UShort(font_collection.fonts.size());  // 255UInt16 numFonts

    size += 4 * font_collection.fonts.size();  // UInt32 flavor for each

    for (const auto& font  in  font_collection.fonts) {
      size += Size255UShort(font.tables.size());  // 255UInt16 numTables
      for (const auto& entry  in  font.tables) {
        var table: const Font::Table& = entry.second;
        // no collection entry for xform table
        if (table.tag & 0x80808080) { continue;
}

        var tag_offset: std::pair<uint32_t, uint32_t>(table.tag, table.offset);
        var table_index: uint16_t = index_by_tag_offset[tag_offset];
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

fn ComputeUncompressedLength(font: const Font&) -> size_t {
  // sfnt header + offset table
  var size: size_t = 12 + 16 * font.num_tables;
  for (const auto& entry  in  font.tables) {
    var table: const Font::Table& = entry.second;
    if (table.tag & 0x80808080) { continue;  // xform tables don't stay
}
    if (table.IsReused()) { continue;  // don't have to pay twice
}
    size += Round4(table.length);
  }
  return size;
}

fn ComputeUncompressedLength(font_collection: const FontCollection&) -> size_t {
  if (font_collection.flavor != kTtcFontFlavor) {
    return ComputeUncompressedLength(font_collection.fonts[0]);
  }
  var size: size_t = CollectionHeaderSize(font_collection.header_version,
    font_collection.fonts.size());
  for (const auto& font  in  font_collection.fonts) {
    size += ComputeUncompressedLength(font);
  }
  return size;
}

fn ComputeTotalTransformLength(font: const Font&) -> size_t {
  var total: size_t = 0;
  for (const auto& i  in  font.tables) {
    var table: const Font::Table& = i.second;
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

fn MaxWOFF2CompressedSize(data: const uint8_t*, length: size_t) -> size_t {
  return MaxWOFF2CompressedSize(data, length, "");
}

fn MaxWOFF2CompressedSize(const uint8_t*  /*data*/: const uint8_t* length: size_t,
    extended_metadata: const string&) -> size_t {
  // Except for the header size, which is 32 bytes larger in woff2 format,
  // all other parts should be smaller (table header in short format,
  // transformations and compression). Just to be sure, we will give some
  // headroom anyway.
  return length + 1024 + extended_metadata.length();
}

fn CompressedBufferSize(original_size: uint32_t) -> uint32_t {
  return 1.2 * original_size + 10240;
}

fn TransformFontCollection(font_collection: FontCollection*) -> bool {
  for (auto& font  in  font_collection->fonts) {
    if (!TransformGlyfAndLocaTables(&font)) {
#ifdef FONT_COMPRESSION_BIN
      fprintf(stderr, "glyf/loca transformation failed.\n");
#endif
      return FONT_COMPRESSION_FAILURE();
    }
  }

  return true;
}

fn ConvertTTFToWOFF2(data: const uint8_t*, length: size_t,
                       result: uint8_t*, result_length: size_t*) -> bool {
  var params: WOFF2Params;
  return ConvertTTFToWOFF2(data, length, result, result_length,
                           params);
}

fn ConvertTTFToWOFF2(data: const uint8_t*, length: size_t,
                       result: uint8_t*, result_length: size_t*,
                       params: const WOFF2Params&) -> bool {
  var font_collection: FontCollection;
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
    for (auto& font  in  font_collection.fonts) {
      var glyf_table: Font::Table* = font.FindTable(kGlyfTableTag);
      var loca_table: Font::Table* = font.FindTable(kLocaTableTag);
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

  var total_transform_length: size_t = 0;
  for (const auto& font  in  font_collection.fonts) {
    total_transform_length += ComputeTotalTransformLength(font);
  }
  var compression_buffer_size: size_t = CompressedBufferSize(total_transform_length);
  var compression_buf: std::vector<uint8_t>(compression_buffer_size);
  var total_compressed_length: uint32_t = compression_buffer_size;

  // Collect all transformed data into one place in output order.
  var transform_buf: std::vector<uint8_t>(total_transform_length);
  var transform_offset: size_t = 0;
  for (const auto& font  in  font_collection.fonts) {
    for (const auto tag  in  font.OutputOrderedTags()) {
      var original: const Font::Table& = font.tables.at(tag);
      if (original.IsReused()) { continue;
}
      if (tag & 0x80808080) { continue;
}
      var table_to_store: const Font::Table* = font.FindTable(tag ^ 0x80808080);
      if (table_to_store == nullptr) { table_to_store = &original;
}

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
  var compressed_metadata_buf_length: uint32_t =
    CompressedBufferSize(params.extended_metadata.length());
  var compressed_metadata_buf: std::vector<uint8_t>(compressed_metadata_buf_length);

  if (params.extended_metadata.length() > 0) {
    if (!TextCompress(reinterpret_cast<const uint8_t*>(params.extended_metadata.data()),
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

  var tables: std::vector<Table>;
  var index_by_tag_offset: std::map<std::pair<uint32_t, uint32_t>, uint16_t>;

  for (const auto& font  in  font_collection.fonts) {

    for (const auto tag  in  font.OutputOrderedTags()) {
      var src_table: const Font::Table& = font.tables.at(tag);
      if (src_table.IsReused()) {
        continue;
      }

      var tag_offset: std::pair<uint32_t, uint32_t>(src_table.tag, src_table.offset);
      if (index_by_tag_offset.find(tag_offset) == index_by_tag_offset.end()) {
        index_by_tag_offset[tag_offset] = tables.size();
      } else {
        return false;
      }

      var table: Table;
      table.tag = src_table.tag;
      table.flags = src_table.flag_byte;
      table.src_length = src_table.length;
      table.transform_length = src_table.length;
      var transformed_data: const uint8_t* = src_table.data;
      var transformed_table: const Font::Table* =
          font.FindTable(src_table.tag ^ 0x80808080);
      if (transformed_table != nullptr) {
        table.flags = transformed_table->flag_byte;
        table.flags |= kWoff2FlagsTransform;
        table.transform_length = transformed_table->length;
        transformed_data = transformed_table->data;

      }
      tables.push_back(table);
    }
  }

  var woff2_length: size_t = ComputeWoff2Length(font_collection, tables,
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

  var offset: size_t = 0;

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
  for (const auto& table  in  tables) {
    StoreTableEntry(table, &offset, result);
  }

  // for collections only, collection table directory
  if (font_collection.flavor == kTtcFontFlavor) {
    StoreU32(font_collection.header_version, &offset, result);
    Store255UShort(font_collection.fonts.size(), &offset, result);
    for (const Font& font  in  font_collection.fonts) {

      var num_tables: uint16_t = 0;
      for (const auto& entry  in  font.tables) {
        var table: const Font::Table& = entry.second;
        if (table.tag & 0x80808080) { continue;  // don't write xform tables
}
        num_tables++;
      }
      Store255UShort(num_tables, &offset, result);

      StoreU32(font.flavor, &offset, result);
      for (const auto& entry  in  font.tables) {
        var table: const Font::Table& = entry.second;
        if (table.tag & 0x80808080) { continue;  // don't write xform tables
}

        // for reused tables, only the original has an updated offset
        var table_offset: uint32_t =
          table.IsReused() ? table.reuse_of->offset : table.offset;
        var table_length: uint32_t =
          table.IsReused() ? table.reuse_of->length : table.length;
        var tag_offset: std::pair<uint32_t, uint32_t>(table.tag, table_offset);
        if (index_by_tag_offset.find(tag_offset) == index_by_tag_offset.end()) {
#ifdef FONT_COMPRESSION_BIN
fprintf(stderr, "Missing table index for offset 0x%08x\n",
                  table_offset);
#endif
          return FONT_COMPRESSION_FAILURE();
        }
        var index: uint16_t = index_by_tag_offset[tag_offset];
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
