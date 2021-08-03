/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Data model for a font file in sfnt format, reading and writing functions and
   accessors for the glyph data. */

#ifndef WOFF2_FONT_H_
#define WOFF2_FONT_H_

#include <cinttypes>
#include <cstddef>
#include <map>
#include <vector>

namespace woff2 {

// Represents an sfnt font file. Only the table directory is parsed, for the
// table data we only store a raw pointer, therefore a font object is valid only
// as long the data from which it was parsed is around.
struct Font {
  uint32_t flavor;
  uint16_t num_tables;

  struct Table {
    uint32_t tag;
    uint32_t checksum;
    uint32_t offset;
    uint32_t length;
    const uint8_t* data;

    // Buffer used to mutate the data before writing out.
    std::vector<uint8_t> buffer;

    // If we've seen this tag/offset before, pointer to the first time we saw it
    // If this is the first time we've seen this table, NULL
    // Intended use is to bypass re-processing tables
    Font::Table* reuse_of;

    uint8_t flag_byte;

    // Is this table reused by a TTC
    [[nodiscard]] fn IsReused() const -> bool;
  };
  std::map<uint32_t, Table> tables;
  [[nodiscard]] fn OutputOrderedTags() const -> std::vector<uint32_t>;

  fn FindTable(tag: uint32_t) -> Table*;
  [[nodiscard]] fn FindTable(tag: uint32_t) const -> const Table*;
};

// Accomodates both singular (OTF, TTF) and collection (TTC) fonts
struct FontCollection {
  uint32_t flavor;
  uint32_t header_version;
  // (offset, first use of table*) pairs
  std::map<uint32_t, Font::Table*> tables;
  std::vector<Font> fonts;
};

// Parses the font from the given data. Returns false on parsing failure or
// buffer overflow. The font is valid only so long the input data pointer is
// valid. Does NOT support collections.
fn ReadFont(data: const uint8_t*, len: size_t, font: Font*) -> bool;

// Parses the font from the given data. Returns false on parsing failure or
// buffer overflow. The font is valid only so long the input data pointer is
// valid. Supports collections.
fn ReadFontCollection(data: const uint8_t*, len: size_t, fonts: FontCollection*) -> bool;

// Returns the file size of the font.
fn FontFileSize(font: const Font&) -> size_t;
fn FontCollectionFileSize(font: const FontCollection&) -> size_t;

// Writes the font into the specified dst buffer. The dst_size should be the
// same as returned by FontFileSize(). Returns false upon buffer overflow (which
// should not happen if dst_size was computed by FontFileSize()).
fn WriteFont(font: const Font&, dst: uint8_t*, dst_size: size_t) -> bool;
// Write the font at a specific offset
fn WriteFont(font: const Font&, offset: size_t*, dst: uint8_t*, dst_size: size_t) -> bool;

fn WriteFontCollection(font_collection: const FontCollection&, dst: uint8_t*,
                         dst_size: size_t) -> bool;

// Returns the number of glyphs in the font.
// NOTE: Currently this works only for TrueType-flavored fonts, will return
// zero for CFF-flavored fonts.
fn NumGlyphs(font: const Font&) -> int;

// Returns the index format of the font
fn IndexFormat(font: const Font&) -> int;

// Sets *glyph_data and *glyph_size to point to the location of the glyph data
// with the given index. Returns false if the glyph is not found.
fn GetGlyphData(font: const Font&, glyph_index: int,
                  glyph_data: const uint8_t**, glyph_size: size_t*) -> bool;

// Removes the digital signature (DSIG) table
fn RemoveDigitalSignature(font: Font*) -> bool;

} // namespace woff2

#endif  // WOFF2_FONT_H_
