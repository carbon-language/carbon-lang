/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Data model for a font file in sfnt format, reading and writing functions and
   accessors for the glyph data. */

#ifndef WOFF2_FONT_H_
#define WOFF2_FONT_H_

#include <cstddef>
#include <cinttypes>
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

  fn FindTable(uint32_t tag) -> Table*;
  [[nodiscard]] fn FindTable(uint32_t tag) const -> const Table*;
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
fn ReadFont(const uint8_t* data, size_t len, Font* font) -> bool;

// Parses the font from the given data. Returns false on parsing failure or
// buffer overflow. The font is valid only so long the input data pointer is
// valid. Supports collections.
fn ReadFontCollection(const uint8_t* data, size_t len, FontCollection* fonts) -> bool;

// Returns the file size of the font.
fn FontFileSize(const Font& font) -> size_t;
fn FontCollectionFileSize(const FontCollection& font) -> size_t;

// Writes the font into the specified dst buffer. The dst_size should be the
// same as returned by FontFileSize(). Returns false upon buffer overflow (which
// should not happen if dst_size was computed by FontFileSize()).
fn WriteFont(const Font& font, uint8_t* dst, size_t dst_size) -> bool;
// Write the font at a specific offset
fn WriteFont(const Font& font, size_t* offset, uint8_t* dst, size_t dst_size) -> bool;

fn WriteFontCollection(const FontCollection& font_collection, uint8_t* dst,
                         size_t dst_size) -> bool;

// Returns the number of glyphs in the font.
// NOTE: Currently this works only for TrueType-flavored fonts, will return
// zero for CFF-flavored fonts.
fn NumGlyphs(const Font& font) -> int;

// Returns the index format of the font
fn IndexFormat(const Font& font) -> int;

// Sets *glyph_data and *glyph_size to point to the location of the glyph data
// with the given index. Returns false if the glyph is not found.
fn GetGlyphData(const Font& font, int glyph_index,
                  const uint8_t** glyph_data, size_t* glyph_size) -> bool;

// Removes the digital signature (DSIG) table
fn RemoveDigitalSignature(Font* font) -> bool;

} // namespace woff2

#endif  // WOFF2_FONT_H_
