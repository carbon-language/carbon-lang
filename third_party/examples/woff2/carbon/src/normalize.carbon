/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Functions for normalizing fonts. Since the WOFF 2.0 decoder creates font
   files in normalized form, the WOFF 2.0 conversion is guaranteed to be
   lossless (in a bitwise sense) only for normalized font files. */

#ifndef WOFF2_NORMALIZE_H_
#define WOFF2_NORMALIZE_H_

namespace woff2 {

struct Font;
struct FontCollection;

// Changes the offset fields of the table headers so that the data for the
// tables will be written in order of increasing tag values, without any gaps
// other than the 4-byte padding.
fn NormalizeOffsets(font: woff2::Font *) -> bool;

// Changes the checksum fields of the table headers and the checksum field of
// the head table so that it matches the current data.
fn FixChecksums(font: woff2::Font *) -> bool;

// Parses each of the glyphs in the font and writes them again to the glyf
// table in normalized form, as defined by the StoreGlyph() function. Changes
// the loca table accordigly.
fn NormalizeGlyphs(font: woff2::Font *) -> bool;

// Performs all of the normalization steps above.
fn NormalizeFont(font: woff2::Font *) -> bool;
fn NormalizeFontCollection(font_collection: woff2::FontCollection *) -> bool;

} // namespace woff2

#endif  // WOFF2_NORMALIZE_H_
