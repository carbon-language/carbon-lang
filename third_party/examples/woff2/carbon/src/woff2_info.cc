/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* A commandline tool for dumping info about a woff2 file. */

#include <string>

#include "file.h"
#include "./woff2_common.h"
#include "./buffer.h"
#include "./font.h"
#include "./table_tags.h"
#include "./variable_length.h"

std::string PrintTag(int tag) {
  if (tag & 0x80808080) {
    return std::string("_xfm");  // print _xfm for xform tables (else garbage)
  }
  char printable[] = {
    static_cast<char>((tag >> 24) & 0xFF),
    static_cast<char>((tag >> 16) & 0xFF),
    static_cast<char>((tag >> 8) & 0xFF),
    static_cast<char>(tag & 0xFF)
  };
  return std::string(printable, 4);
}

int main(int argc, char **argv) {
  using std::string;

  if (argc != 2) {
    fprintf(stderr, "One argument, the input filename, must be provided.\n");
    return 1;
  }

  string filename(argv[1]);
  string outfilename = filename.substr(0, filename.find_last_of(".")) + ".woff2";
  fprintf(stdout, "Processing %s => %s\n",
    filename.c_str(), outfilename.c_str());
  string input = woff2::GetFileContent(filename);

  woff2::Buffer file(reinterpret_cast<const uint8_t*>(input.data()),
    input.size());

  printf("WOFF2Header\n");
  uint32_t signature, flavor, length, totalSfntSize, totalCompressedSize;
  uint32_t metaOffset, metaLength, metaOrigLength, privOffset, privLength;
  uint16_t num_tables, reserved, major, minor;
  if (!file.ReadU32(&signature)) return 1;
  if (!file.ReadU32(&flavor)) return 1;
  if (!file.ReadU32(&length)) return 1;
  if (!file.ReadU16(&num_tables)) return 1;
  if (!file.ReadU16(&reserved)) return 1;
  if (!file.ReadU32(&totalSfntSize)) return 1;
  if (!file.ReadU32(&totalCompressedSize)) return 1;
  if (!file.ReadU16(&major)) return 1;
  if (!file.ReadU16(&minor)) return 1;
  if (!file.ReadU32(&metaOffset)) return 1;
  if (!file.ReadU32(&metaLength)) return 1;
  if (!file.ReadU32(&metaOrigLength)) return 1;
  if (!file.ReadU32(&privOffset)) return 1;
  if (!file.ReadU32(&privLength)) return 1;

  if (signature != 0x774F4632) {
    printf("Invalid signature: %08x\n", signature);
    return 1;
  }
  printf("signature           0x%08x\n", signature);
  printf("flavor              0x%08x\n", flavor);
  printf("length              %d\n", length);
  printf("numTables           %d\n", num_tables);
  printf("reserved            %d\n", reserved);
  printf("totalSfntSize       %d\n", totalSfntSize);
  printf("totalCompressedSize %d\n", totalCompressedSize);
  printf("majorVersion        %d\n", major);
  printf("minorVersion        %d\n", minor);
  printf("metaOffset          %d\n", metaOffset);
  printf("metaLength          %d\n", metaLength);
  printf("metaOrigLength      %d\n", metaOrigLength);
  printf("privOffset          %d\n", privOffset);
  printf("privLength          %d\n", privLength);

  std::vector<uint32_t> table_tags;
  printf("TableDirectory starts at +%zu\n", file.offset());
  printf("Entry offset flags tag  origLength txLength\n");
  for (auto i = 0; i < num_tables; i++) {
    size_t offset = file.offset();
    uint8_t flags;
    uint32_t tag, origLength, transformLength;
    if (!file.ReadU8(&flags)) return 1;
    if ((flags & 0x3f) == 0x3f) {
      if (!file.ReadU32(&tag)) return 1;
    } else {
      tag = woff2::kKnownTags[flags & 0x3f];
    }
    table_tags.push_back(tag);
    if (!ReadBase128(&file, &origLength)) return 1;

    printf("%5d %6zu  0x%02x %s %10d", i, offset, flags,
        PrintTag(tag).c_str(), origLength);

    uint8_t xform_version = (flags >> 6) & 0x3;
    if (tag == woff2::kGlyfTableTag || tag == woff2::kLocaTableTag) {
      if (xform_version == 0) {
        if (!ReadBase128(&file, &transformLength)) return 1;
        printf(" %8d", transformLength);
      }
    } else if (xform_version > 0) {
      if (!ReadBase128(&file, &transformLength)) return 1;
      printf(" %8d", transformLength);
    }
    printf("\n");
  }

  // Collection header
  if (flavor == woff2::kTtcFontFlavor) {
    uint32_t version, numFonts;
    if (!file.ReadU32(&version)) return 1;
    if (!woff2::Read255UShort(&file, &numFonts)) return 1;
    printf("CollectionHeader 0x%08x %d fonts\n", version, numFonts);

    for (auto i = 0; i < numFonts; i++) {
      uint32_t numTables, flavor;
      if (!woff2::Read255UShort(&file, &numTables)) return 1;
      if (!file.ReadU32(&flavor)) return 1;
      printf("CollectionFontEntry %d flavor 0x%08x %d tables\n", i, flavor,
          numTables);
      for (auto j = 0; j < numTables; j++) {
        uint32_t table_idx;
        if (!woff2::Read255UShort(&file, &table_idx)) return 1;
        if (table_idx >= table_tags.size()) return 1;
        printf("  %d %s (idx %d)\n", j,
            PrintTag(table_tags[table_idx]).c_str(), table_idx);
      }
    }
  }

  printf("TableDirectory ends at +%zu\n", file.offset());

  return 0;
}
