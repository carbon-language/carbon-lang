//===-- StringTableBuilder.cpp - String table building utility ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;

static int compareBySuffix(StringMapEntry<size_t> *const *AP,
                           StringMapEntry<size_t> *const *BP) {
  StringRef a = (*AP)->first();
  StringRef b = (*BP)->first();
  size_t sizeA = a.size();
  size_t sizeB = b.size();
  size_t len = std::min(sizeA, sizeB);
  for (size_t i = 0; i < len; ++i) {
    char ca = a[sizeA - i - 1];
    char cb = b[sizeB - i - 1];
    if (ca != cb)
      return cb - ca;
  }
  return sizeB - sizeA;
}

void StringTableBuilder::finalize(Kind kind) {
  std::vector<StringMapEntry<size_t> *> Strings;
  Strings.reserve(StringIndexMap.size());
  for (StringMapEntry<size_t> &P : StringIndexMap)
    Strings.push_back(&P);

  array_pod_sort(Strings.begin(), Strings.end(), compareBySuffix);

  switch (kind) {
  case ELF:
  case MachO:
    // Start the table with a NUL byte.
    StringTable += '\x00';
    break;
  case WinCOFF:
    // Make room to write the table size later.
    StringTable.append(4, '\x00');
    break;
  }

  StringRef Previous;
  for (StringMapEntry<size_t> *P : Strings) {
    StringRef s = P->first();
    if (kind == WinCOFF)
      assert(s.size() > COFF::NameSize && "Short string in COFF string table!");

    if (Previous.endswith(s)) {
      P->second = StringTable.size() - 1 - s.size();
      continue;
    }

    P->second = StringTable.size();
    StringTable += s;
    StringTable += '\x00';
    Previous = s;
  }

  switch (kind) {
  case ELF:
    break;
  case MachO:
    // Pad to multiple of 4.
    while (StringTable.size() % 4)
      StringTable += '\x00';
    break;
  case WinCOFF:
    // Write the table size in the first word.
    assert(StringTable.size() <= std::numeric_limits<uint32_t>::max());
    uint32_t size = static_cast<uint32_t>(StringTable.size());
    support::endian::write<uint32_t, support::little, support::unaligned>(
        StringTable.data(), size);
    break;
  }
}

void StringTableBuilder::clear() {
  StringTable.clear();
  StringIndexMap.clear();
}
