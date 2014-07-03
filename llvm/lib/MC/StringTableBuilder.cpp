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

using namespace llvm;

static bool compareBySuffix(StringRef a, StringRef b) {
  size_t sizeA = a.size();
  size_t sizeB = b.size();
  size_t len = std::min(sizeA, sizeB);
  for (size_t i = 0; i < len; ++i) {
    char ca = a[sizeA - i - 1];
    char cb = b[sizeB - i - 1];
    if (ca != cb)
      return ca > cb;
  }
  return sizeA > sizeB;
}

void StringTableBuilder::finalize() {
  SmallVector<StringRef, 8> Strings;
  for (auto i = StringIndexMap.begin(), e = StringIndexMap.end(); i != e; ++i)
    Strings.push_back(i->getKey());

  std::sort(Strings.begin(), Strings.end(), compareBySuffix);

  // FIXME: Starting with a null byte is ELF specific. Generalize this so we
  // can use the class with other object formats.
  StringTable += '\x00';

  StringRef Previous;
  for (StringRef s : Strings) {
    if (Previous.endswith(s)) {
      StringIndexMap[s] = StringTable.size() - 1 - s.size();
      continue;
    }

    StringIndexMap[s] = StringTable.size();
    StringTable += s;
    StringTable += '\x00';
    Previous = s;
  }
}
