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

void StringTableBuilder::finalize() {
  SmallVector<StringRef, 8> Strings;
  for (auto i = StringIndexMap.begin(), e = StringIndexMap.end(); i != e; ++i)
    Strings.push_back(i->getKey());

  // Sort the vector so a string is sorted above its suffixes.
  std::sort(Strings.begin(), Strings.end(), [](StringRef A, StringRef B) {
    typedef std::reverse_iterator<StringRef::iterator> Reverse;
    return !std::lexicographical_compare(Reverse(A.end()), Reverse(A.begin()),
                                         Reverse(B.end()), Reverse(B.begin()));
  });

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
