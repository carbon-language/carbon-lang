//===-- StringTableBuilder.cpp - String table building utility ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

#include <vector>

using namespace llvm;

StringTableBuilder::StringTableBuilder(Kind K) : K(K) {}

static int compareBySuffix(std::pair<StringRef, size_t> *const *AP,
                           std::pair<StringRef, size_t> *const *BP) {
  StringRef A = (*AP)->first;
  StringRef B = (*BP)->first;
  size_t SizeA = A.size();
  size_t SizeB = B.size();
  size_t Len = std::min(SizeA, SizeB);
  for (size_t I = 0; I < Len; ++I) {
    char CA = A[SizeA - I - 1];
    char CB = B[SizeB - I - 1];
    if (CA != CB)
      return CB - CA;
  }
  return SizeB - SizeA;
}

void StringTableBuilder::finalize() {
  std::vector<std::pair<StringRef, size_t> *> Strings;
  Strings.reserve(StringIndexMap.size());
  for (std::pair<StringRef, size_t> &P : StringIndexMap)
    Strings.push_back(&P);

  array_pod_sort(Strings.begin(), Strings.end(), compareBySuffix);

  switch (K) {
  case RAW:
    break;
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
  for (std::pair<StringRef, size_t> *P : Strings) {
    StringRef S = P->first;
    if (K == WinCOFF)
      assert(S.size() > COFF::NameSize && "Short string in COFF string table!");

    if (Previous.endswith(S)) {
      P->second = StringTable.size() - S.size() - (K != RAW);
      continue;
    }

    P->second = StringTable.size();
    StringTable += S;
    if (K != RAW)
      StringTable += '\x00';
    Previous = S;
  }

  switch (K) {
  case RAW:
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
    uint32_t Size = static_cast<uint32_t>(StringTable.size());
    support::endian::write<uint32_t, support::little, support::unaligned>(
        StringTable.data(), Size);
    break;
  }

  Size = StringTable.size();
}

void StringTableBuilder::clear() {
  StringTable.clear();
  StringIndexMap.clear();
}

size_t StringTableBuilder::getOffset(StringRef S) const {
  assert(isFinalized());
  auto I = StringIndexMap.find(S);
  assert(I != StringIndexMap.end() && "String is not in table!");
  return I->second;
}

size_t StringTableBuilder::add(StringRef S) {
  assert(!isFinalized());
  auto P = StringIndexMap.insert(std::make_pair(S, Size));
  if (P.second)
    Size += S.size() + (K != RAW);
  return P.first->second;
}
