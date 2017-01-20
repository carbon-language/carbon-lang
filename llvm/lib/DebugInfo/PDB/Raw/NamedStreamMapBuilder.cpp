//===- NamedStreamMapBuilder.cpp - PDB Named Stream Map Builder -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/NamedStreamMapBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/NamedStreamMap.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstdint>

using namespace llvm;
using namespace llvm::pdb;

NamedStreamMapBuilder::NamedStreamMapBuilder() = default;

void NamedStreamMapBuilder::addMapping(StringRef Name, uint32_t Mapping) {
  Strings.push_back(Name);
  Map.set(Offset, Mapping);
  Offset += Name.size() + 1;
}

uint32_t NamedStreamMapBuilder::calculateSerializedLength() const {
  uint32_t TotalLength = 0;

  // Number of bytes of string data.
  TotalLength += sizeof(support::ulittle32_t);
  // Followed by that many actual bytes of string data.
  TotalLength += Offset;
  // Followed by the mapping from Name to Index.
  TotalLength += Map.calculateSerializedLength();

  return TotalLength;
}

Error NamedStreamMapBuilder::commit(msf::StreamWriter &Writer) const {
  // The first field is the number of bytes of string data.  We've already been
  // keeping a running total of this in `Offset`.
  if (auto EC = Writer.writeInteger(Offset)) // Number of bytes of string data
    return EC;

  // Now all of the string data itself.
  for (auto S : Strings) {
    if (auto EC = Writer.writeZeroString(S))
      return EC;
  }

  // And finally the Linear Map.
  if (auto EC = Map.commit(Writer))
    return EC;

  return Error::success();
}
