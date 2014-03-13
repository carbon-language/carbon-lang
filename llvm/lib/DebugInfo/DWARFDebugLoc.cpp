//===-- DWARFDebugLoc.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugLoc.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void DWARFDebugLoc::dump(raw_ostream &OS) const {
  for (const LocationList &L : Locations) {
    OS << format("0x%8.8x: ", L.Offset);
    const unsigned Indent = 12;
    for (const Entry &E : L.Entries) {
      if (&E != L.Entries.begin())
        OS.indent(Indent);
      OS << "Beginning address offset: " << format("0x%016" PRIx64, E.Begin)
         << '\n';
      OS.indent(Indent) << "   Ending address offset: "
                        << format("0x%016" PRIx64, E.End) << '\n';
      OS.indent(Indent) << "    Location description: ";
      for (unsigned char Loc : E.Loc) {
        OS << format("%2.2x ", Loc);
      }
      OS << "\n\n";
    }
  }
}

void DWARFDebugLoc::parse(DataExtractor data, unsigned AddressSize) {
  uint32_t Offset = 0;
  while (data.isValidOffset(Offset+AddressSize-1)) {
    Locations.resize(Locations.size() + 1);
    LocationList &Loc = Locations.back();
    Loc.Offset = Offset;
    // 2.6.2 Location Lists
    // A location list entry consists of:
    while (true) {
      Entry E;
      RelocAddrMap::const_iterator AI = RelocMap.find(Offset);
      // 1. A beginning address offset. ...
      E.Begin = data.getUnsigned(&Offset, AddressSize);
      if (AI != RelocMap.end())
        E.Begin += AI->second.second;

      AI = RelocMap.find(Offset);
      // 2. An ending address offset. ...
      E.End = data.getUnsigned(&Offset, AddressSize);
      if (AI != RelocMap.end())
        E.End += AI->second.second;

      // The end of any given location list is marked by an end of list entry,
      // which consists of a 0 for the beginning address offset and a 0 for the
      // ending address offset.
      if (E.Begin == 0 && E.End == 0)
        break;

      unsigned Bytes = data.getU16(&Offset);
      // A single location description describing the location of the object...
      StringRef str = data.getData().substr(Offset, Bytes);
      Offset += Bytes;
      E.Loc.reserve(str.size());
      std::copy(str.begin(), str.end(), std::back_inserter(E.Loc));
      Loc.Entries.push_back(std::move(E));
    }
  }
  if (data.isValidOffset(Offset))
    llvm::errs() << "error: failed to consume entire .debug_loc section\n";
}
