//===- PDBSymbolData.cpp - PDB data (e.g. variable) accessors ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"

#include "llvm/Support/Format.h"

using namespace llvm;

PDBSymbolData::PDBSymbolData(const IPDBSession &PDBSession,
                             std::unique_ptr<IPDBRawSymbol> DataSymbol)
    : PDBSymbol(PDBSession, std::move(DataSymbol)) {}

void PDBSymbolData::dump(raw_ostream &OS, int Indent,
                         PDB_DumpLevel Level) const {
  OS.indent(Indent);
  if (Level == PDB_DumpLevel::Compact) {
    PDB_LocType Loc = getLocationType();
    OS << Loc << " data [";
    int Length;
    switch (Loc) {
    case PDB_LocType::Static:
      OS << format_hex(getRelativeVirtualAddress(), 10);
      Length = getLength();
      break;
    case PDB_LocType::TLS:
      OS << getAddressSection() << ":" << format_hex(getAddressOffset(), 10);
      break;
    case PDB_LocType::RegRel:
      OS << getRegisterId() << " + " << getOffset() << "]";
      break;
    case PDB_LocType::ThisRel:
      OS << "this + " << getOffset() << "]";
      break;
    case PDB_LocType::Enregistered:
      OS << getRegisterId() << "]";
      break;
    case PDB_LocType::BitField: {
      uint32_t Offset = getOffset();
      uint32_t BitPos = getBitPosition();
      uint32_t Length = getLength();
      uint32_t StartBits = 8 - BitPos;
      uint32_t MiddleBytes = (Length - StartBits) / 8;
      uint32_t EndBits = Length - StartBits - MiddleBytes * 8;
      OS << format_hex(Offset, 10) << ":" << BitPos;
      OS << " - " << format_hex(Offset + MiddleBytes, 10) << ":" << EndBits;
      break;
    }
    case PDB_LocType::Slot:
      OS << getSlot();
    case PDB_LocType::IlRel:
    case PDB_LocType::MetaData:
    case PDB_LocType::Constant:
    default:
      OS << "???";
    }
    OS << "] ";
  }
  OS << getName() << "\n";
}