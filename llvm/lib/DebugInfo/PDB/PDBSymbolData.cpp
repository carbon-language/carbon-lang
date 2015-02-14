//===- PDBSymbolData.cpp - PDB data (e.g. variable) accessors ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolData.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/Support/Format.h"
#include <utility>

#include <utility>

using namespace llvm;

PDBSymbolData::PDBSymbolData(const IPDBSession &PDBSession,
                             std::unique_ptr<IPDBRawSymbol> DataSymbol)
    : PDBSymbol(PDBSession, std::move(DataSymbol)) {}

void PDBSymbolData::dump(raw_ostream &OS, int Indent,
                         PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  PDB_LocType Loc = getLocationType();
  PDB_DataKind Kind = getDataKind();
  if (Level >= PDB_DumpLevel::Normal) {
    switch (Loc) {
    case PDB_LocType::Static: {
      uint32_t RVA = getRelativeVirtualAddress();
      OS << Kind << " data[";
      if (RVA != 0)
        OS << format_hex(RVA, 10);
      else
        OS << "???";
      break;
    }
    case PDB_LocType::TLS:
      OS << "threadlocal " << Kind << " data[";
      OS << getAddressSection() << ":" << format_hex(getAddressOffset(), 10);
      break;
    case PDB_LocType::RegRel:
      OS << "regrel " << Kind << " data[";
      OS << getRegisterId() << " + " << getOffset();
      break;
    case PDB_LocType::ThisRel: {
      uint32_t Offset = getOffset();
      OS << Kind << " data[this + " << format_hex(Offset, 4);
      break;
    }
    case PDB_LocType::Enregistered:
      OS << "register " << Kind << " data[" << getRegisterId();
      break;
    case PDB_LocType::BitField: {
      OS << "bitfield data[this + ";
      uint32_t Offset = getOffset();
      uint32_t BitPos = getBitPosition();
      uint32_t Length = getLength();
      OS << format_hex(Offset, 4) << ":" << BitPos << "," << Length;
      break;
    }
    case PDB_LocType::Slot:
      OS << getSlot();
      break;
    case PDB_LocType::Constant: {
      OS << "constant data[";
      OS << getValue();
      break;
    }
    case PDB_LocType::IlRel:
    case PDB_LocType::MetaData:
    default:
      OS << "???";
    }
  }

  OS << "] ";
  if (Kind == PDB_DataKind::Member || Kind == PDB_DataKind::StaticMember) {
    uint32_t ClassId = getClassParentId();
    if (auto Class = Session.getSymbolById(ClassId)) {
      if (auto UDT = dyn_cast<PDBSymbolTypeUDT>(Class.get()))
        OS << UDT->getName();
      else
        OS << "{class " << Class->getSymTag() << "}";
      OS << "::";
    }
  }
  OS << getName();
}