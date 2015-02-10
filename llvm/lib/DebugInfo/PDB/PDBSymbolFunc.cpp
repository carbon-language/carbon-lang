//===- PDBSymbolFunc.cpp - --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"

#include "llvm/Support/Format.h"

using namespace llvm;
PDBSymbolFunc::PDBSymbolFunc(const IPDBSession &PDBSession,
                             std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolFunc::dump(raw_ostream &OS, int Indent,
                         PDB_DumpLevel Level) const {
  bool doFullDump = false;
  if (Level == PDB_DumpLevel::Compact) {
    uint32_t FuncStart = getRelativeVirtualAddress();
    uint32_t FuncEnd = FuncStart + getLength();
    auto DebugEndSymbol = findChildren(PDB_SymType::FuncDebugEnd);
    OS << stream_indent(Indent);
    OS << "[" << format_hex(FuncStart, 8);
    if (auto DebugStartEnum = findChildren(PDB_SymType::FuncDebugStart)) {
      if (auto StartSym = DebugStartEnum->getNext()) {
        auto DebugStart = dyn_cast<PDBSymbolFuncDebugStart>(StartSym.get());
        OS << "+" << DebugStart->getRelativeVirtualAddress() - FuncStart;
      }
    }
    OS << " - " << format_hex(FuncEnd, 8);
    if (auto DebugEndEnum = findChildren(PDB_SymType::FuncDebugEnd)) {
      if (auto DebugEndSym = DebugEndEnum->getNext()) {
        auto DebugEnd = dyn_cast<PDBSymbolFuncDebugEnd>(DebugEndSym.get());
        OS << "-" << FuncEnd - DebugEnd->getRelativeVirtualAddress();
      }
    }
    OS << "] ";
    PDB_RegisterId Reg = getLocalBasePointerRegisterId();
    if (Reg == PDB_RegisterId::VFrame)
      OS << "(VFrame)";
    else if (hasFramePointer()) {
      if (Reg == PDB_RegisterId::EBP)
        OS << "(EBP)";
      else
        OS << "(" << (int)Reg << ")";
    } else {
      OS << "(FPO)";
      doFullDump = true;
    }
    OS << " " << getName() << "\n";
  }
  OS.flush();
}
