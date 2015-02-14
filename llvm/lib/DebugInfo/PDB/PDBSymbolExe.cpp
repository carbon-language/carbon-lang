//===- PDBSymbolExe.cpp - ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#include <utility>

using namespace llvm;

PDBSymbolExe::PDBSymbolExe(const IPDBSession &PDBSession,
                           std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolExe::dump(raw_ostream &OS, int Indent,
                        PDB_DumpLevel Level) const {
  std::string FileName(getSymbolsFileName());

  OS << stream_indent(Indent) << "Summary for " << FileName << "\n";

  uint64_t FileSize = 0;
  if (!llvm::sys::fs::file_size(FileName, FileSize))
    OS << stream_indent(Indent + 2) << "Size: " << FileSize << " bytes\n";
  else
    OS << stream_indent(Indent + 2) << "Size: (Unable to obtain file size)\n";
  PDB_UniqueId Guid = getGuid();
  OS << stream_indent(Indent + 2) << "Guid: " << Guid << "\n";
  OS << stream_indent(Indent + 2) << "Age: " << getAge() << "\n";
  OS << stream_indent(Indent + 2) << "Attributes: ";
  if (hasCTypes())
    OS << "HasCTypes ";
  if (hasPrivateSymbols())
    OS << "HasPrivateSymbols ";
  OS << "\n";

  auto ChildrenEnum = findAllChildren();
  OS << stream_indent(Indent + 2) << ChildrenEnum->getChildCount()
     << " children\n";
#if 0
  dumpChildren(OS, PDB_SymType::None, Indent+4);
#else
  dumpChildren(OS, "Compilands", PDB_SymType::Compiland, Indent + 4);
  dumpChildren(OS, "Functions", PDB_SymType::Function, Indent + 4);
  dumpChildren(OS, "Blocks", PDB_SymType::Block, Indent + 4);
  dumpChildren(OS, "Data", PDB_SymType::Data, Indent + 4);
  dumpChildren(OS, "Labels", PDB_SymType::Label, Indent + 4);
  dumpChildren(OS, "Public Symbols", PDB_SymType::PublicSymbol, Indent + 4);
  dumpChildren(OS, "UDTs", PDB_SymType::UDT, Indent + 4);
  dumpChildren(OS, "Enums", PDB_SymType::Enum, Indent + 4);
  dumpChildren(OS, "Function Signatures", PDB_SymType::FunctionSig, Indent + 4);
  dumpChildren(OS, "Typedefs", PDB_SymType::Typedef, Indent + 4);
  dumpChildren(OS, "VTables", PDB_SymType::VTable, Indent + 4);
  dumpChildren(OS, "Thunks", PDB_SymType::Thunk, Indent + 4);
#endif
}

void PDBSymbolExe::dumpChildren(raw_ostream &OS, StringRef Label,
                                PDB_SymType ChildType, int Indent) const {
  auto ChildrenEnum = findAllChildren(ChildType);
  OS << stream_indent(Indent) << Label << ": (" << ChildrenEnum->getChildCount()
     << " items)\n";
  while (auto Child = ChildrenEnum->getNext()) {
    Child->dump(OS, Indent + 2, PDB_DumpLevel::Normal);
    OS << "\n";
  }
}
