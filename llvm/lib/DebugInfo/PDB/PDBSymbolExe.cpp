//===- PDBSymbolExe.cpp - ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
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

  TagStats Stats;
  auto ChildrenEnum = getChildStats(Stats);
  OS << stream_indent(Indent + 2) << "Children: " << Stats << "\n";
  while (auto Child = ChildrenEnum->getNext()) {
    // Skip uninteresting types.  These are useful to print as part of type
    // hierarchies, but as general children of the global scope, they are
    // not very interesting.
    switch (Child->getSymTag()) {
    case PDB_SymType::ArrayType:
    case PDB_SymType::BaseClass:
    case PDB_SymType::BuiltinType:
    case PDB_SymType::CompilandEnv:
    case PDB_SymType::CustomType:
    case PDB_SymType::Dimension:
    case PDB_SymType::Friend:
    case PDB_SymType::ManagedType:
    case PDB_SymType::VTableShape:
    case PDB_SymType::PointerType:
    case PDB_SymType::FunctionSig:
    case PDB_SymType::FunctionArg:
      continue;
    default:
      break;
    }
    Child->dump(OS, Indent + 4, PDB_DumpLevel::Normal);
    OS << "\n";
  }
}
