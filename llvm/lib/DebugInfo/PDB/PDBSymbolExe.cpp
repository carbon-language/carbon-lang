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

#define SKIP_SYMBOL_IF_FLAG_UNSET(Tag, Flag)                                   \
  case PDB_SymType::Tag:                                                       \
    if ((Flags & Flag) == 0)                                                   \
      continue;                                                                \
    break;

PDBSymbolExe::PDBSymbolExe(const IPDBSession &PDBSession,
                           std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolExe::dump(raw_ostream &OS, int Indent,
                        PDB_DumpLevel Level, PDB_DumpFlags Flags) const {
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

  if (Flags & PDB_DF_Children) {
    OS << stream_indent(Indent + 2) << "Dumping types\n";
    if (Flags & PDB_DF_Hidden) {
      // For some reason, for each SymTag T, this dumps more items of type T
      // than are dumped by calling dumpChildren(T).  In other words, there are
      // "hidden" symbols.  For example, it causes functions to be dumped which
      // have no address information, whereas specifically dumping only
      // functions will not find those symbols.
      //
      // My suspicion is that in the underlying DIA call, when you call
      // findChildren, passing a value of SymTagNone means all children
      // recursively, whereas passing a concrete tag value means only immediate
      // children of the global scope.  So perhaps we need to find these
      // mysterious missing values by recursing through the hierarchy.
      //
      // On the other hand, there may just be some symbols that DIA tries to
      // hide from you because it thinks you don't care about them.  However
      // experimentation shows that even vtables, for example, can't be found
      // without an exhaustive search.
      auto ChildrenEnum = findAllChildren();
      OS << stream_indent(Indent + 2) << ChildrenEnum->getChildCount()
         << " symbols";

      while (auto Child = ChildrenEnum->getNext()) {
        switch (Child->getSymTag()) {
          SKIP_SYMBOL_IF_FLAG_UNSET(Function, PDB_DF_Functions)
          SKIP_SYMBOL_IF_FLAG_UNSET(Data, PDB_DF_Data)
          SKIP_SYMBOL_IF_FLAG_UNSET(Label, PDB_DF_Labels)
          SKIP_SYMBOL_IF_FLAG_UNSET(PublicSymbol, PDB_DF_PublicSyms)
          SKIP_SYMBOL_IF_FLAG_UNSET(UDT, PDB_DF_Classes)
          SKIP_SYMBOL_IF_FLAG_UNSET(Enum, PDB_DF_Enums)
          SKIP_SYMBOL_IF_FLAG_UNSET(FunctionSig, PDB_DF_Funcsigs)
          SKIP_SYMBOL_IF_FLAG_UNSET(VTable, PDB_DF_VTables)
          SKIP_SYMBOL_IF_FLAG_UNSET(Thunk, PDB_DF_Thunks)
          SKIP_SYMBOL_IF_FLAG_UNSET(Compiland, PDB_DF_ObjFiles)
        default:
          continue;
        }
        PDB_DumpLevel ChildLevel = (Level == PDB_DumpLevel::Detailed)
                                       ? PDB_DumpLevel::Normal
                                       : PDB_DumpLevel::Compact;
        OS << "\n";
        Child->dump(OS, Indent + 4, ChildLevel, PDB_DF_Children);
      }
    } else {
      if (Flags & PDB_DF_ObjFiles)
        dumpChildren(OS, "Compilands", PDB_SymType::Compiland, Indent + 4);
      if (Flags & PDB_DF_Functions)
        dumpChildren(OS, "Functions", PDB_SymType::Function, Indent + 4);
      if (Flags & PDB_DF_Data)
        dumpChildren(OS, "Data", PDB_SymType::Data, Indent + 4);
      if (Flags & PDB_DF_Labels)
        dumpChildren(OS, "Labels", PDB_SymType::Label, Indent + 4);
      if (Flags & PDB_DF_PublicSyms)
        dumpChildren(OS, "Public Symbols", PDB_SymType::PublicSymbol,
                     Indent + 4);
      if (Flags & PDB_DF_Classes)
        dumpChildren(OS, "UDTs", PDB_SymType::UDT, Indent + 4);
      if (Flags & PDB_DF_Enums)
        dumpChildren(OS, "Enums", PDB_SymType::Enum, Indent + 4);
      if (Flags & PDB_DF_Funcsigs)
        dumpChildren(OS, "Function Signatures", PDB_SymType::FunctionSig,
                     Indent + 4);
      if (Flags & PDB_DF_Typedefs)
        dumpChildren(OS, "Typedefs", PDB_SymType::Typedef, Indent + 4);
      if (Flags & PDB_DF_VTables)
        dumpChildren(OS, "VTables", PDB_SymType::VTable, Indent + 4);
      if (Flags & PDB_DF_Thunks)
        dumpChildren(OS, "Thunks", PDB_SymType::Thunk, Indent + 4);
    }
  }
}

void PDBSymbolExe::dumpChildren(raw_ostream &OS, StringRef Label,
                                PDB_SymType ChildType, int Indent) const {
  auto ChildrenEnum = findAllChildren(ChildType);
  OS << stream_indent(Indent) << Label << ": (" << ChildrenEnum->getChildCount()
     << " items)\n";
  while (auto Child = ChildrenEnum->getNext()) {
    Child->dump(OS, Indent + 2, PDB_DumpLevel::Normal, PDB_DF_None);
    OS << "\n";
  }
}
