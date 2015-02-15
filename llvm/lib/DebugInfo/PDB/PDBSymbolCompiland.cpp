//===- PDBSymbolCompiland.cpp - compiland details --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>
#include <vector>

using namespace llvm;

PDBSymbolCompiland::PDBSymbolCompiland(const IPDBSession &PDBSession,
                                       std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

#define SKIP_SYMBOL_IF_FLAG_UNSET(Tag, Flag) \
  case PDB_SymType::Tag: \
    if ((Flags & Flag) == 0) \
      continue;   \
    break;

void PDBSymbolCompiland::dump(raw_ostream &OS, int Indent,
                              PDB_DumpLevel Level, PDB_DumpFlags Flags) const {
  if (Level == PDB_DumpLevel::Detailed) {
    std::string FullName = getName();
    OS << stream_indent(Indent) << FullName;
    if (Flags & PDB_DF_Children) {
      if (Level >= PDB_DumpLevel::Detailed) {
        auto ChildrenEnum = findAllChildren();
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
          Child->dump(OS, Indent + 2, ChildLevel, PDB_DF_Children);
        }
      }
    }
  } else {
    std::string FullName = getName();
    OS << stream_indent(Indent) << "Compiland: " << FullName;
  }
}
