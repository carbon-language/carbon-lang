//===- PDBSymbolCompiland.cpp - compiland details --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
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

void PDBSymbolCompiland::dump(raw_ostream &OS, int Indent,
                              PDB_DumpLevel Level) const {
  if (Level == PDB_DumpLevel::Detailed) {
    std::string FullName = getName();
    StringRef Name = llvm::sys::path::filename(StringRef(FullName.c_str()));

    OS.indent(Indent);
    OS << "Compiland: " << Name << "\n";

    std::string Source = getSourceFileName();
    std::string Library = getLibraryName();
    if (!Source.empty())
      OS << stream_indent(Indent + 2) << "Source: " << this->getSourceFileName()
         << "\n";
    if (!Library.empty())
      OS << stream_indent(Indent + 2) << "Library: " << this->getLibraryName()
         << "\n";

    TagStats Stats;
    auto ChildrenEnum = getChildStats(Stats);
    OS << stream_indent(Indent + 2) << "Children: " << Stats << "\n";
    if (Level >= PDB_DumpLevel::Detailed) {
      while (auto Child = ChildrenEnum->getNext()) {
        if (llvm::isa<PDBSymbolCompilandDetails>(*Child))
          continue;
        if (llvm::isa<PDBSymbolCompilandEnv>(*Child))
          continue;
        PDB_DumpLevel ChildLevel = (Level == PDB_DumpLevel::Detailed)
                                       ? PDB_DumpLevel::Normal
                                       : PDB_DumpLevel::Compact;
        Child->dump(OS, Indent + 4, ChildLevel);
        OS << "\n";
      }
    }

    auto DetailsEnum(findAllChildren<PDBSymbolCompilandDetails>());
    if (auto CD = DetailsEnum->getNext()) {
      VersionInfo FE;
      VersionInfo BE;
      CD->getFrontEndVersion(FE);
      CD->getBackEndVersion(BE);
      OS << stream_indent(Indent + 2) << "Compiler: " << CD->getCompilerName()
         << "\n";
      OS << stream_indent(Indent + 2) << "Version: " << FE << ", " << BE
         << "\n";

      OS << stream_indent(Indent + 2) << "Lang: " << CD->getLanguage() << "\n";
      OS << stream_indent(Indent + 2) << "Attributes: ";
      if (CD->hasDebugInfo())
        OS << "DebugInfo ";
      if (CD->isDataAligned())
        OS << "DataAligned ";
      if (CD->isLTCG())
        OS << "LTCG ";
      if (CD->hasSecurityChecks())
        OS << "SecurityChecks ";
      if (CD->isHotpatchable())
        OS << "HotPatchable";

      auto Files(Session.getSourceFilesForCompiland(*this));
      OS << "\n";
      OS << stream_indent(Indent + 2) << Files->getChildCount()
         << " source files";
    }
    uint32_t Count = DetailsEnum->getChildCount();
    if (Count > 1) {
      OS << "\n";
      OS << stream_indent(Indent + 2) << "(" << Count - 1 << " more omitted)";
    }
  } else {
    std::string FullName = getName();
    OS << stream_indent(Indent) << "Compiland: " << FullName;
  }
}
