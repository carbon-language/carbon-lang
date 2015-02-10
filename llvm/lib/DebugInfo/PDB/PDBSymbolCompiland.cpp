//===- PDBSymbolCompiland.cpp - compiland details --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>
#include <vector>

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PDBSymbolCompiland::PDBSymbolCompiland(const IPDBSession &PDBSession,
                                       std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolCompiland::dump(raw_ostream &OS, int Indent,
                              PDB_DumpLevel Level) const {
  std::string Name = getName();
  OS << "---- [IDX: " << getSymIndexId() << "] Compiland: " << Name
     << " ----\n";

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
  if (Level >= PDB_DumpLevel::Normal) {
    while (auto Child = ChildrenEnum->getNext()) {
      if (llvm::isa<PDBSymbolCompilandDetails>(*Child))
        continue;
      if (llvm::isa<PDBSymbolCompilandEnv>(*Child))
        continue;
      Child->dump(OS, Indent + 4, PDB_DumpLevel::Compact);
    }
  }

  std::unique_ptr<IPDBEnumSymbols> DetailsEnum(
      findChildren(PDB_SymType::CompilandDetails));
  if (auto DetailsPtr = DetailsEnum->getNext()) {
    const auto *CD = dyn_cast<PDBSymbolCompilandDetails>(DetailsPtr.get());
    assert(CD && "We only asked for compilands, but got something else!");
    VersionInfo FE;
    VersionInfo BE;
    CD->getFrontEndVersion(FE);
    CD->getBackEndVersion(BE);
    OS << stream_indent(Indent + 2) << "Compiler: " << CD->getCompilerName()
       << "\n";
    OS << stream_indent(Indent + 2) << "Version: " << FE << ", " << BE << "\n";

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

    OS << "\n";
    auto Files(Session.getSourceFilesForCompiland(*this));
    if (Level >= PDB_DumpLevel::Detailed) {
      OS << stream_indent(Indent + 2) << Files->getChildCount()
         << " source files:\n";
      while (auto File = Files->getNext())
        File->dump(OS, Indent + 4, PDB_DumpLevel::Compact);
    } else {
      OS << stream_indent(Indent + 2) << Files->getChildCount()
         << " source files\n";
    }
  }
  uint32_t Count = DetailsEnum->getChildCount();
  if (Count > 1)
    OS << stream_indent(Indent + 2) << "(" << Count - 1 << " more omitted)\n";
}
