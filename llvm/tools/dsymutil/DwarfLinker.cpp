//===- tools/dsymutil/DwarfLinker.cpp - Dwarf debug info linker -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "DebugMap.h"

#include "BinaryHolder.h"
#include "DebugMap.h"
#include "dsymutil.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include <string>

namespace llvm {
namespace dsymutil {

namespace {

/// \brief Stores all information relating to a compile unit, be it in
/// its original instance in the object file to its brand new cloned
/// and linked DIE tree.
class CompileUnit {
public:
  /// \brief Information gathered about a DIE in the object file.
  struct DIEInfo {
    uint32_t ParentIdx;
  };

  CompileUnit(DWARFUnit &OrigUnit) : OrigUnit(OrigUnit) {
    Info.resize(OrigUnit.getNumDIEs());
  }

  DWARFUnit &getOrigUnit() { return OrigUnit; }

  DIEInfo &getInfo(unsigned Idx) { return Info[Idx]; }
  const DIEInfo &getInfo(unsigned Idx) const { return Info[Idx]; }

private:
  DWARFUnit &OrigUnit;
  std::vector<DIEInfo> Info; ///< DIE info indexed by DIE index.
};

/// \brief The core of the Dwarf linking logic.
class DwarfLinker {
public:
  DwarfLinker(StringRef OutputFilename, bool Verbose)
      : OutputFilename(OutputFilename), Verbose(Verbose), BinHolder(Verbose) {}

  /// \brief Link the contents of the DebugMap.
  bool link(const DebugMap &);

private:
  /// \brief Called at the start of a debug object link.
  void startDebugObject(DWARFContext &);

  /// \brief Called at the end of a debug object link.
  void endDebugObject();

private:
  std::string OutputFilename;
  bool Verbose;
  BinaryHolder BinHolder;

  /// The units of the current debug map object.
  std::vector<CompileUnit> Units;
};

/// \brief Recursive helper to gather the child->parent relationships in the
/// original compile unit.
void GatherDIEParents(const DWARFDebugInfoEntryMinimal *DIE, unsigned ParentIdx,
                      CompileUnit &CU) {
  unsigned MyIdx = CU.getOrigUnit().getDIEIndex(DIE);
  CU.getInfo(MyIdx).ParentIdx = ParentIdx;

  if (DIE->hasChildren())
    for (auto *Child = DIE->getFirstChild(); Child && !Child->isNULL();
         Child = Child->getSibling())
      GatherDIEParents(Child, MyIdx, CU);
}

void DwarfLinker::startDebugObject(DWARFContext &Dwarf) {
  Units.reserve(Dwarf.getNumCompileUnits());
}

void DwarfLinker::endDebugObject() { Units.clear(); }

bool DwarfLinker::link(const DebugMap &Map) {

  if (Map.begin() == Map.end()) {
    errs() << "Empty debug map.\n";
    return false;
  }

  for (const auto &Obj : Map.objects()) {
    if (Verbose)
      outs() << "DEBUG MAP OBJECT: " << Obj->getObjectFilename() << "\n";
    auto ErrOrObj = BinHolder.GetObjectFile(Obj->getObjectFilename());
    if (std::error_code EC = ErrOrObj.getError()) {
      errs() << Obj->getObjectFilename() << ": " << EC.message() << "\n";
      continue;
    }

    // Setup access to the debug info.
    DWARFContextInMemory DwarfContext(*ErrOrObj);
    startDebugObject(DwarfContext);

    // In a first phase, just read in the debug info and store the DIE
    // parent links that we will use during the next phase.
    for (const auto &CU : DwarfContext.compile_units()) {
      auto *CUDie = CU->getCompileUnitDIE(false);
      if (Verbose) {
        outs() << "Input compilation unit:";
        CUDie->dump(outs(), CU.get(), 0);
      }
      Units.emplace_back(*CU);
      GatherDIEParents(CUDie, 0, Units.back());
    }

    // Clean-up before starting working on the next object.
    endDebugObject();
  }

  return true;
}
}

bool linkDwarf(StringRef OutputFilename, const DebugMap &DM, bool Verbose) {
  DwarfLinker Linker(OutputFilename, Verbose);
  return Linker.link(DM);
}
}
}
