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
#include "llvm/Object/MachO.h"
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

  DWARFUnit &getOrigUnit() const { return OrigUnit; }

  DIEInfo &getInfo(unsigned Idx) { return Info[Idx]; }
  const DIEInfo &getInfo(unsigned Idx) const { return Info[Idx]; }

private:
  DWARFUnit &OrigUnit;
  std::vector<DIEInfo> Info; ///< DIE info indexed by DIE index.
};

/// \brief The core of the Dwarf linking logic.
///
/// The link of the dwarf information from the object files will be
/// driven by the selection of 'root DIEs', which are DIEs that
/// describe variables or functions that are present in the linked
/// binary (and thus have entries in the debug map). All the debug
/// information that will be linked (the DIEs, but also the line
/// tables, ranges, ...) is derived from that set of root DIEs.
///
/// The root DIEs are identified because they contain relocations that
/// correspond to a debug map entry at specific places (the low_pc for
/// a function, the location for a variable). These relocations are
/// called ValidRelocs in the DwarfLinker and are gathered as a very
/// first step when we start processing a DebugMapObject.
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

  /// \defgroup FindValidRelocations Translate debug map into a list
  /// of relevant relocations
  ///
  /// @{
  struct ValidReloc {
    uint32_t Offset;
    uint32_t Size;
    uint64_t Addend;
    const DebugMapObject::DebugMapEntry *Mapping;

    ValidReloc(uint32_t Offset, uint32_t Size, uint64_t Addend,
               const DebugMapObject::DebugMapEntry *Mapping)
        : Offset(Offset), Size(Size), Addend(Addend), Mapping(Mapping) {}

    bool operator<(const ValidReloc &RHS) const { return Offset < RHS.Offset; }
  };

  /// \brief The valid relocations for the current DebugMapObject.
  /// This vector is sorted by relocation offset.
  std::vector<ValidReloc> ValidRelocs;

  /// \brief Index into ValidRelocs of the next relocation to
  /// consider. As we walk the DIEs in acsending file offset and as
  /// ValidRelocs is sorted by file offset, keeping this index
  /// uptodate is all we have to do to have a cheap lookup during the
  /// root DIE selection.
  unsigned NextValidReloc;

  bool findValidRelocsInDebugInfo(const object::ObjectFile &Obj,
                                  const DebugMapObject &DMO);

  bool findValidRelocs(const object::SectionRef &Section,
                       const object::ObjectFile &Obj,
                       const DebugMapObject &DMO);

  void findValidRelocsMachO(const object::SectionRef &Section,
                            const object::MachOObjectFile &Obj,
                            const DebugMapObject &DMO);
  /// @}
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
  NextValidReloc = 0;
}

void DwarfLinker::endDebugObject() {
  Units.clear();
  ValidRelocs.clear();
}

/// \brief Iterate over the relocations of the given \p Section and
/// store the ones that correspond to debug map entries into the
/// ValidRelocs array.
void DwarfLinker::findValidRelocsMachO(const object::SectionRef &Section,
                                       const object::MachOObjectFile &Obj,
                                       const DebugMapObject &DMO) {
  StringRef Contents;
  Section.getContents(Contents);
  DataExtractor Data(Contents, Obj.isLittleEndian(), 0);

  for (const object::RelocationRef &Reloc : Section.relocations()) {
    object::DataRefImpl RelocDataRef = Reloc.getRawDataRefImpl();
    MachO::any_relocation_info MachOReloc = Obj.getRelocation(RelocDataRef);
    unsigned RelocSize = 1 << Obj.getAnyRelocationLength(MachOReloc);
    uint64_t Offset64;
    if ((RelocSize != 4 && RelocSize != 8) || Reloc.getOffset(Offset64)) {
      errs() << "warning: unsupported relocation in debug_info section.\n";
      continue;
    }
    uint32_t Offset = Offset64;
    // Mach-o uses REL relocations, the addend is at the relocation offset.
    uint64_t Addend = Data.getUnsigned(&Offset, RelocSize);

    auto Sym = Reloc.getSymbol();
    if (Sym != Obj.symbol_end()) {
      StringRef SymbolName;
      if (Sym->getName(SymbolName)) {
        errs() << "warning: error getting relocation symbol name.\n";
        continue;
      }
      if (const auto *Mapping = DMO.lookupSymbol(SymbolName))
        ValidRelocs.emplace_back(Offset64, RelocSize, Addend, Mapping);
    } else if (const auto *Mapping = DMO.lookupObjectAddress(Addend)) {
      // Do not store the addend. The addend was the address of the
      // symbol in the object file, the address in the binary that is
      // stored in the debug map doesn't need to be offseted.
      ValidRelocs.emplace_back(Offset64, RelocSize, 0, Mapping);
    }
  }
}

/// \brief Dispatch the valid relocation finding logic to the
/// appropriate handler depending on the object file format.
bool DwarfLinker::findValidRelocs(const object::SectionRef &Section,
                                  const object::ObjectFile &Obj,
                                  const DebugMapObject &DMO) {
  // Dispatch to the right handler depending on the file type.
  if (auto *MachOObj = dyn_cast<object::MachOObjectFile>(&Obj))
    findValidRelocsMachO(Section, *MachOObj, DMO);
  else
    errs() << "warning: unsupported object file type: " << Obj.getFileName()
           << '\n';

  if (ValidRelocs.empty())
    return false;

  // Sort the relocations by offset. We will walk the DIEs linearly in
  // the file, this allows us to just keep an index in the relocation
  // array that we advance during our walk, rather than resorting to
  // some associative container. See DwarfLinker::NextValidReloc.
  std::sort(ValidRelocs.begin(), ValidRelocs.end());
  return true;
}

/// \brief Look for relocations in the debug_info section that match
/// entries in the debug map. These relocations will drive the Dwarf
/// link by indicating which DIEs refer to symbols present in the
/// linked binary.
/// \returns wether there are any valid relocations in the debug info.
bool DwarfLinker::findValidRelocsInDebugInfo(const object::ObjectFile &Obj,
                                             const DebugMapObject &DMO) {
  // Find the debug_info section.
  for (const object::SectionRef &Section : Obj.sections()) {
    StringRef SectionName;
    Section.getName(SectionName);
    SectionName = SectionName.substr(SectionName.find_first_not_of("._"));
    if (SectionName != "debug_info")
      continue;
    return findValidRelocs(Section, Obj, DMO);
  }
  return false;
}

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

    // Look for relocations that correspond to debug map entries.
    if (!findValidRelocsInDebugInfo(*ErrOrObj, *Obj)) {
      if (Verbose)
        outs() << "No valid relocations found. Skipping.\n";
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
