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
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <string>
#include <tuple>

namespace llvm {
namespace dsymutil {

namespace {

void warn(const Twine &Warning, const Twine &Context) {
  errs() << Twine("while processing ") + Context + ":\n";
  errs() << Twine("warning: ") + Warning + "\n";
}

bool error(const Twine &Error, const Twine &Context) {
  errs() << Twine("while processing ") + Context + ":\n";
  errs() << Twine("error: ") + Error + "\n";
  return false;
}

template <typename KeyT, typename ValT>
using HalfOpenIntervalMap =
    IntervalMap<KeyT, ValT, IntervalMapImpl::NodeSizer<KeyT, ValT>::LeafSize,
                IntervalMapHalfOpenInfo<KeyT>>;

typedef HalfOpenIntervalMap<uint64_t, int64_t> FunctionIntervals;

/// \brief Stores all information relating to a compile unit, be it in
/// its original instance in the object file to its brand new cloned
/// and linked DIE tree.
class CompileUnit {
public:
  /// \brief Information gathered about a DIE in the object file.
  struct DIEInfo {
    int64_t AddrAdjust; ///< Address offset to apply to the described entity.
    DIE *Clone;         ///< Cloned version of that DIE.
    uint32_t ParentIdx; ///< The index of this DIE's parent.
    bool Keep;          ///< Is the DIE part of the linked output?
    bool InDebugMap;    ///< Was this DIE's entity found in the map?
  };

  CompileUnit(DWARFUnit &OrigUnit, unsigned ID)
      : OrigUnit(OrigUnit), ID(ID), LowPc(UINT64_MAX), HighPc(0), RangeAlloc(),
        Ranges(RangeAlloc), UnitRangeAttribute(nullptr) {
    Info.resize(OrigUnit.getNumDIEs());
  }

  CompileUnit(CompileUnit &&RHS)
      : OrigUnit(RHS.OrigUnit), Info(std::move(RHS.Info)),
        CUDie(std::move(RHS.CUDie)), StartOffset(RHS.StartOffset),
        NextUnitOffset(RHS.NextUnitOffset), RangeAlloc(), Ranges(RangeAlloc) {
    // The CompileUnit container has been 'reserve()'d with the right
    // size. We cannot move the IntervalMap anyway.
    llvm_unreachable("CompileUnits should not be moved.");
  }

  DWARFUnit &getOrigUnit() const { return OrigUnit; }

  unsigned getUniqueID() const { return ID; }

  DIE *getOutputUnitDIE() const { return CUDie.get(); }
  void setOutputUnitDIE(DIE *Die) { CUDie.reset(Die); }

  DIEInfo &getInfo(unsigned Idx) { return Info[Idx]; }
  const DIEInfo &getInfo(unsigned Idx) const { return Info[Idx]; }

  uint64_t getStartOffset() const { return StartOffset; }
  uint64_t getNextUnitOffset() const { return NextUnitOffset; }
  void setStartOffset(uint64_t DebugInfoSize) { StartOffset = DebugInfoSize; }

  uint64_t getLowPc() const { return LowPc; }
  uint64_t getHighPc() const { return HighPc; }

  DIEInteger *getUnitRangesAttribute() const { return UnitRangeAttribute; }
  const FunctionIntervals &getFunctionRanges() const { return Ranges; }
  const std::vector<DIEInteger *> &getRangesAttributes() const {
    return RangeAttributes;
  }

  const std::vector<std::pair<DIEInteger *, int64_t>> &
  getLocationAttributes() const {
    return LocationAttributes;
  }

  /// \brief Compute the end offset for this unit. Must be
  /// called after the CU's DIEs have been cloned.
  /// \returns the next unit offset (which is also the current
  /// debug_info section size).
  uint64_t computeNextUnitOffset();

  /// \brief Keep track of a forward reference to DIE \p Die in \p
  /// RefUnit by \p Attr. The attribute should be fixed up later to
  /// point to the absolute offset of \p Die in the debug_info section.
  void noteForwardReference(DIE *Die, const CompileUnit *RefUnit,
                            DIEInteger *Attr);

  /// \brief Apply all fixups recored by noteForwardReference().
  void fixupForwardReferences();

  /// \brief Add a function range [\p LowPC, \p HighPC) that is
  /// relocatad by applying offset \p PCOffset.
  void addFunctionRange(uint64_t LowPC, uint64_t HighPC, int64_t PCOffset);

  /// \brief Keep track of a DW_AT_range attribute that we will need to
  /// patch up later.
  void noteRangeAttribute(const DIE &Die, DIEInteger *Attr);

  /// \brief Keep track of a location attribute pointing to a location
  /// list in the debug_loc section.
  void noteLocationAttribute(DIEInteger *Attr, int64_t PcOffset);

  /// \brief Add a name accelerator entry for \p Die with \p Name
  /// which is stored in the string table at \p Offset.
  void addNameAccelerator(const DIE *Die, const char *Name, uint32_t Offset,
                          bool SkipPubnamesSection = false);

  /// \brief Add a type accelerator entry for \p Die with \p Name
  /// which is stored in the string table at \p Offset.
  void addTypeAccelerator(const DIE *Die, const char *Name, uint32_t Offset);

  struct AccelInfo {
    StringRef Name; ///< Name of the entry.
    const DIE *Die; ///< DIE this entry describes.
    uint32_t NameOffset; ///< Offset of Name in the string pool.
    bool SkipPubSection; ///< Emit this entry only in the apple_* sections.

    AccelInfo(StringRef Name, const DIE *Die, uint32_t NameOffset,
              bool SkipPubSection = false)
        : Name(Name), Die(Die), NameOffset(NameOffset),
          SkipPubSection(SkipPubSection) {}
  };

  const std::vector<AccelInfo> &getPubnames() const { return Pubnames; }
  const std::vector<AccelInfo> &getPubtypes() const { return Pubtypes; }

private:
  DWARFUnit &OrigUnit;
  unsigned ID;
  std::vector<DIEInfo> Info;  ///< DIE info indexed by DIE index.
  std::unique_ptr<DIE> CUDie; ///< Root of the linked DIE tree.

  uint64_t StartOffset;
  uint64_t NextUnitOffset;

  uint64_t LowPc;
  uint64_t HighPc;

  /// \brief A list of attributes to fixup with the absolute offset of
  /// a DIE in the debug_info section.
  ///
  /// The offsets for the attributes in this array couldn't be set while
  /// cloning because for cross-cu forward refences the target DIE's
  /// offset isn't known you emit the reference attribute.
  std::vector<std::tuple<DIE *, const CompileUnit *, DIEInteger *>>
      ForwardDIEReferences;

  FunctionIntervals::Allocator RangeAlloc;
  /// \brief The ranges in that interval map are the PC ranges for
  /// functions in this unit, associated with the PC offset to apply
  /// to the addresses to get the linked address.
  FunctionIntervals Ranges;

  /// \brief DW_AT_ranges attributes to patch after we have gathered
  /// all the unit's function addresses.
  /// @{
  std::vector<DIEInteger *> RangeAttributes;
  DIEInteger *UnitRangeAttribute;
  /// @}

  /// \brief Location attributes that need to be transfered from th
  /// original debug_loc section to the liked one. They are stored
  /// along with the PC offset that is to be applied to their
  /// function's address.
  std::vector<std::pair<DIEInteger *, int64_t>> LocationAttributes;

  /// \brief Accelerator entries for the unit, both for the pub*
  /// sections and the apple* ones.
  /// @{
  std::vector<AccelInfo> Pubnames;
  std::vector<AccelInfo> Pubtypes;
  /// @}
};

uint64_t CompileUnit::computeNextUnitOffset() {
  NextUnitOffset = StartOffset + 11 /* Header size */;
  // The root DIE might be null, meaning that the Unit had nothing to
  // contribute to the linked output. In that case, we will emit the
  // unit header without any actual DIE.
  if (CUDie)
    NextUnitOffset += CUDie->getSize();
  return NextUnitOffset;
}

/// \brief Keep track of a forward cross-cu reference from this unit
/// to \p Die that lives in \p RefUnit.
void CompileUnit::noteForwardReference(DIE *Die, const CompileUnit *RefUnit,
                                       DIEInteger *Attr) {
  ForwardDIEReferences.emplace_back(Die, RefUnit, Attr);
}

/// \brief Apply all fixups recorded by noteForwardReference().
void CompileUnit::fixupForwardReferences() {
  for (const auto &Ref : ForwardDIEReferences) {
    DIE *RefDie;
    const CompileUnit *RefUnit;
    DIEInteger *Attr;
    std::tie(RefDie, RefUnit, Attr) = Ref;
    Attr->setValue(RefDie->getOffset() + RefUnit->getStartOffset());
  }
}

void CompileUnit::addFunctionRange(uint64_t FuncLowPc, uint64_t FuncHighPc,
                                   int64_t PcOffset) {
  Ranges.insert(FuncLowPc, FuncHighPc, PcOffset);
  this->LowPc = std::min(LowPc, FuncLowPc + PcOffset);
  this->HighPc = std::max(HighPc, FuncHighPc + PcOffset);
}

void CompileUnit::noteRangeAttribute(const DIE &Die, DIEInteger *Attr) {
  if (Die.getTag() != dwarf::DW_TAG_compile_unit)
    RangeAttributes.push_back(Attr);
  else
    UnitRangeAttribute = Attr;
}

void CompileUnit::noteLocationAttribute(DIEInteger *Attr, int64_t PcOffset) {
  LocationAttributes.emplace_back(Attr, PcOffset);
}

/// \brief Add a name accelerator entry for \p Die with \p Name
/// which is stored in the string table at \p Offset.
void CompileUnit::addNameAccelerator(const DIE *Die, const char *Name,
                                     uint32_t Offset, bool SkipPubSection) {
  Pubnames.emplace_back(Name, Die, Offset, SkipPubSection);
}

/// \brief Add a type accelerator entry for \p Die with \p Name
/// which is stored in the string table at \p Offset.
void CompileUnit::addTypeAccelerator(const DIE *Die, const char *Name,
                                     uint32_t Offset) {
  Pubtypes.emplace_back(Name, Die, Offset, false);
}

/// \brief A string table that doesn't need relocations.
///
/// We are doing a final link, no need for a string table that
/// has relocation entries for every reference to it. This class
/// provides this ablitity by just associating offsets with
/// strings.
class NonRelocatableStringpool {
public:
  /// \brief Entries are stored into the StringMap and simply linked
  /// together through the second element of this pair in order to
  /// keep track of insertion order.
  typedef StringMap<std::pair<uint32_t, StringMapEntryBase *>, BumpPtrAllocator>
      MapTy;

  NonRelocatableStringpool()
      : CurrentEndOffset(0), Sentinel(0), Last(&Sentinel) {
    // Legacy dsymutil puts an empty string at the start of the line
    // table.
    getStringOffset("");
  }

  /// \brief Get the offset of string \p S in the string table. This
  /// can insert a new element or return the offset of a preexisitng
  /// one.
  uint32_t getStringOffset(StringRef S);

  /// \brief Get permanent storage for \p S (but do not necessarily
  /// emit \p S in the output section).
  /// \returns The StringRef that points to permanent storage to use
  /// in place of \p S.
  StringRef internString(StringRef S);

  // \brief Return the first entry of the string table.
  const MapTy::MapEntryTy *getFirstEntry() const {
    return getNextEntry(&Sentinel);
  }

  // \brief Get the entry following \p E in the string table or null
  // if \p E was the last entry.
  const MapTy::MapEntryTy *getNextEntry(const MapTy::MapEntryTy *E) const {
    return static_cast<const MapTy::MapEntryTy *>(E->getValue().second);
  }

  uint64_t getSize() { return CurrentEndOffset; }

private:
  MapTy Strings;
  uint32_t CurrentEndOffset;
  MapTy::MapEntryTy Sentinel, *Last;
};

/// \brief Get the offset of string \p S in the string table. This
/// can insert a new element or return the offset of a preexisitng
/// one.
uint32_t NonRelocatableStringpool::getStringOffset(StringRef S) {
  if (S.empty() && !Strings.empty())
    return 0;

  std::pair<uint32_t, StringMapEntryBase *> Entry(0, nullptr);
  MapTy::iterator It;
  bool Inserted;

  // A non-empty string can't be at offset 0, so if we have an entry
  // with a 0 offset, it must be a previously interned string.
  std::tie(It, Inserted) = Strings.insert(std::make_pair(S, Entry));
  if (Inserted || It->getValue().first == 0) {
    // Set offset and chain at the end of the entries list.
    It->getValue().first = CurrentEndOffset;
    CurrentEndOffset += S.size() + 1; // +1 for the '\0'.
    Last->getValue().second = &*It;
    Last = &*It;
  }
  return It->getValue().first;
}

/// \brief Put \p S into the StringMap so that it gets permanent
/// storage, but do not actually link it in the chain of elements
/// that go into the output section. A latter call to
/// getStringOffset() with the same string will chain it though.
StringRef NonRelocatableStringpool::internString(StringRef S) {
  std::pair<uint32_t, StringMapEntryBase *> Entry(0, nullptr);
  auto InsertResult = Strings.insert(std::make_pair(S, Entry));
  return InsertResult.first->getKey();
}

/// \brief The Dwarf streaming logic
///
/// All interactions with the MC layer that is used to build the debug
/// information binary representation are handled in this class.
class DwarfStreamer {
  /// \defgroup MCObjects MC layer objects constructed by the streamer
  /// @{
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCObjectFileInfo> MOFI;
  std::unique_ptr<MCContext> MC;
  MCAsmBackend *MAB; // Owned by MCStreamer
  std::unique_ptr<MCInstrInfo> MII;
  std::unique_ptr<MCSubtargetInfo> MSTI;
  MCCodeEmitter *MCE; // Owned by MCStreamer
  MCStreamer *MS;     // Owned by AsmPrinter
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<AsmPrinter> Asm;
  /// @}

  /// \brief the file we stream the linked Dwarf to.
  std::unique_ptr<raw_fd_ostream> OutFile;

  uint32_t RangesSectionSize;
  uint32_t LocSectionSize;
  uint32_t LineSectionSize;

  /// \brief Emit the pubnames or pubtypes section contribution for \p
  /// Unit into \p Sec. The data is provided in \p Names.
  void emitPubSectionForUnit(const MCSection *Sec, StringRef Name,
                             const CompileUnit &Unit,
                             const std::vector<CompileUnit::AccelInfo> &Names);

public:
  /// \brief Actually create the streamer and the ouptut file.
  ///
  /// This could be done directly in the constructor, but it feels
  /// more natural to handle errors through return value.
  bool init(Triple TheTriple, StringRef OutputFilename);

  /// \brief Dump the file to the disk.
  bool finish();

  AsmPrinter &getAsmPrinter() const { return *Asm; }

  /// \brief Set the current output section to debug_info and change
  /// the MC Dwarf version to \p DwarfVersion.
  void switchToDebugInfoSection(unsigned DwarfVersion);

  /// \brief Emit the compilation unit header for \p Unit in the
  /// debug_info section.
  ///
  /// As a side effect, this also switches the current Dwarf version
  /// of the MC layer to the one of U.getOrigUnit().
  void emitCompileUnitHeader(CompileUnit &Unit);

  /// \brief Recursively emit the DIE tree rooted at \p Die.
  void emitDIE(DIE &Die);

  /// \brief Emit the abbreviation table \p Abbrevs to the
  /// debug_abbrev section.
  void emitAbbrevs(const std::vector<DIEAbbrev *> &Abbrevs);

  /// \brief Emit the string table described by \p Pool.
  void emitStrings(const NonRelocatableStringpool &Pool);

  /// \brief Emit debug_ranges for \p FuncRange by translating the
  /// original \p Entries.
  void emitRangesEntries(
      int64_t UnitPcOffset, uint64_t OrigLowPc,
      FunctionIntervals::const_iterator FuncRange,
      const std::vector<DWARFDebugRangeList::RangeListEntry> &Entries,
      unsigned AddressSize);

  /// \brief Emit debug_aranges entries for \p Unit and if \p
  /// DoRangesSection is true, also emit the debug_ranges entries for
  /// the DW_TAG_compile_unit's DW_AT_ranges attribute.
  void emitUnitRangesEntries(CompileUnit &Unit, bool DoRangesSection);

  uint32_t getRangesSectionSize() const { return RangesSectionSize; }

  /// \brief Emit the debug_loc contribution for \p Unit by copying
  /// the entries from \p Dwarf and offseting them. Update the
  /// location attributes to point to the new entries.
  void emitLocationsForUnit(const CompileUnit &Unit, DWARFContext &Dwarf);

  /// \brief Emit the line table described in \p Rows into the
  /// debug_line section.
  void emitLineTableForUnit(StringRef PrologueBytes, unsigned MinInstLength,
                            std::vector<DWARFDebugLine::Row> &Rows,
                            unsigned AdddressSize);

  uint32_t getLineSectionSize() const { return LineSectionSize; }

  /// \brief Emit the .debug_pubnames contribution for \p Unit.
  void emitPubNamesForUnit(const CompileUnit &Unit);

  /// \brief Emit the .debug_pubtypes contribution for \p Unit.
  void emitPubTypesForUnit(const CompileUnit &Unit);
};

bool DwarfStreamer::init(Triple TheTriple, StringRef OutputFilename) {
  std::string ErrorStr;
  std::string TripleName;
  StringRef Context = "dwarf streamer init";

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TripleName, TheTriple, ErrorStr);
  if (!TheTarget)
    return error(ErrorStr, Context);
  TripleName = TheTriple.getTriple();

  // Create all the MC Objects.
  MRI.reset(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return error(Twine("no register info for target ") + TripleName, Context);

  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!MAI)
    return error("no asm info for target " + TripleName, Context);

  MOFI.reset(new MCObjectFileInfo);
  MC.reset(new MCContext(MAI.get(), MRI.get(), MOFI.get()));
  MOFI->InitMCObjectFileInfo(TripleName, Reloc::Default, CodeModel::Default,
                             *MC);

  MAB = TheTarget->createMCAsmBackend(*MRI, TripleName, "");
  if (!MAB)
    return error("no asm backend for target " + TripleName, Context);

  MII.reset(TheTarget->createMCInstrInfo());
  if (!MII)
    return error("no instr info info for target " + TripleName, Context);

  MSTI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context);

  MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, *MC);
  if (!MCE)
    return error("no code emitter for target " + TripleName, Context);

  // Create the output file.
  std::error_code EC;
  OutFile =
      llvm::make_unique<raw_fd_ostream>(OutputFilename, EC, sys::fs::F_None);
  if (EC)
    return error(Twine(OutputFilename) + ": " + EC.message(), Context);

  MS = TheTarget->createMCObjectStreamer(TripleName, *MC, *MAB, *OutFile, MCE,
                                         *MSTI, false);
  if (!MS)
    return error("no object streamer for target " + TripleName, Context);

  // Finally create the AsmPrinter we'll use to emit the DIEs.
  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions()));
  if (!TM)
    return error("no target machine for target " + TripleName, Context);

  Asm.reset(TheTarget->createAsmPrinter(*TM, std::unique_ptr<MCStreamer>(MS)));
  if (!Asm)
    return error("no asm printer for target " + TripleName, Context);

  RangesSectionSize = 0;
  LocSectionSize = 0;
  LineSectionSize = 0;

  return true;
}

bool DwarfStreamer::finish() {
  MS->Finish();
  return true;
}

/// \brief Set the current output section to debug_info and change
/// the MC Dwarf version to \p DwarfVersion.
void DwarfStreamer::switchToDebugInfoSection(unsigned DwarfVersion) {
  MS->SwitchSection(MOFI->getDwarfInfoSection());
  MC->setDwarfVersion(DwarfVersion);
}

/// \brief Emit the compilation unit header for \p Unit in the
/// debug_info section.
///
/// A Dwarf scetion header is encoded as:
///  uint32_t   Unit length (omiting this field)
///  uint16_t   Version
///  uint32_t   Abbreviation table offset
///  uint8_t    Address size
///
/// Leading to a total of 11 bytes.
void DwarfStreamer::emitCompileUnitHeader(CompileUnit &Unit) {
  unsigned Version = Unit.getOrigUnit().getVersion();
  switchToDebugInfoSection(Version);

  // Emit size of content not including length itself. The size has
  // already been computed in CompileUnit::computeOffsets(). Substract
  // 4 to that size to account for the length field.
  Asm->EmitInt32(Unit.getNextUnitOffset() - Unit.getStartOffset() - 4);
  Asm->EmitInt16(Version);
  // We share one abbreviations table across all units so it's always at the
  // start of the section.
  Asm->EmitInt32(0);
  Asm->EmitInt8(Unit.getOrigUnit().getAddressByteSize());
}

/// \brief Emit the \p Abbrevs array as the shared abbreviation table
/// for the linked Dwarf file.
void DwarfStreamer::emitAbbrevs(const std::vector<DIEAbbrev *> &Abbrevs) {
  MS->SwitchSection(MOFI->getDwarfAbbrevSection());
  Asm->emitDwarfAbbrevs(Abbrevs);
}

/// \brief Recursively emit the DIE tree rooted at \p Die.
void DwarfStreamer::emitDIE(DIE &Die) {
  MS->SwitchSection(MOFI->getDwarfInfoSection());
  Asm->emitDwarfDIE(Die);
}

/// \brief Emit the debug_str section stored in \p Pool.
void DwarfStreamer::emitStrings(const NonRelocatableStringpool &Pool) {
  Asm->OutStreamer.SwitchSection(MOFI->getDwarfStrSection());
  for (auto *Entry = Pool.getFirstEntry(); Entry;
       Entry = Pool.getNextEntry(Entry))
    Asm->OutStreamer.EmitBytes(
        StringRef(Entry->getKey().data(), Entry->getKey().size() + 1));
}

/// \brief Emit the debug_range section contents for \p FuncRange by
/// translating the original \p Entries. The debug_range section
/// format is totally trivial, consisting just of pairs of address
/// sized addresses describing the ranges.
void DwarfStreamer::emitRangesEntries(
    int64_t UnitPcOffset, uint64_t OrigLowPc,
    FunctionIntervals::const_iterator FuncRange,
    const std::vector<DWARFDebugRangeList::RangeListEntry> &Entries,
    unsigned AddressSize) {
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfRangesSection());

  // Offset each range by the right amount.
  int64_t PcOffset = FuncRange.value() + UnitPcOffset;
  for (const auto &Range : Entries) {
    if (Range.isBaseAddressSelectionEntry(AddressSize)) {
      warn("unsupported base address selection operation",
           "emitting debug_ranges");
      break;
    }
    // Do not emit empty ranges.
    if (Range.StartAddress == Range.EndAddress)
      continue;

    // All range entries should lie in the function range.
    if (!(Range.StartAddress + OrigLowPc >= FuncRange.start() &&
          Range.EndAddress + OrigLowPc <= FuncRange.stop()))
      warn("inconsistent range data.", "emitting debug_ranges");
    MS->EmitIntValue(Range.StartAddress + PcOffset, AddressSize);
    MS->EmitIntValue(Range.EndAddress + PcOffset, AddressSize);
    RangesSectionSize += 2 * AddressSize;
  }

  // Add the terminator entry.
  MS->EmitIntValue(0, AddressSize);
  MS->EmitIntValue(0, AddressSize);
  RangesSectionSize += 2 * AddressSize;
}

/// \brief Emit the debug_aranges contribution of a unit and
/// if \p DoDebugRanges is true the debug_range contents for a
/// compile_unit level DW_AT_ranges attribute (Which are basically the
/// same thing with a different base address).
/// Just aggregate all the ranges gathered inside that unit.
void DwarfStreamer::emitUnitRangesEntries(CompileUnit &Unit,
                                          bool DoDebugRanges) {
  unsigned AddressSize = Unit.getOrigUnit().getAddressByteSize();
  // Gather the ranges in a vector, so that we can simplify them. The
  // IntervalMap will have coalesced the non-linked ranges, but here
  // we want to coalesce the linked addresses.
  std::vector<std::pair<uint64_t, uint64_t>> Ranges;
  const auto &FunctionRanges = Unit.getFunctionRanges();
  for (auto Range = FunctionRanges.begin(), End = FunctionRanges.end();
       Range != End; ++Range)
    Ranges.push_back(std::make_pair(Range.start() + Range.value(),
                                    Range.stop() + Range.value()));

  // The object addresses where sorted, but again, the linked
  // addresses might end up in a different order.
  std::sort(Ranges.begin(), Ranges.end());

  if (!Ranges.empty()) {
    MS->SwitchSection(MC->getObjectFileInfo()->getDwarfARangesSection());

    MCSymbol *BeginLabel = Asm->GetTempSymbol("Barange", Unit.getUniqueID());
    MCSymbol *EndLabel = Asm->GetTempSymbol("Earange", Unit.getUniqueID());

    unsigned HeaderSize =
        sizeof(int32_t) + // Size of contents (w/o this field
        sizeof(int16_t) + // DWARF ARange version number
        sizeof(int32_t) + // Offset of CU in the .debug_info section
        sizeof(int8_t) +  // Pointer Size (in bytes)
        sizeof(int8_t);   // Segment Size (in bytes)

    unsigned TupleSize = AddressSize * 2;
    unsigned Padding = OffsetToAlignment(HeaderSize, TupleSize);

    Asm->EmitLabelDifference(EndLabel, BeginLabel, 4); // Arange length
    Asm->OutStreamer.EmitLabel(BeginLabel);
    Asm->EmitInt16(dwarf::DW_ARANGES_VERSION); // Version number
    Asm->EmitInt32(Unit.getStartOffset());     // Corresponding unit's offset
    Asm->EmitInt8(AddressSize);                // Address size
    Asm->EmitInt8(0);                          // Segment size

    Asm->OutStreamer.EmitFill(Padding, 0x0);

    for (auto Range = Ranges.begin(), End = Ranges.end(); Range != End;
         ++Range) {
      uint64_t RangeStart = Range->first;
      MS->EmitIntValue(RangeStart, AddressSize);
      while ((Range + 1) != End && Range->second == (Range + 1)->first)
        ++Range;
      MS->EmitIntValue(Range->second - RangeStart, AddressSize);
    }

    // Emit terminator
    Asm->OutStreamer.EmitIntValue(0, AddressSize);
    Asm->OutStreamer.EmitIntValue(0, AddressSize);
    Asm->OutStreamer.EmitLabel(EndLabel);
  }

  if (!DoDebugRanges)
    return;

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfRangesSection());
  // Offset each range by the right amount.
  int64_t PcOffset = -Unit.getLowPc();
  // Emit coalesced ranges.
  for (auto Range = Ranges.begin(), End = Ranges.end(); Range != End; ++Range) {
    MS->EmitIntValue(Range->first + PcOffset, AddressSize);
    while (Range + 1 != End && Range->second == (Range + 1)->first)
      ++Range;
    MS->EmitIntValue(Range->second + PcOffset, AddressSize);
    RangesSectionSize += 2 * AddressSize;
  }

  // Add the terminator entry.
  MS->EmitIntValue(0, AddressSize);
  MS->EmitIntValue(0, AddressSize);
  RangesSectionSize += 2 * AddressSize;
}

/// \brief Emit location lists for \p Unit and update attribtues to
/// point to the new entries.
void DwarfStreamer::emitLocationsForUnit(const CompileUnit &Unit,
                                         DWARFContext &Dwarf) {
  const std::vector<std::pair<DIEInteger *, int64_t>> &Attributes =
      Unit.getLocationAttributes();

  if (Attributes.empty())
    return;

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLocSection());

  unsigned AddressSize = Unit.getOrigUnit().getAddressByteSize();
  const DWARFSection &InputSec = Dwarf.getLocSection();
  DataExtractor Data(InputSec.Data, Dwarf.isLittleEndian(), AddressSize);
  DWARFUnit &OrigUnit = Unit.getOrigUnit();
  const auto *OrigUnitDie = OrigUnit.getCompileUnitDIE(false);
  int64_t UnitPcOffset = 0;
  uint64_t OrigLowPc = OrigUnitDie->getAttributeValueAsAddress(
      &OrigUnit, dwarf::DW_AT_low_pc, -1ULL);
  if (OrigLowPc != -1ULL)
    UnitPcOffset = int64_t(OrigLowPc) - Unit.getLowPc();

  for (const auto &Attr : Attributes) {
    uint32_t Offset = Attr.first->getValue();
    Attr.first->setValue(LocSectionSize);
    // This is the quantity to add to the old location address to get
    // the correct address for the new one.
    int64_t LocPcOffset = Attr.second + UnitPcOffset;
    while (Data.isValidOffset(Offset)) {
      uint64_t Low = Data.getUnsigned(&Offset, AddressSize);
      uint64_t High = Data.getUnsigned(&Offset, AddressSize);
      LocSectionSize += 2 * AddressSize;
      if (Low == 0 && High == 0) {
        Asm->OutStreamer.EmitIntValue(0, AddressSize);
        Asm->OutStreamer.EmitIntValue(0, AddressSize);
        break;
      }
      Asm->OutStreamer.EmitIntValue(Low + LocPcOffset, AddressSize);
      Asm->OutStreamer.EmitIntValue(High + LocPcOffset, AddressSize);
      uint64_t Length = Data.getU16(&Offset);
      Asm->OutStreamer.EmitIntValue(Length, 2);
      // Just copy the bytes over.
      Asm->OutStreamer.EmitBytes(
          StringRef(InputSec.Data.substr(Offset, Length)));
      Offset += Length;
      LocSectionSize += Length + 2;
    }
  }
}

void DwarfStreamer::emitLineTableForUnit(StringRef PrologueBytes,
                                         unsigned MinInstLength,
                                         std::vector<DWARFDebugLine::Row> &Rows,
                                         unsigned PointerSize) {
  // Switch to the section where the table will be emitted into.
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLineSection());
  MCSymbol *LineStartSym = MC->CreateTempSymbol();
  MCSymbol *LineEndSym = MC->CreateTempSymbol();

  // The first 4 bytes is the total length of the information for this
  // compilation unit (not including these 4 bytes for the length).
  Asm->EmitLabelDifference(LineEndSym, LineStartSym, 4);
  Asm->OutStreamer.EmitLabel(LineStartSym);
  // Copy Prologue.
  MS->EmitBytes(PrologueBytes);
  LineSectionSize += PrologueBytes.size() + 4;

  SmallString<128> EncodingBuffer;
  raw_svector_ostream EncodingOS(EncodingBuffer);

  if (Rows.empty()) {
    // We only have the dummy entry, dsymutil emits an entry with a 0
    // address in that case.
    MCDwarfLineAddr::Encode(*MC, INT64_MAX, 0, EncodingOS);
    MS->EmitBytes(EncodingOS.str());
    LineSectionSize += EncodingBuffer.size();
    MS->EmitLabel(LineEndSym);
    return;
  }

  // Line table state machine fields
  unsigned FileNum = 1;
  unsigned LastLine = 1;
  unsigned Column = 0;
  unsigned IsStatement = 1;
  unsigned Isa = 0;
  uint64_t Address = -1ULL;

  unsigned RowsSinceLastSequence = 0;

  for (unsigned Idx = 0; Idx < Rows.size(); ++Idx) {
    auto &Row = Rows[Idx];

    int64_t AddressDelta;
    if (Address == -1ULL) {
      MS->EmitIntValue(dwarf::DW_LNS_extended_op, 1);
      MS->EmitULEB128IntValue(PointerSize + 1);
      MS->EmitIntValue(dwarf::DW_LNE_set_address, 1);
      MS->EmitIntValue(Row.Address, PointerSize);
      LineSectionSize += 2 + PointerSize + getULEB128Size(PointerSize + 1);
      AddressDelta = 0;
    } else {
      AddressDelta = (Row.Address - Address) / MinInstLength;
    }

    // FIXME: code copied and transfromed from
    // MCDwarf.cpp::EmitDwarfLineTable. We should find a way to share
    // this code, but the current compatibility requirement with
    // classic dsymutil makes it hard. Revisit that once this
    // requirement is dropped.

    if (FileNum != Row.File) {
      FileNum = Row.File;
      MS->EmitIntValue(dwarf::DW_LNS_set_file, 1);
      MS->EmitULEB128IntValue(FileNum);
      LineSectionSize += 1 + getULEB128Size(FileNum);
    }
    if (Column != Row.Column) {
      Column = Row.Column;
      MS->EmitIntValue(dwarf::DW_LNS_set_column, 1);
      MS->EmitULEB128IntValue(Column);
      LineSectionSize += 1 + getULEB128Size(Column);
    }

    // FIXME: We should handle the discriminator here, but dsymutil
    // doesn' consider it, thus ignore it for now.

    if (Isa != Row.Isa) {
      Isa = Row.Isa;
      MS->EmitIntValue(dwarf::DW_LNS_set_isa, 1);
      MS->EmitULEB128IntValue(Isa);
      LineSectionSize += 1 + getULEB128Size(Isa);
    }
    if (IsStatement != Row.IsStmt) {
      IsStatement = Row.IsStmt;
      MS->EmitIntValue(dwarf::DW_LNS_negate_stmt, 1);
      LineSectionSize += 1;
    }
    if (Row.BasicBlock) {
      MS->EmitIntValue(dwarf::DW_LNS_set_basic_block, 1);
      LineSectionSize += 1;
    }

    if (Row.PrologueEnd) {
      MS->EmitIntValue(dwarf::DW_LNS_set_prologue_end, 1);
      LineSectionSize += 1;
    }

    if (Row.EpilogueBegin) {
      MS->EmitIntValue(dwarf::DW_LNS_set_epilogue_begin, 1);
      LineSectionSize += 1;
    }

    int64_t LineDelta = int64_t(Row.Line) - LastLine;
    if (!Row.EndSequence) {
      MCDwarfLineAddr::Encode(*MC, LineDelta, AddressDelta, EncodingOS);
      MS->EmitBytes(EncodingOS.str());
      LineSectionSize += EncodingBuffer.size();
      EncodingBuffer.resize(0);
      EncodingOS.resync();
      Address = Row.Address;
      LastLine = Row.Line;
      RowsSinceLastSequence++;
    } else {
      if (LineDelta) {
        MS->EmitIntValue(dwarf::DW_LNS_advance_line, 1);
        MS->EmitSLEB128IntValue(LineDelta);
        LineSectionSize += 1 + getSLEB128Size(LineDelta);
      }
      if (AddressDelta) {
        MS->EmitIntValue(dwarf::DW_LNS_advance_pc, 1);
        MS->EmitULEB128IntValue(AddressDelta);
        LineSectionSize += 1 + getULEB128Size(AddressDelta);
      }
      MCDwarfLineAddr::Encode(*MC, INT64_MAX, 0, EncodingOS);
      MS->EmitBytes(EncodingOS.str());
      LineSectionSize += EncodingBuffer.size();
      EncodingBuffer.resize(0);
      EncodingOS.resync();
      Address = -1ULL;
      LastLine = FileNum = IsStatement = 1;
      RowsSinceLastSequence = Column = Isa = 0;
    }
  }

  if (RowsSinceLastSequence) {
    MCDwarfLineAddr::Encode(*MC, INT64_MAX, 0, EncodingOS);
    MS->EmitBytes(EncodingOS.str());
    LineSectionSize += EncodingBuffer.size();
    EncodingBuffer.resize(0);
    EncodingOS.resync();
  }

  MS->EmitLabel(LineEndSym);
}

/// \brief Emit the pubnames or pubtypes section contribution for \p
/// Unit into \p Sec. The data is provided in \p Names.
void DwarfStreamer::emitPubSectionForUnit(
    const MCSection *Sec, StringRef SecName, const CompileUnit &Unit,
    const std::vector<CompileUnit::AccelInfo> &Names) {
  if (Names.empty())
    return;

  // Start the dwarf pubnames section.
  Asm->OutStreamer.SwitchSection(Sec);
  MCSymbol *BeginLabel =
      Asm->GetTempSymbol("pub" + SecName + "_begin", Unit.getUniqueID());
  MCSymbol *EndLabel =
      Asm->GetTempSymbol("pub" + SecName + "_end", Unit.getUniqueID());

  bool HeaderEmitted = false;
  // Emit the pubnames for this compilation unit.
  for (const auto &Name : Names) {
    if (Name.SkipPubSection)
      continue;

    if (!HeaderEmitted) {
      // Emit the header.
      Asm->EmitLabelDifference(EndLabel, BeginLabel, 4); // Length
      Asm->OutStreamer.EmitLabel(BeginLabel);
      Asm->EmitInt16(dwarf::DW_PUBNAMES_VERSION); // Version
      Asm->EmitInt32(Unit.getStartOffset()); // Unit offset
      Asm->EmitInt32(Unit.getNextUnitOffset() - Unit.getStartOffset()); // Size
      HeaderEmitted = true;
    }
    Asm->EmitInt32(Name.Die->getOffset());
    Asm->OutStreamer.EmitBytes(
        StringRef(Name.Name.data(), Name.Name.size() + 1));
  }

  if (!HeaderEmitted)
    return;
  Asm->EmitInt32(0); // End marker.
  Asm->OutStreamer.EmitLabel(EndLabel);
}

/// \brief Emit .debug_pubnames for \p Unit.
void DwarfStreamer::emitPubNamesForUnit(const CompileUnit &Unit) {
  emitPubSectionForUnit(MC->getObjectFileInfo()->getDwarfPubNamesSection(),
                        "names", Unit, Unit.getPubnames());
}

/// \brief Emit .debug_pubtypes for \p Unit.
void DwarfStreamer::emitPubTypesForUnit(const CompileUnit &Unit) {
  emitPubSectionForUnit(MC->getObjectFileInfo()->getDwarfPubTypesSection(),
                        "types", Unit, Unit.getPubtypes());
}

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
  DwarfLinker(StringRef OutputFilename, const LinkOptions &Options)
      : OutputFilename(OutputFilename), Options(Options),
        BinHolder(Options.Verbose) {}

  ~DwarfLinker() {
    for (auto *Abbrev : Abbreviations)
      delete Abbrev;
  }

  /// \brief Link the contents of the DebugMap.
  bool link(const DebugMap &);

private:
  /// \brief Called at the start of a debug object link.
  void startDebugObject(DWARFContext &, DebugMapObject &);

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
  /// root DIE selection and during DIE cloning.
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

  /// \defgroup FindRootDIEs Find DIEs corresponding to debug map entries.
  ///
  /// @{
  /// \brief Recursively walk the \p DIE tree and look for DIEs to
  /// keep. Store that information in \p CU's DIEInfo.
  void lookForDIEsToKeep(const DWARFDebugInfoEntryMinimal &DIE,
                         const DebugMapObject &DMO, CompileUnit &CU,
                         unsigned Flags);

  /// \brief Flags passed to DwarfLinker::lookForDIEsToKeep
  enum TravesalFlags {
    TF_Keep = 1 << 0,            ///< Mark the traversed DIEs as kept.
    TF_InFunctionScope = 1 << 1, ///< Current scope is a fucntion scope.
    TF_DependencyWalk = 1 << 2,  ///< Walking the dependencies of a kept DIE.
    TF_ParentWalk = 1 << 3,      ///< Walking up the parents of a kept DIE.
  };

  /// \brief Mark the passed DIE as well as all the ones it depends on
  /// as kept.
  void keepDIEAndDenpendencies(const DWARFDebugInfoEntryMinimal &DIE,
                               CompileUnit::DIEInfo &MyInfo,
                               const DebugMapObject &DMO, CompileUnit &CU,
                               unsigned Flags);

  unsigned shouldKeepDIE(const DWARFDebugInfoEntryMinimal &DIE,
                         CompileUnit &Unit, CompileUnit::DIEInfo &MyInfo,
                         unsigned Flags);

  unsigned shouldKeepVariableDIE(const DWARFDebugInfoEntryMinimal &DIE,
                                 CompileUnit &Unit,
                                 CompileUnit::DIEInfo &MyInfo, unsigned Flags);

  unsigned shouldKeepSubprogramDIE(const DWARFDebugInfoEntryMinimal &DIE,
                                   CompileUnit &Unit,
                                   CompileUnit::DIEInfo &MyInfo,
                                   unsigned Flags);

  bool hasValidRelocation(uint32_t StartOffset, uint32_t EndOffset,
                          CompileUnit::DIEInfo &Info);
  /// @}

  /// \defgroup Linking Methods used to link the debug information
  ///
  /// @{
  /// \brief Recursively clone \p InputDIE into an tree of DIE objects
  /// where useless (as decided by lookForDIEsToKeep()) bits have been
  /// stripped out and addresses have been rewritten according to the
  /// debug map.
  ///
  /// \param OutOffset is the offset the cloned DIE in the output
  /// compile unit.
  /// \param PCOffset (while cloning a function scope) is the offset
  /// applied to the entry point of the function to get the linked address.
  ///
  /// \returns the root of the cloned tree.
  DIE *cloneDIE(const DWARFDebugInfoEntryMinimal &InputDIE, CompileUnit &U,
                int64_t PCOffset, uint32_t OutOffset);

  typedef DWARFAbbreviationDeclaration::AttributeSpec AttributeSpec;

  /// \brief Information gathered and exchanged between the various
  /// clone*Attributes helpers about the attributes of a particular DIE.
  struct AttributesInfo {
    const char *Name, *MangledName;         ///< Names.
    uint32_t NameOffset, MangledNameOffset; ///< Offsets in the string pool.

    uint64_t OrigHighPc; ///< Value of AT_high_pc in the input DIE
    int64_t PCOffset;    ///< Offset to apply to PC addresses inside a function.

    bool HasLowPc;      ///< Does the DIE have a low_pc attribute?
    bool IsDeclaration; ///< Is this DIE only a declaration?

    AttributesInfo()
        : Name(nullptr), MangledName(nullptr), NameOffset(0),
          MangledNameOffset(0), OrigHighPc(0), PCOffset(0), HasLowPc(false),
          IsDeclaration(false) {}
  };

  /// \brief Helper for cloneDIE.
  unsigned cloneAttribute(DIE &Die, const DWARFDebugInfoEntryMinimal &InputDIE,
                          CompileUnit &U, const DWARFFormValue &Val,
                          const AttributeSpec AttrSpec, unsigned AttrSize,
                          AttributesInfo &AttrInfo);

  /// \brief Helper for cloneDIE.
  unsigned cloneStringAttribute(DIE &Die, AttributeSpec AttrSpec,
                                const DWARFFormValue &Val, const DWARFUnit &U);

  /// \brief Helper for cloneDIE.
  unsigned
  cloneDieReferenceAttribute(DIE &Die,
                             const DWARFDebugInfoEntryMinimal &InputDIE,
                             AttributeSpec AttrSpec, unsigned AttrSize,
                             const DWARFFormValue &Val, CompileUnit &Unit);

  /// \brief Helper for cloneDIE.
  unsigned cloneBlockAttribute(DIE &Die, AttributeSpec AttrSpec,
                               const DWARFFormValue &Val, unsigned AttrSize);

  /// \brief Helper for cloneDIE.
  unsigned cloneAddressAttribute(DIE &Die, AttributeSpec AttrSpec,
                                 const DWARFFormValue &Val,
                                 const CompileUnit &Unit, AttributesInfo &Info);

  /// \brief Helper for cloneDIE.
  unsigned cloneScalarAttribute(DIE &Die,
                                const DWARFDebugInfoEntryMinimal &InputDIE,
                                CompileUnit &U, AttributeSpec AttrSpec,
                                const DWARFFormValue &Val, unsigned AttrSize,
                                AttributesInfo &Info);

  /// \brief Helper for cloneDIE.
  bool applyValidRelocs(MutableArrayRef<char> Data, uint32_t BaseOffset,
                        bool isLittleEndian);

  /// \brief Assign an abbreviation number to \p Abbrev
  void AssignAbbrev(DIEAbbrev &Abbrev);

  /// \brief FoldingSet that uniques the abbreviations.
  FoldingSet<DIEAbbrev> AbbreviationsSet;
  /// \brief Storage for the unique Abbreviations.
  /// This is passed to AsmPrinter::emitDwarfAbbrevs(), thus it cannot
  /// be changed to a vecot of unique_ptrs.
  std::vector<DIEAbbrev *> Abbreviations;

  /// \brief Compute and emit debug_ranges section for \p Unit, and
  /// patch the attributes referencing it.
  void patchRangesForUnit(const CompileUnit &Unit, DWARFContext &Dwarf) const;

  /// \brief Generate and emit the DW_AT_ranges attribute for a
  /// compile_unit if it had one.
  void generateUnitRanges(CompileUnit &Unit) const;

  /// \brief Extract the line tables fromt he original dwarf, extract
  /// the relevant parts according to the linked function ranges and
  /// emit the result in the debug_line section.
  void patchLineTableForUnit(CompileUnit &Unit, DWARFContext &OrigDwarf);

  /// \brief Emit the accelerator entries for \p Unit.
  void emitAcceleratorEntriesForUnit(CompileUnit &Unit);

  /// \brief DIELoc objects that need to be destructed (but not freed!).
  std::vector<DIELoc *> DIELocs;
  /// \brief DIEBlock objects that need to be destructed (but not freed!).
  std::vector<DIEBlock *> DIEBlocks;
  /// \brief Allocator used for all the DIEValue objects.
  BumpPtrAllocator DIEAlloc;
  /// @}

  /// \defgroup Helpers Various helper methods.
  ///
  /// @{
  const DWARFDebugInfoEntryMinimal *
  resolveDIEReference(DWARFFormValue &RefValue, const DWARFUnit &Unit,
                      const DWARFDebugInfoEntryMinimal &DIE,
                      CompileUnit *&ReferencedCU);

  CompileUnit *getUnitForOffset(unsigned Offset);

  bool getDIENames(const DWARFDebugInfoEntryMinimal &Die, DWARFUnit &U,
                   AttributesInfo &Info);

  void reportWarning(const Twine &Warning, const DWARFUnit *Unit = nullptr,
                     const DWARFDebugInfoEntryMinimal *DIE = nullptr) const;

  bool createStreamer(Triple TheTriple, StringRef OutputFilename);
  /// @}

private:
  std::string OutputFilename;
  LinkOptions Options;
  BinaryHolder BinHolder;
  std::unique_ptr<DwarfStreamer> Streamer;

  /// The units of the current debug map object.
  std::vector<CompileUnit> Units;

  /// The debug map object curently under consideration.
  DebugMapObject *CurrentDebugObject;

  /// \brief The Dwarf string pool
  NonRelocatableStringpool StringPool;

  /// \brief This map is keyed by the entry PC of functions in that
  /// debug object and the associated value is a pair storing the
  /// corresponding end PC and the offset to apply to get the linked
  /// address.
  ///
  /// See startDebugObject() for a more complete description of its use.
  std::map<uint64_t, std::pair<uint64_t, int64_t>> Ranges;
};

/// \brief Similar to DWARFUnitSection::getUnitForOffset(), but
/// returning our CompileUnit object instead.
CompileUnit *DwarfLinker::getUnitForOffset(unsigned Offset) {
  auto CU =
      std::upper_bound(Units.begin(), Units.end(), Offset,
                       [](uint32_t LHS, const CompileUnit &RHS) {
                         return LHS < RHS.getOrigUnit().getNextUnitOffset();
                       });
  return CU != Units.end() ? &*CU : nullptr;
}

/// \brief Resolve the DIE attribute reference that has been
/// extracted in \p RefValue. The resulting DIE migh be in another
/// CompileUnit which is stored into \p ReferencedCU.
/// \returns null if resolving fails for any reason.
const DWARFDebugInfoEntryMinimal *DwarfLinker::resolveDIEReference(
    DWARFFormValue &RefValue, const DWARFUnit &Unit,
    const DWARFDebugInfoEntryMinimal &DIE, CompileUnit *&RefCU) {
  assert(RefValue.isFormClass(DWARFFormValue::FC_Reference));
  uint64_t RefOffset = *RefValue.getAsReference(&Unit);

  if ((RefCU = getUnitForOffset(RefOffset)))
    if (const auto *RefDie = RefCU->getOrigUnit().getDIEForOffset(RefOffset))
      return RefDie;

  reportWarning("could not find referenced DIE", &Unit, &DIE);
  return nullptr;
}

/// \brief Get the potential name and mangled name for the entity
/// described by \p Die and store them in \Info if they are not
/// already there.
/// \returns is a name was found.
bool DwarfLinker::getDIENames(const DWARFDebugInfoEntryMinimal &Die,
                              DWARFUnit &U, AttributesInfo &Info) {
  // FIXME: a bit wastefull as the first getName might return the
  // short name.
  if (!Info.MangledName &&
      (Info.MangledName = Die.getName(&U, DINameKind::LinkageName)))
    Info.MangledNameOffset = StringPool.getStringOffset(Info.MangledName);

  if (!Info.Name && (Info.Name = Die.getName(&U, DINameKind::ShortName)))
    Info.NameOffset = StringPool.getStringOffset(Info.Name);

  return Info.Name || Info.MangledName;
}

/// \brief Report a warning to the user, optionaly including
/// information about a specific \p DIE related to the warning.
void DwarfLinker::reportWarning(const Twine &Warning, const DWARFUnit *Unit,
                                const DWARFDebugInfoEntryMinimal *DIE) const {
  StringRef Context = "<debug map>";
  if (CurrentDebugObject)
    Context = CurrentDebugObject->getObjectFilename();
  warn(Warning, Context);

  if (!Options.Verbose || !DIE)
    return;

  errs() << "    in DIE:\n";
  DIE->dump(errs(), const_cast<DWARFUnit *>(Unit), 0 /* RecurseDepth */,
            6 /* Indent */);
}

bool DwarfLinker::createStreamer(Triple TheTriple, StringRef OutputFilename) {
  if (Options.NoOutput)
    return true;

  Streamer = llvm::make_unique<DwarfStreamer>();
  return Streamer->init(TheTriple, OutputFilename);
}

/// \brief Recursive helper to gather the child->parent relationships in the
/// original compile unit.
static void gatherDIEParents(const DWARFDebugInfoEntryMinimal *DIE,
                             unsigned ParentIdx, CompileUnit &CU) {
  unsigned MyIdx = CU.getOrigUnit().getDIEIndex(DIE);
  CU.getInfo(MyIdx).ParentIdx = ParentIdx;

  if (DIE->hasChildren())
    for (auto *Child = DIE->getFirstChild(); Child && !Child->isNULL();
         Child = Child->getSibling())
      gatherDIEParents(Child, MyIdx, CU);
}

static bool dieNeedsChildrenToBeMeaningful(uint32_t Tag) {
  switch (Tag) {
  default:
    return false;
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_lexical_block:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_union_type:
    return true;
  }
  llvm_unreachable("Invalid Tag");
}

void DwarfLinker::startDebugObject(DWARFContext &Dwarf, DebugMapObject &Obj) {
  Units.reserve(Dwarf.getNumCompileUnits());
  NextValidReloc = 0;
  // Iterate over the debug map entries and put all the ones that are
  // functions (because they have a size) into the Ranges map. This
  // map is very similar to the FunctionRanges that are stored in each
  // unit, with 2 notable differences:
  //  - obviously this one is global, while the other ones are per-unit.
  //  - this one contains not only the functions described in the DIE
  // tree, but also the ones that are only in the debug map.
  // The latter information is required to reproduce dsymutil's logic
  // while linking line tables. The cases where this information
  // matters look like bugs that need to be investigated, but for now
  // we need to reproduce dsymutil's behavior.
  // FIXME: Once we understood exactly if that information is needed,
  // maybe totally remove this (or try to use it to do a real
  // -gline-tables-only on Darwin.
  for (const auto &Entry : Obj.symbols()) {
    const auto &Mapping = Entry.getValue();
    if (Mapping.Size)
      Ranges[Mapping.ObjectAddress] = std::make_pair(
          Mapping.ObjectAddress + Mapping.Size,
          int64_t(Mapping.BinaryAddress) - Mapping.ObjectAddress);
  }
}

void DwarfLinker::endDebugObject() {
  Units.clear();
  ValidRelocs.clear();
  Ranges.clear();

  for (auto *Block : DIEBlocks)
    Block->~DIEBlock();
  for (auto *Loc : DIELocs)
    Loc->~DIELoc();

  DIEBlocks.clear();
  DIELocs.clear();
  DIEAlloc.Reset();
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
      reportWarning(" unsupported relocation in debug_info section.");
      continue;
    }
    uint32_t Offset = Offset64;
    // Mach-o uses REL relocations, the addend is at the relocation offset.
    uint64_t Addend = Data.getUnsigned(&Offset, RelocSize);

    auto Sym = Reloc.getSymbol();
    if (Sym != Obj.symbol_end()) {
      StringRef SymbolName;
      if (Sym->getName(SymbolName)) {
        reportWarning("error getting relocation symbol name.");
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
    reportWarning(Twine("unsupported object file type: ") + Obj.getFileName());

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

/// \brief Checks that there is a relocation against an actual debug
/// map entry between \p StartOffset and \p NextOffset.
///
/// This function must be called with offsets in strictly ascending
/// order because it never looks back at relocations it already 'went past'.
/// \returns true and sets Info.InDebugMap if it is the case.
bool DwarfLinker::hasValidRelocation(uint32_t StartOffset, uint32_t EndOffset,
                                     CompileUnit::DIEInfo &Info) {
  assert(NextValidReloc == 0 ||
         StartOffset > ValidRelocs[NextValidReloc - 1].Offset);
  if (NextValidReloc >= ValidRelocs.size())
    return false;

  uint64_t RelocOffset = ValidRelocs[NextValidReloc].Offset;

  // We might need to skip some relocs that we didn't consider. For
  // example the high_pc of a discarded DIE might contain a reloc that
  // is in the list because it actually corresponds to the start of a
  // function that is in the debug map.
  while (RelocOffset < StartOffset && NextValidReloc < ValidRelocs.size() - 1)
    RelocOffset = ValidRelocs[++NextValidReloc].Offset;

  if (RelocOffset < StartOffset || RelocOffset >= EndOffset)
    return false;

  const auto &ValidReloc = ValidRelocs[NextValidReloc++];
  if (Options.Verbose)
    outs() << "Found valid debug map entry: " << ValidReloc.Mapping->getKey()
           << " " << format("\t%016" PRIx64 " => %016" PRIx64,
                            ValidReloc.Mapping->getValue().ObjectAddress,
                            ValidReloc.Mapping->getValue().BinaryAddress);

  Info.AddrAdjust = int64_t(ValidReloc.Mapping->getValue().BinaryAddress) +
                    ValidReloc.Addend -
                    ValidReloc.Mapping->getValue().ObjectAddress;
  Info.InDebugMap = true;
  return true;
}

/// \brief Get the starting and ending (exclusive) offset for the
/// attribute with index \p Idx descibed by \p Abbrev. \p Offset is
/// supposed to point to the position of the first attribute described
/// by \p Abbrev.
/// \return [StartOffset, EndOffset) as a pair.
static std::pair<uint32_t, uint32_t>
getAttributeOffsets(const DWARFAbbreviationDeclaration *Abbrev, unsigned Idx,
                    unsigned Offset, const DWARFUnit &Unit) {
  DataExtractor Data = Unit.getDebugInfoExtractor();

  for (unsigned i = 0; i < Idx; ++i)
    DWARFFormValue::skipValue(Abbrev->getFormByIndex(i), Data, &Offset, &Unit);

  uint32_t End = Offset;
  DWARFFormValue::skipValue(Abbrev->getFormByIndex(Idx), Data, &End, &Unit);

  return std::make_pair(Offset, End);
}

/// \brief Check if a variable describing DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DwarfLinker::shouldKeepVariableDIE(
    const DWARFDebugInfoEntryMinimal &DIE, CompileUnit &Unit,
    CompileUnit::DIEInfo &MyInfo, unsigned Flags) {
  const auto *Abbrev = DIE.getAbbreviationDeclarationPtr();

  // Global variables with constant value can always be kept.
  if (!(Flags & TF_InFunctionScope) &&
      Abbrev->findAttributeIndex(dwarf::DW_AT_const_value) != -1U) {
    MyInfo.InDebugMap = true;
    return Flags | TF_Keep;
  }

  uint32_t LocationIdx = Abbrev->findAttributeIndex(dwarf::DW_AT_location);
  if (LocationIdx == -1U)
    return Flags;

  uint32_t Offset = DIE.getOffset() + getULEB128Size(Abbrev->getCode());
  const DWARFUnit &OrigUnit = Unit.getOrigUnit();
  uint32_t LocationOffset, LocationEndOffset;
  std::tie(LocationOffset, LocationEndOffset) =
      getAttributeOffsets(Abbrev, LocationIdx, Offset, OrigUnit);

  // See if there is a relocation to a valid debug map entry inside
  // this variable's location. The order is important here. We want to
  // always check in the variable has a valid relocation, so that the
  // DIEInfo is filled. However, we don't want a static variable in a
  // function to force us to keep the enclosing function.
  if (!hasValidRelocation(LocationOffset, LocationEndOffset, MyInfo) ||
      (Flags & TF_InFunctionScope))
    return Flags;

  if (Options.Verbose)
    DIE.dump(outs(), const_cast<DWARFUnit *>(&OrigUnit), 0, 8 /* Indent */);

  return Flags | TF_Keep;
}

/// \brief Check if a function describing DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DwarfLinker::shouldKeepSubprogramDIE(
    const DWARFDebugInfoEntryMinimal &DIE, CompileUnit &Unit,
    CompileUnit::DIEInfo &MyInfo, unsigned Flags) {
  const auto *Abbrev = DIE.getAbbreviationDeclarationPtr();

  Flags |= TF_InFunctionScope;

  uint32_t LowPcIdx = Abbrev->findAttributeIndex(dwarf::DW_AT_low_pc);
  if (LowPcIdx == -1U)
    return Flags;

  uint32_t Offset = DIE.getOffset() + getULEB128Size(Abbrev->getCode());
  const DWARFUnit &OrigUnit = Unit.getOrigUnit();
  uint32_t LowPcOffset, LowPcEndOffset;
  std::tie(LowPcOffset, LowPcEndOffset) =
      getAttributeOffsets(Abbrev, LowPcIdx, Offset, OrigUnit);

  uint64_t LowPc =
      DIE.getAttributeValueAsAddress(&OrigUnit, dwarf::DW_AT_low_pc, -1ULL);
  assert(LowPc != -1ULL && "low_pc attribute is not an address.");
  if (LowPc == -1ULL ||
      !hasValidRelocation(LowPcOffset, LowPcEndOffset, MyInfo))
    return Flags;

  if (Options.Verbose)
    DIE.dump(outs(), const_cast<DWARFUnit *>(&OrigUnit), 0, 8 /* Indent */);

  Flags |= TF_Keep;

  DWARFFormValue HighPcValue;
  if (!DIE.getAttributeValue(&OrigUnit, dwarf::DW_AT_high_pc, HighPcValue)) {
    reportWarning("Function without high_pc. Range will be discarded.\n",
                  &OrigUnit, &DIE);
    return Flags;
  }

  uint64_t HighPc;
  if (HighPcValue.isFormClass(DWARFFormValue::FC_Address)) {
    HighPc = *HighPcValue.getAsAddress(&OrigUnit);
  } else {
    assert(HighPcValue.isFormClass(DWARFFormValue::FC_Constant));
    HighPc = LowPc + *HighPcValue.getAsUnsignedConstant();
  }

  // Replace the debug map range with a more accurate one.
  Ranges[LowPc] = std::make_pair(HighPc, MyInfo.AddrAdjust);
  Unit.addFunctionRange(LowPc, HighPc, MyInfo.AddrAdjust);
  return Flags;
}

/// \brief Check if a DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DwarfLinker::shouldKeepDIE(const DWARFDebugInfoEntryMinimal &DIE,
                                    CompileUnit &Unit,
                                    CompileUnit::DIEInfo &MyInfo,
                                    unsigned Flags) {
  switch (DIE.getTag()) {
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_variable:
    return shouldKeepVariableDIE(DIE, Unit, MyInfo, Flags);
  case dwarf::DW_TAG_subprogram:
    return shouldKeepSubprogramDIE(DIE, Unit, MyInfo, Flags);
  case dwarf::DW_TAG_module:
  case dwarf::DW_TAG_imported_module:
  case dwarf::DW_TAG_imported_declaration:
  case dwarf::DW_TAG_imported_unit:
    // We always want to keep these.
    return Flags | TF_Keep;
  }

  return Flags;
}

/// \brief Mark the passed DIE as well as all the ones it depends on
/// as kept.
///
/// This function is called by lookForDIEsToKeep on DIEs that are
/// newly discovered to be needed in the link. It recursively calls
/// back to lookForDIEsToKeep while adding TF_DependencyWalk to the
/// TraversalFlags to inform it that it's not doing the primary DIE
/// tree walk.
void DwarfLinker::keepDIEAndDenpendencies(const DWARFDebugInfoEntryMinimal &DIE,
                                          CompileUnit::DIEInfo &MyInfo,
                                          const DebugMapObject &DMO,
                                          CompileUnit &CU, unsigned Flags) {
  const DWARFUnit &Unit = CU.getOrigUnit();
  MyInfo.Keep = true;

  // First mark all the parent chain as kept.
  unsigned AncestorIdx = MyInfo.ParentIdx;
  while (!CU.getInfo(AncestorIdx).Keep) {
    lookForDIEsToKeep(*Unit.getDIEAtIndex(AncestorIdx), DMO, CU,
                      TF_ParentWalk | TF_Keep | TF_DependencyWalk);
    AncestorIdx = CU.getInfo(AncestorIdx).ParentIdx;
  }

  // Then we need to mark all the DIEs referenced by this DIE's
  // attributes as kept.
  DataExtractor Data = Unit.getDebugInfoExtractor();
  const auto *Abbrev = DIE.getAbbreviationDeclarationPtr();
  uint32_t Offset = DIE.getOffset() + getULEB128Size(Abbrev->getCode());

  // Mark all DIEs referenced through atttributes as kept.
  for (const auto &AttrSpec : Abbrev->attributes()) {
    DWARFFormValue Val(AttrSpec.Form);

    if (!Val.isFormClass(DWARFFormValue::FC_Reference)) {
      DWARFFormValue::skipValue(AttrSpec.Form, Data, &Offset, &Unit);
      continue;
    }

    Val.extractValue(Data, &Offset, &Unit);
    CompileUnit *ReferencedCU;
    if (const auto *RefDIE = resolveDIEReference(Val, Unit, DIE, ReferencedCU))
      lookForDIEsToKeep(*RefDIE, DMO, *ReferencedCU,
                        TF_Keep | TF_DependencyWalk);
  }
}

/// \brief Recursively walk the \p DIE tree and look for DIEs to
/// keep. Store that information in \p CU's DIEInfo.
///
/// This function is the entry point of the DIE selection
/// algorithm. It is expected to walk the DIE tree in file order and
/// (though the mediation of its helper) call hasValidRelocation() on
/// each DIE that might be a 'root DIE' (See DwarfLinker class
/// comment).
/// While walking the dependencies of root DIEs, this function is
/// also called, but during these dependency walks the file order is
/// not respected. The TF_DependencyWalk flag tells us which kind of
/// traversal we are currently doing.
void DwarfLinker::lookForDIEsToKeep(const DWARFDebugInfoEntryMinimal &DIE,
                                    const DebugMapObject &DMO, CompileUnit &CU,
                                    unsigned Flags) {
  unsigned Idx = CU.getOrigUnit().getDIEIndex(&DIE);
  CompileUnit::DIEInfo &MyInfo = CU.getInfo(Idx);
  bool AlreadyKept = MyInfo.Keep;

  // If the Keep flag is set, we are marking a required DIE's
  // dependencies. If our target is already marked as kept, we're all
  // set.
  if ((Flags & TF_DependencyWalk) && AlreadyKept)
    return;

  // We must not call shouldKeepDIE while called from keepDIEAndDenpendencies,
  // because it would screw up the relocation finding logic.
  if (!(Flags & TF_DependencyWalk))
    Flags = shouldKeepDIE(DIE, CU, MyInfo, Flags);

  // If it is a newly kept DIE mark it as well as all its dependencies as kept.
  if (!AlreadyKept && (Flags & TF_Keep))
    keepDIEAndDenpendencies(DIE, MyInfo, DMO, CU, Flags);

  // The TF_ParentWalk flag tells us that we are currently walking up
  // the parent chain of a required DIE, and we don't want to mark all
  // the children of the parents as kept (consider for example a
  // DW_TAG_namespace node in the parent chain). There are however a
  // set of DIE types for which we want to ignore that directive and still
  // walk their children.
  if (dieNeedsChildrenToBeMeaningful(DIE.getTag()))
    Flags &= ~TF_ParentWalk;

  if (!DIE.hasChildren() || (Flags & TF_ParentWalk))
    return;

  for (auto *Child = DIE.getFirstChild(); Child && !Child->isNULL();
       Child = Child->getSibling())
    lookForDIEsToKeep(*Child, DMO, CU, Flags);
}

/// \brief Assign an abbreviation numer to \p Abbrev.
///
/// Our DIEs get freed after every DebugMapObject has been processed,
/// thus the FoldingSet we use to unique DIEAbbrevs cannot refer to
/// the instances hold by the DIEs. When we encounter an abbreviation
/// that we don't know, we create a permanent copy of it.
void DwarfLinker::AssignAbbrev(DIEAbbrev &Abbrev) {
  // Check the set for priors.
  FoldingSetNodeID ID;
  Abbrev.Profile(ID);
  void *InsertToken;
  DIEAbbrev *InSet = AbbreviationsSet.FindNodeOrInsertPos(ID, InsertToken);

  // If it's newly added.
  if (InSet) {
    // Assign existing abbreviation number.
    Abbrev.setNumber(InSet->getNumber());
  } else {
    // Add to abbreviation list.
    Abbreviations.push_back(
        new DIEAbbrev(Abbrev.getTag(), Abbrev.hasChildren()));
    for (const auto &Attr : Abbrev.getData())
      Abbreviations.back()->AddAttribute(Attr.getAttribute(), Attr.getForm());
    AbbreviationsSet.InsertNode(Abbreviations.back(), InsertToken);
    // Assign the unique abbreviation number.
    Abbrev.setNumber(Abbreviations.size());
    Abbreviations.back()->setNumber(Abbreviations.size());
  }
}

/// \brief Clone a string attribute described by \p AttrSpec and add
/// it to \p Die.
/// \returns the size of the new attribute.
unsigned DwarfLinker::cloneStringAttribute(DIE &Die, AttributeSpec AttrSpec,
                                           const DWARFFormValue &Val,
                                           const DWARFUnit &U) {
  // Switch everything to out of line strings.
  const char *String = *Val.getAsCString(&U);
  unsigned Offset = StringPool.getStringOffset(String);
  Die.addValue(dwarf::Attribute(AttrSpec.Attr), dwarf::DW_FORM_strp,
               new (DIEAlloc) DIEInteger(Offset));
  return 4;
}

/// \brief Clone an attribute referencing another DIE and add
/// it to \p Die.
/// \returns the size of the new attribute.
unsigned DwarfLinker::cloneDieReferenceAttribute(
    DIE &Die, const DWARFDebugInfoEntryMinimal &InputDIE,
    AttributeSpec AttrSpec, unsigned AttrSize, const DWARFFormValue &Val,
    CompileUnit &Unit) {
  uint32_t Ref = *Val.getAsReference(&Unit.getOrigUnit());
  DIE *NewRefDie = nullptr;
  CompileUnit *RefUnit = nullptr;
  const DWARFDebugInfoEntryMinimal *RefDie = nullptr;

  if (!(RefUnit = getUnitForOffset(Ref)) ||
      !(RefDie = RefUnit->getOrigUnit().getDIEForOffset(Ref))) {
    const char *AttributeString = dwarf::AttributeString(AttrSpec.Attr);
    if (!AttributeString)
      AttributeString = "DW_AT_???";
    reportWarning(Twine("Missing DIE for ref in attribute ") + AttributeString +
                      ". Dropping.",
                  &Unit.getOrigUnit(), &InputDIE);
    return 0;
  }

  unsigned Idx = RefUnit->getOrigUnit().getDIEIndex(RefDie);
  CompileUnit::DIEInfo &RefInfo = RefUnit->getInfo(Idx);
  if (!RefInfo.Clone) {
    assert(Ref > InputDIE.getOffset());
    // We haven't cloned this DIE yet. Just create an empty one and
    // store it. It'll get really cloned when we process it.
    RefInfo.Clone = new DIE(dwarf::Tag(RefDie->getTag()));
  }
  NewRefDie = RefInfo.Clone;

  if (AttrSpec.Form == dwarf::DW_FORM_ref_addr) {
    // We cannot currently rely on a DIEEntry to emit ref_addr
    // references, because the implementation calls back to DwarfDebug
    // to find the unit offset. (We don't have a DwarfDebug)
    // FIXME: we should be able to design DIEEntry reliance on
    // DwarfDebug away.
    DIEInteger *Attr;
    if (Ref < InputDIE.getOffset()) {
      // We must have already cloned that DIE.
      uint32_t NewRefOffset =
          RefUnit->getStartOffset() + NewRefDie->getOffset();
      Attr = new (DIEAlloc) DIEInteger(NewRefOffset);
    } else {
      // A forward reference. Note and fixup later.
      Attr = new (DIEAlloc) DIEInteger(0xBADDEF);
      Unit.noteForwardReference(NewRefDie, RefUnit, Attr);
    }
    Die.addValue(dwarf::Attribute(AttrSpec.Attr), dwarf::DW_FORM_ref_addr,
                 Attr);
    return AttrSize;
  }

  Die.addValue(dwarf::Attribute(AttrSpec.Attr), dwarf::Form(AttrSpec.Form),
               new (DIEAlloc) DIEEntry(*NewRefDie));
  return AttrSize;
}

/// \brief Clone an attribute of block form (locations, constants) and add
/// it to \p Die.
/// \returns the size of the new attribute.
unsigned DwarfLinker::cloneBlockAttribute(DIE &Die, AttributeSpec AttrSpec,
                                          const DWARFFormValue &Val,
                                          unsigned AttrSize) {
  DIE *Attr;
  DIEValue *Value;
  DIELoc *Loc = nullptr;
  DIEBlock *Block = nullptr;
  // Just copy the block data over.
  if (AttrSpec.Form == dwarf::DW_FORM_exprloc) {
    Loc = new (DIEAlloc) DIELoc();
    DIELocs.push_back(Loc);
  } else {
    Block = new (DIEAlloc) DIEBlock();
    DIEBlocks.push_back(Block);
  }
  Attr = Loc ? static_cast<DIE *>(Loc) : static_cast<DIE *>(Block);
  Value = Loc ? static_cast<DIEValue *>(Loc) : static_cast<DIEValue *>(Block);
  ArrayRef<uint8_t> Bytes = *Val.getAsBlock();
  for (auto Byte : Bytes)
    Attr->addValue(static_cast<dwarf::Attribute>(0), dwarf::DW_FORM_data1,
                   new (DIEAlloc) DIEInteger(Byte));
  // FIXME: If DIEBlock and DIELoc just reuses the Size field of
  // the DIE class, this if could be replaced by
  // Attr->setSize(Bytes.size()).
  if (Streamer) {
    if (Loc)
      Loc->ComputeSize(&Streamer->getAsmPrinter());
    else
      Block->ComputeSize(&Streamer->getAsmPrinter());
  }
  Die.addValue(dwarf::Attribute(AttrSpec.Attr), dwarf::Form(AttrSpec.Form),
               Value);
  return AttrSize;
}

/// \brief Clone an address attribute and add it to \p Die.
/// \returns the size of the new attribute.
unsigned DwarfLinker::cloneAddressAttribute(DIE &Die, AttributeSpec AttrSpec,
                                            const DWARFFormValue &Val,
                                            const CompileUnit &Unit,
                                            AttributesInfo &Info) {
  uint64_t Addr = *Val.getAsAddress(&Unit.getOrigUnit());
  if (AttrSpec.Attr == dwarf::DW_AT_low_pc) {
    if (Die.getTag() == dwarf::DW_TAG_inlined_subroutine ||
        Die.getTag() == dwarf::DW_TAG_lexical_block)
      Addr += Info.PCOffset;
    else if (Die.getTag() == dwarf::DW_TAG_compile_unit) {
      Addr = Unit.getLowPc();
      if (Addr == UINT64_MAX)
        return 0;
    }
    Info.HasLowPc = true;
  } else if (AttrSpec.Attr == dwarf::DW_AT_high_pc) {
    if (Die.getTag() == dwarf::DW_TAG_compile_unit) {
      if (uint64_t HighPc = Unit.getHighPc())
        Addr = HighPc;
      else
        return 0;
    } else
      // If we have a high_pc recorded for the input DIE, use
      // it. Otherwise (when no relocations where applied) just use the
      // one we just decoded.
      Addr = (Info.OrigHighPc ? Info.OrigHighPc : Addr) + Info.PCOffset;
  }

  Die.addValue(static_cast<dwarf::Attribute>(AttrSpec.Attr),
               static_cast<dwarf::Form>(AttrSpec.Form),
               new (DIEAlloc) DIEInteger(Addr));
  return Unit.getOrigUnit().getAddressByteSize();
}

/// \brief Clone a scalar attribute  and add it to \p Die.
/// \returns the size of the new attribute.
unsigned DwarfLinker::cloneScalarAttribute(
    DIE &Die, const DWARFDebugInfoEntryMinimal &InputDIE, CompileUnit &Unit,
    AttributeSpec AttrSpec, const DWARFFormValue &Val, unsigned AttrSize,
    AttributesInfo &Info) {
  uint64_t Value;
  if (AttrSpec.Attr == dwarf::DW_AT_high_pc &&
      Die.getTag() == dwarf::DW_TAG_compile_unit) {
    if (Unit.getLowPc() == -1ULL)
      return 0;
    // Dwarf >= 4 high_pc is an size, not an address.
    Value = Unit.getHighPc() - Unit.getLowPc();
  } else if (AttrSpec.Form == dwarf::DW_FORM_sec_offset)
    Value = *Val.getAsSectionOffset();
  else if (AttrSpec.Form == dwarf::DW_FORM_sdata)
    Value = *Val.getAsSignedConstant();
  else if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = *OptionalValue;
  else {
    reportWarning("Unsupported scalar attribute form. Dropping attribute.",
                  &Unit.getOrigUnit(), &InputDIE);
    return 0;
  }
  DIEInteger *Attr = new (DIEAlloc) DIEInteger(Value);
  if (AttrSpec.Attr == dwarf::DW_AT_ranges)
    Unit.noteRangeAttribute(Die, Attr);
  // A more generic way to check for location attributes would be
  // nice, but it's very unlikely that any other attribute needs a
  // location list.
  else if (AttrSpec.Attr == dwarf::DW_AT_location ||
           AttrSpec.Attr == dwarf::DW_AT_frame_base)
    Unit.noteLocationAttribute(Attr, Info.PCOffset);
  else if (AttrSpec.Attr == dwarf::DW_AT_declaration && Value)
    Info.IsDeclaration = true;

  Die.addValue(dwarf::Attribute(AttrSpec.Attr), dwarf::Form(AttrSpec.Form),
               Attr);
  return AttrSize;
}

/// \brief Clone \p InputDIE's attribute described by \p AttrSpec with
/// value \p Val, and add it to \p Die.
/// \returns the size of the cloned attribute.
unsigned DwarfLinker::cloneAttribute(DIE &Die,
                                     const DWARFDebugInfoEntryMinimal &InputDIE,
                                     CompileUnit &Unit,
                                     const DWARFFormValue &Val,
                                     const AttributeSpec AttrSpec,
                                     unsigned AttrSize, AttributesInfo &Info) {
  const DWARFUnit &U = Unit.getOrigUnit();

  switch (AttrSpec.Form) {
  case dwarf::DW_FORM_strp:
  case dwarf::DW_FORM_string:
    return cloneStringAttribute(Die, AttrSpec, Val, U);
  case dwarf::DW_FORM_ref_addr:
  case dwarf::DW_FORM_ref1:
  case dwarf::DW_FORM_ref2:
  case dwarf::DW_FORM_ref4:
  case dwarf::DW_FORM_ref8:
    return cloneDieReferenceAttribute(Die, InputDIE, AttrSpec, AttrSize, Val,
                                      Unit);
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_block1:
  case dwarf::DW_FORM_block2:
  case dwarf::DW_FORM_block4:
  case dwarf::DW_FORM_exprloc:
    return cloneBlockAttribute(Die, AttrSpec, Val, AttrSize);
  case dwarf::DW_FORM_addr:
    return cloneAddressAttribute(Die, AttrSpec, Val, Unit, Info);
  case dwarf::DW_FORM_data1:
  case dwarf::DW_FORM_data2:
  case dwarf::DW_FORM_data4:
  case dwarf::DW_FORM_data8:
  case dwarf::DW_FORM_udata:
  case dwarf::DW_FORM_sdata:
  case dwarf::DW_FORM_sec_offset:
  case dwarf::DW_FORM_flag:
  case dwarf::DW_FORM_flag_present:
    return cloneScalarAttribute(Die, InputDIE, Unit, AttrSpec, Val, AttrSize,
                                Info);
  default:
    reportWarning("Unsupported attribute form in cloneAttribute. Dropping.", &U,
                  &InputDIE);
  }

  return 0;
}

/// \brief Apply the valid relocations found by findValidRelocs() to
/// the buffer \p Data, taking into account that Data is at \p BaseOffset
/// in the debug_info section.
///
/// Like for findValidRelocs(), this function must be called with
/// monotonic \p BaseOffset values.
///
/// \returns wether any reloc has been applied.
bool DwarfLinker::applyValidRelocs(MutableArrayRef<char> Data,
                                   uint32_t BaseOffset, bool isLittleEndian) {
  assert((NextValidReloc == 0 ||
          BaseOffset > ValidRelocs[NextValidReloc - 1].Offset) &&
         "BaseOffset should only be increasing.");
  if (NextValidReloc >= ValidRelocs.size())
    return false;

  // Skip relocs that haven't been applied.
  while (NextValidReloc < ValidRelocs.size() &&
         ValidRelocs[NextValidReloc].Offset < BaseOffset)
    ++NextValidReloc;

  bool Applied = false;
  uint64_t EndOffset = BaseOffset + Data.size();
  while (NextValidReloc < ValidRelocs.size() &&
         ValidRelocs[NextValidReloc].Offset >= BaseOffset &&
         ValidRelocs[NextValidReloc].Offset < EndOffset) {
    const auto &ValidReloc = ValidRelocs[NextValidReloc++];
    assert(ValidReloc.Offset - BaseOffset < Data.size());
    assert(ValidReloc.Offset - BaseOffset + ValidReloc.Size <= Data.size());
    char Buf[8];
    uint64_t Value = ValidReloc.Mapping->getValue().BinaryAddress;
    Value += ValidReloc.Addend;
    for (unsigned i = 0; i != ValidReloc.Size; ++i) {
      unsigned Index = isLittleEndian ? i : (ValidReloc.Size - i - 1);
      Buf[i] = uint8_t(Value >> (Index * 8));
    }
    assert(ValidReloc.Size <= sizeof(Buf));
    memcpy(&Data[ValidReloc.Offset - BaseOffset], Buf, ValidReloc.Size);
    Applied = true;
  }

  return Applied;
}

static bool isTypeTag(uint16_t Tag) {
  switch (Tag) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_pointer_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_string_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_ptr_to_member_type:
  case dwarf::DW_TAG_set_type:
  case dwarf::DW_TAG_subrange_type:
  case dwarf::DW_TAG_base_type:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_file_type:
  case dwarf::DW_TAG_namelist:
  case dwarf::DW_TAG_packed_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_interface_type:
  case dwarf::DW_TAG_unspecified_type:
  case dwarf::DW_TAG_shared_type:
    return true;
  default:
    break;
  }
  return false;
}

/// \brief Recursively clone \p InputDIE's subtrees that have been
/// selected to appear in the linked output.
///
/// \param OutOffset is the Offset where the newly created DIE will
/// lie in the linked compile unit.
///
/// \returns the cloned DIE object or null if nothing was selected.
DIE *DwarfLinker::cloneDIE(const DWARFDebugInfoEntryMinimal &InputDIE,
                           CompileUnit &Unit, int64_t PCOffset,
                           uint32_t OutOffset) {
  DWARFUnit &U = Unit.getOrigUnit();
  unsigned Idx = U.getDIEIndex(&InputDIE);
  CompileUnit::DIEInfo &Info = Unit.getInfo(Idx);

  // Should the DIE appear in the output?
  if (!Unit.getInfo(Idx).Keep)
    return nullptr;

  uint32_t Offset = InputDIE.getOffset();
  // The DIE might have been already created by a forward reference
  // (see cloneDieReferenceAttribute()).
  DIE *Die = Info.Clone;
  if (!Die)
    Die = Info.Clone = new DIE(dwarf::Tag(InputDIE.getTag()));
  assert(Die->getTag() == InputDIE.getTag());
  Die->setOffset(OutOffset);

  // Extract and clone every attribute.
  DataExtractor Data = U.getDebugInfoExtractor();
  uint32_t NextOffset = U.getDIEAtIndex(Idx + 1)->getOffset();
  AttributesInfo AttrInfo;

  // We could copy the data only if we need to aply a relocation to
  // it. After testing, it seems there is no performance downside to
  // doing the copy unconditionally, and it makes the code simpler.
  SmallString<40> DIECopy(Data.getData().substr(Offset, NextOffset - Offset));
  Data = DataExtractor(DIECopy, Data.isLittleEndian(), Data.getAddressSize());
  // Modify the copy with relocated addresses.
  if (applyValidRelocs(DIECopy, Offset, Data.isLittleEndian())) {
    // If we applied relocations, we store the value of high_pc that was
    // potentially stored in the input DIE. If high_pc is an address
    // (Dwarf version == 2), then it might have been relocated to a
    // totally unrelated value (because the end address in the object
    // file might be start address of another function which got moved
    // independantly by the linker). The computation of the actual
    // high_pc value is done in cloneAddressAttribute().
    AttrInfo.OrigHighPc =
        InputDIE.getAttributeValueAsAddress(&U, dwarf::DW_AT_high_pc, 0);
  }

  // Reset the Offset to 0 as we will be working on the local copy of
  // the data.
  Offset = 0;

  const auto *Abbrev = InputDIE.getAbbreviationDeclarationPtr();
  Offset += getULEB128Size(Abbrev->getCode());

  // We are entering a subprogram. Get and propagate the PCOffset.
  if (Die->getTag() == dwarf::DW_TAG_subprogram)
    PCOffset = Info.AddrAdjust;
  AttrInfo.PCOffset = PCOffset;

  for (const auto &AttrSpec : Abbrev->attributes()) {
    DWARFFormValue Val(AttrSpec.Form);
    uint32_t AttrSize = Offset;
    Val.extractValue(Data, &Offset, &U);
    AttrSize = Offset - AttrSize;

    OutOffset +=
        cloneAttribute(*Die, InputDIE, Unit, Val, AttrSpec, AttrSize, AttrInfo);
  }

  // Look for accelerator entries.
  uint16_t Tag = InputDIE.getTag();
  // FIXME: This is slightly wrong. An inline_subroutine without a
  // low_pc, but with AT_ranges might be interesting to get into the
  // accelerator tables too. For now stick with dsymutil's behavior.
  if ((Info.InDebugMap || AttrInfo.HasLowPc) &&
      Tag != dwarf::DW_TAG_compile_unit &&
      getDIENames(InputDIE, Unit.getOrigUnit(), AttrInfo)) {
    if (AttrInfo.MangledName && AttrInfo.MangledName != AttrInfo.Name)
      Unit.addNameAccelerator(Die, AttrInfo.MangledName,
                              AttrInfo.MangledNameOffset,
                              Tag == dwarf::DW_TAG_inlined_subroutine);
    if (AttrInfo.Name)
      Unit.addNameAccelerator(Die, AttrInfo.Name, AttrInfo.NameOffset,
                              Tag == dwarf::DW_TAG_inlined_subroutine);
  } else if (isTypeTag(Tag) && !AttrInfo.IsDeclaration &&
             getDIENames(InputDIE, Unit.getOrigUnit(), AttrInfo)) {
    Unit.addTypeAccelerator(Die, AttrInfo.Name, AttrInfo.NameOffset);
  }

  DIEAbbrev &NewAbbrev = Die->getAbbrev();
  // If a scope DIE is kept, we must have kept at least one child. If
  // it's not the case, we'll just be emitting one wasteful end of
  // children marker, but things won't break.
  if (InputDIE.hasChildren())
    NewAbbrev.setChildrenFlag(dwarf::DW_CHILDREN_yes);
  // Assign a permanent abbrev number
  AssignAbbrev(Die->getAbbrev());

  // Add the size of the abbreviation number to the output offset.
  OutOffset += getULEB128Size(Die->getAbbrevNumber());

  if (!Abbrev->hasChildren()) {
    // Update our size.
    Die->setSize(OutOffset - Die->getOffset());
    return Die;
  }

  // Recursively clone children.
  for (auto *Child = InputDIE.getFirstChild(); Child && !Child->isNULL();
       Child = Child->getSibling()) {
    if (DIE *Clone = cloneDIE(*Child, Unit, PCOffset, OutOffset)) {
      Die->addChild(std::unique_ptr<DIE>(Clone));
      OutOffset = Clone->getOffset() + Clone->getSize();
    }
  }

  // Account for the end of children marker.
  OutOffset += sizeof(int8_t);
  // Update our size.
  Die->setSize(OutOffset - Die->getOffset());
  return Die;
}

/// \brief Patch the input object file relevant debug_ranges entries
/// and emit them in the output file. Update the relevant attributes
/// to point at the new entries.
void DwarfLinker::patchRangesForUnit(const CompileUnit &Unit,
                                     DWARFContext &OrigDwarf) const {
  DWARFDebugRangeList RangeList;
  const auto &FunctionRanges = Unit.getFunctionRanges();
  unsigned AddressSize = Unit.getOrigUnit().getAddressByteSize();
  DataExtractor RangeExtractor(OrigDwarf.getRangeSection(),
                               OrigDwarf.isLittleEndian(), AddressSize);
  auto InvalidRange = FunctionRanges.end(), CurrRange = InvalidRange;
  DWARFUnit &OrigUnit = Unit.getOrigUnit();
  const auto *OrigUnitDie = OrigUnit.getCompileUnitDIE(false);
  uint64_t OrigLowPc = OrigUnitDie->getAttributeValueAsAddress(
      &OrigUnit, dwarf::DW_AT_low_pc, -1ULL);
  // Ranges addresses are based on the unit's low_pc. Compute the
  // offset we need to apply to adapt to the the new unit's low_pc.
  int64_t UnitPcOffset = 0;
  if (OrigLowPc != -1ULL)
    UnitPcOffset = int64_t(OrigLowPc) - Unit.getLowPc();

  for (const auto &RangeAttribute : Unit.getRangesAttributes()) {
    uint32_t Offset = RangeAttribute->getValue();
    RangeAttribute->setValue(Streamer->getRangesSectionSize());
    RangeList.extract(RangeExtractor, &Offset);
    const auto &Entries = RangeList.getEntries();
    const DWARFDebugRangeList::RangeListEntry &First = Entries.front();

    if (CurrRange == InvalidRange || First.StartAddress < CurrRange.start() ||
        First.StartAddress >= CurrRange.stop()) {
      CurrRange = FunctionRanges.find(First.StartAddress + OrigLowPc);
      if (CurrRange == InvalidRange ||
          CurrRange.start() > First.StartAddress + OrigLowPc) {
        reportWarning("no mapping for range.");
        continue;
      }
    }

    Streamer->emitRangesEntries(UnitPcOffset, OrigLowPc, CurrRange, Entries,
                                AddressSize);
  }
}

/// \brief Generate the debug_aranges entries for \p Unit and if the
/// unit has a DW_AT_ranges attribute, also emit the debug_ranges
/// contribution for this attribute.
/// FIXME: this could actually be done right in patchRangesForUnit,
/// but for the sake of initial bit-for-bit compatibility with legacy
/// dsymutil, we have to do it in a delayed pass.
void DwarfLinker::generateUnitRanges(CompileUnit &Unit) const {
  DIEInteger *Attr = Unit.getUnitRangesAttribute();
  if (Attr)
    Attr->setValue(Streamer->getRangesSectionSize());
  Streamer->emitUnitRangesEntries(Unit, Attr != nullptr);
}

/// \brief Insert the new line info sequence \p Seq into the current
/// set of already linked line info \p Rows.
static void insertLineSequence(std::vector<DWARFDebugLine::Row> &Seq,
                               std::vector<DWARFDebugLine::Row> &Rows) {
  if (Seq.empty())
    return;

  if (!Rows.empty() && Rows.back().Address < Seq.front().Address) {
    Rows.insert(Rows.end(), Seq.begin(), Seq.end());
    Seq.clear();
    return;
  }

  auto InsertPoint = std::lower_bound(
      Rows.begin(), Rows.end(), Seq.front(),
      [](const DWARFDebugLine::Row &LHS, const DWARFDebugLine::Row &RHS) {
        return LHS.Address < RHS.Address;
      });

  // FIXME: this only removes the unneeded end_sequence if the
  // sequences have been inserted in order. using a global sort like
  // described in patchLineTableForUnit() and delaying the end_sequene
  // elimination to emitLineTableForUnit() we can get rid of all of them.
  if (InsertPoint != Rows.end() &&
      InsertPoint->Address == Seq.front().Address && InsertPoint->EndSequence) {
    *InsertPoint = Seq.front();
    Rows.insert(InsertPoint + 1, Seq.begin() + 1, Seq.end());
  } else {
    Rows.insert(InsertPoint, Seq.begin(), Seq.end());
  }

  Seq.clear();
}

/// \brief Extract the line table for \p Unit from \p OrigDwarf, and
/// recreate a relocated version of these for the address ranges that
/// are present in the binary.
void DwarfLinker::patchLineTableForUnit(CompileUnit &Unit,
                                        DWARFContext &OrigDwarf) {
  const DWARFDebugInfoEntryMinimal *CUDie =
      Unit.getOrigUnit().getCompileUnitDIE();
  uint64_t StmtList = CUDie->getAttributeValueAsSectionOffset(
      &Unit.getOrigUnit(), dwarf::DW_AT_stmt_list, -1ULL);
  if (StmtList == -1ULL)
    return;

  // Update the cloned DW_AT_stmt_list with the correct debug_line offset.
  if (auto *OutputDIE = Unit.getOutputUnitDIE()) {
    const auto &Abbrev = OutputDIE->getAbbrev().getData();
    auto Stmt = std::find_if(
        Abbrev.begin(), Abbrev.end(), [](const DIEAbbrevData &AbbrevData) {
          return AbbrevData.getAttribute() == dwarf::DW_AT_stmt_list;
        });
    assert(Stmt < Abbrev.end() && "Didn't find DW_AT_stmt_list in cloned DIE!");
    DIEInteger *StmtAttr =
        cast<DIEInteger>(OutputDIE->getValues()[Stmt - Abbrev.begin()]);
    StmtAttr->setValue(Streamer->getLineSectionSize());
  }

  // Parse the original line info for the unit.
  DWARFDebugLine::LineTable LineTable;
  uint32_t StmtOffset = StmtList;
  StringRef LineData = OrigDwarf.getLineSection().Data;
  DataExtractor LineExtractor(LineData, OrigDwarf.isLittleEndian(),
                              Unit.getOrigUnit().getAddressByteSize());
  LineTable.parse(LineExtractor, &OrigDwarf.getLineSection().Relocs,
                  &StmtOffset);

  // This vector is the output line table.
  std::vector<DWARFDebugLine::Row> NewRows;
  NewRows.reserve(LineTable.Rows.size());

  // Current sequence of rows being extracted, before being inserted
  // in NewRows.
  std::vector<DWARFDebugLine::Row> Seq;
  const auto &FunctionRanges = Unit.getFunctionRanges();
  auto InvalidRange = FunctionRanges.end(), CurrRange = InvalidRange;

  // FIXME: This logic is meant to generate exactly the same output as
  // Darwin's classic dsynutil. There is a nicer way to implement this
  // by simply putting all the relocated line info in NewRows and simply
  // sorting NewRows before passing it to emitLineTableForUnit. This
  // should be correct as sequences for a function should stay
  // together in the sorted output. There are a few corner cases that
  // look suspicious though, and that required to implement the logic
  // this way. Revisit that once initial validation is finished.

  // Iterate over the object file line info and extract the sequences
  // that correspond to linked functions.
  for (auto &Row : LineTable.Rows) {
    // Check wether we stepped out of the range. The range is
    // half-open, but consider accept the end address of the range if
    // it is marked as end_sequence in the input (because in that
    // case, the relocation offset is accurate and that entry won't
    // serve as the start of another function).
    if (CurrRange == InvalidRange || Row.Address < CurrRange.start() ||
        Row.Address > CurrRange.stop() ||
        (Row.Address == CurrRange.stop() && !Row.EndSequence)) {
      // We just stepped out of a known range. Insert a end_sequence
      // corresponding to the end of the range.
      uint64_t StopAddress = CurrRange != InvalidRange
                                 ? CurrRange.stop() + CurrRange.value()
                                 : -1ULL;
      CurrRange = FunctionRanges.find(Row.Address);
      bool CurrRangeValid =
          CurrRange != InvalidRange && CurrRange.start() <= Row.Address;
      if (!CurrRangeValid) {
        CurrRange = InvalidRange;
        if (StopAddress != -1ULL) {
          // Try harder by looking in the DebugMapObject function
          // ranges map. There are corner cases where this finds a
          // valid entry. It's unclear if this is right or wrong, but
          // for now do as dsymutil.
          // FIXME: Understand exactly what cases this addresses and
          // potentially remove it along with the Ranges map.
          auto Range = Ranges.lower_bound(Row.Address);
          if (Range != Ranges.begin() && Range != Ranges.end())
            --Range;

          if (Range != Ranges.end() && Range->first <= Row.Address &&
              Range->second.first >= Row.Address) {
            StopAddress = Row.Address + Range->second.second;
          }
        }
      }
      if (StopAddress != -1ULL && !Seq.empty()) {
        // Insert end sequence row with the computed end address, but
        // the same line as the previous one.
        Seq.emplace_back(Seq.back());
        Seq.back().Address = StopAddress;
        Seq.back().EndSequence = 1;
        Seq.back().PrologueEnd = 0;
        Seq.back().BasicBlock = 0;
        Seq.back().EpilogueBegin = 0;
        insertLineSequence(Seq, NewRows);
      }

      if (!CurrRangeValid)
        continue;
    }

    // Ignore empty sequences.
    if (Row.EndSequence && Seq.empty())
      continue;

    // Relocate row address and add it to the current sequence.
    Row.Address += CurrRange.value();
    Seq.emplace_back(Row);

    if (Row.EndSequence)
      insertLineSequence(Seq, NewRows);
  }

  // Finished extracting, now emit the line tables.
  uint32_t PrologueEnd = StmtList + 10 + LineTable.Prologue.PrologueLength;
  // FIXME: LLVM hardcodes it's prologue values. We just copy the
  // prologue over and that works because we act as both producer and
  // consumer. It would be nicer to have a real configurable line
  // table emitter.
  if (LineTable.Prologue.Version != 2 ||
      LineTable.Prologue.DefaultIsStmt != DWARF2_LINE_DEFAULT_IS_STMT ||
      LineTable.Prologue.LineBase != -5 || LineTable.Prologue.LineRange != 14 ||
      LineTable.Prologue.OpcodeBase != 13)
    reportWarning("line table paramters mismatch. Cannot emit.");
  else
    Streamer->emitLineTableForUnit(LineData.slice(StmtList + 4, PrologueEnd),
                                   LineTable.Prologue.MinInstLength, NewRows,
                                   Unit.getOrigUnit().getAddressByteSize());
}

void DwarfLinker::emitAcceleratorEntriesForUnit(CompileUnit &Unit) {
  Streamer->emitPubNamesForUnit(Unit);
  Streamer->emitPubTypesForUnit(Unit);
}

bool DwarfLinker::link(const DebugMap &Map) {

  if (Map.begin() == Map.end()) {
    errs() << "Empty debug map.\n";
    return false;
  }

  if (!createStreamer(Map.getTriple(), OutputFilename))
    return false;

  // Size of the DIEs (and headers) generated for the linked output.
  uint64_t OutputDebugInfoSize = 0;
  // A unique ID that identifies each compile unit.
  unsigned UnitID = 0;
  for (const auto &Obj : Map.objects()) {
    CurrentDebugObject = Obj.get();

    if (Options.Verbose)
      outs() << "DEBUG MAP OBJECT: " << Obj->getObjectFilename() << "\n";
    auto ErrOrObj = BinHolder.GetObjectFile(Obj->getObjectFilename());
    if (std::error_code EC = ErrOrObj.getError()) {
      reportWarning(Twine(Obj->getObjectFilename()) + ": " + EC.message());
      continue;
    }

    // Look for relocations that correspond to debug map entries.
    if (!findValidRelocsInDebugInfo(*ErrOrObj, *Obj)) {
      if (Options.Verbose)
        outs() << "No valid relocations found. Skipping.\n";
      continue;
    }

    // Setup access to the debug info.
    DWARFContextInMemory DwarfContext(*ErrOrObj);
    startDebugObject(DwarfContext, *Obj);

    // In a first phase, just read in the debug info and store the DIE
    // parent links that we will use during the next phase.
    for (const auto &CU : DwarfContext.compile_units()) {
      auto *CUDie = CU->getCompileUnitDIE(false);
      if (Options.Verbose) {
        outs() << "Input compilation unit:";
        CUDie->dump(outs(), CU.get(), 0);
      }
      Units.emplace_back(*CU, UnitID++);
      gatherDIEParents(CUDie, 0, Units.back());
    }

    // Then mark all the DIEs that need to be present in the linked
    // output and collect some information about them. Note that this
    // loop can not be merged with the previous one becaue cross-cu
    // references require the ParentIdx to be setup for every CU in
    // the object file before calling this.
    for (auto &CurrentUnit : Units)
      lookForDIEsToKeep(*CurrentUnit.getOrigUnit().getCompileUnitDIE(), *Obj,
                        CurrentUnit, 0);

    // The calls to applyValidRelocs inside cloneDIE will walk the
    // reloc array again (in the same way findValidRelocsInDebugInfo()
    // did). We need to reset the NextValidReloc index to the beginning.
    NextValidReloc = 0;

    // Construct the output DIE tree by cloning the DIEs we chose to
    // keep above. If there are no valid relocs, then there's nothing
    // to clone/emit.
    if (!ValidRelocs.empty())
      for (auto &CurrentUnit : Units) {
        const auto *InputDIE = CurrentUnit.getOrigUnit().getCompileUnitDIE();
        CurrentUnit.setStartOffset(OutputDebugInfoSize);
        DIE *OutputDIE = cloneDIE(*InputDIE, CurrentUnit, 0 /* PCOffset */,
                                  11 /* Unit Header size */);
        CurrentUnit.setOutputUnitDIE(OutputDIE);
        OutputDebugInfoSize = CurrentUnit.computeNextUnitOffset();
        if (Options.NoOutput)
          continue;
        // FIXME: for compatibility with the classic dsymutil, we emit
        // an empty line table for the unit, even if the unit doesn't
        // actually exist in the DIE tree.
        patchLineTableForUnit(CurrentUnit, DwarfContext);
        if (!OutputDIE)
          continue;
        patchRangesForUnit(CurrentUnit, DwarfContext);
        Streamer->emitLocationsForUnit(CurrentUnit, DwarfContext);
        emitAcceleratorEntriesForUnit(CurrentUnit);
      }

    // Emit all the compile unit's debug information.
    if (!ValidRelocs.empty() && !Options.NoOutput)
      for (auto &CurrentUnit : Units) {
        generateUnitRanges(CurrentUnit);
        CurrentUnit.fixupForwardReferences();
        Streamer->emitCompileUnitHeader(CurrentUnit);
        if (!CurrentUnit.getOutputUnitDIE())
          continue;
        Streamer->emitDIE(*CurrentUnit.getOutputUnitDIE());
      }

    // Clean-up before starting working on the next object.
    endDebugObject();
  }

  // Emit everything that's global.
  if (!Options.NoOutput) {
    Streamer->emitAbbrevs(Abbreviations);
    Streamer->emitStrings(StringPool);
  }

  return Options.NoOutput ? true : Streamer->finish();
}
}

bool linkDwarf(StringRef OutputFilename, const DebugMap &DM,
               const LinkOptions &Options) {
  DwarfLinker Linker(OutputFilename, Options);
  return Linker.link(DM);
}
}
}
