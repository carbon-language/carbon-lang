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

  CompileUnit(DWARFUnit &OrigUnit) : OrigUnit(OrigUnit) {
    Info.resize(OrigUnit.getNumDIEs());
  }

  // Workaround MSVC not supporting implicit move ops
  CompileUnit(CompileUnit &&RHS)
      : OrigUnit(RHS.OrigUnit), Info(std::move(RHS.Info)),
        CUDie(std::move(RHS.CUDie)), StartOffset(RHS.StartOffset),
        NextUnitOffset(RHS.NextUnitOffset) {}

  DWARFUnit &getOrigUnit() const { return OrigUnit; }

  DIE *getOutputUnitDIE() const { return CUDie.get(); }
  void setOutputUnitDIE(DIE *Die) { CUDie.reset(Die); }

  DIEInfo &getInfo(unsigned Idx) { return Info[Idx]; }
  const DIEInfo &getInfo(unsigned Idx) const { return Info[Idx]; }

  uint64_t getStartOffset() const { return StartOffset; }
  uint64_t getNextUnitOffset() const { return NextUnitOffset; }

  void setStartOffset(uint64_t DebugInfoSize) { StartOffset = DebugInfoSize; }

  /// \brief Compute the end offset for this unit. Must be
  /// called after the CU's DIEs have been cloned.
  /// \returns the next unit offset (which is also the current
  /// debug_info section size).
  uint64_t computeNextUnitOffset();

  /// \brief Keep track of a forward reference to DIE \p Die by
  /// \p Attr. The attribute should be fixed up later to point to the
  /// absolute offset of \p Die in the debug_info section.
  void noteForwardReference(DIE *Die, DIEInteger *Attr);

  /// \brief Apply all fixups recored by noteForwardReference().
  void fixupForwardReferences();

private:
  DWARFUnit &OrigUnit;
  std::vector<DIEInfo> Info;  ///< DIE info indexed by DIE index.
  std::unique_ptr<DIE> CUDie; ///< Root of the linked DIE tree.

  uint64_t StartOffset;
  uint64_t NextUnitOffset;

  /// \brief A list of attributes to fixup with the absolute offset of
  /// a DIE in the debug_info section.
  ///
  /// The offsets for the attributes in this array couldn't be set while
  /// cloning because for forward refences the target DIE's offset isn't
  /// known you emit the reference attribute.
  std::vector<std::pair<DIE *, DIEInteger *>> ForwardDIEReferences;
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

/// \brief Keep track of a forward reference to \p Die.
void CompileUnit::noteForwardReference(DIE *Die, DIEInteger *Attr) {
  ForwardDIEReferences.emplace_back(Die, Attr);
}

/// \brief Apply all fixups recorded by noteForwardReference().
void CompileUnit::fixupForwardReferences() {
  for (const auto &Ref : ForwardDIEReferences)
    Ref.second->setValue(Ref.first->getOffset() + getStartOffset());
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
    uint64_t OrigHighPc; ///< Value of AT_high_pc in the input DIE
    int64_t PCOffset;    ///< Offset to apply to PC addresses inside a function.

    AttributesInfo() : OrigHighPc(0), PCOffset(0) {}
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
                             const DWARFFormValue &Val, const DWARFUnit &U);

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
                                const DWARFUnit &U, AttributeSpec AttrSpec,
                                const DWARFFormValue &Val, unsigned AttrSize);

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

  void reportWarning(const Twine &Warning, const DWARFUnit *Unit = nullptr,
                     const DWARFDebugInfoEntryMinimal *DIE = nullptr);

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

/// \brief Report a warning to the user, optionaly including
/// information about a specific \p DIE related to the warning.
void DwarfLinker::reportWarning(const Twine &Warning, const DWARFUnit *Unit,
                                const DWARFDebugInfoEntryMinimal *DIE) {
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

void DwarfLinker::startDebugObject(DWARFContext &Dwarf) {
  Units.reserve(Dwarf.getNumCompileUnits());
  NextValidReloc = 0;
}

void DwarfLinker::endDebugObject() {
  Units.clear();
  ValidRelocs.clear();

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

  return Flags | TF_Keep;
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
    const DWARFUnit &U) {
  uint32_t Ref = *Val.getAsReference(&U);
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
                  &U, &InputDIE);
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
      RefUnit->noteForwardReference(NewRefDie, Attr);
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
  if (AttrSpec.Attr == dwarf::DW_FORM_exprloc) {
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
  int64_t Addr = *Val.getAsAddress(&Unit.getOrigUnit());
  if (AttrSpec.Attr == dwarf::DW_AT_low_pc) {
    if (Die.getTag() == dwarf::DW_TAG_inlined_subroutine ||
        Die.getTag() == dwarf::DW_TAG_lexical_block)
      Addr += Info.PCOffset;
  } else if (AttrSpec.Attr == dwarf::DW_AT_high_pc) {
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
    DIE &Die, const DWARFDebugInfoEntryMinimal &InputDIE, const DWARFUnit &U,
    AttributeSpec AttrSpec, const DWARFFormValue &Val, unsigned AttrSize) {
  uint64_t Value;
  if (AttrSpec.Form == dwarf::DW_FORM_sec_offset)
    Value = *Val.getAsSectionOffset();
  else if (AttrSpec.Form == dwarf::DW_FORM_sdata)
    Value = *Val.getAsSignedConstant();
  else if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = *OptionalValue;
  else {
    reportWarning("Unsupported scalar attribute form. Dropping attribute.", &U,
                  &InputDIE);
    return 0;
  }
  Die.addValue(dwarf::Attribute(AttrSpec.Attr), dwarf::Form(AttrSpec.Form),
               new (DIEAlloc) DIEInteger(Value));
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
                                      U);
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
    return cloneScalarAttribute(Die, InputDIE, U, AttrSpec, Val, AttrSize);
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

bool DwarfLinker::link(const DebugMap &Map) {

  if (Map.begin() == Map.end()) {
    errs() << "Empty debug map.\n";
    return false;
  }

  if (!createStreamer(Map.getTriple(), OutputFilename))
    return false;

  // Size of the DIEs (and headers) generated for the linked output.
  uint64_t OutputDebugInfoSize = 0;

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
    startDebugObject(DwarfContext);

    // In a first phase, just read in the debug info and store the DIE
    // parent links that we will use during the next phase.
    for (const auto &CU : DwarfContext.compile_units()) {
      auto *CUDie = CU->getCompileUnitDIE(false);
      if (Options.Verbose) {
        outs() << "Input compilation unit:";
        CUDie->dump(outs(), CU.get(), 0);
      }
      Units.emplace_back(*CU);
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
      }

    // Emit all the compile unit's debug information.
    if (!ValidRelocs.empty() && !Options.NoOutput)
      for (auto &CurrentUnit : Units) {
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
