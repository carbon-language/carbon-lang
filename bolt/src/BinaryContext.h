//===--- BinaryContext.h  - Interface for machine-level context -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Context for processing binary executables in files and/or memory.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_CONTEXT_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_CONTEXT_H

#include "BinaryData.h"
#include "BinarySection.h"
#include "DebugData.h"
#include "JumpTable.h"
#include "MCPlusBuilder.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include <functional>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace llvm {

class DWARFDebugInfoEntryMinimal;

using namespace object;

namespace bolt {

class BinaryFunction;
class BinaryBasicBlock;
class DataReader;

/// Helper function to truncate a \p Value to given size in \p Bytes.
inline int64_t truncateToSize(int64_t Value, unsigned Bytes) {
  return Value & ((uint64_t) (int64_t) -1 >> (64 - Bytes * 8));
}

/// Filter iterator.
template <typename ItrType,
          typename PredType = std::function<bool (const ItrType &)>>
class FilterIterator
  : public std::iterator<std::bidirectional_iterator_tag,
                         typename std::iterator_traits<ItrType>::value_type> {
  using Iterator = FilterIterator;
  using T = typename std::iterator_traits<ItrType>::reference;
  using PointerT = typename std::iterator_traits<ItrType>::pointer;

  PredType Pred;
  ItrType Itr, End;

  void prev() {
    while (!Pred(--Itr))
      ;
  }
  void next() {
    ++Itr;
    nextMatching();
  }
  void nextMatching() {
    while (Itr != End && !Pred(Itr))
      ++Itr;
  }
public:
  Iterator &operator++() { next(); return *this; }
  Iterator &operator--() { prev(); return *this; }
  Iterator operator++(int) { auto Tmp(Itr); next(); return Tmp; }
  Iterator operator--(int) { auto Tmp(Itr); prev(); return Tmp; }
  bool operator==(const Iterator& Other) const {
    return Itr == Other.Itr;
  }
  bool operator!=(const Iterator& Other) const {
    return !operator==(Other);
  }
  T operator*() { return *Itr; }
  PointerT operator->() { return &operator*(); }
  FilterIterator(PredType Pred, ItrType Itr, ItrType End)
    : Pred(Pred), Itr(Itr), End(End) {
    nextMatching();
  }
};

class BinaryContext {
  BinaryContext() = delete;

  /// Set of all sections.
  struct CompareSections {
    bool operator()(const BinarySection *A, const BinarySection *B) const {
      return *A < *B;
    }
  };
  using SectionSetType = std::set<BinarySection *, CompareSections>;
  SectionSetType Sections;

  using SectionIterator = pointee_iterator<SectionSetType::iterator>;
  using SectionConstIterator = pointee_iterator<SectionSetType::const_iterator>;

  using FilteredSectionIterator = FilterIterator<SectionIterator>;
  using FilteredSectionConstIterator = FilterIterator<SectionConstIterator>;

  /// Map virtual address to a section.  It is possible to have more than one
  /// section mapped to the same address, e.g. non-allocatable sections.
  using AddressToSectionMapType = std::multimap<uint64_t, BinarySection *>;
  AddressToSectionMapType AddressToSection;

  /// multimap of section name to BinarySection object.  Some binaries
  /// have multiple sections with the same name.
  using NameToSectionMapType = std::multimap<std::string, BinarySection *>;
  NameToSectionMapType NameToSection;

  /// Low level section registration.
  BinarySection &registerSection(BinarySection *Section);

  /// Store all functions in the binary, sorted by original address.
  std::map<uint64_t, BinaryFunction> BinaryFunctions;

  /// Functions injected by BOLT
  std::vector<BinaryFunction *> InjectedBinaryFunctions;

  /// Jump tables for all functions mapped by address.
  std::map<uint64_t, JumpTable *> JumpTables;

public:
  /// [name] -> [BinaryData*] map used for global symbol resolution.
  using SymbolMapType = std::map<std::string, BinaryData *>;
  SymbolMapType GlobalSymbols;

  /// [address] -> [BinaryData], ...
  /// Addresses never change.
  /// Note: it is important that clients do not hold on to instances of
  /// BinaryData* while the map is still being modified during BinaryFunction
  /// disassembly.  This is because of the possibility that a regular
  /// BinaryData is later discovered to be a JumpTable.
  using BinaryDataMapType = std::map<uint64_t, BinaryData *>;
  using binary_data_iterator = BinaryDataMapType::iterator;
  using binary_data_const_iterator = BinaryDataMapType::const_iterator;
  BinaryDataMapType BinaryDataMap;

  using FilteredBinaryDataConstIterator =
    FilterIterator<binary_data_const_iterator>;
  using FilteredBinaryDataIterator = FilterIterator<binary_data_iterator>;

  /// Return BinaryFunction containing a given \p Address or nullptr if
  /// no registered function has it.
  ///
  /// In a binary a function has somewhat vague  boundaries. E.g. a function can
  /// refer to the first byte past the end of the function, and it will still be
  /// referring to this function, not the function following it in the address
  /// space. Thus we have the following flags that allow to lookup for
  /// a function where a caller has more context for the search.
  ///
  /// If \p CheckPastEnd is true and the \p Address falls on a byte
  /// immediately following the last byte of some function and there's no other
  /// function that starts there, then return the function as the one containing
  /// the \p Address. This is useful when we need to locate functions for
  /// references pointing immediately past a function body.
  ///
  /// If \p UseMaxSize is true, then include the space between this function
  /// body and the next object in address ranges that we check.
  BinaryFunction *getBinaryFunctionContainingAddress(uint64_t Address,
                                                     bool CheckPastEnd = false,
                                                     bool UseMaxSize = false,
                                                     bool Shallow = false);

  /// Return BinaryFunction which has a fragment that starts at a given
  /// \p Address. If the BinaryFunction is a child fragment, then return its
  /// parent unless \p Shallow parameter is set to true.
  BinaryFunction *getBinaryFunctionAtAddress(uint64_t Address,
                                             bool Shallow = false);

  const BinaryFunction *getBinaryFunctionAtAddress(uint64_t Address,
                                                   bool Shallow = false) const {
    return const_cast<BinaryContext *>(this)->
        getBinaryFunctionAtAddress(Address, Shallow);
  }

  /// Return JumpTable containing a given \p Address.
  JumpTable *getJumpTableContainingAddress(uint64_t Address) {
    auto JTI = JumpTables.upper_bound(Address);
    if (JTI == JumpTables.begin())
      return nullptr;
    --JTI;
    if (JTI->first + JTI->second->getSize() > Address) {
      return JTI->second;
    }
    return nullptr;
  }

  /// [MCSymbol] -> [BinaryFunction]
  ///
  /// As we fold identical functions, multiple symbols can point
  /// to the same BinaryFunction.
  std::unordered_map<const MCSymbol *,
                     BinaryFunction *> SymbolToFunctionMap;

  /// Look up the symbol entry that contains the given \p Address (based on
  /// the start address and size for each symbol).  Returns a pointer to
  /// the BinaryData for that symbol.  If no data is found, nullptr is returned.
  const BinaryData *getBinaryDataContainingAddressImpl(uint64_t Address,
                                                       bool IncludeEnd,
                                                       bool BestFit) const;

  /// Update the Parent fields in BinaryDatas after adding a new entry into
  /// \p BinaryDataMap.
  void updateObjectNesting(BinaryDataMapType::iterator GAI);

  /// Validate that if object address ranges overlap that the object with
  /// the larger range is a parent of the object with the smaller range.
  bool validateObjectNesting() const;

  /// Validate that there are no top level "holes" in each section
  /// and that all relocations with a section are mapped to a valid
  /// top level BinaryData.
  bool validateHoles() const;

  /// Produce output address ranges based on input ranges for some module.
  DebugAddressRangesVector translateModuleAddressRanges(
      const DWARFAddressRangesVector &InputRanges) const;

  /// Get a bogus "absolute" section that will be associated with all
  /// absolute BinaryDatas.
  BinarySection &absoluteSection();

  /// Process "holes" in between known BinaryData objects.  For now,
  /// symbols are padded with the space before the next BinaryData object.
  void fixBinaryDataHoles();

  /// Generate names based on data hashes for unknown symbols.
  void generateSymbolHashes();

  /// Populate \p GlobalMemData.  This should be done after all symbol discovery
  /// is complete, e.g. after building CFGs for all functions.
  void assignMemData();

  /// Construct BinaryFunction object and add it to internal maps.
  BinaryFunction *createBinaryFunction(const std::string &Name,
                                       BinarySection &Section,
                                       uint64_t Address,
                                       uint64_t Size,
                                       bool IsSimple,
                                       uint64_t SymbolSize = 0,
                                       uint16_t Alignment = 0);

  /// Return all functions for this rewrite instance.
  std::map<uint64_t, BinaryFunction> &getBinaryFunctions() {
    return BinaryFunctions;
  }

  /// Return all functions for this rewrite instance.
  const std::map<uint64_t, BinaryFunction> &getBinaryFunctions() const {
    return BinaryFunctions;
  }

  /// Create BOLT-injected function
  BinaryFunction *createInjectedBinaryFunction(const std::string &Name,
                                               bool IsSimple = true);

  std::vector<BinaryFunction *> &getInjectedBinaryFunctions() {
    return InjectedBinaryFunctions;
  }

  /// Construct a jump table for \p Function at \p Address.
  /// May create an embedded jump table and return its label as the second
  /// element of the pair.
  std::pair<JumpTable *, const MCSymbol *>
  createJumpTable(BinaryFunction &Function,
                  uint64_t Address,
                  JumpTable::JumpTableType Type,
                  JumpTable::OffsetEntriesType &&OffsetEntries);

  /// Generate a unique name for jump table at a given \p Address belonging
  /// to function \p BF.
  std::string generateJumpTableName(const BinaryFunction &BF, uint64_t Address);

public:
  /// Regular page size.
  static constexpr unsigned RegularPageSize = 0x1000;

  /// Huge page size to use.
  static constexpr unsigned HugePageSize = 0x200000;

  /// Map address to a constant island owner (constant data in code section)
  std::map<uint64_t, BinaryFunction *> AddressToConstantIslandMap;

  /// A map from jump table address to insertion order.  Used for generating
  /// jump table names.
  std::map<uint64_t, size_t> JumpTableIds;

  /// Set of addresses in the code that are not a function start, and are
  /// referenced from outside of containing function. E.g. this could happen
  /// when a function has more than a single entry point.
  std::set<std::pair<BinaryFunction *, uint64_t>> InterproceduralReferences;

  std::unique_ptr<MCContext> Ctx;

  std::unique_ptr<DWARFContext> DwCtx;

  std::unique_ptr<Triple> TheTriple;

  const Target *TheTarget;

  std::string TripleName;

  std::unique_ptr<MCCodeEmitter> MCE;

  std::unique_ptr<MCObjectFileInfo> MOFI;

  std::unique_ptr<const MCAsmInfo> AsmInfo;

  std::unique_ptr<const MCInstrInfo> MII;

  std::unique_ptr<const MCSubtargetInfo> STI;

  std::unique_ptr<MCInstPrinter> InstPrinter;

  std::unique_ptr<const MCInstrAnalysis> MIA;

  std::unique_ptr<MCPlusBuilder> MIB;

  std::unique_ptr<const MCRegisterInfo> MRI;

  std::unique_ptr<MCDisassembler> DisAsm;

  std::unique_ptr<MCAsmBackend> MAB;

  DataReader &DR;

  /// Indicates if relocations are available for usage.
  bool HasRelocations{false};

  /// Is the binary always loaded at a fixed address.
  bool HasFixedLoadAddress{true};

  /// Sum of execution count of all functions
  uint64_t SumExecutionCount{0};

  /// Number of functions with profile information
  uint64_t NumProfiledFuncs{0};

  /// Total hotness score according to profiling data for this binary.
  uint64_t TotalScore{0};

  /// Binary-wide stats for macro-fusion.
  uint64_t MissedMacroFusionPairs{0};
  uint64_t MissedMacroFusionExecCount{0};

  // Address of the first allocated segment.
  uint64_t FirstAllocAddress{std::numeric_limits<uint64_t>::max()};

  /// Track next available address for new allocatable sections. RewriteInstance
  /// sets this prior to running BOLT passes, so layout passes are aware of the
  /// final addresses functions will have.
  uint64_t LayoutStartAddress{0};

  /// Old .text info.
  uint64_t OldTextSectionAddress{0};
  uint64_t OldTextSectionOffset{0};
  uint64_t OldTextSectionSize{0};

  /// Page alignment used for code layout.
  uint64_t PageAlign{HugePageSize};

  /// True if the binary requires immediate relocation processing.
  bool RequiresZNow{false};

  /// List of functions that always trap.
  std::vector<const BinaryFunction *> TrappedFunctions;

  /// Map SDT locations to SDT markers info
  std::unordered_map<uint64_t, SDTMarkerInfo> SDTMarkers;

  BinaryContext(std::unique_ptr<MCContext> Ctx,
                std::unique_ptr<DWARFContext> DwCtx,
                std::unique_ptr<Triple> TheTriple,
                const Target *TheTarget,
                std::string TripleName,
                std::unique_ptr<MCCodeEmitter> MCE,
                std::unique_ptr<MCObjectFileInfo> MOFI,
                std::unique_ptr<const MCAsmInfo> AsmInfo,
                std::unique_ptr<const MCInstrInfo> MII,
                std::unique_ptr<const MCSubtargetInfo> STI,
                std::unique_ptr<MCInstPrinter> InstPrinter,
                std::unique_ptr<const MCInstrAnalysis> MIA,
                std::unique_ptr<MCPlusBuilder> MIB,
                std::unique_ptr<const MCRegisterInfo> MRI,
                std::unique_ptr<MCDisassembler> DisAsm,
                DataReader &DR);

  ~BinaryContext();

  std::unique_ptr<MCObjectWriter> createObjectWriter(raw_pwrite_stream &OS);

  bool isAArch64() const {
    return TheTriple->getArch() == llvm::Triple::aarch64;
  }

  bool isX86() const {
    return TheTriple->getArch() == llvm::Triple::x86 ||
           TheTriple->getArch() == llvm::Triple::x86_64;
  }

  /// Iterate over all BinaryData.
  iterator_range<binary_data_const_iterator> getBinaryData() const {
    return make_range(BinaryDataMap.begin(), BinaryDataMap.end());
  }

  /// Iterate over all BinaryData.
  iterator_range<binary_data_iterator> getBinaryData() {
    return make_range(BinaryDataMap.begin(), BinaryDataMap.end());
  }

  /// Iterate over all BinaryData associated with the given \p Section.
  iterator_range<FilteredBinaryDataConstIterator>
  getBinaryDataForSection(const BinarySection &Section) const {
    auto Begin = BinaryDataMap.lower_bound(Section.getAddress());
    if (Begin != BinaryDataMap.begin()) {
      --Begin;
    }
    auto End = BinaryDataMap.upper_bound(Section.getEndAddress());
    auto pred =
      [&Section](const binary_data_const_iterator &Itr) -> bool {
        return Itr->second->getSection() == Section;
      };
    return make_range(FilteredBinaryDataConstIterator(pred, Begin, End),
                      FilteredBinaryDataConstIterator(pred, End, End));
  }

  /// Iterate over all BinaryData associated with the given \p Section.
  iterator_range<FilteredBinaryDataIterator>
  getBinaryDataForSection(BinarySection &Section) {
    auto Begin = BinaryDataMap.lower_bound(Section.getAddress());
    if (Begin != BinaryDataMap.begin()) {
      --Begin;
    }
    auto End = BinaryDataMap.upper_bound(Section.getEndAddress());
    auto pred = [&Section](const binary_data_iterator &Itr) -> bool {
      return Itr->second->getSection() == Section;
    };
    return make_range(FilteredBinaryDataIterator(pred, Begin, End),
                      FilteredBinaryDataIterator(pred, End, End));
  }

  /// Iterate over all the sub-symbols of /p BD (if any).
  iterator_range<binary_data_iterator> getSubBinaryData(BinaryData *BD);

  /// Clear the global symbol address -> name(s) map.
  void clearBinaryData() {
    GlobalSymbols.clear();
    for (auto &Entry : BinaryDataMap) {
      delete Entry.second;
    }
    BinaryDataMap.clear();
  }


  /// Return a global symbol registered at a given \p Address and \p Size.
  /// If no symbol exists, create one with unique name using \p Prefix.
  /// If there are multiple symbols registered at the \p Address, then
  /// return the first one.
  MCSymbol *getOrCreateGlobalSymbol(uint64_t Address,
                                    Twine Prefix,
                                    uint64_t Size = 0,
                                    uint16_t Alignment = 0,
                                    unsigned Flags = 0);

  /// Register a symbol with \p Name at a given \p Address and \p Size.
  MCSymbol *registerNameAtAddress(StringRef Name,
                                  uint64_t Address,
                                  BinaryData* BD);

  /// Register a symbol with \p Name at a given \p Address, \p Size and
  /// /p Flags.  See llvm::SymbolRef::Flags for definition of /p Flags.
  MCSymbol *registerNameAtAddress(StringRef Name,
                                  uint64_t Address,
                                  uint64_t Size,
                                  uint16_t Alignment,
                                  unsigned Flags = 0);

  /// Return BinaryData registered at a given \p Address or nullptr if no
  /// global symbol was registered at the location.
  const BinaryData *getBinaryDataAtAddress(uint64_t Address) const {
    auto NI = BinaryDataMap.find(Address);
    return NI != BinaryDataMap.end() ? NI->second : nullptr;
  }

  BinaryData *getBinaryDataAtAddress(uint64_t Address) {
    auto NI = BinaryDataMap.find(Address);
    return NI != BinaryDataMap.end() ? NI->second : nullptr;
  }

  /// Look up the symbol entry that contains the given \p Address (based on
  /// the start address and size for each symbol).  Returns a pointer to
  /// the BinaryData for that symbol.  If no data is found, nullptr is returned.
  const BinaryData *getBinaryDataContainingAddress(uint64_t Address,
                                                   bool IncludeEnd = false,
                                                   bool BestFit = false) const {
    return getBinaryDataContainingAddressImpl(Address, IncludeEnd, BestFit);
  }

  BinaryData *getBinaryDataContainingAddress(uint64_t Address,
                                             bool IncludeEnd = false,
                                             bool BestFit = false) {
    return const_cast<BinaryData *>(getBinaryDataContainingAddressImpl(Address,
                                                                       IncludeEnd,
                                                                       BestFit));
  }

  /// Return BinaryData for the given \p Name or nullptr if no
  /// global symbol with that name exists.
  const BinaryData *getBinaryDataByName(StringRef Name) const {
    auto Itr = GlobalSymbols.find(Name);
    return Itr != GlobalSymbols.end() ? Itr->second : nullptr;
  }

  BinaryData *getBinaryDataByName(StringRef Name) {
    auto Itr = GlobalSymbols.find(Name);
    return Itr != GlobalSymbols.end() ? Itr->second : nullptr;
  }

  MCSymbol *getHotTextStartSymbol() const {
    return Ctx->getOrCreateSymbol("__hot_start");
  }

  MCSymbol *getHotTextEndSymbol() const {
    return Ctx->getOrCreateSymbol("__hot_end");
  }

  MCSection *getTextSection() const {
    return MOFI->getTextSection();
  }

  /// Return code section with a given name.
  MCSection *getCodeSection(StringRef SectionName) const {
    return Ctx->getELFSection(SectionName,
                              ELF::SHT_PROGBITS,
                              ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
  }

  /// \name Pre-assigned Section Names
  /// @{

  const char *getMainCodeSectionName() const {
    return ".text";
  }

  const char *getColdCodeSectionName() const {
    return ".text.cold";
  }

  const char *getHotTextMoverSectionName() const {
    return ".text.mover";
  }

  const char *getInjectedCodeSectionName() const {
    return ".text.injected";
  }

  const char *getInjectedColdCodeSectionName() const {
    return ".text.injected.cold";
  }

  ErrorOr<BinarySection &> getGdbIndexSection() const {
    return getUniqueSectionByName(".gdb_index");
  }

  /// @}

  /// Perform any necessary post processing on the symbol table after
  /// function disassembly is complete.  This processing fixes top
  /// level data holes and makes sure the symbol table is valid.
  /// It also assigns all memory profiling info to the appropriate
  /// BinaryData objects.
  void postProcessSymbolTable();

  /// Set the size of the global symbol located at \p Address.  Return
  /// false if no symbol exists, true otherwise.
  bool setBinaryDataSize(uint64_t Address, uint64_t Size);

  /// Print the global symbol table.
  void printGlobalSymbols(raw_ostream& OS) const;

  /// Get the raw bytes for a given function.
  ErrorOr<ArrayRef<uint8_t>>
  getFunctionData(const BinaryFunction &Function) const;

  /// Register information about the given \p Section so we can look up
  /// sections by address.
  BinarySection &registerSection(SectionRef Section);

  /// Register a copy of /p OriginalSection under a different name.
  BinarySection &registerSection(StringRef SectionName,
                                 const BinarySection &OriginalSection);

  /// Register or update the information for the section with the given
  /// /p Name.  If the section already exists, the information in the
  /// section will be updated with the new data.
  BinarySection &registerOrUpdateSection(StringRef Name,
                                         unsigned ELFType,
                                         unsigned ELFFlags,
                                         uint8_t *Data = nullptr,
                                         uint64_t Size = 0,
                                         unsigned Alignment = 1,
                                         bool IsLocal = false);

  /// Register the information for the note (non-allocatable) section
  /// with the given /p Name.  If the section already exists, the
  /// information in the section will be updated with the new data.
  BinarySection &registerOrUpdateNoteSection(StringRef Name,
                                             uint8_t *Data = nullptr,
                                             uint64_t Size = 0,
                                             unsigned Alignment = 1,
                                             bool IsReadOnly = true,
                                             unsigned ELFType = ELF::SHT_PROGBITS,
                                             bool IsLocal = false) {
    return registerOrUpdateSection(Name, ELFType,
                                   BinarySection::getFlags(IsReadOnly),
                                   Data, Size, Alignment, IsLocal);
  }

  /// Remove the given /p Section from the set of all sections.  Return
  /// true if the section was removed (and deleted), otherwise false.
  bool deregisterSection(BinarySection &Section);

  /// Iterate over all registered sections.
  iterator_range<FilteredSectionIterator> sections() {
    auto notNull = [](const SectionIterator &Itr) {
      return (bool)*Itr;
    };
    return make_range(FilteredSectionIterator(notNull,
                                              Sections.begin(),
                                              Sections.end()),
                      FilteredSectionIterator(notNull,
                                              Sections.end(),
                                              Sections.end()));
  }

  /// Iterate over all registered sections.
  iterator_range<FilteredSectionConstIterator> sections() const {
    return const_cast<BinaryContext *>(this)->sections();
  }

  /// Iterate over all registered allocatable sections.
  iterator_range<FilteredSectionIterator> allocatableSections() {
    auto isAllocatable = [](const SectionIterator &Itr) {
      return *Itr && Itr->isAllocatable();
    };
    return make_range(FilteredSectionIterator(isAllocatable,
                                              Sections.begin(),
                                              Sections.end()),
                      FilteredSectionIterator(isAllocatable,
                                              Sections.end(),
                                              Sections.end()));
  }

  /// Iterate over all registered code sections.
  iterator_range<FilteredSectionIterator> textSections() {
    auto isText = [](const SectionIterator &Itr) {
      return *Itr && Itr->isAllocatable() && Itr->isText();
    };
    return make_range(FilteredSectionIterator(isText,
                                              Sections.begin(),
                                              Sections.end()),
                      FilteredSectionIterator(isText,
                                              Sections.end(),
                                              Sections.end()));
  }

  /// Iterate over all registered allocatable sections.
  iterator_range<FilteredSectionConstIterator> allocatableSections() const {
    return const_cast<BinaryContext *>(this)->allocatableSections();
  }

  /// Iterate over all registered non-allocatable sections.
  iterator_range<FilteredSectionIterator> nonAllocatableSections() {
    auto notAllocated = [](const SectionIterator &Itr) {
      return *Itr && !Itr->isAllocatable();
    };
    return make_range(FilteredSectionIterator(notAllocated,
                                              Sections.begin(),
                                              Sections.end()),
                      FilteredSectionIterator(notAllocated,
                                              Sections.end(),
                                              Sections.end()));
  }

  /// Iterate over all registered non-allocatable sections.
  iterator_range<FilteredSectionConstIterator> nonAllocatableSections() const {
    return const_cast<BinaryContext *>(this)->nonAllocatableSections();
  }

  /// Iterate over all allocatable relocation sections.
  iterator_range<FilteredSectionIterator> allocatableRelaSections() {
    auto isAllocatableRela = [](const SectionIterator &Itr) {
      return *Itr && Itr->isAllocatable() && Itr->isRela();
    };
    return make_range(FilteredSectionIterator(isAllocatableRela,
                                              Sections.begin(),
                                              Sections.end()),
                      FilteredSectionIterator(isAllocatableRela,
                                              Sections.end(),
                                              Sections.end()));
  }

  /// Check if the address belongs to this binary's static allocation space.
  bool containsAddress(uint64_t Address) const {
    return Address >= FirstAllocAddress && Address < LayoutStartAddress;
  }

  /// Return section name containing the given \p Address.
  ErrorOr<StringRef> getSectionNameForAddress(uint64_t Address) const;

  /// Print all sections.
  void printSections(raw_ostream& OS) const;

  /// Return largest section containing the given \p Address.  These
  /// functions only work for allocatable sections, i.e. ones with non-zero
  /// addresses.
  ErrorOr<BinarySection &> getSectionForAddress(uint64_t Address);
  ErrorOr<const BinarySection &> getSectionForAddress(uint64_t Address) const;

  /// Return section(s) associated with given \p Name.
  iterator_range<NameToSectionMapType::iterator>
  getSectionByName(StringRef Name) {
    return make_range(NameToSection.equal_range(Name));
  }
  iterator_range<NameToSectionMapType::const_iterator>
  getSectionByName(StringRef Name) const {
    return make_range(NameToSection.equal_range(Name));
  }

  /// Return the unique section associated with given \p Name.
  /// If there is more than one section with the same name, return an error
  /// object.
  ErrorOr<BinarySection &> getUniqueSectionByName(StringRef SectionName) const {
    auto Sections = getSectionByName(SectionName);
    if (Sections.begin() != Sections.end() &&
        std::next(Sections.begin()) == Sections.end())
      return *Sections.begin()->second;
    return std::make_error_code(std::errc::bad_address);
  }

  /// Return an unsigned value of \p Size stored at \p Address. The address has
  /// to be a valid statically allocated address for the binary.
  ErrorOr<uint64_t> getUnsignedValueAtAddress(uint64_t Address,
                                              size_t Size) const;

  /// Return a signed value of \p Size stored at \p Address. The address has
  /// to be a valid statically allocated address for the binary.
  ErrorOr<uint64_t> getSignedValueAtAddress(uint64_t Address,
                                            size_t Size) const;

  /// Special case of getUnsignedValueAtAddress() that uses a pointer size.
  ErrorOr<uint64_t> getPointerAtAddress(uint64_t Address) const {
    return getUnsignedValueAtAddress(Address, AsmInfo->getCodePointerSize());
  }

  /// Replaces all references to \p ChildBF with \p ParentBF. \p ChildBF is then
  /// removed from the list of functions \p BFs. The profile data of \p ChildBF
  /// is merged into that of \p ParentBF.
  void foldFunction(BinaryFunction &ChildBF, BinaryFunction &ParentBF);

  /// Add a Section relocation at a given \p Address.
  void addRelocation(uint64_t Address, MCSymbol *Symbol, uint64_t Type,
                     uint64_t Addend = 0, uint64_t Value = 0);

  /// Remove registered relocation at a given \p Address.
  bool removeRelocationAt(uint64_t Address);

  /// Return a relocation registered at a given \p Address, or nullptr if there
  /// is no relocation at such address.
  const Relocation *getRelocationAt(uint64_t Address);

  const BinaryFunction *getFunctionForSymbol(const MCSymbol *Symbol) const {
    auto BFI = SymbolToFunctionMap.find(Symbol);
    return BFI == SymbolToFunctionMap.end() ? nullptr : BFI->second;
  }

  BinaryFunction *getFunctionForSymbol(const MCSymbol *Symbol) {
    auto BFI = SymbolToFunctionMap.find(Symbol);
    return BFI == SymbolToFunctionMap.end() ? nullptr : BFI->second;
  }

  /// Associate the symbol \p Sym with the function \p BF for lookups with
  /// getFunctionForSymbol().
  void setSymbolToFunctionMap(const MCSymbol *Sym, BinaryFunction *BF) {
    SymbolToFunctionMap[Sym] = BF;
  }

  /// Populate some internal data structures with debug info.
  void preprocessDebugInfo();

  /// Add a filename entry from SrcCUID to DestCUID.
  unsigned addDebugFilenameToUnit(const uint32_t DestCUID,
                                  const uint32_t SrcCUID,
                                  unsigned FileIndex);

  /// Return functions in output layout order
  std::vector<BinaryFunction *> getSortedFunctions();

  /// Do the best effort to calculate the size of the function by emitting
  /// its code, and relaxing branch instructions.
  ///
  /// Return the pair where the first size is for the main part, and the second
  /// size is for the cold one.
  std::pair<size_t, size_t> calculateEmittedSize(BinaryFunction &BF);

  /// Calculate the size of the instruction \p Inst.
  uint64_t computeInstructionSize(const MCInst &Inst) const {
    SmallString<256> Code;
    SmallVector<MCFixup, 4> Fixups;
    raw_svector_ostream VecOS(Code);
    MCE->encodeInstruction(Inst, VecOS, Fixups, *STI);

    return Code.size();
  }

  /// Compute the native code size for a range of instructions.
  /// Note: this can be imprecise wrt the final binary since happening prior to
  /// relaxation, as well as wrt the original binary because of opcode
  /// shortening.
  template <typename Itr>
  uint64_t computeCodeSize(Itr Beg, Itr End) const {
    uint64_t Size = 0;
    while (Beg != End) {
      if (!MII->get(Beg->getOpcode()).isPseudo())
        Size += computeInstructionSize(*Beg);
      ++Beg;
    }
    return Size;
  }

  /// Return a function execution count threshold for determining whether
  /// the function is 'hot'. Consider it hot if count is above the average exec
  /// count of profiled functions.
  uint64_t getHotThreshold() const {
    static uint64_t Threshold{0};
    if (Threshold == 0) {
      Threshold =
          NumProfiledFuncs ? SumExecutionCount / (2 * NumProfiledFuncs) : 1;
    }
    return Threshold;
  }

  /// Return true if instruction \p Inst requires an offset for further
  /// processing (e.g. assigning a profile).
  bool keepOffsetForInstruction(const MCInst &Inst) const {
    if (MIB->isCall(Inst) || MIB->isBranch(Inst) || MIB->isReturn(Inst) ||
        MIB->isPrefix(Inst) || MIB->isIndirectBranch(Inst)) {
      return true;
    }
    return false;
  }

  /// Print the string name for a CFI operation.
  static void printCFI(raw_ostream &OS, const MCCFIInstruction &Inst);

  /// Print a single MCInst in native format.  If Function is non-null,
  /// the instruction will be annotated with CFI and possibly DWARF line table
  /// info.
  /// If printMCInst is true, the instruction is also printed in the
  /// architecture independent format.
  void printInstruction(raw_ostream &OS,
                        const MCInst &Instruction,
                        uint64_t Offset = 0,
                        const BinaryFunction *Function = nullptr,
                        bool PrintMCInst = false,
                        bool PrintMemData = false,
                        bool PrintRelocations = false) const;

  /// Print a range of instructions.
  template <typename Itr>
  uint64_t printInstructions(raw_ostream &OS,
                             Itr Begin,
                             Itr End,
                             uint64_t Offset = 0,
                             const BinaryFunction *Function = nullptr,
                             bool PrintMCInst = false,
                             bool PrintMemData = false,
                             bool PrintRelocations = false) const {
    while (Begin != End) {
      printInstruction(OS, *Begin, Offset, Function, PrintMCInst,
                       PrintMemData, PrintRelocations);
      Offset += computeCodeSize(Begin, Begin + 1);
      ++Begin;
    }
    return Offset;
  }

  void exitWithBugReport(StringRef Message,
                         const BinaryFunction &Function) const;
};

} // namespace bolt
} // namespace llvm

#endif
