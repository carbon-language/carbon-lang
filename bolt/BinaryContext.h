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

#include "BinarySection.h"
#include "DebugData.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
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
class DataReader;

class BinaryContext {
  BinaryContext() = delete;

  /// Set of all sections.
  using SectionSetType = std::set<BinarySection>;
  SectionSetType Sections;

  /// Map virtual address to a section.  It is possible to have more than one
  /// section mapped to the same address, e.g. non-allocatable sections.
  using AddressToSectionMapType = std::multimap<uint64_t, BinarySection *>;
  AddressToSectionMapType AddressToSection;

  /// multimap of section name to BinarySection object.  Some binaries
  /// have multiple sections with the same name.
  using NameToSectionMapType = std::multimap<std::string, BinarySection *>;
  NameToSectionMapType NameToSection;
public:

  /// [name] -> [address] map used for global symbol resolution.
  typedef std::map<std::string, uint64_t> SymbolMapType;
  SymbolMapType GlobalSymbols;

  /// [address] -> [name1], [name2], ...
  /// Global addresses never change.
  std::multimap<uint64_t, std::string> GlobalAddresses;

  /// [MCSymbol] -> [BinaryFunction]
  ///
  /// As we fold identical functions, multiple symbols can point
  /// to the same BinaryFunction.
  std::unordered_map<const MCSymbol *,
                     BinaryFunction *> SymbolToFunctionMap;

  /// Map address to a constant island owner (constant data in code section)
  std::map<uint64_t, BinaryFunction *> AddressToConstantIslandMap;

  /// Set of addresses in the code that are not a function start, and are
  /// referenced from outside of containing function. E.g. this could happen
  /// when a function has more than a single entry point.
  std::set<uint64_t> InterproceduralReferences;

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

  std::unique_ptr<const MCRegisterInfo> MRI;

  std::unique_ptr<MCDisassembler> DisAsm;

  std::unique_ptr<MCAsmBackend> MAB;

  std::function<void(std::error_code)> ErrorCheck;

  DataReader &DR;

  /// Indicates if relocations are availabe for usage.
  bool HasRelocations{false};

  /// Sum of execution count of all functions
  uint64_t SumExecutionCount{0};

  /// Number of functions with profile information
  uint64_t NumProfiledFuncs{0};

  /// Total hotness score according to profiling data for this binary.
  uint64_t TotalScore{0};

  /// Track next available address for new allocatable sections. RewriteInstance
  /// sets this prior to running BOLT passes, so layout passes are aware of the
  /// final addresses functions will have.
  uint64_t LayoutStartAddress{0};

  /// Old .text info.
  uint64_t OldTextSectionAddress{0};
  uint64_t OldTextSectionOffset{0};
  uint64_t OldTextSectionSize{0};

  /// True if the binary requires immediate relocation processing.
  bool RequiresZNow{false};

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
                std::unique_ptr<const MCRegisterInfo> MRI,
                std::unique_ptr<MCDisassembler> DisAsm,
                DataReader &DR) :
      Ctx(std::move(Ctx)),
      DwCtx(std::move(DwCtx)),
      TheTriple(std::move(TheTriple)),
      TheTarget(TheTarget),
      TripleName(TripleName),
      MCE(std::move(MCE)),
      MOFI(std::move(MOFI)),
      AsmInfo(std::move(AsmInfo)),
      MII(std::move(MII)),
      STI(std::move(STI)),
      InstPrinter(std::move(InstPrinter)),
      MIA(std::move(MIA)),
      MRI(std::move(MRI)),
      DisAsm(std::move(DisAsm)),
      DR(DR) {
    Relocation::Arch = this->TheTriple->getArch();
  }

  ~BinaryContext();

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS);

  /// Return a global symbol registered at a given \p Address. If no symbol
  /// exists, create one with unique name using \p Prefix.
  /// If there are multiple symbols registered at the \p Address, then
  /// return the first one.
  MCSymbol *getOrCreateGlobalSymbol(uint64_t Address, Twine Prefix);

  /// Return MCSymbol registered at a given \p Address or nullptr if no
  /// global symbol was registered at the location.
  MCSymbol *getGlobalSymbolAtAddress(uint64_t Address) const;

  /// Find the address of the global symbol with the given \p Name.
  /// return an error if no such symbol exists.
  ErrorOr<uint64_t> getAddressForGlobalSymbol(StringRef Name) const {
    auto Itr = GlobalSymbols.find(Name);
    if (Itr != GlobalSymbols.end())
      return Itr->second;
    return std::make_error_code(std::errc::bad_address);
  }

  /// Return MCSymbol for the given \p Name or nullptr if no
  /// global symbol with that name exists.
  MCSymbol *getGlobalSymbolByName(const std::string &Name) const;

  /// Print the global symbol table.
  void printGlobalSymbols(raw_ostream& OS) const;

  /// Get the raw bytes for a given function.
  ErrorOr<ArrayRef<uint8_t>>
  getFunctionData(const BinaryFunction &Function) const;

  /// Register information about the given section so we can look up
  /// sections for addresses.
  BinarySection &registerSection(SectionRef Section);

  iterator_range<SectionSetType::iterator> sections() {
    return make_range(Sections.begin(), Sections.end());
  }

  iterator_range<SectionSetType::const_iterator> sections() const {
    return make_range(Sections.begin(), Sections.end());
  }

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

  /// Return the unique (allocatable) section associated with given \p Name.
  /// If there is more than one section with the same name, return an error
  /// object.
  ErrorOr<BinarySection &> getUniqueSectionByName(StringRef SectionName) {
    auto Sections = getSectionByName(SectionName);
    if (Sections.begin() != Sections.end() &&
        std::next(Sections.begin()) == Sections.end())
      return *Sections.begin()->second;
    return std::make_error_code(std::errc::bad_address);
  }
  ErrorOr<const BinarySection &>
  getUniqueSectionByName(StringRef SectionName) const {
    auto Sections = getSectionByName(SectionName);
    if (Sections.begin() != Sections.end() &&
        std::next(Sections.begin()) == Sections.end())
      return *Sections.begin()->second;
    return std::make_error_code(std::errc::bad_address);
  }

  /// Given \p Address in the binary, extract and return a pointer value at that
  /// address. The address has to be a valid statically allocated address for
  /// the binary.
  ErrorOr<uint64_t> extractPointerAtAddress(uint64_t Address) const;

  /// Register a symbol with \p Name at a given \p Address.
  MCSymbol *registerNameAtAddress(const std::string &Name, uint64_t Address) {
    // Check if the Name was already registered.
    const auto GSI = GlobalSymbols.find(Name);
    if (GSI != GlobalSymbols.end()) {
      assert(GSI->second == Address && "addresses do not match");
      auto *Symbol = Ctx->lookupSymbol(Name);
      assert(Symbol && "symbol should be registered with MCContext");

      return Symbol;
    }

    // Add the name to global symbols map.
    GlobalSymbols[Name] = Address;

    // Add to the reverse map. There could multiple names at the same address.
    GlobalAddresses.emplace(std::make_pair(Address, Name));

    // Register the name with MCContext.
    return Ctx->getOrCreateSymbol(Name);
  }

  /// Replaces all references to \p ChildBF with \p ParentBF. \p ChildBF is then
  /// removed from the list of functions \p BFs. The profile data of \p ChildBF
  /// is merged into that of \p ParentBF.
  void foldFunction(BinaryFunction &ChildBF,
                    BinaryFunction &ParentBF,
                    std::map<uint64_t, BinaryFunction> &BFs);

  /// Add relocation for \p Section at a given \p Offset.
  void addSectionRelocation(BinarySection &Section, uint64_t Offset,
                            MCSymbol *Symbol, uint64_t Type,
                            uint64_t Addend = 0);

  /// Add a relocation at a given \p Address.
  void addRelocation(uint64_t Address, MCSymbol *Symbol, uint64_t Type,
                     uint64_t Addend = 0);

  /// Remove registered relocation at a given \p Address.
  void removeRelocationAt(uint64_t Address);

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

  /// Populate some internal data structures with debug info.
  void preprocessDebugInfo(
      std::map<uint64_t, BinaryFunction> &BinaryFunctions);

  /// Add a filename entry from SrcCUID to DestCUID.
  unsigned addDebugFilenameToUnit(const uint32_t DestCUID,
                                  const uint32_t SrcCUID,
                                  unsigned FileIndex);

  /// Return functions in output layout order
  static std::vector<BinaryFunction *>
  getSortedFunctions(std::map<uint64_t, BinaryFunction> &BinaryFunctions);

  /// Compute the native code size for a range of instructions.
  /// Note: this can be imprecise wrt the final binary since happening prior to
  /// relaxation, as well as wrt the original binary because of opcode
  /// shortening.
  template <typename Itr>
  uint64_t computeCodeSize(Itr Beg, Itr End) const {
    uint64_t Size = 0;
    while (Beg != End) {
      // Calculate the size of the instruction.
      SmallString<256> Code;
      SmallVector<MCFixup, 4> Fixups;
      raw_svector_ostream VecOS(Code);
      if (MIA->isCFI(*Beg) || MIA->isEHLabel(*Beg)) {
        ++Beg;
        continue;
      }
      MCE->encodeInstruction(*Beg++, VecOS, Fixups, *STI);
      Size += Code.size();
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
    if (MIA->isCall(Inst) || MIA->isBranch(Inst) || MIA->isReturn(Inst) ||
        MIA->isPrefix(Inst) || MIA->isIndirectBranch(Inst)) {
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
};

} // namespace bolt
} // namespace llvm

#endif
