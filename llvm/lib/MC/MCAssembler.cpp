//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "assembler"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace llvm;

class MachObjectWriter;

STATISTIC(EmittedFragments, "Number of emitted assembler fragments");

// FIXME FIXME FIXME: There are number of places in this file where we convert
// what is a 64-bit assembler value used for computation into a value in the
// object file, which may truncate it. We should detect that truncation where
// invalid and report errors back.

static void WriteFileData(raw_ostream &OS, const MCSectionData &SD,
                          MachObjectWriter &MOW);

/// isVirtualSection - Check if this is a section which does not actually exist
/// in the object file.
static bool isVirtualSection(const MCSection &Section) {
  // FIXME: Lame.
  const MCSectionMachO &SMO = static_cast<const MCSectionMachO&>(Section);
  unsigned Type = SMO.getTypeAndAttributes() & MCSectionMachO::SECTION_TYPE;
  return (Type == MCSectionMachO::S_ZEROFILL);
}

class MachObjectWriter {
  // See <mach-o/loader.h>.
  enum {
    Header_Magic32 = 0xFEEDFACE,
    Header_Magic64 = 0xFEEDFACF
  };

  static const unsigned Header32Size = 28;
  static const unsigned Header64Size = 32;
  static const unsigned SegmentLoadCommand32Size = 56;
  static const unsigned Section32Size = 68;
  static const unsigned SymtabLoadCommandSize = 24;
  static const unsigned DysymtabLoadCommandSize = 80;
  static const unsigned Nlist32Size = 12;
  static const unsigned RelocationInfoSize = 8;

  enum HeaderFileType {
    HFT_Object = 0x1
  };

  enum HeaderFlags {
    HF_SubsectionsViaSymbols = 0x2000
  };

  enum LoadCommandType {
    LCT_Segment = 0x1,
    LCT_Symtab = 0x2,
    LCT_Dysymtab = 0xb
  };

  // See <mach-o/nlist.h>.
  enum SymbolTypeType {
    STT_Undefined = 0x00,
    STT_Absolute  = 0x02,
    STT_Section   = 0x0e
  };

  enum SymbolTypeFlags {
    // If any of these bits are set, then the entry is a stab entry number (see
    // <mach-o/stab.h>. Otherwise the other masks apply.
    STF_StabsEntryMask = 0xe0,

    STF_TypeMask       = 0x0e,
    STF_External       = 0x01,
    STF_PrivateExtern  = 0x10
  };

  /// IndirectSymbolFlags - Flags for encoding special values in the indirect
  /// symbol entry.
  enum IndirectSymbolFlags {
    ISF_Local    = 0x80000000,
    ISF_Absolute = 0x40000000
  };

  /// RelocationFlags - Special flags for addresses.
  enum RelocationFlags {
    RF_Scattered = 0x80000000
  };

  enum RelocationInfoType {
    RIT_Vanilla             = 0,
    RIT_Pair                = 1,
    RIT_Difference          = 2,
    RIT_PreboundLazyPointer = 3,
    RIT_LocalDifference     = 4
  };

  /// MachSymbolData - Helper struct for containing some precomputed information
  /// on symbols.
  struct MachSymbolData {
    MCSymbolData *SymbolData;
    uint64_t StringIndex;
    uint8_t SectionIndex;

    // Support lexicographic sorting.
    bool operator<(const MachSymbolData &RHS) const {
      const std::string &Name = SymbolData->getSymbol().getName();
      return Name < RHS.SymbolData->getSymbol().getName();
    }
  };

  raw_ostream &OS;
  bool IsLSB;

public:
  MachObjectWriter(raw_ostream &_OS, bool _IsLSB = true)
    : OS(_OS), IsLSB(_IsLSB) {
  }

  /// @name Helper Methods
  /// @{

  void Write8(uint8_t Value) {
    OS << char(Value);
  }

  void Write16(uint16_t Value) {
    if (IsLSB) {
      Write8(uint8_t(Value >> 0));
      Write8(uint8_t(Value >> 8));
    } else {
      Write8(uint8_t(Value >> 8));
      Write8(uint8_t(Value >> 0));
    }
  }

  void Write32(uint32_t Value) {
    if (IsLSB) {
      Write16(uint16_t(Value >> 0));
      Write16(uint16_t(Value >> 16));
    } else {
      Write16(uint16_t(Value >> 16));
      Write16(uint16_t(Value >> 0));
    }
  }

  void Write64(uint64_t Value) {
    if (IsLSB) {
      Write32(uint32_t(Value >> 0));
      Write32(uint32_t(Value >> 32));
    } else {
      Write32(uint32_t(Value >> 32));
      Write32(uint32_t(Value >> 0));
    }
  }

  void WriteZeros(unsigned N) {
    const char Zeros[16] = { 0 };

    for (unsigned i = 0, e = N / 16; i != e; ++i)
      OS << StringRef(Zeros, 16);

    OS << StringRef(Zeros, N % 16);
  }

  void WriteString(StringRef Str, unsigned ZeroFillSize = 0) {
    OS << Str;
    if (ZeroFillSize)
      WriteZeros(ZeroFillSize - Str.size());
  }

  /// @}

  void WriteHeader32(unsigned NumLoadCommands, unsigned LoadCommandsSize,
                     bool SubsectionsViaSymbols) {
    uint32_t Flags = 0;

    if (SubsectionsViaSymbols)
      Flags |= HF_SubsectionsViaSymbols;

    // struct mach_header (28 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(Header_Magic32);

    // FIXME: Support cputype.
    Write32(MachO::CPUTypeI386);
    // FIXME: Support cpusubtype.
    Write32(MachO::CPUSubType_I386_ALL);
    Write32(HFT_Object);
    Write32(NumLoadCommands);    // Object files have a single load command, the
                                 // segment.
    Write32(LoadCommandsSize);
    Write32(Flags);

    assert(OS.tell() - Start == Header32Size);
  }

  /// WriteSegmentLoadCommand32 - Write a 32-bit segment load command.
  ///
  /// \arg NumSections - The number of sections in this segment.
  /// \arg SectionDataSize - The total size of the sections.
  void WriteSegmentLoadCommand32(unsigned NumSections,
                                 uint64_t VMSize,
                                 uint64_t SectionDataStartOffset,
                                 uint64_t SectionDataSize) {
    // struct segment_command (56 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(LCT_Segment);
    Write32(SegmentLoadCommand32Size + NumSections * Section32Size);

    WriteString("", 16);
    Write32(0); // vmaddr
    Write32(VMSize); // vmsize
    Write32(SectionDataStartOffset); // file offset
    Write32(SectionDataSize); // file size
    Write32(0x7); // maxprot
    Write32(0x7); // initprot
    Write32(NumSections);
    Write32(0); // flags

    assert(OS.tell() - Start == SegmentLoadCommand32Size);
  }

  void WriteSection32(const MCSectionData &SD, uint64_t FileOffset,
                      uint64_t RelocationsStart, unsigned NumRelocations) {
    // The offset is unused for virtual sections.
    if (isVirtualSection(SD.getSection())) {
      assert(SD.getFileSize() == 0 && "Invalid file size!");
      FileOffset = 0;
    }

    // struct section (68 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    // FIXME: cast<> support!
    const MCSectionMachO &Section =
      static_cast<const MCSectionMachO&>(SD.getSection());
    WriteString(Section.getSectionName(), 16);
    WriteString(Section.getSegmentName(), 16);
    Write32(SD.getAddress()); // address
    Write32(SD.getSize()); // size
    Write32(FileOffset);

    assert(isPowerOf2_32(SD.getAlignment()) && "Invalid alignment!");
    Write32(Log2_32(SD.getAlignment()));
    Write32(NumRelocations ? RelocationsStart : 0);
    Write32(NumRelocations);
    Write32(Section.getTypeAndAttributes());
    Write32(0); // reserved1
    Write32(Section.getStubSize()); // reserved2

    assert(OS.tell() - Start == Section32Size);
  }

  void WriteSymtabLoadCommand(uint32_t SymbolOffset, uint32_t NumSymbols,
                              uint32_t StringTableOffset,
                              uint32_t StringTableSize) {
    // struct symtab_command (24 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(LCT_Symtab);
    Write32(SymtabLoadCommandSize);
    Write32(SymbolOffset);
    Write32(NumSymbols);
    Write32(StringTableOffset);
    Write32(StringTableSize);

    assert(OS.tell() - Start == SymtabLoadCommandSize);
  }

  void WriteDysymtabLoadCommand(uint32_t FirstLocalSymbol,
                                uint32_t NumLocalSymbols,
                                uint32_t FirstExternalSymbol,
                                uint32_t NumExternalSymbols,
                                uint32_t FirstUndefinedSymbol,
                                uint32_t NumUndefinedSymbols,
                                uint32_t IndirectSymbolOffset,
                                uint32_t NumIndirectSymbols) {
    // struct dysymtab_command (80 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(LCT_Dysymtab);
    Write32(DysymtabLoadCommandSize);
    Write32(FirstLocalSymbol);
    Write32(NumLocalSymbols);
    Write32(FirstExternalSymbol);
    Write32(NumExternalSymbols);
    Write32(FirstUndefinedSymbol);
    Write32(NumUndefinedSymbols);
    Write32(0); // tocoff
    Write32(0); // ntoc
    Write32(0); // modtaboff
    Write32(0); // nmodtab
    Write32(0); // extrefsymoff
    Write32(0); // nextrefsyms
    Write32(IndirectSymbolOffset);
    Write32(NumIndirectSymbols);
    Write32(0); // extreloff
    Write32(0); // nextrel
    Write32(0); // locreloff
    Write32(0); // nlocrel

    assert(OS.tell() - Start == DysymtabLoadCommandSize);
  }

  void WriteNlist32(MachSymbolData &MSD) {
    MCSymbolData &Data = *MSD.SymbolData;
    const MCSymbol &Symbol = Data.getSymbol();
    uint8_t Type = 0;
    uint16_t Flags = Data.getFlags();
    uint32_t Address = 0;

    // Set the N_TYPE bits. See <mach-o/nlist.h>.
    //
    // FIXME: Are the prebound or indirect fields possible here?
    if (Symbol.isUndefined())
      Type = STT_Undefined;
    else if (Symbol.isAbsolute())
      Type = STT_Absolute;
    else
      Type = STT_Section;

    // FIXME: Set STAB bits.

    if (Data.isPrivateExtern())
      Type |= STF_PrivateExtern;

    // Set external bit.
    if (Data.isExternal() || Symbol.isUndefined())
      Type |= STF_External;

    // Compute the symbol address.
    if (Symbol.isDefined()) {
      if (Symbol.isAbsolute()) {
        llvm_unreachable("FIXME: Not yet implemented!");
      } else {
        Address = Data.getFragment()->getAddress() + Data.getOffset();
      }
    } else if (Data.isCommon()) {
      // Common symbols are encoded with the size in the address
      // field, and their alignment in the flags.
      Address = Data.getCommonSize();

      // Common alignment is packed into the 'desc' bits.
      if (unsigned Align = Data.getCommonAlignment()) {
        unsigned Log2Size = Log2_32(Align);
        assert((1U << Log2Size) == Align && "Invalid 'common' alignment!");
        if (Log2Size > 15)
          llvm_report_error("invalid 'common' alignment '" +
                            Twine(Align) + "'");
        // FIXME: Keep this mask with the SymbolFlags enumeration.
        Flags = (Flags & 0xF0FF) | (Log2Size << 8);
      }
    }

    // struct nlist (12 bytes)

    Write32(MSD.StringIndex);
    Write8(Type);
    Write8(MSD.SectionIndex);

    // The Mach-O streamer uses the lowest 16-bits of the flags for the 'desc'
    // value.
    Write16(Flags);
    Write32(Address);
  }

  struct MachRelocationEntry {
    uint32_t Word0;
    uint32_t Word1;
  };
  void ComputeScatteredRelocationInfo(MCAssembler &Asm,
                                      MCSectionData::Fixup &Fixup,
                                      const MCValue &Target,
                             DenseMap<const MCSymbol*,MCSymbolData*> &SymbolMap,
                                     std::vector<MachRelocationEntry> &Relocs) {
    uint32_t Address = Fixup.Fragment->getOffset() + Fixup.Offset;
    unsigned IsPCRel = 0;
    unsigned Type = RIT_Vanilla;

    // See <reloc.h>.
    const MCSymbol *A = Target.getSymA();
    MCSymbolData *SD = SymbolMap.lookup(A);
    uint32_t Value = SD->getFragment()->getAddress() + SD->getOffset();
    uint32_t Value2 = 0;

    if (const MCSymbol *B = Target.getSymB()) {
      Type = RIT_LocalDifference;

      MCSymbolData *SD = SymbolMap.lookup(B);
      Value2 = SD->getFragment()->getAddress() + SD->getOffset();
    }

    unsigned Log2Size = Log2_32(Fixup.Size);
    assert((1U << Log2Size) == Fixup.Size && "Invalid fixup size!");

    // The value which goes in the fixup is current value of the expression.
    Fixup.FixedValue = Value - Value2 + Target.getConstant();

    MachRelocationEntry MRE;
    MRE.Word0 = ((Address   <<  0) |
                 (Type      << 24) |
                 (Log2Size  << 28) |
                 (IsPCRel   << 30) |
                 RF_Scattered);
    MRE.Word1 = Value;
    Relocs.push_back(MRE);

    if (Type == RIT_LocalDifference) {
      Type = RIT_Pair;

      MachRelocationEntry MRE;
      MRE.Word0 = ((0         <<  0) |
                   (Type      << 24) |
                   (Log2Size  << 28) |
                   (0   << 30) |
                   RF_Scattered);
      MRE.Word1 = Value2;
      Relocs.push_back(MRE);
    }
  }

  void ComputeRelocationInfo(MCAssembler &Asm,
                             MCSectionData::Fixup &Fixup,
                             DenseMap<const MCSymbol*,MCSymbolData*> &SymbolMap,
                             std::vector<MachRelocationEntry> &Relocs) {
    MCValue Target;
    if (!Fixup.Value->EvaluateAsRelocatable(Target))
      llvm_report_error("expected relocatable expression");

    // If this is a difference or a local symbol plus an offset, then we need a
    // scattered relocation entry.
    if (Target.getSymB() ||
        (Target.getSymA() && !Target.getSymA()->isUndefined() &&
         Target.getConstant()))
      return ComputeScatteredRelocationInfo(Asm, Fixup, Target,
                                            SymbolMap, Relocs);

    // See <reloc.h>.
    uint32_t Address = Fixup.Fragment->getOffset() + Fixup.Offset;
    uint32_t Value = 0;
    unsigned Index = 0;
    unsigned IsPCRel = 0;
    unsigned IsExtern = 0;
    unsigned Type = 0;

    if (Target.isAbsolute()) { // constant
      // SymbolNum of 0 indicates the absolute section.
      Type = RIT_Vanilla;
      Value = 0;
      llvm_unreachable("FIXME: Not yet implemented!");
    } else {
      const MCSymbol *Symbol = Target.getSymA();
      MCSymbolData *SD = SymbolMap.lookup(Symbol);

      if (Symbol->isUndefined()) {
        IsExtern = 1;
        Index = SD->getIndex();
        Value = 0;
      } else {
        // The index is the section ordinal.
        //
        // FIXME: O(N)
        Index = 1;
        for (MCAssembler::iterator it = Asm.begin(),
               ie = Asm.end(); it != ie; ++it, ++Index)
          if (&*it == SD->getFragment()->getParent())
            break;
        Value = SD->getFragment()->getAddress() + SD->getOffset();
      }

      Type = RIT_Vanilla;
    }

    // The value which goes in the fixup is current value of the expression.
    Fixup.FixedValue = Value + Target.getConstant();

    unsigned Log2Size = Log2_32(Fixup.Size);
    assert((1U << Log2Size) == Fixup.Size && "Invalid fixup size!");

    // struct relocation_info (8 bytes)
    MachRelocationEntry MRE;
    MRE.Word0 = Address;
    MRE.Word1 = ((Index     <<  0) |
                 (IsPCRel   << 24) |
                 (Log2Size  << 25) |
                 (IsExtern  << 27) |
                 (Type      << 28));
    Relocs.push_back(MRE);
  }

  void BindIndirectSymbols(MCAssembler &Asm,
                           DenseMap<const MCSymbol*,MCSymbolData*> &SymbolMap) {
    // This is the point where 'as' creates actual symbols for indirect symbols
    // (in the following two passes). It would be easier for us to do this
    // sooner when we see the attribute, but that makes getting the order in the
    // symbol table much more complicated than it is worth.
    //
    // FIXME: Revisit this when the dust settles.

    // Bind non lazy symbol pointers first.
    for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
           ie = Asm.indirect_symbol_end(); it != ie; ++it) {
      // FIXME: cast<> support!
      const MCSectionMachO &Section =
        static_cast<const MCSectionMachO&>(it->SectionData->getSection());

      unsigned Type =
        Section.getTypeAndAttributes() & MCSectionMachO::SECTION_TYPE;
      if (Type != MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS)
        continue;

      MCSymbolData *&Entry = SymbolMap[it->Symbol];
      if (!Entry)
        Entry = new MCSymbolData(*it->Symbol, 0, 0, &Asm);
    }

    // Then lazy symbol pointers and symbol stubs.
    for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
           ie = Asm.indirect_symbol_end(); it != ie; ++it) {
      // FIXME: cast<> support!
      const MCSectionMachO &Section =
        static_cast<const MCSectionMachO&>(it->SectionData->getSection());

      unsigned Type =
        Section.getTypeAndAttributes() & MCSectionMachO::SECTION_TYPE;
      if (Type != MCSectionMachO::S_LAZY_SYMBOL_POINTERS &&
          Type != MCSectionMachO::S_SYMBOL_STUBS)
        continue;

      MCSymbolData *&Entry = SymbolMap[it->Symbol];
      if (!Entry) {
        Entry = new MCSymbolData(*it->Symbol, 0, 0, &Asm);

        // Set the symbol type to undefined lazy, but only on construction.
        //
        // FIXME: Do not hardcode.
        Entry->setFlags(Entry->getFlags() | 0x0001);
      }
    }
  }

  /// ComputeSymbolTable - Compute the symbol table data
  ///
  /// \param StringTable [out] - The string table data.
  /// \param StringIndexMap [out] - Map from symbol names to offsets in the
  /// string table.
  void ComputeSymbolTable(MCAssembler &Asm, SmallString<256> &StringTable,
                          std::vector<MachSymbolData> &LocalSymbolData,
                          std::vector<MachSymbolData> &ExternalSymbolData,
                          std::vector<MachSymbolData> &UndefinedSymbolData) {
    // Build section lookup table.
    DenseMap<const MCSection*, uint8_t> SectionIndexMap;
    unsigned Index = 1;
    for (MCAssembler::iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it, ++Index)
      SectionIndexMap[&it->getSection()] = Index;
    assert(Index <= 256 && "Too many sections!");

    // Index 0 is always the empty string.
    StringMap<uint64_t> StringIndexMap;
    StringTable += '\x00';

    // Build the symbol arrays and the string table, but only for non-local
    // symbols.
    //
    // The particular order that we collect the symbols and create the string
    // table, then sort the symbols is chosen to match 'as'. Even though it
    // doesn't matter for correctness, this is important for letting us diff .o
    // files.
    for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
           ie = Asm.symbol_end(); it != ie; ++it) {
      const MCSymbol &Symbol = it->getSymbol();

      // Ignore assembler temporaries.
      if (it->getSymbol().isTemporary())
        continue;

      if (!it->isExternal() && !Symbol.isUndefined())
        continue;

      uint64_t &Entry = StringIndexMap[Symbol.getName()];
      if (!Entry) {
        Entry = StringTable.size();
        StringTable += Symbol.getName();
        StringTable += '\x00';
      }

      MachSymbolData MSD;
      MSD.SymbolData = it;
      MSD.StringIndex = Entry;

      if (Symbol.isUndefined()) {
        MSD.SectionIndex = 0;
        UndefinedSymbolData.push_back(MSD);
      } else if (Symbol.isAbsolute()) {
        MSD.SectionIndex = 0;
        ExternalSymbolData.push_back(MSD);
      } else {
        MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
        assert(MSD.SectionIndex && "Invalid section index!");
        ExternalSymbolData.push_back(MSD);
      }
    }

    // Now add the data for local symbols.
    for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
           ie = Asm.symbol_end(); it != ie; ++it) {
      const MCSymbol &Symbol = it->getSymbol();

      // Ignore assembler temporaries.
      if (it->getSymbol().isTemporary())
        continue;

      if (it->isExternal() || Symbol.isUndefined())
        continue;

      uint64_t &Entry = StringIndexMap[Symbol.getName()];
      if (!Entry) {
        Entry = StringTable.size();
        StringTable += Symbol.getName();
        StringTable += '\x00';
      }

      MachSymbolData MSD;
      MSD.SymbolData = it;
      MSD.StringIndex = Entry;

      if (Symbol.isAbsolute()) {
        MSD.SectionIndex = 0;
        LocalSymbolData.push_back(MSD);
      } else {
        MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
        assert(MSD.SectionIndex && "Invalid section index!");
        LocalSymbolData.push_back(MSD);
      }
    }

    // External and undefined symbols are required to be in lexicographic order.
    std::sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
    std::sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

    // Set the symbol indices.
    Index = 0;
    for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
      LocalSymbolData[i].SymbolData->setIndex(Index++);
    for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
      ExternalSymbolData[i].SymbolData->setIndex(Index++);
    for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
      UndefinedSymbolData[i].SymbolData->setIndex(Index++);

    // The string table is padded to a multiple of 4.
    while (StringTable.size() % 4)
      StringTable += '\x00';
  }

  void WriteObject(MCAssembler &Asm) {
    unsigned NumSections = Asm.size();

    // Compute the symbol -> symbol data map.
    //
    // FIXME: This should not be here.
    DenseMap<const MCSymbol*, MCSymbolData *> SymbolMap;
    for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
           ie = Asm.symbol_end(); it != ie; ++it)
      SymbolMap[&it->getSymbol()] = it;

    // Create symbol data for any indirect symbols.
    BindIndirectSymbols(Asm, SymbolMap);

    // Compute symbol table information.
    SmallString<256> StringTable;
    std::vector<MachSymbolData> LocalSymbolData;
    std::vector<MachSymbolData> ExternalSymbolData;
    std::vector<MachSymbolData> UndefinedSymbolData;
    unsigned NumSymbols = Asm.symbol_size();

    // No symbol table command is written if there are no symbols.
    if (NumSymbols)
      ComputeSymbolTable(Asm, StringTable, LocalSymbolData, ExternalSymbolData,
                         UndefinedSymbolData);

    // The section data starts after the header, the segment load command (and
    // section headers) and the symbol table.
    unsigned NumLoadCommands = 1;
    uint64_t LoadCommandsSize =
      SegmentLoadCommand32Size + NumSections * Section32Size;

    // Add the symbol table load command sizes, if used.
    if (NumSymbols) {
      NumLoadCommands += 2;
      LoadCommandsSize += SymtabLoadCommandSize + DysymtabLoadCommandSize;
    }

    // Compute the total size of the section data, as well as its file size and
    // vm size.
    uint64_t SectionDataStart = Header32Size + LoadCommandsSize;
    uint64_t SectionDataSize = 0;
    uint64_t SectionDataFileSize = 0;
    uint64_t VMSize = 0;
    for (MCAssembler::iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it) {
      MCSectionData &SD = *it;

      VMSize = std::max(VMSize, SD.getAddress() + SD.getSize());

      if (isVirtualSection(SD.getSection()))
        continue;

      SectionDataSize = std::max(SectionDataSize,
                                 SD.getAddress() + SD.getSize());
      SectionDataFileSize = std::max(SectionDataFileSize,
                                     SD.getAddress() + SD.getFileSize());
    }

    // The section data is passed to 4 bytes.
    //
    // FIXME: Is this machine dependent?
    unsigned SectionDataPadding = OffsetToAlignment(SectionDataFileSize, 4);
    SectionDataFileSize += SectionDataPadding;

    // Write the prolog, starting with the header and load command...
    WriteHeader32(NumLoadCommands, LoadCommandsSize,
                  Asm.getSubsectionsViaSymbols());
    WriteSegmentLoadCommand32(NumSections, VMSize,
                              SectionDataStart, SectionDataSize);

    // ... and then the section headers.
    //
    // We also compute the section relocations while we do this. Note that
    // compute relocation info will also update the fixup to have the correct
    // value; this will be overwrite the appropriate data in the fragment when
    // it is written.
    std::vector<MachRelocationEntry> RelocInfos;
    uint64_t RelocTableEnd = SectionDataStart + SectionDataFileSize;
    for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie;
         ++it) {
      MCSectionData &SD = *it;

      // The assembler writes relocations in the reverse order they were seen.
      //
      // FIXME: It is probably more complicated than this.
      unsigned NumRelocsStart = RelocInfos.size();
      for (unsigned i = 0, e = SD.fixup_size(); i != e; ++i)
        ComputeRelocationInfo(Asm, SD.getFixups()[e - i - 1], SymbolMap,
                              RelocInfos);

      unsigned NumRelocs = RelocInfos.size() - NumRelocsStart;
      uint64_t SectionStart = SectionDataStart + SD.getAddress();
      WriteSection32(SD, SectionStart, RelocTableEnd, NumRelocs);
      RelocTableEnd += NumRelocs * RelocationInfoSize;
    }

    // Write the symbol table load command, if used.
    if (NumSymbols) {
      unsigned FirstLocalSymbol = 0;
      unsigned NumLocalSymbols = LocalSymbolData.size();
      unsigned FirstExternalSymbol = FirstLocalSymbol + NumLocalSymbols;
      unsigned NumExternalSymbols = ExternalSymbolData.size();
      unsigned FirstUndefinedSymbol = FirstExternalSymbol + NumExternalSymbols;
      unsigned NumUndefinedSymbols = UndefinedSymbolData.size();
      unsigned NumIndirectSymbols = Asm.indirect_symbol_size();
      unsigned NumSymTabSymbols =
        NumLocalSymbols + NumExternalSymbols + NumUndefinedSymbols;
      uint64_t IndirectSymbolSize = NumIndirectSymbols * 4;
      uint64_t IndirectSymbolOffset = 0;

      // If used, the indirect symbols are written after the section data.
      if (NumIndirectSymbols)
        IndirectSymbolOffset = RelocTableEnd;

      // The symbol table is written after the indirect symbol data.
      uint64_t SymbolTableOffset = RelocTableEnd + IndirectSymbolSize;

      // The string table is written after symbol table.
      uint64_t StringTableOffset =
        SymbolTableOffset + NumSymTabSymbols * Nlist32Size;
      WriteSymtabLoadCommand(SymbolTableOffset, NumSymTabSymbols,
                             StringTableOffset, StringTable.size());

      WriteDysymtabLoadCommand(FirstLocalSymbol, NumLocalSymbols,
                               FirstExternalSymbol, NumExternalSymbols,
                               FirstUndefinedSymbol, NumUndefinedSymbols,
                               IndirectSymbolOffset, NumIndirectSymbols);
    }

    // Write the actual section data.
    for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
      WriteFileData(OS, *it, *this);

    // Write the extra padding.
    WriteZeros(SectionDataPadding);

    // Write the relocation entries.
    for (unsigned i = 0, e = RelocInfos.size(); i != e; ++i) {
      Write32(RelocInfos[i].Word0);
      Write32(RelocInfos[i].Word1);
    }

    // Write the symbol table data, if used.
    if (NumSymbols) {
      // Write the indirect symbol entries.
      for (MCAssembler::indirect_symbol_iterator
             it = Asm.indirect_symbol_begin(),
             ie = Asm.indirect_symbol_end(); it != ie; ++it) {
        // Indirect symbols in the non lazy symbol pointer section have some
        // special handling.
        const MCSectionMachO &Section =
          static_cast<const MCSectionMachO&>(it->SectionData->getSection());
        unsigned Type =
          Section.getTypeAndAttributes() & MCSectionMachO::SECTION_TYPE;
        if (Type == MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS) {
          // If this symbol is defined and internal, mark it as such.
          if (it->Symbol->isDefined() &&
              !SymbolMap.lookup(it->Symbol)->isExternal()) {
            uint32_t Flags = ISF_Local;
            if (it->Symbol->isAbsolute())
              Flags |= ISF_Absolute;
            Write32(Flags);
            continue;
          }
        }

        Write32(SymbolMap[it->Symbol]->getIndex());
      }

      // FIXME: Check that offsets match computed ones.

      // Write the symbol table entries.
      for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
        WriteNlist32(LocalSymbolData[i]);
      for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
        WriteNlist32(ExternalSymbolData[i]);
      for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
        WriteNlist32(UndefinedSymbolData[i]);

      // Write the string table.
      OS << StringTable.str();
    }
  }
};

/* *** */

MCFragment::MCFragment() : Kind(FragmentType(~0)) {
}

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *_Parent)
  : Kind(_Kind),
    Parent(_Parent),
    FileSize(~UINT64_C(0))
{
  if (Parent)
    Parent->getFragmentList().push_back(this);
}

MCFragment::~MCFragment() {
}

uint64_t MCFragment::getAddress() const {
  assert(getParent() && "Missing Section!");
  return getParent()->getAddress() + Offset;
}

/* *** */

MCSectionData::MCSectionData() : Section(0) {}

MCSectionData::MCSectionData(const MCSection &_Section, MCAssembler *A)
  : Section(&_Section),
    Alignment(1),
    Address(~UINT64_C(0)),
    Size(~UINT64_C(0)),
    FileSize(~UINT64_C(0)),
    LastFixupLookup(~0)
{
  if (A)
    A->getSectionList().push_back(this);
}

const MCSectionData::Fixup *
MCSectionData::LookupFixup(const MCFragment *Fragment, uint64_t Offset) const {
  // Use a one level cache to turn the common case of accessing the fixups in
  // order into O(1) instead of O(N).
  unsigned i = LastFixupLookup, Count = Fixups.size(), End = Fixups.size();
  if (i >= End)
    i = 0;
  while (Count--) {
    const Fixup &F = Fixups[i];
    if (F.Fragment == Fragment && F.Offset == Offset) {
      LastFixupLookup = i;
      return &F;
    }

    ++i;
    if (i == End)
      i = 0;
  }

  return 0;
}

/* *** */

MCSymbolData::MCSymbolData() : Symbol(0) {}

MCSymbolData::MCSymbolData(const MCSymbol &_Symbol, MCFragment *_Fragment,
                           uint64_t _Offset, MCAssembler *A)
  : Symbol(&_Symbol), Fragment(_Fragment), Offset(_Offset),
    IsExternal(false), IsPrivateExtern(false),
    CommonSize(0), CommonAlign(0), Flags(0), Index(0)
{
  if (A)
    A->getSymbolList().push_back(this);
}

/* *** */

MCAssembler::MCAssembler(MCContext &_Context, raw_ostream &_OS)
  : Context(_Context), OS(_OS), SubsectionsViaSymbols(false)
{
}

MCAssembler::~MCAssembler() {
}

void MCAssembler::LayoutSection(MCSectionData &SD) {
  uint64_t Address = SD.getAddress();

  for (MCSectionData::iterator it = SD.begin(), ie = SD.end(); it != ie; ++it) {
    MCFragment &F = *it;

    F.setOffset(Address - SD.getAddress());

    // Evaluate fragment size.
    switch (F.getKind()) {
    case MCFragment::FT_Align: {
      MCAlignFragment &AF = cast<MCAlignFragment>(F);

      uint64_t Size = OffsetToAlignment(Address, AF.getAlignment());
      if (Size > AF.getMaxBytesToEmit())
        AF.setFileSize(0);
      else
        AF.setFileSize(Size);
      break;
    }

    case MCFragment::FT_Data:
      F.setFileSize(F.getMaxFileSize());
      break;

    case MCFragment::FT_Fill: {
      MCFillFragment &FF = cast<MCFillFragment>(F);

      F.setFileSize(F.getMaxFileSize());

      MCValue Target;
      if (!FF.getValue().EvaluateAsRelocatable(Target))
        llvm_report_error("expected relocatable expression");

      // If the fill value is constant, thats it.
      if (Target.isAbsolute())
        break;

      // Otherwise, add fixups for the values.
      for (uint64_t i = 0, e = FF.getCount(); i != e; ++i) {
        MCSectionData::Fixup Fix(F, i * FF.getValueSize(),
                                 FF.getValue(),FF.getValueSize());
        SD.getFixups().push_back(Fix);
      }
      break;
    }

    case MCFragment::FT_Org: {
      MCOrgFragment &OF = cast<MCOrgFragment>(F);

      MCValue Target;
      if (!OF.getOffset().EvaluateAsRelocatable(Target))
        llvm_report_error("expected relocatable expression");

      if (!Target.isAbsolute())
        llvm_unreachable("FIXME: Not yet implemented!");
      uint64_t OrgOffset = Target.getConstant();
      uint64_t Offset = Address - SD.getAddress();

      // FIXME: We need a way to communicate this error.
      if (OrgOffset < Offset)
        llvm_report_error("invalid .org offset '" + Twine(OrgOffset) +
                          "' (at offset '" + Twine(Offset) + "'");

      F.setFileSize(OrgOffset - Offset);
      break;
    }

    case MCFragment::FT_ZeroFill: {
      MCZeroFillFragment &ZFF = cast<MCZeroFillFragment>(F);

      // Align the fragment offset; it is safe to adjust the offset freely since
      // this is only in virtual sections.
      uint64_t Aligned = RoundUpToAlignment(Address, ZFF.getAlignment());
      F.setOffset(Aligned - SD.getAddress());

      // FIXME: This is misnamed.
      F.setFileSize(ZFF.getSize());
      break;
    }
    }

    Address += F.getFileSize();
  }

  // Set the section sizes.
  SD.setSize(Address - SD.getAddress());
  if (isVirtualSection(SD.getSection()))
    SD.setFileSize(0);
  else
    SD.setFileSize(Address - SD.getAddress());
}

/// WriteFileData - Write the \arg F data to the output file.
static void WriteFileData(raw_ostream &OS, const MCFragment &F,
                          MachObjectWriter &MOW) {
  uint64_t Start = OS.tell();
  (void) Start;

  ++EmittedFragments;

  // FIXME: Embed in fragments instead?
  switch (F.getKind()) {
  case MCFragment::FT_Align: {
    MCAlignFragment &AF = cast<MCAlignFragment>(F);
    uint64_t Count = AF.getFileSize() / AF.getValueSize();

    // FIXME: This error shouldn't actually occur (the front end should emit
    // multiple .align directives to enforce the semantics it wants), but is
    // severe enough that we want to report it. How to handle this?
    if (Count * AF.getValueSize() != AF.getFileSize())
      llvm_report_error("undefined .align directive, value size '" +
                        Twine(AF.getValueSize()) +
                        "' is not a divisor of padding size '" +
                        Twine(AF.getFileSize()) + "'");

    for (uint64_t i = 0; i != Count; ++i) {
      switch (AF.getValueSize()) {
      default:
        assert(0 && "Invalid size!");
      case 1: MOW.Write8 (uint8_t (AF.getValue())); break;
      case 2: MOW.Write16(uint16_t(AF.getValue())); break;
      case 4: MOW.Write32(uint32_t(AF.getValue())); break;
      case 8: MOW.Write64(uint64_t(AF.getValue())); break;
      }
    }
    break;
  }

  case MCFragment::FT_Data:
    OS << cast<MCDataFragment>(F).getContents().str();
    break;

  case MCFragment::FT_Fill: {
    MCFillFragment &FF = cast<MCFillFragment>(F);

    int64_t Value = 0;

    MCValue Target;
    if (!FF.getValue().EvaluateAsRelocatable(Target))
      llvm_report_error("expected relocatable expression");

    if (Target.isAbsolute())
      Value = Target.getConstant();
    for (uint64_t i = 0, e = FF.getCount(); i != e; ++i) {
      if (!Target.isAbsolute()) {
        // Find the fixup.
        //
        // FIXME: Find a better way to write in the fixes.
        const MCSectionData::Fixup *Fixup =
          F.getParent()->LookupFixup(&F, i * FF.getValueSize());
        assert(Fixup && "Missing fixup for fill value!");
        Value = Fixup->FixedValue;
      }

      switch (FF.getValueSize()) {
      default:
        assert(0 && "Invalid size!");
      case 1: MOW.Write8 (uint8_t (Value)); break;
      case 2: MOW.Write16(uint16_t(Value)); break;
      case 4: MOW.Write32(uint32_t(Value)); break;
      case 8: MOW.Write64(uint64_t(Value)); break;
      }
    }
    break;
  }

  case MCFragment::FT_Org: {
    MCOrgFragment &OF = cast<MCOrgFragment>(F);

    for (uint64_t i = 0, e = OF.getFileSize(); i != e; ++i)
      MOW.Write8(uint8_t(OF.getValue()));

    break;
  }

  case MCFragment::FT_ZeroFill: {
    assert(0 && "Invalid zero fill fragment in concrete section!");
    break;
  }
  }

  assert(OS.tell() - Start == F.getFileSize());
}

/// WriteFileData - Write the \arg SD data to the output file.
static void WriteFileData(raw_ostream &OS, const MCSectionData &SD,
                          MachObjectWriter &MOW) {
  // Ignore virtual sections.
  if (isVirtualSection(SD.getSection())) {
    assert(SD.getFileSize() == 0);
    return;
  }

  uint64_t Start = OS.tell();
  (void) Start;

  for (MCSectionData::const_iterator it = SD.begin(),
         ie = SD.end(); it != ie; ++it)
    WriteFileData(OS, *it, MOW);

  // Add section padding.
  assert(SD.getFileSize() >= SD.getSize() && "Invalid section sizes!");
  MOW.WriteZeros(SD.getFileSize() - SD.getSize());

  assert(OS.tell() - Start == SD.getFileSize());
}

void MCAssembler::Finish() {
  // Layout the concrete sections and fragments.
  uint64_t Address = 0;
  MCSectionData *Prev = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;

    // Skip virtual sections.
    if (isVirtualSection(SD.getSection()))
      continue;

    // Align this section if necessary by adding padding bytes to the previous
    // section.
    if (uint64_t Pad = OffsetToAlignment(Address, it->getAlignment())) {
      assert(Prev && "Missing prev section!");
      Prev->setFileSize(Prev->getFileSize() + Pad);
      Address += Pad;
    }

    // Layout the section fragments and its size.
    SD.setAddress(Address);
    LayoutSection(SD);
    Address += SD.getFileSize();

    Prev = &SD;
  }

  // Layout the virtual sections.
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;

    if (!isVirtualSection(SD.getSection()))
      continue;

    SD.setAddress(Address);
    LayoutSection(SD);
    Address += SD.getSize();
  }

  // Write the object file.
  MachObjectWriter MOW(OS);
  MOW.WriteObject(*this);

  OS.flush();
}
