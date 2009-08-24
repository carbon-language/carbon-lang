//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Target/TargetMachOWriterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace llvm;

class MachObjectWriter;

static void WriteFileData(raw_ostream &OS, const MCSectionData &SD,
                          MachObjectWriter &MOW);

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

  enum HeaderFileType {
    HFT_Object = 0x1
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

  void WriteString(const StringRef &Str, unsigned ZeroFillSize = 0) {
    OS << Str;
    if (ZeroFillSize)
      WriteZeros(ZeroFillSize - Str.size());
  }

  /// @}
  
  void WriteHeader32(unsigned NumLoadCommands, unsigned LoadCommandsSize) {
    // struct mach_header (28 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(Header_Magic32);

    // FIXME: Support cputype.
    Write32(TargetMachOWriterInfo::HDR_CPU_TYPE_I386);

    // FIXME: Support cpusubtype.
    Write32(TargetMachOWriterInfo::HDR_CPU_SUBTYPE_I386_ALL);

    Write32(HFT_Object);

    // Object files have a single load command, the segment.
    Write32(NumLoadCommands);
    Write32(LoadCommandsSize);
    Write32(0); // Flags

    assert(OS.tell() - Start == Header32Size);
  }

  /// WriteSegmentLoadCommand32 - Write a 32-bit segment load command.
  ///
  /// \arg NumSections - The number of sections in this segment.
  /// \arg SectionDataSize - The total size of the sections.
  void WriteSegmentLoadCommand32(unsigned NumSections,
                                 uint64_t SectionDataStartOffset,
                                 uint64_t SectionDataSize) {
    // struct segment_command (56 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(LCT_Segment);
    Write32(SegmentLoadCommand32Size + NumSections * Section32Size);

    WriteString("", 16);
    Write32(0); // vmaddr
    Write32(SectionDataSize); // vmsize
    Write32(SectionDataStartOffset); // file offset
    Write32(SectionDataSize); // file size
    Write32(0x7); // maxprot
    Write32(0x7); // initprot
    Write32(NumSections);
    Write32(0); // flags

    assert(OS.tell() - Start == SegmentLoadCommand32Size);
  }

  void WriteSection32(const MCSectionData &SD, uint64_t FileOffset) {
    // struct section (68 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    // FIXME: cast<> support!
    const MCSectionMachO &Section =
      static_cast<const MCSectionMachO&>(SD.getSection());
    WriteString(Section.getSectionName(), 16);
    WriteString(Section.getSegmentName(), 16);
    Write32(0); // address
    Write32(SD.getFileSize()); // size
    Write32(FileOffset);

    assert(isPowerOf2_32(SD.getAlignment()) && "Invalid alignment!");
    Write32(Log2_32(SD.getAlignment()));
    Write32(0); // file offset of relocation entries
    Write32(0); // number of relocation entrions
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
    MCSymbol &Symbol = MSD.SymbolData->getSymbol();
    uint8_t Type = 0;

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

    // FIXME: Set private external bit.

    // Set external bit.
    if (MSD.SymbolData->isExternal())
      Type |= STF_External;

    // struct nlist (12 bytes)

    Write32(MSD.StringIndex);
    Write8(Type);
    Write8(MSD.SectionIndex);
    Write16(0); // FIXME: Desc
    Write32(0); // FIXME: Value
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
      MCSymbol &Symbol = it->getSymbol();

      if (!it->isExternal())
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
      MCSymbol &Symbol = it->getSymbol();

      if (it->isExternal())
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

      assert(!Symbol.isUndefined() && "Local symbol can not be undefined!");
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

    // The string table is padded to a multiple of 4.
    //
    // FIXME: Check to see if this varies per arch.
    while (StringTable.size() % 4)
      StringTable += '\x00';
  }

  void WriteObject(MCAssembler &Asm) {
    unsigned NumSections = Asm.size();

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

    // Compute the file offsets for all the sections in advance, so that we can
    // write things out in order.
    SmallVector<uint64_t, 16> SectionFileOffsets;
    SectionFileOffsets.resize(NumSections);
  
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

    uint64_t FileOffset = Header32Size + LoadCommandsSize;
    uint64_t SectionDataStartOffset = FileOffset;
    uint64_t SectionDataSize = 0;
    unsigned Index = 0;
    for (MCAssembler::iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it, ++Index) {
      SectionFileOffsets[Index] = FileOffset;
      FileOffset += it->getFileSize();
      SectionDataSize += it->getFileSize();
    }

    // Write the prolog, starting with the header and load command...
    WriteHeader32(NumLoadCommands, LoadCommandsSize);
    WriteSegmentLoadCommand32(NumSections, SectionDataStartOffset,
                              SectionDataSize);
  
    // ... and then the section headers.
    Index = 0;
    for (MCAssembler::iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it, ++Index)
      WriteSection32(*it, SectionFileOffsets[Index]);

    // Write the symbol table load command, if used.
    if (NumSymbols) {
      // The string table is written after all the section data.
      uint64_t SymbolTableOffset = SectionDataStartOffset + SectionDataSize;
      uint64_t StringTableOffset =
        SymbolTableOffset + NumSymbols * Nlist32Size;
      WriteSymtabLoadCommand(SymbolTableOffset, NumSymbols,
                             StringTableOffset, StringTable.size());

      unsigned FirstLocalSymbol = 0;
      unsigned NumLocalSymbols = LocalSymbolData.size();
      unsigned FirstExternalSymbol = FirstLocalSymbol + NumLocalSymbols;
      unsigned NumExternalSymbols = ExternalSymbolData.size();
      unsigned FirstUndefinedSymbol = FirstExternalSymbol + NumExternalSymbols;
      unsigned NumUndefinedSymbols = UndefinedSymbolData.size();
      // FIXME: Get correct symbol indices and counts for indirect symbols.
      unsigned IndirectSymbolOffset = 0;
      unsigned NumIndirectSymbols = 0;
      WriteDysymtabLoadCommand(FirstLocalSymbol, NumLocalSymbols,
                               FirstExternalSymbol, NumExternalSymbols,
                               FirstUndefinedSymbol, NumUndefinedSymbols,
                               IndirectSymbolOffset, NumIndirectSymbols);
    }

    // Write the actual section data.
    for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
      WriteFileData(OS, *it, *this);

    // Write the symbol table data, if used.
    if (NumSymbols) {
      // FIXME: Check that offsets match computed ones.

      // FIXME: Some of these are ordered by name to help the linker.

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

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *SD)
  : Kind(_Kind),
    FileSize(~UINT64_C(0))
{
  if (SD)
    SD->getFragmentList().push_back(this);
}

MCFragment::~MCFragment() {
}

/* *** */

MCSectionData::MCSectionData() : Section(*(MCSection*)0) {}

MCSectionData::MCSectionData(const MCSection &_Section, MCAssembler *A)
  : Section(_Section),
    Alignment(1),
    FileSize(~UINT64_C(0))
{
  if (A)
    A->getSectionList().push_back(this);
}

/* *** */

MCSymbolData::MCSymbolData() : Symbol(*(MCSymbol*)0) {}

MCSymbolData::MCSymbolData(MCSymbol &_Symbol, MCFragment *_Fragment,
                           uint64_t _Offset, MCAssembler *A)
  : Symbol(_Symbol), Fragment(_Fragment), Offset(_Offset),
    IsExternal(false)
{
  if (A)
    A->getSymbolList().push_back(this);
}

/* *** */

MCAssembler::MCAssembler(raw_ostream &_OS) : OS(_OS) {}

MCAssembler::~MCAssembler() {
}

void MCAssembler::LayoutSection(MCSectionData &SD) {
  uint64_t Offset = 0;

  for (MCSectionData::iterator it = SD.begin(), ie = SD.end(); it != ie; ++it) {
    MCFragment &F = *it;

    F.setOffset(Offset);

    // Evaluate fragment size.
    switch (F.getKind()) {
    case MCFragment::FT_Align: {
      MCAlignFragment &AF = cast<MCAlignFragment>(F);
      
      uint64_t AlignedOffset = RoundUpToAlignment(Offset, AF.getAlignment());
      uint64_t PaddingBytes = AlignedOffset - Offset;

      if (PaddingBytes > AF.getMaxBytesToEmit())
        AF.setFileSize(0);
      else
        AF.setFileSize(PaddingBytes);
      break;
    }

    case MCFragment::FT_Data:
    case MCFragment::FT_Fill:
      F.setFileSize(F.getMaxFileSize());
      break;

    case MCFragment::FT_Org: {
      MCOrgFragment &OF = cast<MCOrgFragment>(F);

      if (!OF.getOffset().isAbsolute())
        llvm_unreachable("FIXME: Not yet implemented!");
      uint64_t OrgOffset = OF.getOffset().getConstant();

      // FIXME: We need a way to communicate this error.
      if (OrgOffset < Offset)
        llvm_report_error("invalid .org offset '" + Twine(OrgOffset) + 
                          "' (section offset '" + Twine(Offset) + "'");
        
      F.setFileSize(OrgOffset - Offset);
      break;
    }      
    }

    Offset += F.getFileSize();
  }

  // FIXME: Pad section?
  SD.setFileSize(Offset);
}

/// WriteFileData - Write the \arg F data to the output file.
static void WriteFileData(raw_ostream &OS, const MCFragment &F,
                          MachObjectWriter &MOW) {
  uint64_t Start = OS.tell();
  (void) Start;
    
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

    if (!FF.getValue().isAbsolute())
      llvm_unreachable("FIXME: Not yet implemented!");
    int64_t Value = FF.getValue().getConstant();

    for (uint64_t i = 0, e = FF.getCount(); i != e; ++i) {
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
  }

  assert(OS.tell() - Start == F.getFileSize());
}

/// WriteFileData - Write the \arg SD data to the output file.
static void WriteFileData(raw_ostream &OS, const MCSectionData &SD,
                          MachObjectWriter &MOW) {
  uint64_t Start = OS.tell();
  (void) Start;
      
  for (MCSectionData::const_iterator it = SD.begin(),
         ie = SD.end(); it != ie; ++it)
    WriteFileData(OS, *it, MOW);

  assert(OS.tell() - Start == SD.getFileSize());
}

void MCAssembler::Finish() {
  // Layout the sections and fragments.
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    LayoutSection(*it);

  // Write the object file.
  MachObjectWriter MOW(OS);
  MOW.WriteObject(*this);

  OS.flush();
}
