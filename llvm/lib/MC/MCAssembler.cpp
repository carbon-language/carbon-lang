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
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachOWriterInfo.h"

using namespace llvm;

namespace {

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

  enum HeaderFileType {
    HFT_Object = 0x1
  };

  enum LoadCommandType {
    LCT_Segment = 0x1
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
  
  static unsigned getPrologSize32(unsigned NumSections) {
    return Header32Size + SegmentLoadCommand32Size + 
      NumSections * Section32Size;
  }

  void WriteHeader32(unsigned NumSections) {
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
    Write32(1);
    Write32(SegmentLoadCommand32Size + NumSections * Section32Size);
    Write32(0); // Flags

    assert(OS.tell() - Start == Header32Size);
  }

  void WriteLoadCommandHeader(uint32_t Cmd, uint32_t CmdSize) {
    assert((CmdSize & 0x3) == 0 && "Invalid size!");

    Write32(Cmd);
    Write32(CmdSize);
  }

  /// WriteSegmentLoadCommand32 - Write a 32-bit segment load command.
  ///
  /// \arg NumSections - The number of sections in this segment.
  /// \arg SectionDataSize - The total size of the sections.
  void WriteSegmentLoadCommand32(unsigned NumSections,
                                 uint64_t SectionDataSize) {
    // struct segment_command (56 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(LCT_Segment);
    Write32(SegmentLoadCommand32Size + NumSections * Section32Size);

    WriteString("", 16);
    Write32(0); // vmaddr
    Write32(SectionDataSize); // vmsize
    Write32(Header32Size + SegmentLoadCommand32Size + 
            NumSections * Section32Size); // file offset
    Write32(SectionDataSize); // file size
    Write32(0x7); // maxprot
    Write32(0x7); // initprot
    Write32(NumSections);
    Write32(0); // flags

    assert(OS.tell() - Start == SegmentLoadCommand32Size);
  }

  void WriteSection32(const MCSectionData &SD) {
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
    Write32(SD.getFileOffset());

    assert(isPowerOf2_32(SD.getAlignment()) && "Invalid alignment!");
    Write32(Log2_32(SD.getAlignment()));
    Write32(0); // file offset of relocation entries
    Write32(0); // number of relocation entrions
    Write32(Section.getTypeAndAttributes());
    Write32(0); // reserved1
    Write32(Section.getStubSize()); // reserved2

    assert(OS.tell() - Start == Section32Size);
  }
};

}

/* *** */

MCFragment::MCFragment() : Kind(FragmentType(~0)) {
}

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *SD)
  : Kind(_Kind),
    FileOffset(~UINT64_C(0)),
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
    FileOffset(~UINT64_C(0)),
    FileSize(~UINT64_C(0))
{
  if (A)
    A->getSectionList().push_back(this);
}

/* *** */

MCAssembler::MCAssembler(raw_ostream &_OS) : OS(_OS) {}

MCAssembler::~MCAssembler() {
}

void MCAssembler::LayoutSection(MCSectionData &SD) {
  uint64_t Offset = SD.getFileOffset();

  for (MCSectionData::iterator it = SD.begin(), ie = SD.end(); it != ie; ++it) {
    MCFragment &F = *it;
    F.setFileOffset(Offset);
    F.setFileSize(F.getMaxFileSize());
    Offset += F.getFileSize();
  }

  // FIXME: Pad section?
  SD.setFileSize(Offset - SD.getFileOffset());
}

/// WriteFileData - Write the \arg F data to the output file.
static void WriteFileData(raw_ostream &OS, const MCFragment &F,
                          MachObjectWriter &MOW) {
  uint64_t Start = OS.tell();
  (void) Start;
    
  // FIXME: Embed in fragments instead?
  switch (F.getKind()) {
  default:
    assert(0 && "Invalid section kind!");

  case MCFragment::FT_Data:
    OS << cast<MCDataFragment>(F).getContents().str();
    break;

  case MCFragment::FT_Align:
    llvm_unreachable("FIXME: Not yet implemented!");

  case MCFragment::FT_Fill: {
    MCFillFragment &FF = cast<MCFillFragment>(F);

    if (!FF.getValue().isAbsolute())
      llvm_unreachable("FIXME: Not yet implemented!");

    for (uint64_t i = 0, e = FF.getCount(); i != e; ++i) {
      switch (FF.getValueSize()) {
      default:
        assert(0 && "Invalid size!");
      case 1: MOW.Write8 (uint8_t (FF.getValue().getConstant())); break;
      case 2: MOW.Write16(uint16_t(FF.getValue().getConstant())); break;
      case 4: MOW.Write32(uint32_t(FF.getValue().getConstant())); break;
      case 8: MOW.Write64(uint64_t(FF.getValue().getConstant())); break;
      }
    }
    break;
  }
    
  case MCFragment::FT_Org:
    llvm_unreachable("FIXME: Not yet implemented!");
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
  unsigned NumSections = Sections.size();

  // Layout the sections and fragments.
  uint64_t Offset = MachObjectWriter::getPrologSize32(NumSections);
  uint64_t SectionDataSize = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    it->setFileOffset(Offset);

    LayoutSection(*it);

    Offset += it->getFileSize();
    SectionDataSize += it->getFileSize();
  }

  MachObjectWriter MOW(OS);

  // Write the prolog, starting with the header and load command...
  MOW.WriteHeader32(NumSections);
  MOW.WriteSegmentLoadCommand32(NumSections, SectionDataSize);
  
  // ... and then the section headers.
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    MOW.WriteSection32(*it);

  // Finally, write the section data.
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    WriteFileData(OS, *it, MOW);

  OS.flush();
}
