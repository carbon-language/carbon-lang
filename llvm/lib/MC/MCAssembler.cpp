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

  void Write32(uint32_t Value) {
    if (IsLSB) {
      OS << char(Value >> 0);
      OS << char(Value >> 8);
      OS << char(Value >> 16);
      OS << char(Value >> 24);
    } else {
      OS << char(Value >> 24);
      OS << char(Value >> 16);
      OS << char(Value >> 8);
      OS << char(Value >> 0);
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

  void WriteSegmentLoadCommand32(unsigned NumSections) {
    // struct segment_command (56 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(LCT_Segment);
    Write32(SegmentLoadCommand32Size + NumSections * Section32Size);

    WriteString("", 16);
    Write32(0); // vmaddr
    Write32(0); // vmsize
    Write32(Header32Size + SegmentLoadCommand32Size + 
            NumSections * Section32Size); // file offset
    Write32(0); // file size
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

MCFragment::MCFragment(MCSectionData *SD)
{
  if (SD)
    SD->getFragmentList().push_back(this);
}

/* *** */

MCSectionData::MCSectionData() : Section(*(MCSection*)0) {}

MCSectionData::MCSectionData(const MCSection &_Section, MCAssembler *A)
  : Section(_Section),
    Alignment(1),
    FileOffset(0),
    FileSize(0)
{
  if (A)
    A->getSectionList().push_back(this);
}

void MCSectionData::WriteFileData(raw_ostream &OS) const {
  
}

/* *** */

MCAssembler::MCAssembler(raw_ostream &_OS) : OS(_OS) {}

MCAssembler::~MCAssembler() {
}

void MCAssembler::Finish() {
  unsigned NumSections = Sections.size();
  
  // Compute the file offsets so we can write in a single pass.
  uint64_t Offset = MachObjectWriter::getPrologSize32(NumSections);
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    it->setFileOffset(Offset);
    Offset += it->getFileSize();
  }

  MachObjectWriter MOW(OS);

  // Write the prolog, starting with the header and load command...
  MOW.WriteHeader32(NumSections);
  MOW.WriteSegmentLoadCommand32(NumSections);
  
  // ... and then the section headers.
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    MOW.WriteSection32(*it);

  // Finally, write the section data.
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    it->WriteFileData(OS);
  
  OS.flush();
}
