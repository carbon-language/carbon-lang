//===-- DWARFDebugFrame.h - Parsing of .debug_frame -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugFrame.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace dwarf;


/// \brief Abstract frame entry defining the common interface concrete
/// entries implement.
class llvm::FrameEntry {
public:
  enum FrameKind {FK_CIE, FK_FDE};
  FrameEntry(FrameKind K, DataExtractor D, uint64_t Offset, uint64_t Length)
    : Kind(K), Data(D), Offset(Offset), Length(Length) {}

  virtual ~FrameEntry() {
  }

  FrameKind getKind() const { return Kind; }

  virtual void dumpHeader(raw_ostream &OS) const = 0;

protected:
  const FrameKind Kind;

  /// \brief The data stream holding the section from which the entry was
  /// parsed.
  DataExtractor Data;

  /// \brief Offset of this entry in the section.
  uint64_t Offset;

  /// \brief Entry length as specified in DWARF.
  uint64_t Length;
};


/// \brief DWARF Common Information Entry (CIE)
class CIE : public FrameEntry {
public:
  // CIEs (and FDEs) are simply container classes, so the only sensible way to
  // create them is by providing the full parsed contents in the constructor.
  CIE(DataExtractor D, uint64_t Offset, uint64_t Length, uint8_t Version,
      SmallString<8> Augmentation, uint64_t CodeAlignmentFactor,
      int64_t DataAlignmentFactor, uint64_t ReturnAddressRegister)
   : FrameEntry(FK_CIE, D, Offset, Length), Version(Version),
     Augmentation(Augmentation), CodeAlignmentFactor(CodeAlignmentFactor),
     DataAlignmentFactor(DataAlignmentFactor),
     ReturnAddressRegister(ReturnAddressRegister) {}

  ~CIE() {
  }

  void dumpHeader(raw_ostream &OS) const {
    OS << format("%08x %08x %08x CIE",
                 (uint32_t)Offset, (uint32_t)Length, DW_CIE_ID)
       << "\n";
    OS << format("  Version:               %d\n", Version);
    OS << "  Augmentation:          \"" << Augmentation << "\"\n";
    OS << format("  Code alignment factor: %u\n", (uint32_t)CodeAlignmentFactor);
    OS << format("  Data alignment factor: %d\n", (int32_t)DataAlignmentFactor);
    OS << format("  Return address column: %d\n", (int32_t)ReturnAddressRegister);
    OS << "\n";
  }

  static bool classof(const FrameEntry *FE) {
    return FE->getKind() == FK_CIE;
  } 

private:
  /// The following fields are defined in section 6.4.1 of the DWARF standard v3
  uint8_t Version;
  SmallString<8> Augmentation;
  uint64_t CodeAlignmentFactor;
  int64_t DataAlignmentFactor;
  uint64_t ReturnAddressRegister;
};


/// \brief DWARF Frame Description Entry (FDE)
class FDE : public FrameEntry {
public:
  // Each FDE has a CIE it's "linked to". Our FDE contains is constructed with
  // an offset to the CIE (provided by parsing the FDE header). The CIE itself
  // is obtained lazily once it's actually required.
  FDE(DataExtractor D, uint64_t Offset, uint64_t Length,
      int64_t LinkedCIEOffset, uint64_t InitialLocation, uint64_t AddressRange)
   : FrameEntry(FK_FDE, D, Offset, Length), LinkedCIEOffset(LinkedCIEOffset),
     InitialLocation(InitialLocation), AddressRange(AddressRange),
     LinkedCIE(NULL) {}

  ~FDE() {
  }

  void dumpHeader(raw_ostream &OS) const {
    OS << format("%08x %08x %08x FDE ",
                 (uint32_t)Offset, (uint32_t)Length, (int32_t)LinkedCIEOffset);
    OS << format("cie=%08x pc=%08x...%08x\n",
                 (int32_t)LinkedCIEOffset,
                 (uint32_t)InitialLocation,
                 (uint32_t)InitialLocation + (uint32_t)AddressRange);
    OS << "\n";
    if (LinkedCIE) {
      OS << format("%p\n", LinkedCIE);
    }
  }

  static bool classof(const FrameEntry *FE) {
    return FE->getKind() == FK_FDE;
  } 
private:

  /// The following fields are defined in section 6.4.1 of the DWARF standard v3
  uint64_t LinkedCIEOffset;
  uint64_t InitialLocation;
  uint64_t AddressRange;
  CIE *LinkedCIE;
};


DWARFDebugFrame::DWARFDebugFrame() {
}


DWARFDebugFrame::~DWARFDebugFrame() {
  for (EntryVector::iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    delete *I;
  }
}


static void LLVM_ATTRIBUTE_UNUSED dumpDataAux(DataExtractor Data,
                                              uint32_t Offset, int Length) {
  errs() << "DUMP: ";
  for (int i = 0; i < Length; ++i) {
    uint8_t c = Data.getU8(&Offset);
    errs().write_hex(c); errs() << " ";
  }
  errs() << "\n";
}


void DWARFDebugFrame::parse(DataExtractor Data) {
  uint32_t Offset = 0;

  while (Data.isValidOffset(Offset)) {
    uint32_t StartOffset = Offset;

    bool IsDWARF64 = false;
    uint64_t Length = Data.getU32(&Offset);
    uint64_t Id;

    if (Length == UINT32_MAX) {
      // DWARF-64 is distinguished by the first 32 bits of the initial length
      // field being 0xffffffff. Then, the next 64 bits are the actual entry
      // length.
      IsDWARF64 = true;
      Length = Data.getU64(&Offset);
    }

    // At this point, Offset points to the next field after Length.
    // Length is the structure size excluding itself. Compute an offset one
    // past the end of the structure (needed to know how many instructions to
    // read).
    // TODO: For honest DWARF64 support, DataExtractor will have to treat
    //       offset_ptr as uint64_t*
    uint32_t EndStructureOffset = Offset + static_cast<uint32_t>(Length);

    // The Id field's size depends on the DWARF format
    Id = Data.getUnsigned(&Offset, IsDWARF64 ? 8 : 4);
    bool IsCIE = ((IsDWARF64 && Id == DW64_CIE_ID) || Id == DW_CIE_ID);

    if (IsCIE) {
      // Note: this is specifically DWARFv3 CIE header structure. It was
      // changed in DWARFv4.
      uint8_t Version = Data.getU8(&Offset);
      const char *Augmentation = Data.getCStr(&Offset);
      uint64_t CodeAlignmentFactor = Data.getULEB128(&Offset);
      int64_t DataAlignmentFactor = Data.getSLEB128(&Offset);
      uint64_t ReturnAddressRegister = Data.getULEB128(&Offset);

      CIE *NewCIE = new CIE(Data, StartOffset, Length, Version,
                            StringRef(Augmentation), CodeAlignmentFactor,
                            DataAlignmentFactor, ReturnAddressRegister);
      Entries.push_back(NewCIE);
    } else {
      // FDE
      uint64_t CIEPointer = Id;
      uint64_t InitialLocation = Data.getAddress(&Offset);
      uint64_t AddressRange = Data.getAddress(&Offset);

      FDE *NewFDE = new FDE(Data, StartOffset, Length, CIEPointer,
                            InitialLocation, AddressRange);
      Entries.push_back(NewFDE);
    }

    Offset = EndStructureOffset;
  }
}


void DWARFDebugFrame::dump(raw_ostream &OS) const {
  OS << "\n";
  for (EntryVector::const_iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    (*I)->dumpHeader(OS);
  }
}

