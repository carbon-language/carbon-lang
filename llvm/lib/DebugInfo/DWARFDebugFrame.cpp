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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

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
  virtual uint64_t getOffset() const { return Offset; }

  /// \brief Parse and store a sequence of CFI instructions from our data
  /// stream, starting at *Offset and ending at EndOffset. If everything
  /// goes well, *Offset should be equal to EndOffset when this method
  /// returns. Otherwise, an error occurred.
  virtual void parseInstructions(uint32_t *Offset, uint32_t EndOffset);

  /// \brief Dump the entry header to the given output stream.
  virtual void dumpHeader(raw_ostream &OS) const = 0;

  /// \brief Dump the entry's instructions to the given output stream.
  virtual void dumpInstructions(raw_ostream &OS) const;

protected:
  const FrameKind Kind;

  /// \brief The data stream holding the section from which the entry was
  /// parsed.
  DataExtractor Data;

  /// \brief Offset of this entry in the section.
  uint64_t Offset;

  /// \brief Entry length as specified in DWARF.
  uint64_t Length;

  /// An entry may contain CFI instructions. An instruction consists of an
  /// opcode and an optional sequence of operands.
  typedef std::vector<uint64_t> Operands;
  struct Instruction {
    Instruction(uint8_t Opcode)
      : Opcode(Opcode)
    {}

    uint8_t Opcode;
    Operands Ops;
  };

  std::vector<Instruction> Instructions;

  /// Convenience methods to add a new instruction with the given opcode and
  /// operands to the Instructions vector.
  void addInstruction(uint8_t Opcode) {
    Instructions.push_back(Instruction(Opcode));
  }

  void addInstruction(uint8_t Opcode, uint64_t Operand1) {
    Instructions.push_back(Instruction(Opcode));
    Instructions.back().Ops.push_back(Operand1);
  }

  void addInstruction(uint8_t Opcode, uint64_t Operand1, uint64_t Operand2) {
    Instructions.push_back(Instruction(Opcode));
    Instructions.back().Ops.push_back(Operand1);
    Instructions.back().Ops.push_back(Operand2);
  }
};


// See DWARF standard v3, section 7.23
const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;
const uint8_t DWARF_CFI_PRIMARY_OPERAND_MASK = 0x3f;


void FrameEntry::parseInstructions(uint32_t *Offset, uint32_t EndOffset) {
  while (*Offset < EndOffset) {
    uint8_t Opcode = Data.getU8(Offset);
    // Some instructions have a primary opcode encoded in the top bits.
    uint8_t Primary = Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK;

    if (Primary) {
      // If it's a primary opcode, the first operand is encoded in the bottom
      // bits of the opcode itself.
      uint64_t Op1 = Opcode & DWARF_CFI_PRIMARY_OPERAND_MASK;
      switch (Primary) {
        default: llvm_unreachable("Impossible primary CFI opcode");
        case DW_CFA_advance_loc:
        case DW_CFA_restore:
          addInstruction(Primary, Op1);
          break;
        case DW_CFA_offset:
          addInstruction(Primary, Op1, Data.getULEB128(Offset));
          break;
      }
    } else {
      // Extended opcode - its value is Opcode itself.
      switch (Opcode) {
        default: llvm_unreachable("Invalid extended CFI opcode");
        case DW_CFA_nop:
        case DW_CFA_remember_state:
        case DW_CFA_restore_state:
        case DW_CFA_GNU_window_save:
          // No operands
          addInstruction(Opcode);
          break;
        case DW_CFA_set_loc:
          // Operands: Address
          addInstruction(Opcode, Data.getAddress(Offset));
          break;
        case DW_CFA_advance_loc1:
          // Operands: 1-byte delta
          addInstruction(Opcode, Data.getU8(Offset));
          break;
        case DW_CFA_advance_loc2:
          // Operands: 2-byte delta
          addInstruction(Opcode, Data.getU16(Offset));
          break;
        case DW_CFA_advance_loc4:
          // Operands: 4-byte delta
          addInstruction(Opcode, Data.getU32(Offset));
          break;
        case DW_CFA_restore_extended:
        case DW_CFA_undefined:
        case DW_CFA_same_value:
        case DW_CFA_def_cfa_register:
        case DW_CFA_def_cfa_offset:
          // Operands: ULEB128
          addInstruction(Opcode, Data.getULEB128(Offset));
          break;
        case DW_CFA_def_cfa_offset_sf:
          // Operands: SLEB128
          addInstruction(Opcode, Data.getSLEB128(Offset));
          break;
        case DW_CFA_offset_extended:
        case DW_CFA_register:
        case DW_CFA_def_cfa:
        case DW_CFA_val_offset:
          // Operands: ULEB128, ULEB128
          addInstruction(Opcode, Data.getULEB128(Offset),
                                 Data.getULEB128(Offset));
          break;
        case DW_CFA_offset_extended_sf:
        case DW_CFA_def_cfa_sf:
        case DW_CFA_val_offset_sf:
          // Operands: ULEB128, SLEB128
          addInstruction(Opcode, Data.getULEB128(Offset),
                                 Data.getSLEB128(Offset));
          break;
        case DW_CFA_def_cfa_expression:
        case DW_CFA_expression:
        case DW_CFA_val_expression:
          // TODO: implement this
          report_fatal_error("Values with expressions not implemented yet!");
      }
    }
  }
}


void FrameEntry::dumpInstructions(raw_ostream &OS) const {
  // TODO: at the moment only instruction names are dumped. Expand this to
  // dump operands as well.
  for (std::vector<Instruction>::const_iterator I = Instructions.begin(),
                                                E = Instructions.end();
       I != E; ++I) {
    uint8_t Opcode = I->Opcode;
    if (Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK)
      Opcode &= DWARF_CFI_PRIMARY_OPCODE_MASK;
    OS << "  " << CallFrameString(Opcode) << ":\n";
  }
}


namespace {
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

  void dumpHeader(raw_ostream &OS) const override {
    OS << format("%08x %08x %08x CIE",
                 (uint32_t)Offset, (uint32_t)Length, DW_CIE_ID)
       << "\n";
    OS << format("  Version:               %d\n", Version);
    OS << "  Augmentation:          \"" << Augmentation << "\"\n";
    OS << format("  Code alignment factor: %u\n",
                 (uint32_t)CodeAlignmentFactor);
    OS << format("  Data alignment factor: %d\n",
                 (int32_t)DataAlignmentFactor);
    OS << format("  Return address column: %d\n",
                 (int32_t)ReturnAddressRegister);
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

  void dumpHeader(raw_ostream &OS) const override {
    OS << format("%08x %08x %08x FDE ",
                 (uint32_t)Offset, (uint32_t)Length, (int32_t)LinkedCIEOffset);
    OS << format("cie=%08x pc=%08x...%08x\n",
                 (int32_t)LinkedCIEOffset,
                 (uint32_t)InitialLocation,
                 (uint32_t)InitialLocation + (uint32_t)AddressRange);
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
} // end anonymous namespace


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

    FrameEntry *Entry = 0;
    if (IsCIE) {
      // Note: this is specifically DWARFv3 CIE header structure. It was
      // changed in DWARFv4. We currently don't support reading DWARFv4
      // here because LLVM itself does not emit it (and LLDB doesn't
      // support it either).
      uint8_t Version = Data.getU8(&Offset);
      const char *Augmentation = Data.getCStr(&Offset);
      uint64_t CodeAlignmentFactor = Data.getULEB128(&Offset);
      int64_t DataAlignmentFactor = Data.getSLEB128(&Offset);
      uint64_t ReturnAddressRegister = Data.getULEB128(&Offset);

      Entry = new CIE(Data, StartOffset, Length, Version,
                      StringRef(Augmentation), CodeAlignmentFactor,
                      DataAlignmentFactor, ReturnAddressRegister);
    } else {
      // FDE
      uint64_t CIEPointer = Id;
      uint64_t InitialLocation = Data.getAddress(&Offset);
      uint64_t AddressRange = Data.getAddress(&Offset);

      Entry = new FDE(Data, StartOffset, Length, CIEPointer,
                      InitialLocation, AddressRange);
    }

    assert(Entry && "Expected Entry to be populated with CIE or FDE");
    Entry->parseInstructions(&Offset, EndStructureOffset);

    if (Offset == EndStructureOffset) {
      // Entry instrucitons parsed successfully.
      Entries.push_back(Entry);
    } else {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << format("Parsing entry instructions at %lx failed",
                   Entry->getOffset());
      report_fatal_error(Str);
    }
  }
}


void DWARFDebugFrame::dump(raw_ostream &OS) const {
  OS << "\n";
  for (EntryVector::const_iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    FrameEntry *Entry = *I;
    Entry->dumpHeader(OS);
    Entry->dumpInstructions(OS);
    OS << "\n";
  }
}

