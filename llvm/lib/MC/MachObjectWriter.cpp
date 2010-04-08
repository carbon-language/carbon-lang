//===- lib/MC/MachObjectWriter.cpp - Mach-O File Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MachObjectWriter.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachO.h"
#include "llvm/Target/TargetAsmBackend.h"

// FIXME: Gross.
#include "../Target/X86/X86FixupKinds.h"

#include <vector>
using namespace llvm;

static unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  default: llvm_unreachable("invalid fixup kind!");
  case X86::reloc_pcrel_1byte:
  case FK_Data_1: return 0;
  case FK_Data_2: return 1;
  case X86::reloc_pcrel_4byte:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
  case FK_Data_4: return 2;
  case FK_Data_8: return 3;
  }
}

static bool isFixupKindPCRel(unsigned Kind) {
  switch (Kind) {
  default:
    return false;
  case X86::reloc_pcrel_1byte:
  case X86::reloc_pcrel_4byte:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
    return true;
  }
}

static bool isFixupKindRIPRel(unsigned Kind) {
  return Kind == X86::reloc_riprel_4byte ||
    Kind == X86::reloc_riprel_4byte_movq_load;
}

namespace {

class MachObjectWriterImpl {
  // See <mach-o/loader.h>.
  enum {
    Header_Magic32 = 0xFEEDFACE,
    Header_Magic64 = 0xFEEDFACF
  };

  enum {
    Header32Size = 28,
    Header64Size = 32,
    SegmentLoadCommand32Size = 56,
    SegmentLoadCommand64Size = 72,
    Section32Size = 68,
    Section64Size = 80,
    SymtabLoadCommandSize = 24,
    DysymtabLoadCommandSize = 80,
    Nlist32Size = 12,
    Nlist64Size = 16,
    RelocationInfoSize = 8
  };

  enum HeaderFileType {
    HFT_Object = 0x1
  };

  enum HeaderFlags {
    HF_SubsectionsViaSymbols = 0x2000
  };

  enum LoadCommandType {
    LCT_Segment = 0x1,
    LCT_Symtab = 0x2,
    LCT_Dysymtab = 0xb,
    LCT_Segment64 = 0x19
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

  /// X86_64 uses its own relocation types.
  enum RelocationInfoTypeX86_64 {
    RIT_X86_64_Unsigned   = 0,
    RIT_X86_64_Signed     = 1,
    RIT_X86_64_Branch     = 2,
    RIT_X86_64_GOTLoad    = 3,
    RIT_X86_64_GOT        = 4,
    RIT_X86_64_Subtractor = 5,
    RIT_X86_64_Signed1    = 6,
    RIT_X86_64_Signed2    = 7,
    RIT_X86_64_Signed4    = 8
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

  /// @name Relocation Data
  /// @{

  struct MachRelocationEntry {
    uint32_t Word0;
    uint32_t Word1;
  };

  llvm::DenseMap<const MCSectionData*,
                 std::vector<MachRelocationEntry> > Relocations;

  /// @}
  /// @name Symbol Table Data
  /// @{

  SmallString<256> StringTable;
  std::vector<MachSymbolData> LocalSymbolData;
  std::vector<MachSymbolData> ExternalSymbolData;
  std::vector<MachSymbolData> UndefinedSymbolData;

  /// @}

  MachObjectWriter *Writer;

  raw_ostream &OS;

  unsigned Is64Bit : 1;

public:
  MachObjectWriterImpl(MachObjectWriter *_Writer, bool _Is64Bit)
    : Writer(_Writer), OS(Writer->getStream()), Is64Bit(_Is64Bit) {
  }

  void Write8(uint8_t Value) { Writer->Write8(Value); }
  void Write16(uint16_t Value) { Writer->Write16(Value); }
  void Write32(uint32_t Value) { Writer->Write32(Value); }
  void Write64(uint64_t Value) { Writer->Write64(Value); }
  void WriteZeros(unsigned N) { Writer->WriteZeros(N); }
  void WriteBytes(StringRef Str, unsigned ZeroFillSize = 0) {
    Writer->WriteBytes(Str, ZeroFillSize);
  }

  void WriteHeader(unsigned NumLoadCommands, unsigned LoadCommandsSize,
                   bool SubsectionsViaSymbols) {
    uint32_t Flags = 0;

    if (SubsectionsViaSymbols)
      Flags |= HF_SubsectionsViaSymbols;

    // struct mach_header (28 bytes) or
    // struct mach_header_64 (32 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    Write32(Is64Bit ? Header_Magic64 : Header_Magic32);

    // FIXME: Support cputype.
    Write32(Is64Bit ? MachO::CPUTypeX86_64 : MachO::CPUTypeI386);
    // FIXME: Support cpusubtype.
    Write32(MachO::CPUSubType_I386_ALL);
    Write32(HFT_Object);
    Write32(NumLoadCommands);    // Object files have a single load command, the
                                 // segment.
    Write32(LoadCommandsSize);
    Write32(Flags);
    if (Is64Bit)
      Write32(0); // reserved

    assert(OS.tell() - Start == Is64Bit ? Header64Size : Header32Size);
  }

  /// WriteSegmentLoadCommand - Write a segment load command.
  ///
  /// \arg NumSections - The number of sections in this segment.
  /// \arg SectionDataSize - The total size of the sections.
  void WriteSegmentLoadCommand(unsigned NumSections,
                               uint64_t VMSize,
                               uint64_t SectionDataStartOffset,
                               uint64_t SectionDataSize) {
    // struct segment_command (56 bytes) or
    // struct segment_command_64 (72 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    unsigned SegmentLoadCommandSize = Is64Bit ? SegmentLoadCommand64Size :
      SegmentLoadCommand32Size;
    Write32(Is64Bit ? LCT_Segment64 : LCT_Segment);
    Write32(SegmentLoadCommandSize +
            NumSections * (Is64Bit ? Section64Size : Section32Size));

    WriteBytes("", 16);
    if (Is64Bit) {
      Write64(0); // vmaddr
      Write64(VMSize); // vmsize
      Write64(SectionDataStartOffset); // file offset
      Write64(SectionDataSize); // file size
    } else {
      Write32(0); // vmaddr
      Write32(VMSize); // vmsize
      Write32(SectionDataStartOffset); // file offset
      Write32(SectionDataSize); // file size
    }
    Write32(0x7); // maxprot
    Write32(0x7); // initprot
    Write32(NumSections);
    Write32(0); // flags

    assert(OS.tell() - Start == SegmentLoadCommandSize);
  }

  void WriteSection(const MCAssembler &Asm, const MCAsmLayout &Layout,
                    const MCSectionData &SD, uint64_t FileOffset,
                    uint64_t RelocationsStart, unsigned NumRelocations) {
    uint64_t SectionSize = Layout.getSectionSize(&SD);

    // The offset is unused for virtual sections.
    if (Asm.getBackend().isVirtualSection(SD.getSection())) {
      assert(Layout.getSectionFileSize(&SD) == 0 && "Invalid file size!");
      FileOffset = 0;
    }

    // struct section (68 bytes) or
    // struct section_64 (80 bytes)

    uint64_t Start = OS.tell();
    (void) Start;

    // FIXME: cast<> support!
    const MCSectionMachO &Section =
      static_cast<const MCSectionMachO&>(SD.getSection());
    WriteBytes(Section.getSectionName(), 16);
    WriteBytes(Section.getSegmentName(), 16);
    if (Is64Bit) {
      Write64(Layout.getSectionAddress(&SD)); // address
      Write64(SectionSize); // size
    } else {
      Write32(Layout.getSectionAddress(&SD)); // address
      Write32(SectionSize); // size
    }
    Write32(FileOffset);

    unsigned Flags = Section.getTypeAndAttributes();
    if (SD.hasInstructions())
      Flags |= MCSectionMachO::S_ATTR_SOME_INSTRUCTIONS;

    assert(isPowerOf2_32(SD.getAlignment()) && "Invalid alignment!");
    Write32(Log2_32(SD.getAlignment()));
    Write32(NumRelocations ? RelocationsStart : 0);
    Write32(NumRelocations);
    Write32(Flags);
    Write32(0); // reserved1
    Write32(Section.getStubSize()); // reserved2
    if (Is64Bit)
      Write32(0); // reserved3

    assert(OS.tell() - Start == Is64Bit ? Section64Size : Section32Size);
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

  void WriteNlist(MachSymbolData &MSD, const MCAsmLayout &Layout) {
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
        Address = Layout.getSymbolAddress(&Data);
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
          report_fatal_error("invalid 'common' alignment '" +
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
    if (Is64Bit)
      Write64(Address);
    else
      Write32(Address);
  }

  // FIXME: We really need to improve the relocation validation. Basically, we
  // want to implement a separate computation which evaluates the relocation
  // entry as the linker would, and verifies that the resultant fixup value is
  // exactly what the encoder wanted. This will catch several classes of
  // problems:
  //
  //  - Relocation entry bugs, the two algorithms are unlikely to have the same
  //    exact bug.
  //
  //  - Relaxation issues, where we forget to relax something.
  //
  //  - Input errors, where something cannot be correctly encoded. 'as' allows
  //    these through in many cases.

  void RecordX86_64Relocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                              const MCFragment *Fragment,
                              const MCAsmFixup &Fixup, MCValue Target,
                              uint64_t &FixedValue) {
    unsigned IsPCRel = isFixupKindPCRel(Fixup.Kind);
    unsigned IsRIPRel = isFixupKindRIPRel(Fixup.Kind);
    unsigned Log2Size = getFixupKindLog2Size(Fixup.Kind);

    // See <reloc.h>.
    uint32_t Address = Layout.getFragmentOffset(Fragment) + Fixup.Offset;
    int64_t Value = 0;
    unsigned Index = 0;
    unsigned IsExtern = 0;
    unsigned Type = 0;

    Value = Target.getConstant();

    if (IsPCRel) {
      // Compensate for the relocation offset, Darwin x86_64 relocations only
      // have the addend and appear to have attempted to define it to be the
      // actual expression addend without the PCrel bias. However, instructions
      // with data following the relocation are not accomodated for (see comment
      // below regarding SIGNED{1,2,4}), so it isn't exactly that either.
      Value += 1LL << Log2Size;
    }

    if (Target.isAbsolute()) { // constant
      // SymbolNum of 0 indicates the absolute section.
      Type = RIT_X86_64_Unsigned;
      Index = 0;

      // FIXME: I believe this is broken, I don't think the linker can
      // understand it. I think it would require a local relocation, but I'm not
      // sure if that would work either. The official way to get an absolute
      // PCrel relocation is to use an absolute symbol (which we don't support
      // yet).
      if (IsPCRel) {
        IsExtern = 1;
        Type = RIT_X86_64_Branch;
      }
    } else if (Target.getSymB()) { // A - B + constant
      const MCSymbol *A = &Target.getSymA()->getSymbol();
      MCSymbolData &A_SD = Asm.getSymbolData(*A);
      const MCSymbolData *A_Base = Asm.getAtom(Layout, &A_SD);

      const MCSymbol *B = &Target.getSymB()->getSymbol();
      MCSymbolData &B_SD = Asm.getSymbolData(*B);
      const MCSymbolData *B_Base = Asm.getAtom(Layout, &B_SD);

      // Neither symbol can be modified.
      if (Target.getSymA()->getKind() != MCSymbolRefExpr::VK_None ||
          Target.getSymB()->getKind() != MCSymbolRefExpr::VK_None)
        report_fatal_error("unsupported relocation of modified symbol");

      // We don't support PCrel relocations of differences. Darwin 'as' doesn't
      // implement most of these correctly.
      if (IsPCRel)
        report_fatal_error("unsupported pc-relative relocation of difference");

      // We don't currently support any situation where one or both of the
      // symbols would require a local relocation. This is almost certainly
      // unused and may not be possible to encode correctly.
      if (!A_Base || !B_Base)
        report_fatal_error("unsupported local relocations in difference");

      // Darwin 'as' doesn't emit correct relocations for this (it ends up with
      // a single SIGNED relocation); reject it for now.
      if (A_Base == B_Base)
        report_fatal_error("unsupported relocation with identical base");

      Value += Layout.getSymbolAddress(&A_SD) - Layout.getSymbolAddress(A_Base);
      Value -= Layout.getSymbolAddress(&B_SD) - Layout.getSymbolAddress(B_Base);

      Index = A_Base->getIndex();
      IsExtern = 1;
      Type = RIT_X86_64_Unsigned;

      MachRelocationEntry MRE;
      MRE.Word0 = Address;
      MRE.Word1 = ((Index     <<  0) |
                   (IsPCRel   << 24) |
                   (Log2Size  << 25) |
                   (IsExtern  << 27) |
                   (Type      << 28));
      Relocations[Fragment->getParent()].push_back(MRE);

      Index = B_Base->getIndex();
      IsExtern = 1;
      Type = RIT_X86_64_Subtractor;
    } else {
      const MCSymbol *Symbol = &Target.getSymA()->getSymbol();
      MCSymbolData &SD = Asm.getSymbolData(*Symbol);
      const MCSymbolData *Base = Asm.getAtom(Layout, &SD);

      // x86_64 almost always uses external relocations, except when there is no
      // symbol to use as a base address (a local symbol with no preceeding
      // non-local symbol).
      if (Base) {
        Index = Base->getIndex();
        IsExtern = 1;

        // Add the local offset, if needed.
        if (Base != &SD)
          Value += Layout.getSymbolAddress(&SD) - Layout.getSymbolAddress(Base);
      } else {
        // The index is the section ordinal (1-based).
        Index = SD.getFragment()->getParent()->getOrdinal() + 1;
        IsExtern = 0;
        Value += Layout.getSymbolAddress(&SD);

        if (IsPCRel)
          Value -= Address + (1 << Log2Size);
      }

      MCSymbolRefExpr::VariantKind Modifier = Target.getSymA()->getKind();
      if (IsPCRel) {
        if (IsRIPRel) {
          if (Modifier == MCSymbolRefExpr::VK_GOTPCREL) {
            // x86_64 distinguishes movq foo@GOTPCREL so that the linker can
            // rewrite the movq to an leaq at link time if the symbol ends up in
            // the same linkage unit.
            if (unsigned(Fixup.Kind) == X86::reloc_riprel_4byte_movq_load)
              Type = RIT_X86_64_GOTLoad;
            else
              Type = RIT_X86_64_GOT;
          } else if (Modifier != MCSymbolRefExpr::VK_None)
            report_fatal_error("unsupported symbol modifier in relocation");
          else
            Type = RIT_X86_64_Signed;
        } else {
          if (Modifier != MCSymbolRefExpr::VK_None)
            report_fatal_error("unsupported symbol modifier in branch "
                              "relocation");

          Type = RIT_X86_64_Branch;
        }

        // The Darwin x86_64 relocation format has a problem where it cannot
        // encode an address (L<foo> + <constant>) which is outside the atom
        // containing L<foo>. Generally, this shouldn't occur but it does happen
        // when we have a RIPrel instruction with data following the relocation
        // entry (e.g., movb $012, L0(%rip)). Even with the PCrel adjustment
        // Darwin x86_64 uses, the offset is still negative and the linker has
        // no way to recognize this.
        //
        // To work around this, Darwin uses several special relocation types to
        // indicate the offsets. However, the specification or implementation of
        // these seems to also be incomplete; they should adjust the addend as
        // well based on the actual encoded instruction (the additional bias),
        // but instead appear to just look at the final offset.
        if (IsRIPRel) {
          switch (-(Target.getConstant() + (1LL << Log2Size))) {
          case 1: Type = RIT_X86_64_Signed1; break;
          case 2: Type = RIT_X86_64_Signed2; break;
          case 4: Type = RIT_X86_64_Signed4; break;
          }
        }
      } else {
        if (Modifier == MCSymbolRefExpr::VK_GOT) {
          Type = RIT_X86_64_GOT;
        } else if (Modifier == MCSymbolRefExpr::VK_GOTPCREL) {
          // GOTPCREL is allowed as a modifier on non-PCrel instructions, in
          // which case all we do is set the PCrel bit in the relocation entry;
          // this is used with exception handling, for example. The source is
          // required to include any necessary offset directly.
          Type = RIT_X86_64_GOT;
          IsPCRel = 1;
        } else if (Modifier != MCSymbolRefExpr::VK_None)
          report_fatal_error("unsupported symbol modifier in relocation");
        else
          Type = RIT_X86_64_Unsigned;
      }
    }

    // x86_64 always writes custom values into the fixups.
    FixedValue = Value;

    // struct relocation_info (8 bytes)
    MachRelocationEntry MRE;
    MRE.Word0 = Address;
    MRE.Word1 = ((Index     <<  0) |
                 (IsPCRel   << 24) |
                 (Log2Size  << 25) |
                 (IsExtern  << 27) |
                 (Type      << 28));
    Relocations[Fragment->getParent()].push_back(MRE);
  }

  void RecordScatteredRelocation(const MCAssembler &Asm,
                                 const MCAsmLayout &Layout,
                                 const MCFragment *Fragment,
                                 const MCAsmFixup &Fixup, MCValue Target,
                                 uint64_t &FixedValue) {
    uint32_t Address = Layout.getFragmentOffset(Fragment) + Fixup.Offset;
    unsigned IsPCRel = isFixupKindPCRel(Fixup.Kind);
    unsigned Log2Size = getFixupKindLog2Size(Fixup.Kind);
    unsigned Type = RIT_Vanilla;

    // See <reloc.h>.
    const MCSymbol *A = &Target.getSymA()->getSymbol();
    MCSymbolData *A_SD = &Asm.getSymbolData(*A);

    if (!A_SD->getFragment())
      report_fatal_error("symbol '" + A->getName() +
                        "' can not be undefined in a subtraction expression");

    uint32_t Value = Layout.getSymbolAddress(A_SD);
    uint32_t Value2 = 0;

    if (const MCSymbolRefExpr *B = Target.getSymB()) {
      MCSymbolData *B_SD = &Asm.getSymbolData(B->getSymbol());

      if (!B_SD->getFragment())
        report_fatal_error("symbol '" + B->getSymbol().getName() +
                          "' can not be undefined in a subtraction expression");

      // Select the appropriate difference relocation type.
      //
      // Note that there is no longer any semantic difference between these two
      // relocation types from the linkers point of view, this is done solely
      // for pedantic compatibility with 'as'.
      Type = A_SD->isExternal() ? RIT_Difference : RIT_LocalDifference;
      Value2 = Layout.getSymbolAddress(B_SD);
    }

    // Relocations are written out in reverse order, so the PAIR comes first.
    if (Type == RIT_Difference || Type == RIT_LocalDifference) {
      MachRelocationEntry MRE;
      MRE.Word0 = ((0         <<  0) |
                   (RIT_Pair  << 24) |
                   (Log2Size  << 28) |
                   (IsPCRel   << 30) |
                   RF_Scattered);
      MRE.Word1 = Value2;
      Relocations[Fragment->getParent()].push_back(MRE);
    }

    MachRelocationEntry MRE;
    MRE.Word0 = ((Address   <<  0) |
                 (Type      << 24) |
                 (Log2Size  << 28) |
                 (IsPCRel   << 30) |
                 RF_Scattered);
    MRE.Word1 = Value;
    Relocations[Fragment->getParent()].push_back(MRE);
  }

  void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCAsmFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) {
    if (Is64Bit) {
      RecordX86_64Relocation(Asm, Layout, Fragment, Fixup, Target, FixedValue);
      return;
    }

    unsigned IsPCRel = isFixupKindPCRel(Fixup.Kind);
    unsigned Log2Size = getFixupKindLog2Size(Fixup.Kind);

    // If this is a difference or a defined symbol plus an offset, then we need
    // a scattered relocation entry.
    uint32_t Offset = Target.getConstant();
    if (IsPCRel)
      Offset += 1 << Log2Size;
    if (Target.getSymB() ||
        (Target.getSymA() && !Target.getSymA()->getSymbol().isUndefined() &&
         Offset)) {
      RecordScatteredRelocation(Asm, Layout, Fragment, Fixup,Target,FixedValue);
      return;
    }

    // See <reloc.h>.
    uint32_t Address = Layout.getFragmentOffset(Fragment) + Fixup.Offset;
    uint32_t Value = 0;
    unsigned Index = 0;
    unsigned IsExtern = 0;
    unsigned Type = 0;

    if (Target.isAbsolute()) { // constant
      // SymbolNum of 0 indicates the absolute section.
      //
      // FIXME: Currently, these are never generated (see code below). I cannot
      // find a case where they are actually emitted.
      Type = RIT_Vanilla;
      Value = 0;
    } else {
      const MCSymbol *Symbol = &Target.getSymA()->getSymbol();
      MCSymbolData *SD = &Asm.getSymbolData(*Symbol);

      if (Symbol->isUndefined()) {
        IsExtern = 1;
        Index = SD->getIndex();
        Value = 0;
      } else {
        // The index is the section ordinal (1-based).
        Index = SD->getFragment()->getParent()->getOrdinal() + 1;
        Value = Layout.getSymbolAddress(SD);
      }

      Type = RIT_Vanilla;
    }

    // struct relocation_info (8 bytes)
    MachRelocationEntry MRE;
    MRE.Word0 = Address;
    MRE.Word1 = ((Index     <<  0) |
                 (IsPCRel   << 24) |
                 (Log2Size  << 25) |
                 (IsExtern  << 27) |
                 (Type      << 28));
    Relocations[Fragment->getParent()].push_back(MRE);
  }

  void BindIndirectSymbols(MCAssembler &Asm) {
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

      if (Section.getType() != MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS)
        continue;

      Asm.getOrCreateSymbolData(*it->Symbol);
    }

    // Then lazy symbol pointers and symbol stubs.
    for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
           ie = Asm.indirect_symbol_end(); it != ie; ++it) {
      // FIXME: cast<> support!
      const MCSectionMachO &Section =
        static_cast<const MCSectionMachO&>(it->SectionData->getSection());

      if (Section.getType() != MCSectionMachO::S_LAZY_SYMBOL_POINTERS &&
          Section.getType() != MCSectionMachO::S_SYMBOL_STUBS)
        continue;

      // Set the symbol type to undefined lazy, but only on construction.
      //
      // FIXME: Do not hardcode.
      bool Created;
      MCSymbolData &Entry = Asm.getOrCreateSymbolData(*it->Symbol, &Created);
      if (Created)
        Entry.setFlags(Entry.getFlags() | 0x0001);
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

      // Ignore non-linker visible symbols.
      if (!Asm.isSymbolLinkerVisible(it))
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

      // Ignore non-linker visible symbols.
      if (!Asm.isSymbolLinkerVisible(it))
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

  void ExecutePostLayoutBinding(MCAssembler &Asm) {
    // Create symbol data for any indirect symbols.
    BindIndirectSymbols(Asm);

    // Compute symbol table information and bind symbol indices.
    ComputeSymbolTable(Asm, StringTable, LocalSymbolData, ExternalSymbolData,
                       UndefinedSymbolData);
  }

  void WriteObject(const MCAssembler &Asm, const MCAsmLayout &Layout) {
    unsigned NumSections = Asm.size();

    // The section data starts after the header, the segment load command (and
    // section headers) and the symbol table.
    unsigned NumLoadCommands = 1;
    uint64_t LoadCommandsSize = Is64Bit ?
      SegmentLoadCommand64Size + NumSections * Section64Size :
      SegmentLoadCommand32Size + NumSections * Section32Size;

    // Add the symbol table load command sizes, if used.
    unsigned NumSymbols = LocalSymbolData.size() + ExternalSymbolData.size() +
      UndefinedSymbolData.size();
    if (NumSymbols) {
      NumLoadCommands += 2;
      LoadCommandsSize += SymtabLoadCommandSize + DysymtabLoadCommandSize;
    }

    // Compute the total size of the section data, as well as its file size and
    // vm size.
    uint64_t SectionDataStart = (Is64Bit ? Header64Size : Header32Size)
      + LoadCommandsSize;
    uint64_t SectionDataSize = 0;
    uint64_t SectionDataFileSize = 0;
    uint64_t VMSize = 0;
    for (MCAssembler::const_iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it) {
      const MCSectionData &SD = *it;
      uint64_t Address = Layout.getSectionAddress(&SD);
      uint64_t Size = Layout.getSectionSize(&SD);
      uint64_t FileSize = Layout.getSectionFileSize(&SD);

      VMSize = std::max(VMSize, Address + Size);

      if (Asm.getBackend().isVirtualSection(SD.getSection()))
        continue;

      SectionDataSize = std::max(SectionDataSize, Address + Size);
      SectionDataFileSize = std::max(SectionDataFileSize, Address + FileSize);
    }

    // The section data is padded to 4 bytes.
    //
    // FIXME: Is this machine dependent?
    unsigned SectionDataPadding = OffsetToAlignment(SectionDataFileSize, 4);
    SectionDataFileSize += SectionDataPadding;

    // Write the prolog, starting with the header and load command...
    WriteHeader(NumLoadCommands, LoadCommandsSize,
                Asm.getSubsectionsViaSymbols());
    WriteSegmentLoadCommand(NumSections, VMSize,
                            SectionDataStart, SectionDataSize);

    // ... and then the section headers.
    uint64_t RelocTableEnd = SectionDataStart + SectionDataFileSize;
    for (MCAssembler::const_iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it) {
      std::vector<MachRelocationEntry> &Relocs = Relocations[it];
      unsigned NumRelocs = Relocs.size();
      uint64_t SectionStart = SectionDataStart + Layout.getSectionAddress(it);
      WriteSection(Asm, Layout, *it, SectionStart, RelocTableEnd, NumRelocs);
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
        SymbolTableOffset + NumSymTabSymbols * (Is64Bit ? Nlist64Size :
                                                Nlist32Size);
      WriteSymtabLoadCommand(SymbolTableOffset, NumSymTabSymbols,
                             StringTableOffset, StringTable.size());

      WriteDysymtabLoadCommand(FirstLocalSymbol, NumLocalSymbols,
                               FirstExternalSymbol, NumExternalSymbols,
                               FirstUndefinedSymbol, NumUndefinedSymbols,
                               IndirectSymbolOffset, NumIndirectSymbols);
    }

    // Write the actual section data.
    for (MCAssembler::const_iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it)
      Asm.WriteSectionData(it, Layout, Writer);

    // Write the extra padding.
    WriteZeros(SectionDataPadding);

    // Write the relocation entries.
    for (MCAssembler::const_iterator it = Asm.begin(),
           ie = Asm.end(); it != ie; ++it) {
      // Write the section relocation entries, in reverse order to match 'as'
      // (approximately, the exact algorithm is more complicated than this).
      std::vector<MachRelocationEntry> &Relocs = Relocations[it];
      for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
        Write32(Relocs[e - i - 1].Word0);
        Write32(Relocs[e - i - 1].Word1);
      }
    }

    // Write the symbol table data, if used.
    if (NumSymbols) {
      // Write the indirect symbol entries.
      for (MCAssembler::const_indirect_symbol_iterator
             it = Asm.indirect_symbol_begin(),
             ie = Asm.indirect_symbol_end(); it != ie; ++it) {
        // Indirect symbols in the non lazy symbol pointer section have some
        // special handling.
        const MCSectionMachO &Section =
          static_cast<const MCSectionMachO&>(it->SectionData->getSection());
        if (Section.getType() == MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS) {
          // If this symbol is defined and internal, mark it as such.
          if (it->Symbol->isDefined() &&
              !Asm.getSymbolData(*it->Symbol).isExternal()) {
            uint32_t Flags = ISF_Local;
            if (it->Symbol->isAbsolute())
              Flags |= ISF_Absolute;
            Write32(Flags);
            continue;
          }
        }

        Write32(Asm.getSymbolData(*it->Symbol).getIndex());
      }

      // FIXME: Check that offsets match computed ones.

      // Write the symbol table entries.
      for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
        WriteNlist(LocalSymbolData[i], Layout);
      for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
        WriteNlist(ExternalSymbolData[i], Layout);
      for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
        WriteNlist(UndefinedSymbolData[i], Layout);

      // Write the string table.
      OS << StringTable.str();
    }
  }
};

}

MachObjectWriter::MachObjectWriter(raw_ostream &OS,
                                   bool Is64Bit,
                                   bool IsLittleEndian)
  : MCObjectWriter(OS, IsLittleEndian)
{
  Impl = new MachObjectWriterImpl(this, Is64Bit);
}

MachObjectWriter::~MachObjectWriter() {
  delete (MachObjectWriterImpl*) Impl;
}

void MachObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm) {
  ((MachObjectWriterImpl*) Impl)->ExecutePostLayoutBinding(Asm);
}

void MachObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const MCFragment *Fragment,
                                        const MCAsmFixup &Fixup, MCValue Target,
                                        uint64_t &FixedValue) {
  ((MachObjectWriterImpl*) Impl)->RecordRelocation(Asm, Layout, Fragment, Fixup,
                                                   Target, FixedValue);
}

void MachObjectWriter::WriteObject(const MCAssembler &Asm,
                                   const MCAsmLayout &Layout) {
  ((MachObjectWriterImpl*) Impl)->WriteObject(Asm, Layout);
}
