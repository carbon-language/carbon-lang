//===- lib/MC/MCDwarf.cpp - MCDwarf implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FoldingSet.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"
#include "llvm/Target/TargetAsmInfo.h"
using namespace llvm;

// Given a special op, return the address skip amount (in units of
// DWARF2_LINE_MIN_INSN_LENGTH.
#define SPECIAL_ADDR(op) (((op) - DWARF2_LINE_OPCODE_BASE)/DWARF2_LINE_RANGE)

// The maximum address skip amount that can be encoded with a special op.
#define MAX_SPECIAL_ADDR_DELTA		SPECIAL_ADDR(255)

// First special line opcode - leave room for the standard opcodes.
// Note: If you want to change this, you'll have to update the
// "standard_opcode_lengths" table that is emitted in DwarfFileTable::Emit().  
#define DWARF2_LINE_OPCODE_BASE		13

// Minimum line offset in a special line info. opcode.  This value
// was chosen to give a reasonable range of values.
#define DWARF2_LINE_BASE		-5

// Range of line offsets in a special line info. opcode.
# define DWARF2_LINE_RANGE		14

// Define the architecture-dependent minimum instruction length (in bytes).
// This value should be rather too small than too big.
# define DWARF2_LINE_MIN_INSN_LENGTH	1

// Note: when DWARF2_LINE_MIN_INSN_LENGTH == 1 which is the current setting,
// this routine is a nop and will be optimized away.
static inline uint64_t ScaleAddrDelta(uint64_t AddrDelta)
{
  if (DWARF2_LINE_MIN_INSN_LENGTH == 1)
    return AddrDelta;
  if (AddrDelta % DWARF2_LINE_MIN_INSN_LENGTH != 0) {
    // TODO: report this error, but really only once.
    ;
  }
  return AddrDelta / DWARF2_LINE_MIN_INSN_LENGTH;
}

//
// This is called when an instruction is assembled into the specified section
// and if there is information from the last .loc directive that has yet to have
// a line entry made for it is made.
//
void MCLineEntry::Make(MCStreamer *MCOS, const MCSection *Section) {
  if (!MCOS->getContext().getDwarfLocSeen())
    return;

  // Create a symbol at in the current section for use in the line entry.
  MCSymbol *LineSym = MCOS->getContext().CreateTempSymbol();
  // Set the value of the symbol to use for the MCLineEntry.
  MCOS->EmitLabel(LineSym);

  // Get the current .loc info saved in the context.
  const MCDwarfLoc &DwarfLoc = MCOS->getContext().getCurrentDwarfLoc();

  // Create a (local) line entry with the symbol and the current .loc info.
  MCLineEntry LineEntry(LineSym, DwarfLoc);

  // clear DwarfLocSeen saying the current .loc info is now used.
  MCOS->getContext().ClearDwarfLocSeen();

  // Get the MCLineSection for this section, if one does not exist for this
  // section create it.
  const DenseMap<const MCSection *, MCLineSection *> &MCLineSections =
    MCOS->getContext().getMCLineSections();
  MCLineSection *LineSection = MCLineSections.lookup(Section);
  if (!LineSection) {
    // Create a new MCLineSection.  This will be deleted after the dwarf line
    // table is created using it by iterating through the MCLineSections
    // DenseMap.
    LineSection = new MCLineSection;
    // Save a pointer to the new LineSection into the MCLineSections DenseMap.
    MCOS->getContext().addMCLineSection(Section, LineSection);
  }

  // Add the line entry to this section's entries.
  LineSection->addLineEntry(LineEntry);
}

//
// This helper routine returns an expression of End - Start + IntVal .
// 
static inline const MCExpr *MakeStartMinusEndExpr(const MCStreamer &MCOS,
                                                  const MCSymbol &Start,
                                                  const MCSymbol &End,
                                                  int IntVal) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *Res =
    MCSymbolRefExpr::Create(&End, Variant, MCOS.getContext());
  const MCExpr *RHS =
    MCSymbolRefExpr::Create(&Start, Variant, MCOS.getContext());
  const MCExpr *Res1 =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, Res, RHS, MCOS.getContext());
  const MCExpr *Res2 =
    MCConstantExpr::Create(IntVal, MCOS.getContext());
  const MCExpr *Res3 =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, Res1, Res2, MCOS.getContext());
  return Res3;
}

//
// This emits the Dwarf line table for the specified section from the entries
// in the LineSection.
//
static inline void EmitDwarfLineTable(MCStreamer *MCOS,
                                      const MCSection *Section,
                                      const MCLineSection *LineSection) {
  unsigned FileNum = 1;
  unsigned LastLine = 1;
  unsigned Column = 0;
  unsigned Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
  unsigned Isa = 0;
  MCSymbol *LastLabel = NULL;

  // Loop through each MCLineEntry and encode the dwarf line number table.
  for (MCLineSection::const_iterator
         it = LineSection->getMCLineEntries()->begin(),
         ie = LineSection->getMCLineEntries()->end(); it != ie; ++it) {

    if (FileNum != it->getFileNum()) {
      FileNum = it->getFileNum();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_file, 1);
      MCOS->EmitULEB128IntValue(FileNum);
    }
    if (Column != it->getColumn()) {
      Column = it->getColumn();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_column, 1);
      MCOS->EmitULEB128IntValue(Column);
    }
    if (Isa != it->getIsa()) {
      Isa = it->getIsa();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_isa, 1);
      MCOS->EmitULEB128IntValue(Isa);
    }
    if ((it->getFlags() ^ Flags) & DWARF2_FLAG_IS_STMT) {
      Flags = it->getFlags();
      MCOS->EmitIntValue(dwarf::DW_LNS_negate_stmt, 1);
    }
    if (it->getFlags() & DWARF2_FLAG_BASIC_BLOCK)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_basic_block, 1);
    if (it->getFlags() & DWARF2_FLAG_PROLOGUE_END)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_prologue_end, 1);
    if (it->getFlags() & DWARF2_FLAG_EPILOGUE_BEGIN)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_epilogue_begin, 1);

    int64_t LineDelta = static_cast<int64_t>(it->getLine()) - LastLine;
    MCSymbol *Label = it->getLabel();

    // At this point we want to emit/create the sequence to encode the delta in
    // line numbers and the increment of the address from the previous Label
    // and the current Label.
    MCOS->EmitDwarfAdvanceLineAddr(LineDelta, LastLabel, Label);

    LastLine = it->getLine();
    LastLabel = Label;
  }

  // Emit a DW_LNE_end_sequence for the end of the section.
  // Using the pointer Section create a temporary label at the end of the
  // section and use that and the LastLabel to compute the address delta
  // and use INT64_MAX as the line delta which is the signal that this is
  // actually a DW_LNE_end_sequence.

  // Switch to the section to be able to create a symbol at its end.
  MCOS->SwitchSection(Section);

  MCContext &context = MCOS->getContext();
  // Create a symbol at the end of the section.
  MCSymbol *SectionEnd = context.CreateTempSymbol();
  // Set the value of the symbol, as we are at the end of the section.
  MCOS->EmitLabel(SectionEnd);

  // Switch back the the dwarf line section.
  MCOS->SwitchSection(context.getTargetAsmInfo().getDwarfLineSection());

  MCOS->EmitDwarfAdvanceLineAddr(INT64_MAX, LastLabel, SectionEnd);
}

//
// This emits the Dwarf file and the line tables.
//
void MCDwarfFileTable::Emit(MCStreamer *MCOS) {
  MCContext &context = MCOS->getContext();
  // Switch to the section where the table will be emitted into.
  MCOS->SwitchSection(context.getTargetAsmInfo().getDwarfLineSection());

  // Create a symbol at the beginning of this section.
  MCSymbol *LineStartSym = context.CreateTempSymbol();
  // Set the value of the symbol, as we are at the start of the section.
  MCOS->EmitLabel(LineStartSym);

  // Create a symbol for the end of the section (to be set when we get there).
  MCSymbol *LineEndSym = context.CreateTempSymbol();

  // The first 4 bytes is the total length of the information for this
  // compilation unit (not including these 4 bytes for the length).
  MCOS->EmitAbsValue(MakeStartMinusEndExpr(*MCOS, *LineStartSym, *LineEndSym,4),
                     4);

  // Next 2 bytes is the Version, which is Dwarf 2.
  MCOS->EmitIntValue(2, 2);

  // Create a symbol for the end of the prologue (to be set when we get there).
  MCSymbol *ProEndSym = context.CreateTempSymbol(); // Lprologue_end

  // Length of the prologue, is the next 4 bytes.  Which is the start of the
  // section to the end of the prologue.  Not including the 4 bytes for the
  // total length, the 2 bytes for the version, and these 4 bytes for the
  // length of the prologue.
  MCOS->EmitAbsValue(MakeStartMinusEndExpr(*MCOS, *LineStartSym, *ProEndSym,
                                        (4 + 2 + 4)),
                  4, 0);

  // Parameters of the state machine, are next.
  MCOS->EmitIntValue(DWARF2_LINE_MIN_INSN_LENGTH, 1);
  MCOS->EmitIntValue(DWARF2_LINE_DEFAULT_IS_STMT, 1);
  MCOS->EmitIntValue(DWARF2_LINE_BASE, 1);
  MCOS->EmitIntValue(DWARF2_LINE_RANGE, 1);
  MCOS->EmitIntValue(DWARF2_LINE_OPCODE_BASE, 1);

  // Standard opcode lengths
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_copy
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_advance_pc
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_advance_line
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_set_file
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_set_column
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_negate_stmt
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_set_basic_block
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_const_add_pc
  MCOS->EmitIntValue(1, 1); // length of DW_LNS_fixed_advance_pc
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_set_prologue_end
  MCOS->EmitIntValue(0, 1); // length of DW_LNS_set_epilogue_begin
  MCOS->EmitIntValue(1, 1); // DW_LNS_set_isa

  // Put out the directory and file tables.

  // First the directory table.
  const std::vector<StringRef> &MCDwarfDirs =
    context.getMCDwarfDirs();
  for (unsigned i = 0; i < MCDwarfDirs.size(); i++) {
    MCOS->EmitBytes(MCDwarfDirs[i], 0); // the DirectoryName
    MCOS->EmitBytes(StringRef("\0", 1), 0); // the null term. of the string
  }
  MCOS->EmitIntValue(0, 1); // Terminate the directory list

  // Second the file table.
  const std::vector<MCDwarfFile *> &MCDwarfFiles =
    MCOS->getContext().getMCDwarfFiles();
  for (unsigned i = 1; i < MCDwarfFiles.size(); i++) {
    MCOS->EmitBytes(MCDwarfFiles[i]->getName(), 0); // FileName
    MCOS->EmitBytes(StringRef("\0", 1), 0); // the null term. of the string
    // the Directory num
    MCOS->EmitULEB128IntValue(MCDwarfFiles[i]->getDirIndex());
    MCOS->EmitIntValue(0, 1); // last modification timestamp (always 0)
    MCOS->EmitIntValue(0, 1); // filesize (always 0)
  }
  MCOS->EmitIntValue(0, 1); // Terminate the file list

  // This is the end of the prologue, so set the value of the symbol at the
  // end of the prologue (that was used in a previous expression).
  MCOS->EmitLabel(ProEndSym);

  // Put out the line tables.
  const DenseMap<const MCSection *, MCLineSection *> &MCLineSections =
    MCOS->getContext().getMCLineSections();
  const std::vector<const MCSection *> &MCLineSectionOrder =
    MCOS->getContext().getMCLineSectionOrder();
  for (std::vector<const MCSection*>::const_iterator it =
	MCLineSectionOrder.begin(), ie = MCLineSectionOrder.end(); it != ie;
       ++it) {
    const MCSection *Sec = *it;
    const MCLineSection *Line = MCLineSections.lookup(Sec);
    EmitDwarfLineTable(MCOS, Sec, Line);

    // Now delete the MCLineSections that were created in MCLineEntry::Make()
    // and used to emit the line table.
    delete Line;
  }

  if (MCOS->getContext().getAsmInfo().getLinkerRequiresNonEmptyDwarfLines()
      && MCLineSectionOrder.begin() == MCLineSectionOrder.end()) {
    // The darwin9 linker has a bug (see PR8715). For for 32-bit architectures
    // it requires:  
    // total_length >= prologue_length + 10
    // We are 4 bytes short, since we have total_length = 51 and
    // prologue_length = 45

    // The regular end_sequence should be sufficient.
    MCDwarfLineAddr::Emit(MCOS, INT64_MAX, 0);
  }

  // This is the end of the section, so set the value of the symbol at the end
  // of this section (that was used in a previous expression).
  MCOS->EmitLabel(LineEndSym);
}

/// Utility function to write the encoding to an object writer.
void MCDwarfLineAddr::Write(MCObjectWriter *OW, int64_t LineDelta,
                            uint64_t AddrDelta) {
  SmallString<256> Tmp;
  raw_svector_ostream OS(Tmp);
  MCDwarfLineAddr::Encode(LineDelta, AddrDelta, OS);
  OW->WriteBytes(OS.str());
}

/// Utility function to emit the encoding to a streamer.
void MCDwarfLineAddr::Emit(MCStreamer *MCOS, int64_t LineDelta,
                           uint64_t AddrDelta) {
  SmallString<256> Tmp;
  raw_svector_ostream OS(Tmp);
  MCDwarfLineAddr::Encode(LineDelta, AddrDelta, OS);
  MCOS->EmitBytes(OS.str(), /*AddrSpace=*/0);
}

/// Utility function to encode a Dwarf pair of LineDelta and AddrDeltas.
void MCDwarfLineAddr::Encode(int64_t LineDelta, uint64_t AddrDelta,
                             raw_ostream &OS) {
  uint64_t Temp, Opcode;
  bool NeedCopy = false;

  // Scale the address delta by the minimum instruction length.
  AddrDelta = ScaleAddrDelta(AddrDelta);

  // A LineDelta of INT64_MAX is a signal that this is actually a
  // DW_LNE_end_sequence. We cannot use special opcodes here, since we want the 
  // end_sequence to emit the matrix entry.
  if (LineDelta == INT64_MAX) {
    if (AddrDelta == MAX_SPECIAL_ADDR_DELTA)
      OS << char(dwarf::DW_LNS_const_add_pc);
    else {
      OS << char(dwarf::DW_LNS_advance_pc);
      SmallString<32> Tmp;
      raw_svector_ostream OSE(Tmp);
      MCObjectWriter::EncodeULEB128(AddrDelta, OSE);
      OS << OSE.str();
    }
    OS << char(dwarf::DW_LNS_extended_op);
    OS << char(1);
    OS << char(dwarf::DW_LNE_end_sequence);
    return;
  }

  // Bias the line delta by the base.
  Temp = LineDelta - DWARF2_LINE_BASE;

  // If the line increment is out of range of a special opcode, we must encode
  // it with DW_LNS_advance_line.
  if (Temp >= DWARF2_LINE_RANGE) {
    OS << char(dwarf::DW_LNS_advance_line);
    SmallString<32> Tmp;
    raw_svector_ostream OSE(Tmp);
    MCObjectWriter::EncodeSLEB128(LineDelta, OSE);
    OS << OSE.str();

    LineDelta = 0;
    Temp = 0 - DWARF2_LINE_BASE;
    NeedCopy = true;
  }

  // Use DW_LNS_copy instead of a "line +0, addr +0" special opcode.
  if (LineDelta == 0 && AddrDelta == 0) {
    OS << char(dwarf::DW_LNS_copy);
    return;
  }

  // Bias the opcode by the special opcode base.
  Temp += DWARF2_LINE_OPCODE_BASE;

  // Avoid overflow when addr_delta is large.
  if (AddrDelta < 256 + MAX_SPECIAL_ADDR_DELTA) {
    // Try using a special opcode.
    Opcode = Temp + AddrDelta * DWARF2_LINE_RANGE;
    if (Opcode <= 255) {
      OS << char(Opcode);
      return;
    }

    // Try using DW_LNS_const_add_pc followed by special op.
    Opcode = Temp + (AddrDelta - MAX_SPECIAL_ADDR_DELTA) * DWARF2_LINE_RANGE;
    if (Opcode <= 255) {
      OS << char(dwarf::DW_LNS_const_add_pc);
      OS << char(Opcode);
      return;
    }
  }

  // Otherwise use DW_LNS_advance_pc.
  OS << char(dwarf::DW_LNS_advance_pc);
  SmallString<32> Tmp;
  raw_svector_ostream OSE(Tmp);
  MCObjectWriter::EncodeULEB128(AddrDelta, OSE);
  OS << OSE.str();

  if (NeedCopy)
    OS << char(dwarf::DW_LNS_copy);
  else
    OS << char(Temp);
}

void MCDwarfFile::print(raw_ostream &OS) const {
  OS << '"' << getName() << '"';
}

void MCDwarfFile::dump() const {
  print(dbgs());
}

static int getDataAlignmentFactor(MCStreamer &streamer) {
  MCContext &context = streamer.getContext();
  const TargetAsmInfo &asmInfo = context.getTargetAsmInfo();
  int size = asmInfo.getPointerSize();
  if (asmInfo.getStackGrowthDirection() == TargetFrameInfo::StackGrowsUp)
    return size;
 else
   return -size;
}

/// EmitFrameMoves - Emit frame instructions to describe the layout of the
/// frame.
static void EmitFrameMoves(MCStreamer &streamer,
                           const std::vector<MachineMove> &Moves,
                           MCSymbol *BaseLabel, bool isEH) {
  MCContext &context = streamer.getContext();
  const TargetAsmInfo &asmInfo = context.getTargetAsmInfo();
  int dataAlignmentFactor = getDataAlignmentFactor(streamer);

  for (unsigned i = 0, N = Moves.size(); i < N; ++i) {
    const MachineMove &Move = Moves[i];
    MCSymbol *Label = Move.getLabel();
    // Throw out move if the label is invalid.
    if (Label && !Label->isDefined()) continue; // Not emitted, in dead code.

    const MachineLocation &Dst = Move.getDestination();
    const MachineLocation &Src = Move.getSource();

    // Advance row if new location.
    if (BaseLabel && Label) {
      MCSymbol *ThisSym = Label;
      if (ThisSym != BaseLabel) {
        streamer.EmitIntValue(dwarf::DW_CFA_advance_loc4, 1);
        const MCExpr *Length = MakeStartMinusEndExpr(streamer, *BaseLabel,
                                                     *ThisSym, 4);
        streamer.EmitValue(Length, 4);
        BaseLabel = ThisSym;
      }
    }

    // If advancing cfa.
    if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
      assert(!Src.isReg() && "Machine move not supported yet.");

      if (Src.getReg() == MachineLocation::VirtualFP) {
        streamer.EmitIntValue(dwarf::DW_CFA_def_cfa_offset, 1);
      } else {
        streamer.EmitIntValue(dwarf::DW_CFA_def_cfa, 1);
        streamer.EmitULEB128IntValue(asmInfo.getDwarfRegNum(Src.getReg(),
                                                            isEH));
      }

      streamer.EmitULEB128IntValue(-Src.getOffset(), 1);
      continue;
    }

    if (Src.isReg() && Src.getReg() == MachineLocation::VirtualFP) {
      assert(Dst.isReg() && "Machine move not supported yet.");
      streamer.EmitIntValue(dwarf::DW_CFA_def_cfa_register, 1);
      streamer.EmitULEB128IntValue(asmInfo.getDwarfRegNum(Dst.getReg(), isEH));
      continue;
    }

    unsigned Reg = asmInfo.getDwarfRegNum(Src.getReg(), isEH);
    int Offset = Dst.getOffset() / dataAlignmentFactor;

    if (Offset < 0) {
      streamer.EmitIntValue(dwarf::DW_CFA_offset_extended_sf, 1);
      streamer.EmitULEB128IntValue(Reg);
      streamer.EmitSLEB128IntValue(Offset);
    } else if (Reg < 64) {
      streamer.EmitIntValue(dwarf::DW_CFA_offset + Reg, 1);
      streamer.EmitULEB128IntValue(Offset, 1);
    } else {
      streamer.EmitIntValue(dwarf::DW_CFA_offset_extended, 1);
      streamer.EmitULEB128IntValue(Reg, 1);
      streamer.EmitULEB128IntValue(Offset, 1);
    }
  }
}

static void EmitSymbol(MCStreamer &streamer, const MCSymbol &symbol,
                       unsigned symbolEncoding) {
  MCContext &context = streamer.getContext();
  const TargetAsmInfo &asmInfo = context.getTargetAsmInfo();
  unsigned format = symbolEncoding & 0x0f;
  unsigned application = symbolEncoding & 0xf0;
  unsigned size;
  switch (format) {
  default:
    assert(0 && "Unknown Encoding");
    break;
  case dwarf::DW_EH_PE_absptr:
  case dwarf::DW_EH_PE_signed:
    size = asmInfo.getPointerSize();
    break;
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    size = 2;
    break;
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    size = 4;
    break;
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    size = 8;
    break;
  }
  switch (application) {
  default:
    assert(0 && "Unknown Encoding");
    break;
  case 0:
  case dwarf::DW_EH_PE_indirect:
    streamer.EmitSymbolValue(&symbol, size);
    break;
  case dwarf::DW_EH_PE_pcrel:
    streamer.EmitPCRelSymbolValue(&symbol, size);
    break;
  }
}

static const MCSymbol &EmitCIE(MCStreamer &streamer,
                               const MCSymbol *personality,
                               unsigned personalityEncoding,
                               const MCSymbol *lsda,
                               unsigned lsdaEncoding) {
  MCContext &context = streamer.getContext();
  const TargetAsmInfo &asmInfo = context.getTargetAsmInfo();
  const MCSection &section = *asmInfo.getEHFrameSection();
  streamer.SwitchSection(&section);
  MCSymbol *sectionStart = streamer.getContext().CreateTempSymbol();
  MCSymbol *sectionEnd = streamer.getContext().CreateTempSymbol();

  // Length
  const MCExpr *Length = MakeStartMinusEndExpr(streamer, *sectionStart,
                                               *sectionEnd, 4);
  streamer.EmitLabel(sectionStart);
  streamer.EmitValue(Length, 4);

  // CIE ID
  streamer.EmitIntValue(0, 4);

  // Version
  streamer.EmitIntValue(dwarf::DW_CIE_VERSION, 1);

  // Augmentation String
  SmallString<8> Augmentation;
  Augmentation += "z";
  if (personality)
    Augmentation += "P";
  if (lsda)
    Augmentation += "L";
  Augmentation += "R";
  streamer.EmitBytes(Augmentation.str(), 0);
  streamer.EmitIntValue(0, 1);

  // Code Alignment Factor
  streamer.EmitULEB128IntValue(1);

  // Data Alignment Factor
  streamer.EmitSLEB128IntValue(getDataAlignmentFactor(streamer));

  // Return Address Register
  streamer.EmitULEB128IntValue(asmInfo.getDwarfRARegNum(true));

  // Augmentation Data Length (optional)
  MCSymbol *augmentationStart = streamer.getContext().CreateTempSymbol();
  MCSymbol *augmentationEnd = streamer.getContext().CreateTempSymbol();
  const MCExpr *augmentationLength = MakeStartMinusEndExpr(streamer,
                                                           *augmentationStart,
                                                           *augmentationEnd, 0);
  streamer.EmitULEB128Value(augmentationLength);

  // Augmentation Data (optional)
  streamer.EmitLabel(augmentationStart);
  if (personality) {
    // Personality Encoding
    streamer.EmitIntValue(personalityEncoding, 1);
    // Personality
    EmitSymbol(streamer, *personality, personalityEncoding);
  }
  if (lsda) {
    // LSDA Encoding
    streamer.EmitIntValue(lsdaEncoding, 1);
  }
  // Encoding of the FDE pointers
  streamer.EmitIntValue(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4, 1);
  streamer.EmitLabel(augmentationEnd);

  // Initial Instructions

  const std::vector<MachineMove> Moves = asmInfo.getInitialFrameState();

  EmitFrameMoves(streamer, Moves, NULL, true);

  // Padding
  streamer.EmitValueToAlignment(4);

  streamer.EmitLabel(sectionEnd);
  return *sectionStart;
}

static MCSymbol *EmitFDE(MCStreamer &streamer,
                         const MCSymbol &cieStart,
                         const MCDwarfFrameInfo &frame) {
  MCContext &context = streamer.getContext();
  MCSymbol *fdeStart = context.CreateTempSymbol();
  MCSymbol *fdeEnd = context.CreateTempSymbol();

  // Length
  const MCExpr *Length = MakeStartMinusEndExpr(streamer, *fdeStart, *fdeEnd, 0);
  streamer.EmitValue(Length, 4);

  streamer.EmitLabel(fdeStart);
  // CIE Pointer
  const MCExpr *offset = MakeStartMinusEndExpr(streamer, cieStart, *fdeStart,
                                               0);
  streamer.EmitValue(offset, 4);

  // PC Begin
  streamer.EmitPCRelSymbolValue(frame.Begin, 4);

  // PC Range
  const MCExpr *Range = MakeStartMinusEndExpr(streamer, *frame.Begin,
                                              *frame.End, 0);
  streamer.EmitValue(Range, 4);

  // Augmentation Data Length
  MCSymbol *augmentationStart = streamer.getContext().CreateTempSymbol();
  MCSymbol *augmentationEnd = streamer.getContext().CreateTempSymbol();
  const MCExpr *augmentationLength = MakeStartMinusEndExpr(streamer,
                                                           *augmentationStart,
                                                           *augmentationEnd, 0);
  streamer.EmitULEB128Value(augmentationLength);

  // Augmentation Data
  streamer.EmitLabel(augmentationStart);
  if (frame.Lsda)
    EmitSymbol(streamer, *frame.Lsda, frame.LsdaEncoding);
  streamer.EmitLabel(augmentationEnd);
  // Call Frame Instructions

  // Padding
  streamer.EmitValueToAlignment(4);

  return fdeEnd;
}

struct CIEKey {
  static const CIEKey EmptyKey;
  static const CIEKey TombstoneKey;

  CIEKey(const MCSymbol* Personality_, unsigned PersonalityEncoding_,
         unsigned LsdaEncoding_) : Personality(Personality_),
                                   PersonalityEncoding(PersonalityEncoding_),
                                   LsdaEncoding(LsdaEncoding_) {
  }
  const MCSymbol* Personality;
  unsigned PersonalityEncoding;
  unsigned LsdaEncoding;
};

const CIEKey CIEKey::EmptyKey(0, 0, -1);
const CIEKey CIEKey::TombstoneKey(0, -1, 0);

namespace llvm {
  template <>
  struct DenseMapInfo<CIEKey> {
    static CIEKey getEmptyKey() {
      return CIEKey::EmptyKey;
    }
    static CIEKey getTombstoneKey() {
      return CIEKey::TombstoneKey;
    }
    static unsigned getHashValue(const CIEKey &Key) {
      FoldingSetNodeID ID;
      ID.AddPointer(Key.Personality);
      ID.AddInteger(Key.PersonalityEncoding);
      ID.AddInteger(Key.LsdaEncoding);
      return ID.ComputeHash();
    }
    static bool isEqual(const CIEKey &LHS,
                        const CIEKey &RHS) {
      return LHS.Personality == RHS.Personality &&
        LHS.PersonalityEncoding == RHS.PersonalityEncoding &&
        LHS.LsdaEncoding == RHS.LsdaEncoding;
    }
  };
}

void MCDwarfFrameEmitter::Emit(MCStreamer &streamer) {
  const MCContext &context = streamer.getContext();
  const TargetAsmInfo &asmInfo = context.getTargetAsmInfo();
  MCSymbol *fdeEnd = NULL;
  DenseMap<CIEKey, const MCSymbol*> CIEStarts;

  for (unsigned i = 0, n = streamer.getNumFrameInfos(); i < n; ++i) {
    const MCDwarfFrameInfo &frame = streamer.getFrameInfo(i);
    CIEKey key(frame.Personality, frame.PersonalityEncoding,
               frame.LsdaEncoding);
    const MCSymbol *&cieStart = CIEStarts[key];
    if (!cieStart)
      cieStart = &EmitCIE(streamer, frame.Personality,
                          frame.PersonalityEncoding, frame.Lsda,
                          frame.LsdaEncoding);
    fdeEnd = EmitFDE(streamer, *cieStart, frame);
    if (i != n - 1)
      streamer.EmitLabel(fdeEnd);
  }

  streamer.EmitValueToAlignment(asmInfo.getPointerSize());
  if (fdeEnd)
    streamer.EmitLabel(fdeEnd);
}
