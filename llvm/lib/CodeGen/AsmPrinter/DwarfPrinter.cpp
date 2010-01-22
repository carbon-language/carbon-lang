//===--- lib/CodeGen/DwarfPrinter.cpp - Dwarf Printer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Emit general DWARF directives.
// 
//===----------------------------------------------------------------------===//

#include "DwarfPrinter.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

DwarfPrinter::DwarfPrinter(raw_ostream &OS, AsmPrinter *A, const MCAsmInfo *T,
                           const char *flavor)
: O(OS), Asm(A), MAI(T), TD(Asm->TM.getTargetData()),
  RI(Asm->TM.getRegisterInfo()), M(NULL), MF(NULL), MMI(NULL),
  SubprogramCount(0), Flavor(flavor), SetCounter(1) {}

void DwarfPrinter::PrintRelDirective(bool Force32Bit, bool isInSection) const {
  if (isInSection && MAI->getDwarfSectionOffsetDirective())
    O << MAI->getDwarfSectionOffsetDirective();
  else if (Force32Bit || TD->getPointerSize() == sizeof(int32_t))
    O << MAI->getData32bitsDirective();
  else
    O << MAI->getData64bitsDirective();
}

static const char *DecodeDWARFEncoding(unsigned Encoding) {
  switch (Encoding) {
  case dwarf::DW_EH_PE_absptr: return "absptr";
  case dwarf::DW_EH_PE_omit:   return "omit";
  case dwarf::DW_EH_PE_pcrel:  return "pcrel";
  case dwarf::DW_EH_PE_udata4: return "udata4";
  case dwarf::DW_EH_PE_udata8: return "udata8";
  case dwarf::DW_EH_PE_sdata4: return "sdata4";
  case dwarf::DW_EH_PE_sdata8: return "sdata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata4: return "pcrel udata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4: return "pcrel sdata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata8: return "pcrel udata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8: return "pcrel sdata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_udata4:
    return "indirect pcrel udata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_sdata4:
    return "indirect pcrel sdata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_udata8:
    return "indirect pcrel udata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_sdata8:
    return "indirect pcrel sdata8";
  }
  
  return "<unknown encoding>";
}

/// EmitEncodingByte - Emit a .byte 42 directive that corresponds to an
/// encoding.  If verbose assembly output is enabled, we output comments
/// describing the encoding.  Desc is an optional string saying what the
/// encoding is specifying (e.g. "LSDA").
void DwarfPrinter::EmitEncodingByte(unsigned Val, const char *Desc) {
  if (Asm->VerboseAsm) {
    if (Desc != 0)
      Asm->OutStreamer.AddComment(Twine(Desc)+" Encoding = " +
                                  Twine(DecodeDWARFEncoding(Val)));
    else
      Asm->OutStreamer.AddComment(Twine("Encoding = ") +
                                  DecodeDWARFEncoding(Val));
  }

  Asm->OutStreamer.EmitIntValue(Val, 1, 0/*addrspace*/);
}

/// PrintSLEB128 - Print a series of hexadecimal values (separated by commas)
/// representing a signed leb128 value.
static void PrintSLEB128(MCStreamer &O, int Value) {
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    
    O.EmitIntValue(Byte, 1, /*addrspace*/0);
  } while (IsMore);
}

/// EmitSLEB128 - print the specified signed leb128 value.
void DwarfPrinter::EmitSLEB128(int Value, const char *Desc) const {
  if (Asm->VerboseAsm && Desc)
    Asm->OutStreamer.AddComment(Desc);
    
  if (MAI->hasLEB128()) {
    O << "\t.sleb128\t" << Value;
    Asm->OutStreamer.AddBlankLine();
  } else {
    PrintSLEB128(Asm->OutStreamer, Value);
  }
}



/// PrintLabelName - Print label name in form used by Dwarf writer.
///
void DwarfPrinter::PrintLabelName(const char *Tag, unsigned Number) const {
  O << MAI->getPrivateGlobalPrefix() << Tag;
  if (Number) O << Number;
}
void DwarfPrinter::PrintLabelName(const char *Tag, unsigned Number,
                                  const char *Suffix) const {
  O << MAI->getPrivateGlobalPrefix() << Tag;
  if (Number) O << Number;
  O << Suffix;
}

/// EmitLabel - Emit location label for internal use by Dwarf.
///
void DwarfPrinter::EmitLabel(const char *Tag, unsigned Number) const {
  PrintLabelName(Tag, Number);
  O << ":\n";
}

/// EmitReference - Emit a reference to a label.
///
void DwarfPrinter::EmitReference(const char *Tag, unsigned Number,
                                 bool IsPCRelative, bool Force32Bit) const {
  PrintRelDirective(Force32Bit);
  PrintLabelName(Tag, Number);
  if (IsPCRelative) O << "-" << MAI->getPCSymbol();
}
void DwarfPrinter::EmitReference(const std::string &Name, bool IsPCRelative,
                                 bool Force32Bit) const {
  PrintRelDirective(Force32Bit);
  O << Name;
  if (IsPCRelative) O << "-" << MAI->getPCSymbol();
}

void DwarfPrinter::EmitReference(const MCSymbol *Sym, bool IsPCRelative,
                                 bool Force32Bit) const {
  PrintRelDirective(Force32Bit);
  O << *Sym;
  if (IsPCRelative) O << "-" << MAI->getPCSymbol();
}

/// EmitDifference - Emit the difference between two labels.  Some assemblers do
/// not behave with absolute expressions with data directives, so there is an
/// option (needsSet) to use an intermediary set expression.
void DwarfPrinter::EmitDifference(const char *TagHi, unsigned NumberHi,
                                  const char *TagLo, unsigned NumberLo,
                                  bool IsSmall) {
  if (MAI->needsSet()) {
    O << "\t.set\t";
    PrintLabelName("set", SetCounter, Flavor);
    O << ",";
    PrintLabelName(TagHi, NumberHi);
    O << "-";
    PrintLabelName(TagLo, NumberLo);
    O << "\n";

    PrintRelDirective(IsSmall);
    PrintLabelName("set", SetCounter, Flavor);
    ++SetCounter;
  } else {
    PrintRelDirective(IsSmall);
    PrintLabelName(TagHi, NumberHi);
    O << "-";
    PrintLabelName(TagLo, NumberLo);
  }
}

void DwarfPrinter::EmitSectionOffset(const char* Label, const char* Section,
                                     unsigned LabelNumber,
                                     unsigned SectionNumber,
                                     bool IsSmall, bool isEH,
                                     bool useSet) {
  bool printAbsolute = false;
  if (isEH)
    printAbsolute = MAI->isAbsoluteEHSectionOffsets();
  else
    printAbsolute = MAI->isAbsoluteDebugSectionOffsets();

  if (MAI->needsSet() && useSet) {
    O << "\t.set\t";
    PrintLabelName("set", SetCounter, Flavor);
    O << ",";
    PrintLabelName(Label, LabelNumber);

    if (!printAbsolute) {
      O << "-";
      PrintLabelName(Section, SectionNumber);
    }

    O << "\n";
    PrintRelDirective(IsSmall);
    PrintLabelName("set", SetCounter, Flavor);
    ++SetCounter;
  } else {
    PrintRelDirective(IsSmall, true);
    PrintLabelName(Label, LabelNumber);

    if (!printAbsolute) {
      O << "-";
      PrintLabelName(Section, SectionNumber);
    }
  }
}

/// EmitFrameMoves - Emit frame instructions to describe the layout of the
/// frame.
void DwarfPrinter::EmitFrameMoves(const char *BaseLabel, unsigned BaseLabelID,
                                  const std::vector<MachineMove> &Moves,
                                  bool isEH) {
  int stackGrowth =
    Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
    TargetFrameInfo::StackGrowsUp ?
    TD->getPointerSize() : -TD->getPointerSize();
  bool IsLocal = BaseLabel && strcmp(BaseLabel, "label") == 0;

  for (unsigned i = 0, N = Moves.size(); i < N; ++i) {
    const MachineMove &Move = Moves[i];
    unsigned LabelID = Move.getLabelID();

    if (LabelID) {
      LabelID = MMI->MappedLabel(LabelID);

      // Throw out move if the label is invalid.
      if (!LabelID) continue;
    }

    const MachineLocation &Dst = Move.getDestination();
    const MachineLocation &Src = Move.getSource();

    // Advance row if new location.
    if (BaseLabel && LabelID && (BaseLabelID != LabelID || !IsLocal)) {
      Asm->EmitInt8(dwarf::DW_CFA_advance_loc4);
      Asm->EOL("DW_CFA_advance_loc4");
      EmitDifference("label", LabelID, BaseLabel, BaseLabelID, true);
      Asm->O << '\n';

      BaseLabelID = LabelID;
      BaseLabel = "label";
      IsLocal = true;
    }

    // If advancing cfa.
    if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
      if (!Src.isReg()) {
        if (Src.getReg() == MachineLocation::VirtualFP) {
          Asm->EmitInt8(dwarf::DW_CFA_def_cfa_offset);
          Asm->EOL("DW_CFA_def_cfa_offset");
        } else {
          Asm->EmitInt8(dwarf::DW_CFA_def_cfa);
          Asm->EOL("DW_CFA_def_cfa");
          Asm->EmitULEB128Bytes(RI->getDwarfRegNum(Src.getReg(), isEH));
          Asm->EOL("Register");
        }

        int Offset = -Src.getOffset();

        Asm->EmitULEB128Bytes(Offset);
        Asm->EOL("Offset");
      } else {
        llvm_unreachable("Machine move not supported yet.");
      }
    } else if (Src.isReg() &&
               Src.getReg() == MachineLocation::VirtualFP) {
      if (Dst.isReg()) {
        Asm->EmitInt8(dwarf::DW_CFA_def_cfa_register);
        Asm->EOL("DW_CFA_def_cfa_register");
        Asm->EmitULEB128Bytes(RI->getDwarfRegNum(Dst.getReg(), isEH));
        Asm->EOL("Register");
      } else {
        llvm_unreachable("Machine move not supported yet.");
      }
    } else {
      unsigned Reg = RI->getDwarfRegNum(Src.getReg(), isEH);
      int Offset = Dst.getOffset() / stackGrowth;

      if (Offset < 0) {
        Asm->EmitInt8(dwarf::DW_CFA_offset_extended_sf);
        Asm->EOL("DW_CFA_offset_extended_sf");
        Asm->EmitULEB128Bytes(Reg);
        Asm->EOL("Reg");
        EmitSLEB128(Offset, "Offset");
      } else if (Reg < 64) {
        Asm->EmitInt8(dwarf::DW_CFA_offset + Reg);
        Asm->EOL("DW_CFA_offset + Reg (" + Twine(Reg) + ")");
        Asm->EmitULEB128Bytes(Offset);
        Asm->EOL("Offset");
      } else {
        Asm->EmitInt8(dwarf::DW_CFA_offset_extended);
        Asm->EOL("DW_CFA_offset_extended");
        Asm->EmitULEB128Bytes(Reg);
        Asm->EOL("Reg");
        Asm->EmitULEB128Bytes(Offset);
        Asm->EOL("Offset");
      }
    }
  }
}
