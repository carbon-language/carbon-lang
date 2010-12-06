//===-- AsmPrinterDwarf.cpp - AsmPrinter Dwarf Support --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dwarf emissions parts of AsmPrinter.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Dwarf.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Dwarf Emission Helper Routines
//===----------------------------------------------------------------------===//

/// EmitSLEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitSLEB128(int Value, const char *Desc) const {
  if (isVerbose() && Desc)
    OutStreamer.AddComment(Desc);
    
  if (MAI->hasLEB128()) {
    OutStreamer.EmitSLEB128IntValue(Value);
    return;
  }

  // If we don't have .sleb128, emit as .bytes.
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    OutStreamer.EmitIntValue(Byte, 1, /*addrspace*/0);
  } while (IsMore);
}

/// EmitULEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitULEB128(unsigned Value, const char *Desc,
                             unsigned PadTo) const {
  if (isVerbose() && Desc)
    OutStreamer.AddComment(Desc);

  // FIXME: Should we add a PadTo option to the streamer?
  if (MAI->hasLEB128() && PadTo == 0) {
    OutStreamer.EmitULEB128IntValue(Value); 
    return;
  }
  
  // If we don't have .uleb128 or we want to emit padding, emit as .bytes.
  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    if (Value || PadTo != 0) Byte |= 0x80;
    OutStreamer.EmitIntValue(Byte, 1, /*addrspace*/0);
  } while (Value);

  if (PadTo) {
    if (PadTo > 1)
      OutStreamer.EmitFill(PadTo - 1, 0x80/*fillval*/, 0/*addrspace*/);
    OutStreamer.EmitFill(1, 0/*fillval*/, 0/*addrspace*/);
  }
}

/// EmitCFAByte - Emit a .byte 42 directive for a DW_CFA_xxx value.
void AsmPrinter::EmitCFAByte(unsigned Val) const {
  if (isVerbose()) {
    if (Val >= dwarf::DW_CFA_offset && Val < dwarf::DW_CFA_offset+64)
      OutStreamer.AddComment("DW_CFA_offset + Reg (" + 
                             Twine(Val-dwarf::DW_CFA_offset) + ")");
    else
      OutStreamer.AddComment(dwarf::CallFrameString(Val));
  }
  OutStreamer.EmitIntValue(Val, 1, 0/*addrspace*/);
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
void AsmPrinter::EmitEncodingByte(unsigned Val, const char *Desc) const {
  if (isVerbose()) {
    if (Desc != 0)
      OutStreamer.AddComment(Twine(Desc)+" Encoding = " +
                             Twine(DecodeDWARFEncoding(Val)));
    else
      OutStreamer.AddComment(Twine("Encoding = ") +
                             DecodeDWARFEncoding(Val));
  }
  
  OutStreamer.EmitIntValue(Val, 1, 0/*addrspace*/);
}

/// GetSizeOfEncodedValue - Return the size of the encoding in bytes.
unsigned AsmPrinter::GetSizeOfEncodedValue(unsigned Encoding) const {
  if (Encoding == dwarf::DW_EH_PE_omit)
    return 0;
  
  switch (Encoding & 0x07) {
  default: assert(0 && "Invalid encoded value.");
  case dwarf::DW_EH_PE_absptr: return TM.getTargetData()->getPointerSize();
  case dwarf::DW_EH_PE_udata2: return 2;
  case dwarf::DW_EH_PE_udata4: return 4;
  case dwarf::DW_EH_PE_udata8: return 8;
  }
}

void AsmPrinter::EmitReference(const MCSymbol *Sym, unsigned Encoding) const {
  const TargetLoweringObjectFile &TLOF = getObjFileLowering();
  
  const MCExpr *Exp =
    TLOF.getExprForDwarfReference(Sym, Mang, MMI, Encoding, OutStreamer);
  OutStreamer.EmitAbsValue(Exp, GetSizeOfEncodedValue(Encoding));
}

void AsmPrinter::EmitReference(const GlobalValue *GV, unsigned Encoding)const{
  const TargetLoweringObjectFile &TLOF = getObjFileLowering();
  
  const MCExpr *Exp =
    TLOF.getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, OutStreamer);
  OutStreamer.EmitValue(Exp, GetSizeOfEncodedValue(Encoding), /*addrspace*/0);
}

/// EmitSectionOffset - Emit the 4-byte offset of Label from the start of its
/// section.  This can be done with a special directive if the target supports
/// it (e.g. cygwin) or by emitting it as an offset from a label at the start
/// of the section.
///
/// SectionLabel is a temporary label emitted at the start of the section that
/// Label lives in.
void AsmPrinter::EmitSectionOffset(const MCSymbol *Label,
                                   const MCSymbol *SectionLabel) const {
  // On COFF targets, we have to emit the special .secrel32 directive.
  if (const char *SecOffDir = MAI->getDwarfSectionOffsetDirective()) {
    // FIXME: MCize.
    OutStreamer.EmitRawText(SecOffDir + Twine(Label->getName()));
    return;
  }
  
  // Get the section that we're referring to, based on SectionLabel.
  const MCSection &Section = SectionLabel->getSection();
  
  // If Label has already been emitted, verify that it is in the same section as
  // section label for sanity.
  assert((!Label->isInSection() || &Label->getSection() == &Section) &&
         "Section offset using wrong section base for label");
  
  // If the section in question will end up with an address of 0 anyway, we can
  // just emit an absolute reference to save a relocation.
  if (Section.isBaseAddressKnownZero()) {
    OutStreamer.EmitSymbolValue(Label, 4, 0/*AddrSpace*/);
    return;
  }
  
  // Otherwise, emit it as a label difference from the start of the section.
  EmitLabelDifference(Label, SectionLabel, 4);
}

//===----------------------------------------------------------------------===//
// Dwarf Lowering Routines
//===----------------------------------------------------------------------===//


/// EmitFrameMoves - Emit frame instructions to describe the layout of the
/// frame.
void AsmPrinter::EmitFrameMoves(const std::vector<MachineMove> &Moves,
                                MCSymbol *BaseLabel, bool isEH) const {
  const TargetRegisterInfo *RI = TM.getRegisterInfo();
  
  int stackGrowth = TM.getTargetData()->getPointerSize();
  if (TM.getFrameInfo()->getStackGrowthDirection() !=
      TargetFrameInfo::StackGrowsUp)
    stackGrowth *= -1;
  
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
        EmitCFAByte(dwarf::DW_CFA_advance_loc4);
        EmitLabelDifference(ThisSym, BaseLabel, 4);
        BaseLabel = ThisSym;
      }
    }
    
    // If advancing cfa.
    if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
      assert(!Src.isReg() && "Machine move not supported yet.");
      
      if (Src.getReg() == MachineLocation::VirtualFP) {
        EmitCFAByte(dwarf::DW_CFA_def_cfa_offset);
      } else {
        EmitCFAByte(dwarf::DW_CFA_def_cfa);
        EmitULEB128(RI->getDwarfRegNum(Src.getReg(), isEH), "Register");
      }
      
      EmitULEB128(-Src.getOffset(), "Offset");
      continue;
    }
    
    if (Src.isReg() && Src.getReg() == MachineLocation::VirtualFP) {
      assert(Dst.isReg() && "Machine move not supported yet.");
      EmitCFAByte(dwarf::DW_CFA_def_cfa_register);
      EmitULEB128(RI->getDwarfRegNum(Dst.getReg(), isEH), "Register");
      continue;
    }
    
    unsigned Reg = RI->getDwarfRegNum(Src.getReg(), isEH);
    int Offset = Dst.getOffset() / stackGrowth;
    
    if (Offset < 0) {
      EmitCFAByte(dwarf::DW_CFA_offset_extended_sf);
      EmitULEB128(Reg, "Reg");
      EmitSLEB128(Offset, "Offset");
    } else if (Reg < 64) {
      EmitCFAByte(dwarf::DW_CFA_offset + Reg);
      EmitULEB128(Offset, "Offset");
    } else {
      EmitCFAByte(dwarf::DW_CFA_offset_extended);
      EmitULEB128(Reg, "Reg");
      EmitULEB128(Offset, "Offset");
    }
  }
}
