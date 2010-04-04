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
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;

DwarfPrinter::DwarfPrinter(AsmPrinter *A)
: Asm(A), MAI(A->MAI), TD(Asm->TM.getTargetData()),
  RI(Asm->TM.getRegisterInfo()), M(NULL), MF(NULL), MMI(NULL),
  SubprogramCount(0) {
}


/// EmitSectionOffset - Emit the 4-byte offset of Label from the start of its
/// section.  This can be done with a special directive if the target supports
/// it (e.g. cygwin) or by emitting it as an offset from a label at the start
/// of the section.
///
/// SectionLabel is a temporary label emitted at the start of the section that
/// Label lives in.
void DwarfPrinter::EmitSectionOffset(const MCSymbol *Label,
                                     const MCSymbol *SectionLabel) const {
  // On COFF targets, we have to emit the special .secrel32 directive.
  if (const char *SecOffDir = MAI->getDwarfSectionOffsetDirective()) {
    // FIXME: MCize.
    Asm->OutStreamer.EmitRawText(SecOffDir + Twine(Label->getName()));
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
    Asm->OutStreamer.EmitSymbolValue(Label, 4, 0/*AddrSpace*/);
    return;
  }

  // Otherwise, emit it as a label difference from the start of the section.
  Asm->EmitLabelDifference(Label, SectionLabel, 4);
}

/// EmitFrameMoves - Emit frame instructions to describe the layout of the
/// frame.
void DwarfPrinter::EmitFrameMoves(MCSymbol *BaseLabel,
                                  const std::vector<MachineMove> &Moves,
                                  bool isEH) {
  int stackGrowth = TD->getPointerSize();
  if (Asm->TM.getFrameInfo()->getStackGrowthDirection() !=
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
        Asm->EmitCFAByte(dwarf::DW_CFA_advance_loc4);
        Asm->EmitLabelDifference(ThisSym, BaseLabel, 4);
        BaseLabel = ThisSym;
      }
    }

    // If advancing cfa.
    if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
      if (!Src.isReg()) {
        if (Src.getReg() == MachineLocation::VirtualFP) {
          Asm->EmitCFAByte(dwarf::DW_CFA_def_cfa_offset);
        } else {
          Asm->EmitCFAByte(dwarf::DW_CFA_def_cfa);
          Asm->EmitULEB128(RI->getDwarfRegNum(Src.getReg(), isEH), "Register");
        }

        int Offset = -Src.getOffset();
        Asm->EmitULEB128(Offset, "Offset");
      } else {
        llvm_unreachable("Machine move not supported yet.");
      }
    } else if (Src.isReg() &&
               Src.getReg() == MachineLocation::VirtualFP) {
      if (Dst.isReg()) {
        Asm->EmitCFAByte(dwarf::DW_CFA_def_cfa_register);
        Asm->EmitULEB128(RI->getDwarfRegNum(Dst.getReg(), isEH), "Register");
      } else {
        llvm_unreachable("Machine move not supported yet.");
      }
    } else {
      unsigned Reg = RI->getDwarfRegNum(Src.getReg(), isEH);
      int Offset = Dst.getOffset() / stackGrowth;

      if (Offset < 0) {
        Asm->EmitCFAByte(dwarf::DW_CFA_offset_extended_sf);
        Asm->EmitULEB128(Reg, "Reg");
        Asm->EmitSLEB128(Offset, "Offset");
      } else if (Reg < 64) {
        Asm->EmitCFAByte(dwarf::DW_CFA_offset + Reg);
        Asm->EmitULEB128(Offset, "Offset");
      } else {
        Asm->EmitCFAByte(dwarf::DW_CFA_offset_extended);
        Asm->EmitULEB128(Reg, "Reg");
        Asm->EmitULEB128(Offset, "Offset");
      }
    }
  }
}
