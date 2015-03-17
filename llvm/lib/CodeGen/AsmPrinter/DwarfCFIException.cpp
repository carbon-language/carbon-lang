//===-- CodeGen/AsmPrinter/DwarfException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing DWARF exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

DwarfCFIExceptionBase::DwarfCFIExceptionBase(AsmPrinter *A)
    : EHStreamer(A), shouldEmitCFI(false) {}

void DwarfCFIExceptionBase::markFunctionEnd() {
  if (shouldEmitCFI)
    Asm->OutStreamer.EmitCFIEndProc();

  if (MMI->getLandingPads().empty())
    return;

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();
}

DwarfCFIException::DwarfCFIException(AsmPrinter *A)
    : DwarfCFIExceptionBase(A), shouldEmitPersonality(false),
      shouldEmitLSDA(false), shouldEmitMoves(false),
      moveTypeModule(AsmPrinter::CFI_M_None) {}

DwarfCFIException::~DwarfCFIException() {}

/// endModule - Emit all exception information that should come after the
/// content.
void DwarfCFIException::endModule() {
  if (moveTypeModule == AsmPrinter::CFI_M_Debug)
    Asm->OutStreamer.EmitCFISections(false, true);

  // SjLj uses this pass and it doesn't need this info.
  if (!Asm->MAI->usesCFIForEH())
    return;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  unsigned PerEncoding = TLOF.getPersonalityEncoding();

  if ((PerEncoding & 0x80) != dwarf::DW_EH_PE_indirect)
    return;

  // Emit references to all used personality functions
  const std::vector<const Function*> &Personalities = MMI->getPersonalities();
  for (size_t i = 0, e = Personalities.size(); i != e; ++i) {
    if (!Personalities[i])
      continue;
    MCSymbol *Sym = Asm->getSymbol(Personalities[i]);
    TLOF.emitPersonalityValue(Asm->OutStreamer, Asm->TM, Sym);
  }
}

void DwarfCFIException::beginFunction(const MachineFunction *MF) {
  shouldEmitMoves = shouldEmitPersonality = shouldEmitLSDA = false;

  // If any landing pads survive, we need an EH table.
  bool hasLandingPads = !MMI->getLandingPads().empty();

  // See if we need frame move info.
  AsmPrinter::CFIMoveType MoveType = Asm->needsCFIMoves();
  if (MoveType == AsmPrinter::CFI_M_EH ||
      (MoveType == AsmPrinter::CFI_M_Debug &&
       moveTypeModule == AsmPrinter::CFI_M_None))
    moveTypeModule = MoveType;

  shouldEmitMoves = MoveType != AsmPrinter::CFI_M_None;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  unsigned PerEncoding = TLOF.getPersonalityEncoding();
  const Function *Per = MMI->getPersonality();

  shouldEmitPersonality = hasLandingPads &&
    PerEncoding != dwarf::DW_EH_PE_omit && Per;

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  shouldEmitLSDA = shouldEmitPersonality &&
    LSDAEncoding != dwarf::DW_EH_PE_omit;

  shouldEmitCFI = shouldEmitPersonality || shouldEmitMoves;
  if (!shouldEmitCFI)
    return;

  Asm->OutStreamer.EmitCFIStartProc(/*IsSimple=*/false);

  // Indicate personality routine, if any.
  if (!shouldEmitPersonality)
    return;

  const MCSymbol *Sym =
      TLOF.getCFIPersonalitySymbol(Per, *Asm->Mang, Asm->TM, MMI);
  Asm->OutStreamer.EmitCFIPersonality(Sym, PerEncoding);

  // Provide LSDA information.
  if (!shouldEmitLSDA)
    return;

  Asm->OutStreamer.EmitCFILsda(Asm->getCurExceptionSym(), LSDAEncoding);
}

/// endFunction - Gather and emit post-function exception information.
///
void DwarfCFIException::endFunction(const MachineFunction *) {
  if (!shouldEmitPersonality)
    return;

  emitExceptionTable();
}
