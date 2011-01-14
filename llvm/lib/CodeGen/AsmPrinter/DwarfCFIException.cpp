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
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

DwarfCFIException::DwarfCFIException(AsmPrinter *A)
  : DwarfException(A),
    shouldEmitTable(false), shouldEmitMoves(false), shouldEmitTableModule(false)
    {}

DwarfCFIException::~DwarfCFIException() {}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfCFIException::EndModule() {
  if (Asm->MAI->getExceptionHandlingType() != ExceptionHandling::Dwarf)
    return;

  if (!shouldEmitTableModule)
    return;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  unsigned PerEncoding = TLOF.getPersonalityEncoding();

  // Begin eh frame section.
  Asm->OutStreamer.SwitchSection(TLOF.getEHFrameSection());

  // Emit references to all used personality functions
  const std::vector<const Function*> &Personalities = MMI->getPersonalities();
  for (size_t i = 0, e = Personalities.size(); i != e; ++i) {
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("personality", i));
    Asm->EmitReference(Personalities[i], PerEncoding);
  }
}

/// BeginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void DwarfCFIException::BeginFunction(const MachineFunction *MF) {
  shouldEmitTable = shouldEmitMoves = false;

  // If any landing pads survive, we need an EH table.
  shouldEmitTable = !MMI->getLandingPads().empty();

  // See if we need frame move info.
  shouldEmitMoves =
    !Asm->MF->getFunction()->doesNotThrow() || UnwindTablesMandatory;

  if (shouldEmitMoves || shouldEmitTable)
    // Assumes in correct section after the entry point.
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_begin",
                                                  Asm->getFunctionNumber()));

  shouldEmitTableModule |= shouldEmitTable;

  if (shouldEmitMoves) {
    const TargetFrameLowering *TFL = Asm->TM.getFrameLowering();
    Asm->OutStreamer.EmitCFIStartProc();

    // Indicate locations of general callee saved registers in frame.
    std::vector<MachineMove> Moves;
    TFL->getInitialFrameState(Moves);
    Asm->EmitCFIFrameMoves(Moves);
    Asm->EmitCFIFrameMoves(MMI->getFrameMoves());
  }

  if (!shouldEmitTable)
    return;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  // Provide LSDA information.
  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  if (LSDAEncoding != dwarf::DW_EH_PE_omit)
    Asm->OutStreamer.EmitCFILsda(Asm->GetTempSymbol("exception",
                                                    Asm->getFunctionNumber()),
                                 LSDAEncoding);

  // Indicate personality routine, if any.
  unsigned PerEncoding = TLOF.getPersonalityEncoding();
  if (PerEncoding != dwarf::DW_EH_PE_omit &&
      MMI->getPersonalities()[MMI->getPersonalityIndex()])
    Asm->OutStreamer.EmitCFIPersonality(Asm->GetTempSymbol("personality",
                                                    MMI->getPersonalityIndex()),
                                        PerEncoding);
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfCFIException::EndFunction() {
  if (!shouldEmitMoves && !shouldEmitTable) return;

  if (shouldEmitMoves)
    Asm->OutStreamer.EmitCFIEndProc();

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_end",
                                                Asm->getFunctionNumber()));

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  if (shouldEmitTable)
    EmitExceptionTable();
}
