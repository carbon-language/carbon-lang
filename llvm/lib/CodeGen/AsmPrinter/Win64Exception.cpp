//===-- CodeGen/AsmPrinter/Win64Exception.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Win64 exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "Win64Exception.h"
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
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

Win64Exception::Win64Exception(AsmPrinter *A)
  : EHStreamer(A), shouldEmitPersonality(false), shouldEmitLSDA(false),
    shouldEmitMoves(false) {}

Win64Exception::~Win64Exception() {}

/// endModule - Emit all exception information that should come after the
/// content.
void Win64Exception::endModule() {
}

void Win64Exception::beginFunction(const MachineFunction *MF) {
  EHStreamer::beginFunction(MF);

  shouldEmitMoves = shouldEmitPersonality = shouldEmitLSDA = false;

  // If any landing pads survive, we need an EH table.
  bool hasLandingPads = !MMI->getLandingPads().empty();

  shouldEmitMoves = Asm->needsSEHMoves();

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  unsigned PerEncoding = TLOF.getPersonalityEncoding();
  const Function *Per = MF->getMMI().getPersonality();

  shouldEmitPersonality = hasLandingPads &&
    PerEncoding != dwarf::DW_EH_PE_omit && Per;

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  shouldEmitLSDA = shouldEmitPersonality &&
    LSDAEncoding != dwarf::DW_EH_PE_omit;

  if (!shouldEmitPersonality && !shouldEmitMoves)
    return;

  Asm->OutStreamer.EmitWinCFIStartProc(Asm->CurrentFnSym);

  if (!shouldEmitPersonality)
    return;

  const MCSymbol *PersHandlerSym =
      TLOF.getCFIPersonalitySymbol(Per, *Asm->Mang, Asm->TM, MMI);
  Asm->OutStreamer.EmitWinEHHandler(PersHandlerSym, true, true);
}

/// endFunction - Gather and emit post-function exception information.
///
void Win64Exception::endFunction(const MachineFunction *) {
  if (!shouldEmitPersonality && !shouldEmitMoves)
    return;

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  if (shouldEmitPersonality) {
    Asm->OutStreamer.PushSection();

    // Emit an UNWIND_INFO struct describing the prologue.
    Asm->OutStreamer.EmitWinEHHandlerData();

    // Emit the tables appropriate to the personality function in use. If we
    // don't recognize the personality, assume it uses an Itanium-style LSDA.
    EHPersonality Per = MMI->getPersonalityType();
    if (Per == EHPersonality::MSVC_Win64SEH)
      emitCSpecificHandlerTable();
    else
      emitExceptionTable();

    Asm->OutStreamer.PopSection();
  }
  Asm->OutStreamer.EmitWinCFIEndProc();
}

const MCSymbolRefExpr *Win64Exception::createImageRel32(const MCSymbol *Value) {
  return MCSymbolRefExpr::Create(Value, MCSymbolRefExpr::VK_COFF_IMGREL32,
                                 Asm->OutContext);
}

/// Emit the language-specific data that __C_specific_handler expects.  This
/// handler lives in the x64 Microsoft C runtime and allows catching or cleaning
/// up after faults with __try, __except, and __finally.  The typeinfo values
/// are not really RTTI data, but pointers to filter functions that return an
/// integer (1, 0, or -1) indicating how to handle the exception. For __finally
/// blocks and other cleanups, the landing pad label is zero, and the filter
/// function is actually a cleanup handler with the same prototype.  A catch-all
/// entry is modeled with a null filter function field and a non-zero landing
/// pad label.
///
/// Possible filter function return values:
///   EXCEPTION_EXECUTE_HANDLER (1):
///     Jump to the landing pad label after cleanups.
///   EXCEPTION_CONTINUE_SEARCH (0):
///     Continue searching this table or continue unwinding.
///   EXCEPTION_CONTINUE_EXECUTION (-1):
///     Resume execution at the trapping PC.
///
/// Inferred table structure:
///   struct Table {
///     int NumEntries;
///     struct Entry {
///       imagerel32 LabelStart;
///       imagerel32 LabelEnd;
///       imagerel32 FilterOrFinally;  // One means catch-all.
///       imagerel32 LabelLPad;        // Zero means __finally.
///     } Entries[NumEntries];
///   };
void Win64Exception::emitCSpecificHandlerTable() {
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();

  // Simplifying assumptions for first implementation:
  // - Cleanups are not implemented.
  // - Filters are not implemented.

  // The Itanium LSDA table sorts similar landing pads together to simplify the
  // actions table, but we don't need that.
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  LandingPads.reserve(PadInfos.size());
  for (const auto &LP : PadInfos)
    LandingPads.push_back(&LP);

  // Compute label ranges for call sites as we would for the Itanium LSDA, but
  // use an all zero action table because we aren't using these actions.
  SmallVector<unsigned, 64> FirstActions;
  FirstActions.resize(LandingPads.size());
  SmallVector<CallSiteEntry, 64> CallSites;
  computeCallSiteTable(CallSites, LandingPads, FirstActions);

  MCSymbol *EHFuncBeginSym = Asm->getFunctionBegin();
  MCSymbol *EHFuncEndSym = Asm->getFunctionEnd();

  // Emit the number of table entries.
  unsigned NumEntries = 0;
  for (const CallSiteEntry &CSE : CallSites) {
    if (!CSE.LPad)
      continue; // Ignore gaps.
    for (int Selector : CSE.LPad->TypeIds) {
      // Ignore C++ filter clauses in SEH.
      // FIXME: Implement cleanup clauses.
      if (isCatchEHSelector(Selector))
        ++NumEntries;
    }
  }
  Asm->OutStreamer.EmitIntValue(NumEntries, 4);

  // Emit the four-label records for each call site entry. The table has to be
  // sorted in layout order, and the call sites should already be sorted.
  for (const CallSiteEntry &CSE : CallSites) {
    // Ignore gaps. Unlike the Itanium model, unwinding through a frame without
    // an EH table entry will propagate the exception rather than terminating
    // the program.
    if (!CSE.LPad)
      continue;
    const LandingPadInfo *LPad = CSE.LPad;

    // Compute the label range. We may reuse the function begin and end labels
    // rather than forming new ones.
    const MCExpr *Begin =
        createImageRel32(CSE.BeginLabel ? CSE.BeginLabel : EHFuncBeginSym);
    const MCExpr *End;
    if (CSE.EndLabel) {
      // The interval is half-open, so we have to add one to include the return
      // address of the last invoke in the range.
      End = MCBinaryExpr::CreateAdd(createImageRel32(CSE.EndLabel),
                                    MCConstantExpr::Create(1, Asm->OutContext),
                                    Asm->OutContext);
    } else {
      End = createImageRel32(EHFuncEndSym);
    }

    // These aren't really type info globals, they are actually pointers to
    // filter functions ordered by selector. The zero selector is used for
    // cleanups, so slot zero corresponds to selector 1.
    const std::vector<const GlobalValue *> &SelectorToFilter = MMI->getTypeInfos();

    // Do a parallel iteration across typeids and clause labels, skipping filter
    // clauses.
    size_t NextClauseLabel = 0;
    for (size_t I = 0, E = LPad->TypeIds.size(); I < E; ++I) {
      // AddLandingPadInfo stores the clauses in reverse, but there is a FIXME
      // to change that.
      int Selector = LPad->TypeIds[E - I - 1];

      // Ignore C++ filter clauses in SEH.
      // FIXME: Implement cleanup clauses.
      if (!isCatchEHSelector(Selector))
        continue;

      Asm->OutStreamer.EmitValue(Begin, 4);
      Asm->OutStreamer.EmitValue(End, 4);
      if (isCatchEHSelector(Selector)) {
        assert(unsigned(Selector - 1) < SelectorToFilter.size());
        const GlobalValue *TI = SelectorToFilter[Selector - 1];
        if (TI) // Emit the filter function pointer.
          Asm->OutStreamer.EmitValue(createImageRel32(Asm->getSymbol(TI)), 4);
        else  // Otherwise, this is a "catch i8* null", or catch all.
          Asm->OutStreamer.EmitIntValue(1, 4);
      }
      MCSymbol *ClauseLabel = LPad->ClauseLabels[NextClauseLabel++];
      Asm->OutStreamer.EmitValue(createImageRel32(ClauseLabel), 4);
    }
  }
}
